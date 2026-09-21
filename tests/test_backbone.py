"""PLAN.md section 5 items 2 (model half) and 3 (padding mask), against the real backbone.

These tests load the 85M-parameter ECG-FM checkpoint once (session fixture) and run a
handful of forward passes on two short records on CPU.  Nothing here trains.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
import torch

from chagas.config import CONV_DOWNSAMPLE, default_config
from chagas.model import build_padding_mask, masked_mean, token_lengths
from chagas.preprocessing import load_record, preprocess_array

pytestmark = pytest.mark.backbone


# ---------------------------------------------------------------------------
# item 3 -- padding mask
# ---------------------------------------------------------------------------
def test_token_length_formula_matches_fairseq(shared_module):
    """``valid_len // 16`` must be exactly fairseq-signals' own conv length formula.

    ECG-FM's frontend is ``[(256, 2, 2)] * 4`` and fairseq computes
    ``floor((L - kernel)/stride + 1)`` once per layer.  For kernel = stride = 2 that is
    ``floor(L/2)``; nested four deep it is ``L // 16``.  Asserted against the method
    rather than trusting the derivation, including the awkward cases (odd lengths, a
    length that produces zero tokens).
    """
    encoder = shared_module.feature_extractor.ecg_fm_model.encoder
    lengths = torch.tensor(
        [0, 1, 15, 16, 17, 31, 100, 1000, 2500, 2825, 3650, 4999, 5000],
        dtype=torch.long,
    )
    expected = encoder._get_feat_extract_output_lengths(lengths)
    assert torch.equal(token_lengths(lengths), expected), (
        lengths.tolist(), token_lengths(lengths).tolist(), expected.tolist()
    )
    assert CONV_DOWNSAMPLE == 16


def test_masked_mean_ignores_padded_positions():
    """The pooling itself, as an exact unit test on synthetic hidden states."""
    torch.manual_seed(0)
    x = torch.randn(2, 10, 4)
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[0, 7:] = True
    pooled = masked_mean(x, mask)
    assert torch.allclose(pooled[0], x[0, :7].mean(dim=0), atol=1e-6)
    assert torch.allclose(pooled[1], x[1].mean(dim=0), atol=1e-6)
    # Garbage in the padded region must not move the answer at all.
    x2 = x.clone()
    x2[0, 7:] = 1e3
    assert torch.allclose(masked_mean(x2, mask)[0], pooled[0], atol=1e-6)


def test_build_padding_mask_is_a_prefix_mask():
    mask = build_padding_mask(torch.tensor([3, 5]), 5)
    assert mask.tolist() == [[False, False, False, True, True], [False] * 5]


def test_encoder_receives_and_honours_the_padding_mask(shared_module):
    """End to end: the encoder's token mask must match ``valid_len // 16`` exactly, and
    the pooled feature must be the masked mean of the aggregated hidden states."""
    torch.manual_seed(0)
    extractor = shared_module.feature_extractor
    x = torch.zeros(2, 12, 5000)
    x[0, :, :3650] = torch.randn(12, 3650)
    x[1] = torch.randn(12, 5000)
    valid_len = torch.tensor([3650, 5000])

    captured = {}
    base = extractor.ecg_fm_model.encoder
    original = base.forward

    def spy(**kwargs):
        out = original(**kwargs)
        captured["padding_mask"] = out["padding_mask"]
        return out

    base.forward = spy  # type: ignore[assignment]
    try:
        with torch.no_grad():
            pooled = extractor(x, valid_len)
    finally:
        base.forward = original  # type: ignore[assignment]

    token_mask = captured["padding_mask"]
    assert token_mask is not None
    valid_tokens = (~token_mask).sum(dim=1)
    assert valid_tokens.tolist() == [3650 // 16, 5000 // 16]
    assert pooled.shape == (2, extractor.embed_dim)
    assert torch.isfinite(pooled).all()

    # And the mask actually changes the pooled value for the padded record.
    with torch.no_grad():
        unmasked = extractor(x, None)
    delta = float((pooled[0] - unmasked[0]).abs().max())
    assert delta > 1e-3, "the padding mask made no difference; is it reaching the encoder?"
    print(f"\n[item 3] masked vs unmasked pooled feature, padded record: max |delta| = {delta:.4f}")


def test_padding_length_sensitivity_is_reported(shared_module):
    """How much does the amount of zero padding move the pooled feature?

    **Deviation from PLAN.md, recorded deliberately.**  The plan asked for a 7 s record
    padded to 8 s and to 10 s to give *identical* pooled features (atol 1e-5).  That is
    impossible with this backbone, and the reason is worth writing down: ECG-FM's conv
    frontend begins with ``Fp32GroupNorm(256, 256)``, i.e. a per-channel instance norm
    **over the time axis**, and neither it nor the k=128 convolutional positional
    encoding is padding-aware.  Changing the amount of trailing zero padding therefore
    changes the normalisation statistics of every channel, and the encoder's inputs move
    at valid positions too.

    The padding mask still does its job -- it removes padded tokens from attention and
    from pooling -- and this test asserts that the masked pooling is no *worse* than the
    unmasked mean, and prints both numbers.  The practical consequence is recorded in
    METHODS.md: because every record is padded to the same 5000 samples, the effect is a
    systematic (not random) function of record length.
    """
    torch.manual_seed(0)
    extractor = shared_module.feature_extractor
    core = torch.randn(12, 3500)  # 7 s at 500 Hz

    def run(total: int, use_mask: bool):
        x = torch.zeros(1, 12, total)
        x[0, :, :3500] = core
        with torch.no_grad():
            return extractor(x, torch.tensor([3500]) if use_mask else None)

    masked_delta = float((run(4000, True) - run(5000, True)).abs().max())
    unmasked_delta = float((run(4000, False) - run(5000, False)).abs().max())
    scale = float(run(5000, True).abs().mean())
    print(
        f"\n[item 3] 7 s record padded to 8 s vs 10 s: "
        f"max |delta pooled| with mask = {masked_delta:.4f}, "
        f"without mask = {unmasked_delta:.4f} (feature scale {scale:.4f})"
    )
    assert masked_delta <= unmasked_delta + 1e-6


# ---------------------------------------------------------------------------
# item 2 -- model-level resample invariance
# ---------------------------------------------------------------------------
def test_resample_invariance_of_the_logit(shared_module, real_records):
    """A rate round trip must not move the model's logit much.

    The plan's bound is ``|delta logit| < 0.25``.  There is no trained Chagas checkpoint
    to measure against at this point in the work, so this runs with the shipped backbone
    and a seeded, randomly initialised head -- which is the right conservative choice:
    an untrained head has no reason to be *less* sensitive than a trained one.  The
    measured number is printed so a later run against a trained checkpoint can be
    compared with it.
    """
    from scipy.signal import resample_poly

    cfg = default_config()
    cfg.aug.enabled = False
    records = []
    for path in real_records:
        with open(path + ".hea") as fh:
            if fh.readline().split()[2] == "400":
                records.append(path)
        if len(records) >= 2:
            break
    if not records:
        records = real_records[:2]

    worst = 0.0
    for path in records:
        signal, fs, _h = load_record(path)
        ref_sig, ref_len = preprocess_array(signal, fs, cfg.preprocess, train=False)
        demo = torch.zeros(1, 5)
        with torch.no_grad():
            ref_logit = float(
                shared_module(
                    torch.from_numpy(ref_sig).unsqueeze(0), torch.tensor([ref_len]), demo
                )["logits"]
            )
        for target in (300.0, 600.0, 1000.0):
            ratio = Fraction(target / fs).limit_denominator(1000)
            resampled = resample_poly(signal, ratio.numerator, ratio.denominator, axis=-1)
            sig, vlen = preprocess_array(resampled, target, cfg.preprocess, train=False)
            with torch.no_grad():
                logit = float(
                    shared_module(
                        torch.from_numpy(sig).unsqueeze(0), torch.tensor([vlen]), demo
                    )["logits"]
                )
            worst = max(worst, abs(logit - ref_logit))
    print(f"\n[item 2, model] worst |delta logit| over rate round trips: {worst:.5f}")
    assert worst < 0.25, f"rate round trip moved the logit by {worst:.4f}"
