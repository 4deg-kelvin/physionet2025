#!/usr/bin/env python
"""PLAN.md section 5 item 11: does one stage-2 optimiser step fit in a T4's 14.5 GiB?

Stage 2 unfreezes all 12 transformer layers of the 85M-parameter ECG-FM backbone and
trains on 16 x 12 x 5000 inputs in fp16.  This runs exactly that -- forward, backward,
AdamW step, repeated a few times so the optimiser's moment buffers are allocated -- with
the CUDA allocator capped to 14.5 GiB, and prints ``torch.cuda.max_memory_allocated()``.

If the peak exceeds the cap, rerun with ``--gradient-checkpointing`` (the automatic
fallback PLAN.md section 3.1 specifies) and put ``model.gradient_checkpointing: true``
into the shipped config.

bf16 is asserted **not** to be in use: SM 7.5 (T4, TitanRTX) has fp16 tensor cores but
no native bfloat16.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chagas.config import load_config  # noqa: E402
from chagas.losses import weighted_soft_bce  # noqa: E402
from chagas.model import ChagasModule  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="")
    ap.add_argument("--cap-gib", type=float, default=14.5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--samples", type=int, default=5000)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--accumulate", type=int, default=4)
    ap.add_argument("--gradient-checkpointing", action="store_true")
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        print("no CUDA device; this smoke test must run on a GPU node", flush=True)
        return 2

    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    torch.cuda.set_per_process_memory_fraction(min(args.cap_gib / total, 1.0), 0)
    torch.cuda.reset_peak_memory_stats(0)
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"device: {name}, sm {capability}, {total:.1f} GiB total", flush=True)
    print(f"allocator capped at {args.cap_gib} GiB (emulating the grader's T4)", flush=True)

    cfg = load_config(args.config or None)
    assert "bf16" not in str(cfg.train.precision), "bf16 is unsupported on SM 7.5"
    if args.gradient_checkpointing:
        cfg.model.gradient_checkpointing = True
    cfg.train.two_stage = True

    module = ChagasModule(cfg=cfg.to_dict(), stage=1).cuda()
    module.enter_stage_two()
    module.train()

    groups = module.param_groups()
    optimizer = torch.optim.AdamW(groups, lr=cfg.train.stage2_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    b, n = args.batch_size, args.samples
    signal = torch.randn(b, 12, n, device="cuda")
    valid_len = torch.randint(int(0.6 * n), n + 1, (b,), device="cuda")
    demographics = torch.randn(b, 5, device="cuda")
    targets = (torch.rand(b, device="cuda") > 0.8).float()

    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"trainable parameters in stage 2: {trainable / 1e6:.1f} M", flush=True)
    print(
        f"config: batch {b} x {n} samples, accumulate {args.accumulate}, "
        f"precision {cfg.train.precision}, "
        f"gradient_checkpointing={cfg.model.gradient_checkpointing}",
        flush=True,
    )

    started = time.time()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        for _micro in range(args.accumulate):
            with torch.autocast("cuda", dtype=torch.float16):
                out = module(signal, valid_len, demographics)
                loss = weighted_soft_bce(out["logits"], targets, cfg.train.pos_weight)
                loss = loss / args.accumulate
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in module.parameters() if p.requires_grad], cfg.train.gradient_clip_val
        )
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(0) / 2 ** 30
        print(
            f"step {step + 1}/{args.steps}: loss {float(loss) * args.accumulate:.4f}, "
            f"peak allocated {peak:.2f} GiB",
            flush=True,
        )

    elapsed = time.time() - started
    peak = torch.cuda.max_memory_allocated(0) / 2 ** 30
    reserved = torch.cuda.max_memory_reserved(0) / 2 ** 30
    per_step = elapsed / max(1, args.steps)
    print("", flush=True)
    print(f"PEAK ALLOCATED : {peak:.2f} GiB   (cap {args.cap_gib} GiB)", flush=True)
    print(f"PEAK RESERVED  : {reserved:.2f} GiB", flush=True)
    print(
        f"TIME PER OPTIMISER STEP (batch {b} x {args.accumulate} = "
        f"{b * args.accumulate} effective): {per_step:.2f} s", flush=True
    )
    ok = peak < args.cap_gib
    print(f"RESULT: {'PASS' if ok else 'FAIL'} -- stage 2 "
          f"{'fits' if ok else 'does NOT fit'} in the T4 budget", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
