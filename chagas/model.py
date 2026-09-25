"""The model (PLAN.md section 3.6).

The team's core idea is preserved unchanged:

    ECG-FM backbone -> learnable aggregation over all 12 transformer layers
    -> masked mean pooling -> 768-d feature -> MLP head, with age/sex side information.

The layer outputs are captured with forward hooks because the fairseq-signals
classification wrapper's ``forward()`` returns ``res["x"].detach()``, which would block
gradients, and because ``TransformerEncoder.extract_features`` collects per-layer
results into a local list and then discards them.

What is new here relative to commit a72c51c:

* a **real token-level padding mask** built from ``valid_len`` and passed into the
  encoder, so a 7.3 s record's zero padding is excluded from attention and from the
  pooling average (audit defect D3);
* **FiLM** demographic conditioning (Perez et al., AAAI 2018) with an explicit
  age-missing indicator, replacing feature concatenation;
* an optional **input-dependent layer gate** (Nguyen et al., arXiv:2509.00102); the
  team's static softmax stays the default;
* optional **auxiliary heads** (team) and a **domain-adversarial source head**
  (Ganin et al., JMLR 2016) behind config flags, default off;
* correct train/eval semantics across the two-stage schedule, and optional activation
  checkpointing for the T4 memory budget.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn

from . import metrics as M
from .config import CONV_DOWNSAMPLE, ExperimentCfg
from .losses import (
    GradientReversal,
    hard_negative_ranking_loss,
    masked_aux_bce,
    weighted_soft_bce,
)

__all__ = [
    "ECGFMFeatureExtractor",
    "DemographicFiLM",
    "ChagasModule",
    "token_lengths",
    "build_padding_mask",
]


# ---------------------------------------------------------------------------
# padding-mask helpers
# ---------------------------------------------------------------------------
def token_lengths(valid_len: Tensor) -> Tensor:
    """Number of encoder tokens produced by ``valid_len`` raw samples.

    ECG-FM's conv frontend is ``[(256, 2, 2)] * 4`` -- four kernel-2 stride-2
    convolutions.  fairseq-signals computes ``floor((L - kernel)/stride + 1)`` once per
    layer, which for kernel = stride = 2 is ``floor(L/2)``; nested four deep that is
    exactly ``L // 16``.  ``tests/test_backbone.py`` asserts this against
    ``ECGTransformerModel._get_feat_extract_output_lengths`` itself rather than trusting
    the derivation.
    """
    return torch.div(valid_len, CONV_DOWNSAMPLE, rounding_mode="floor")


def build_padding_mask(valid_len: Tensor, total_len: int) -> Tensor:
    """Sample-level mask, ``True`` where the sample is zero padding.

    The valid region is always a prefix (see :func:`chagas.preprocessing.window_and_pad`
    and :func:`chagas.augment.crop_shift`), which is the shape fairseq-signals'
    ``get_embeddings`` assumes when it converts this to a token-level mask.
    """
    idx = torch.arange(total_len, device=valid_len.device).unsqueeze(0)
    return idx >= valid_len.reshape(-1, 1)


def masked_mean(x: Tensor, padding_mask: Optional[Tensor]) -> Tensor:
    """Mean over time of ``(B, T, C)``, ignoring positions where ``padding_mask`` is True."""
    if padding_mask is None:
        return x.mean(dim=1)
    valid = (~padding_mask).to(x.dtype).unsqueeze(-1)  # (B, T, 1)
    denom = valid.sum(dim=1).clamp(min=1.0)
    return (x * valid).sum(dim=1) / denom


# ---------------------------------------------------------------------------
# gradient checkpointing
# ---------------------------------------------------------------------------
class _CheckpointedLayer(nn.Module):
    """Wrap a ``TransformerEncoderLayer`` in activation checkpointing.

    Non-reentrant checkpointing is used so the returned tensors stay attached to the
    autograd graph, which matters here: the layer-aggregation hook is registered on
    *this* wrapper, so it sees the graph-connected output exactly once per forward.  A
    hook on the inner layer would fire twice (forward and recompute) and would capture
    the detached copy.

    Enabling this changes the module tree (``layers.i.layer.*``), so the flag is part of
    :class:`chagas.config.ModelCfg` and is saved in the checkpoint's hparams; the model
    is always rebuilt with the same value.
    """

    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, x: Tensor, **kwargs: Any):  # type: ignore[override]
        if self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                self.layer, x, use_reentrant=False, **kwargs
            )
        return self.layer(x, **kwargs)


# ---------------------------------------------------------------------------
# feature extractor
# ---------------------------------------------------------------------------
class ECGFMFeatureExtractor(nn.Module):
    """ECG-FM backbone + learnable aggregation over all 12 transformer layers.

    (team) The hook-based capture and the learnable softmax over layers are the team's
    design and are unchanged.  ECG-FM itself: McKeen et al., *ECG-FM: an open
    electrocardiogram foundation model*, JAMIA Open 2025, arXiv:2408.05178.
    """

    def __init__(
        self,
        checkpoint_path: str,
        *,
        freeze_encoder: bool = True,
        aggregation: str = "static_softmax",
        use_padding_mask: bool = True,
        gradient_checkpointing: bool = False,
        train_conv_frontend: bool = False,
    ) -> None:
        super().__init__()
        from fairseq_signals.models import build_model_from_checkpoint

        self.ecg_fm_model = build_model_from_checkpoint(str(checkpoint_path))
        self.freeze_encoder = bool(freeze_encoder)
        self.aggregation = str(aggregation)
        self.use_padding_mask = bool(use_padding_mask)
        self.train_conv_frontend = bool(train_conv_frontend)

        encoder = self.transformer_encoder
        if float(getattr(encoder, "layerdrop", 0.0)) != 0.0:
            # Layer aggregation needs all 12 layers on every forward; layerdrop would
            # silently drop some and the hook count check below would fire mid-epoch.
            raise ValueError(
                f"encoder_layerdrop must be 0.0 for layer aggregation, got {encoder.layerdrop}"
            )

        if gradient_checkpointing:
            encoder.layers = nn.ModuleList(
                [_CheckpointedLayer(layer) for layer in encoder.layers]
            )
        self.gradient_checkpointing = bool(gradient_checkpointing)

        self.num_layers = len(encoder.layers)
        self.embed_dim = int(self.ecg_fm_model.encoder.cfg.encoder_embed_dim)

        # Static learnable softmax over layers; zeros -> uniform. (team)
        self.layer_weights = nn.Parameter(torch.zeros(self.num_layers))
        if self.aggregation == "pma":
            # Input-dependent gate, adapted from Nguyen et al., "Exploiting a
            # Mixture-of-Layers in an ECG Foundation Model", arXiv:2509.00102: a learned
            # seed query scores each layer's pooled summary, so the mixture can differ
            # per record instead of being one global vector.
            self.gate_proj = nn.Linear(self.embed_dim, self.embed_dim)
            self.gate_seed = nn.Parameter(torch.randn(self.embed_dim) * 0.02)

        # Hook storage, keyed by layer index so a missing layer cannot silently shift
        # the mapping between layer_weights[i] and the layer it weights. (team)
        self._layer_outputs: Dict[int, Tensor] = {}
        self._hooks: List[Any] = []
        for i, layer in enumerate(encoder.layers):
            self._hooks.append(layer.register_forward_hook(self._make_hook(i)))

        self.set_frozen(self.freeze_encoder)

    # -- structure ---------------------------------------------------------
    @property
    def transformer_encoder(self) -> nn.Module:
        """The 12-layer ``TransformerEncoder``, reached through its real owner.

        Exposed as a property, not an attribute: assigning an already-registered
        submodule to a second attribute name makes ``state_dict()`` emit the whole ~85M
        parameter stack twice. (team)
        """
        return self.ecg_fm_model.encoder.encoder

    @property
    def transformer_layers(self) -> nn.ModuleList:
        return self.transformer_encoder.layers

    def _make_hook(self, layer_idx: int):
        def hook_fn(module: nn.Module, inputs: Any, output: Any) -> None:
            # TransformerEncoderLayer returns (x, (attn, layer_result)); x is (T, B, C).
            self._layer_outputs[layer_idx] = output[0]
        return hook_fn

    # -- freezing ----------------------------------------------------------
    def set_frozen(self, frozen: bool) -> None:
        """Freeze or unfreeze the backbone.

        Stage 2 unfreezes the **12 transformer layers only**.  The conv frontend stays
        out of the graph because the checkpoint ships ``feature_grad_mult = 0.0``, which
        runs it under ``torch.no_grad()``; ``train_conv_frontend=True`` overrides that
        (PLAN.md section 3.1).  The classification wrapper's 17-class ``proj`` head is
        dead weight on this path and is never trained.
        """
        self.freeze_encoder = bool(frozen)
        for p in self.ecg_fm_model.parameters():
            p.requires_grad = False
        if not frozen:
            for p in self.transformer_encoder.parameters():
                p.requires_grad = True
            if self.train_conv_frontend:
                for p in self.ecg_fm_model.encoder.feature_extractor.parameters():
                    p.requires_grad = True
                self.ecg_fm_model.encoder.feature_grad_mult = 1.0
        self.ecg_fm_model.train(not frozen and self.training)

    def unfreeze(self) -> None:
        self.set_frozen(False)

    def train(self, mode: bool = True) -> "ECGFMFeatureExtractor":
        """Keep a *frozen* encoder in eval mode, and let an unfrozen one train.

        ``nn.Module.train()`` is recursive, so Lightning calling ``model.train()`` each
        epoch would otherwise re-enable the frozen encoder's dropout and make training
        see noisy embeddings that inference never sees.  Once stage 2 unfreezes the
        backbone the override must get out of the way, so the checkpoint's
        ``activation_dropout = 0.1`` actually regularises (``encoder_layerdrop`` is 0.0
        and is asserted at construction, so all 12 layers still run every forward).
        """
        super().train(mode)
        if self.freeze_encoder:
            self.ecg_fm_model.eval()
        return self

    # -- forward -----------------------------------------------------------
    def _aggregate(self, padding_mask: Optional[Tensor]) -> Tensor:
        """Combine the captured per-layer outputs into one ``(B, T, C)`` tensor."""
        if len(self._layer_outputs) != self.num_layers:
            # Only reachable if a layer was skipped.  NotImplementedError, not
            # RuntimeError: team_code.run_model degrades a generic exception to
            # (None, None) -- which scores 0 for every record -- but deliberately
            # re-raises NotImplementedError as "do not emit predictions". (team)
            raise NotImplementedError(
                f"Captured {len(self._layer_outputs)} layer outputs but expected "
                f"{self.num_layers}; layer aggregation requires every layer to run."
            )
        if self.aggregation == "pma":
            # (L, T, B, C) -> (L, B, T, C).  Only the input-dependent gate needs every
            # layer materialised at once.
            stacked = torch.stack(
                [self._layer_outputs[i] for i in range(self.num_layers)], dim=0
            ).transpose(1, 2)
            summaries = torch.stack(
                [masked_mean(stacked[i], padding_mask) for i in range(self.num_layers)],
                dim=1,
            )  # (B, L, C)
            scores = self.gate_proj(summaries) @ self.gate_seed / math.sqrt(self.embed_dim)
            weights = torch.softmax(scores, dim=1)  # (B, L)
            return torch.einsum("lbtc,bl->btc", stacked, weights.to(stacked.dtype))

        # Static softmax over layers (team).  Accumulated in a loop rather than stacked:
        # the per-layer activations are already retained by autograd, and stacking would
        # add a second full copy of all 12 of them to peak memory during stage 2.
        weights = torch.softmax(self.layer_weights, dim=0)
        aggregated = self._layer_outputs[0] * weights[0].to(self._layer_outputs[0].dtype)
        for i in range(1, self.num_layers):
            aggregated = aggregated + weights[i].to(aggregated.dtype) * self._layer_outputs[i]
        return aggregated.transpose(0, 1)  # (T, B, C) -> (B, T, C)

    def forward(self, x: Tensor, valid_len: Optional[Tensor] = None) -> Tensor:
        """``(B, 12, T)`` -> ``(B, embed_dim)`` pooled feature."""
        self._layer_outputs = {}

        padding_mask = None
        if self.use_padding_mask and valid_len is not None:
            padding_mask = build_padding_mask(
                valid_len.to(device=x.device, dtype=torch.long), x.shape[-1]
            )
            if not bool(padding_mask.any()):
                padding_mask = None

        # Call ECGTransformerModel directly, bypassing the classification wrapper's
        # forward(), which detaches the encoder output. (team)
        base_encoder = self.ecg_fm_model.encoder
        if self.freeze_encoder:
            with torch.no_grad():
                out = base_encoder(source=x, padding_mask=padding_mask)
        else:
            out = base_encoder(source=x, padding_mask=padding_mask)

        token_mask = out.get("padding_mask", None)  # (B, T') bool, True = pad
        aggregated = self._aggregate(token_mask)
        self._layer_outputs = {}  # drop 12 activation tensors between steps
        return masked_mean(aggregated, token_mask)

    def layer_weight_values(self) -> np.ndarray:
        """The learned softmax over layers, for logging (the PMA gate has no static one)."""
        return torch.softmax(self.layer_weights.detach().float(), dim=0).cpu().numpy()


# ---------------------------------------------------------------------------
# demographics
# ---------------------------------------------------------------------------
class DemographicFiLM(nn.Module):
    """Feature-wise Linear Modulation of the pooled ECG feature by age and sex.

    Perez, Strub, de Vries, Dumoulin & Courville, *FiLM: Visual Reasoning with a General
    Conditioning Layer*, AAAI 2018.  Both the CinC 2025 winner (Van Santvliet et al.)
    and the Physiol. Meas. 2026 model used FiLM instead of concatenation; the latter
    reports +0.022.

    The output layer is zero-initialised so ``gamma = 1`` and ``beta = 0`` at step 0 and
    training starts from the identity -- the conditioner can only earn its influence.
    """

    def __init__(self, n_features: int, dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden)
        self.fc2 = nn.Linear(hidden, 2 * dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, feature: Tensor, demographics: Tensor) -> Tensor:
        gamma, beta = self.fc2(F.relu(self.fc1(demographics.to(feature.dtype)))).chunk(2, dim=-1)
        return feature * (1.0 + gamma) + beta


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------
class ChagasModule(pl.LightningModule):
    """Lightning wrapper: feature extractor + FiLM + heads + the two-stage schedule."""

    def __init__(
        self,
        cfg: Optional[Dict[str, Any]] = None,
        checkpoint_override: Optional[str] = None,
        stage: int = 1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        exp = ExperimentCfg.from_dict(cfg or {})
        self.cfg = exp
        self.stage = int(stage)

        backbone_path = checkpoint_override or str(exp.checkpoint_path)
        self.feature_extractor = ECGFMFeatureExtractor(
            backbone_path,
            # Stage 1 trains the aggregation weights, FiLM and the head on a frozen
            # backbone; stage 2 unfreezes the 12 transformer layers.  ``two_stage:
            # false`` simply never reaches stage 2 (the "ship stage-1-only" fallback of
            # PLAN.md section 6).
            freeze_encoder=(self.stage == 1),
            aggregation=exp.model.aggregation,
            use_padding_mask=exp.model.use_padding_mask,
            gradient_checkpointing=exp.model.gradient_checkpointing,
            train_conv_frontend=exp.model.train_conv_frontend,
        )
        dim = self.feature_extractor.embed_dim
        n_demo = 5  # [age/100, age_missing, sex_f, sex_m, sex_unknown]

        self.film: Optional[DemographicFiLM] = None
        trunk_in = dim
        if exp.model.demographics == "film":
            self.film = DemographicFiLM(n_demo, dim, exp.model.film_hidden)
        elif exp.model.demographics == "concat":
            trunk_in = dim + n_demo  # the team's legacy path, kept for the ablation

        # Head: Dropout -> Linear(dim, 256) -> ReLU -> Dropout -> Linear(256, 1). (team)
        self.trunk = nn.Sequential(
            nn.Dropout(exp.model.dropout),
            nn.Linear(trunk_in, exp.model.hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(exp.model.dropout), nn.Linear(exp.model.hidden_dim, 1)
        )

        self.n_aux = len(exp.labels.aux_columns) if exp.model.aux_heads else 0
        self.aux_head = (
            nn.Linear(exp.model.hidden_dim, self.n_aux) if self.n_aux else None
        )
        self.grl: Optional[GradientReversal] = None
        self.domain_head: Optional[nn.Module] = None
        if exp.model.grl:
            self.grl = GradientReversal(0.0)
            self.domain_head = nn.Sequential(
                nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, 3)
            )

        # Decision threshold, persisted with the weights so inference cannot drift from
        # the value the validation split produced (audit defect D9).
        self.register_buffer("threshold", torch.tensor(0.5, dtype=torch.float32))

        self._val_probs: List[np.ndarray] = []
        self._val_labels: List[np.ndarray] = []
        self._val_losses: List[float] = []
        self.last_val_probs: Optional[np.ndarray] = None
        # One dict per validation epoch (stage, epoch, metrics); summarised into
        # chagas_manifest.json by training.py so a finished run is self-describing.
        self.val_history: list = []

    # -- forward -----------------------------------------------------------
    def forward(
        self,
        signal: Tensor,
        valid_len: Optional[Tensor] = None,
        demographics: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        pooled = self.feature_extractor(signal, valid_len)
        feature = pooled
        if self.film is not None and demographics is not None:
            feature = self.film(pooled, demographics)
        elif self.cfg.model.demographics == "concat" and demographics is not None:
            feature = torch.cat([pooled, demographics.to(pooled.dtype)], dim=1)
        hidden = self.trunk(feature)
        out: Dict[str, Tensor] = {"logits": self.head(hidden).squeeze(-1)}
        if self.aux_head is not None:
            out["aux_logits"] = self.aux_head(hidden)
        if self.domain_head is not None and self.grl is not None:
            out["domain_logits"] = self.domain_head(self.grl(pooled))
        return out

    def predict_proba(
        self,
        signal: Tensor,
        valid_len: Optional[Tensor] = None,
        demographics: Optional[Tensor] = None,
    ) -> Tensor:
        return torch.sigmoid(self(signal, valid_len, demographics)["logits"])

    # -- loss ---------------------------------------------------------------
    def _compute_loss(
        self, out: Dict[str, Tensor], batch: Dict[str, Tensor]
    ) -> Tuple[Tensor, Dict[str, float]]:
        train_cfg = self.cfg.train
        logits = out["logits"]
        soft = batch["label_soft"].to(logits.dtype)
        hard = batch["label_hard"].to(logits.dtype)
        loss = weighted_soft_bce(logits, soft, train_cfg.pos_weight)
        parts = {"bce": float(loss.detach())}

        if train_cfg.ranking_loss:
            ranking = hard_negative_ranking_loss(
                logits, soft, hard, train_cfg.ranking_margin, train_cfg.ranking_top_frac
            )
            loss = loss + train_cfg.ranking_weight * ranking
            parts["ranking"] = float(ranking.detach())

        if "aux_logits" in out and "aux" in batch:
            aux = masked_aux_bce(out["aux_logits"], batch["aux"], batch["aux_mask"])
            loss = loss + self.cfg.model.aux_weight * aux
            parts["aux"] = float(aux.detach())

        if "domain_logits" in out and "source_id" in batch:
            domain = F.cross_entropy(out["domain_logits"], batch["source_id"].long())
            loss = loss + self.cfg.model.grl_weight * domain
            parts["domain"] = float(domain.detach())

        return loss, parts

    def _unpack(self, batch: Any) -> Optional[Dict[str, Tensor]]:
        """``collate`` returns ``None`` when every record in the batch failed to load."""
        if batch is None or batch["signal"].numel() == 0:
            return None
        return batch

    # -- steps --------------------------------------------------------------
    def training_step(self, batch: Any, batch_idx: int) -> Optional[Tensor]:
        data = self._unpack(batch)
        if data is None:
            return None
        self._update_grl_lambda()
        out = self(data["signal"], data.get("valid_len"), data.get("demographics"))
        loss, parts = self._compute_loss(out, data)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=data["signal"].shape[0])
        for name, value in parts.items():
            self.log(f"train_{name}", value, on_step=False, on_epoch=True, batch_size=data["signal"].shape[0])
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        data = self._unpack(batch)
        if data is None:
            return
        out = self(data["signal"], data.get("valid_len"), data.get("demographics"))
        loss, _ = self._compute_loss(out, data)
        self._val_losses.append(float(loss.detach()))
        # Metrics take sigmoid probabilities and HARD labels (audit defect D10).
        self._val_probs.append(torch.sigmoid(out["logits"].detach().float()).cpu().numpy())
        self._val_labels.append(data["label_hard"].detach().float().cpu().numpy())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=data["signal"].shape[0])

    def on_validation_epoch_start(self) -> None:
        self._val_probs.clear()
        self._val_labels.clear()
        self._val_losses.clear()

    def on_validation_epoch_end(self) -> None:
        if not self._val_probs:
            return
        probs = np.concatenate([p.reshape(-1) for p in self._val_probs])
        labels = np.concatenate([l.reshape(-1) for l in self._val_labels])
        self.last_val_probs = probs
        challenge = M.challenge_score(labels, probs)
        auroc, auprc = M.compute_auc(labels, probs)
        n = int(probs.size)
        self.log("val_challenge", float(np.nan_to_num(challenge, nan=0.0)), prog_bar=True, batch_size=n)
        self.log("val_auroc", float(np.nan_to_num(auroc, nan=0.5)), prog_bar=True, batch_size=n)
        self.log("val_auprc", float(np.nan_to_num(auprc, nan=0.0)), batch_size=n)
        if not self.trainer.sanity_checking:
            self.update_threshold(probs)
            record = {
                "stage": int(getattr(self, "stage", 0) or 0),
                "epoch": int(self.current_epoch),
                "val_challenge": float(np.nan_to_num(challenge, nan=0.0)),
                "val_auroc": float(np.nan_to_num(auroc, nan=0.5)),
                "val_auprc": float(np.nan_to_num(auprc, nan=0.0)),
                "n": n,
            }
            self.val_history.append(record)
            # One line per validation epoch. With the progress bar disabled this is the
            # only per-epoch progress signal in a SLURM log (DECISIONS.md, 2026-09-22).
            print(
                f"[chagas] stage {record['stage']} epoch {record['epoch']} "
                f"val_challenge={record['val_challenge']:.4f} "
                f"val_auroc={record['val_auroc']:.4f} val_auprc={record['val_auprc']:.4f} n={n}",
                flush=True,
            )

    def update_threshold(self, probs: np.ndarray) -> float:
        """Set the decision threshold to the ``threshold_quantile`` of validation scores.

        The metric's operating point is the top 5% of the ranked cohort, so the binary
        column should flag roughly 5% of records -- not the 50% a hard 0.5 threshold
        produces on a model trained with heavy positive re-weighting (audit defect D9).
        """
        if probs.size == 0:
            return float(self.threshold)
        value = float(np.quantile(probs, self.cfg.train.threshold_quantile))
        self.threshold.fill_(value)
        return value

    # -- domain-adversarial ramp -------------------------------------------
    def _update_grl_lambda(self) -> None:
        if self.grl is None:
            return
        total = max(1, int(getattr(self.trainer, "estimated_stepping_batches", 1) or 1))
        progress = min(1.0, float(self.global_step) / float(total))
        # Ganin et al. 2016's schedule, rescaled to lambda_max.
        self.grl.lambd = self.cfg.model.grl_lambda_max * (
            2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
        )

    # -- serialisation ------------------------------------------------------
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Carry the calibrated threshold, the learned layer weights and the config.

        ``threshold`` is also a buffer (so it is in ``state_dict``); this block makes it
        readable without instantiating the 85M-parameter backbone, which the evaluation
        scripts want.
        """
        checkpoint["chagas"] = {
            "version": 2,
            "config": self.cfg.to_dict(),
            "threshold": float(self.threshold),
            "layer_weights": self.feature_extractor.layer_weight_values().tolist(),
            "stage": int(self.stage),
        }

    # -- stage switching ----------------------------------------------------
    def enter_stage_two(self) -> None:
        """Unfreeze the 12 transformer layers for stage 2 (PLAN.md section 3.1)."""
        self.stage = 2
        self.hparams["stage"] = 2
        self.feature_extractor.unfreeze()

    # -- optimiser ----------------------------------------------------------
    def param_groups(self) -> List[Dict[str, Any]]:
        """Parameter groups for the current stage.

        Stage 1 -- one group: aggregation weights, FiLM and the head at ``stage1_lr``.

        Stage 2 -- layer-wise LR decay (Howard & Ruder 2018 ULMFiT; Clark et al. 2020
        ELECTRA): with ``n = 12`` layers, transformer layer ``k`` (1-based) gets
        ``stage2_lr * llrd ** (n - k)``, so the top layer gets the base rate and layer 1
        gets ``base * 0.85**11``.  The head, FiLM and layer weights get
        ``stage2_head_lr``.  Everything still frozen (the conv frontend, the wrapper's
        dead ``proj``) is excluded, because AdamW would otherwise build state for
        parameters that never receive a gradient.
        """
        train_cfg = self.cfg.train
        no_decay_suffixes = ("bias", "layer_weights", "gate_seed")

        def split(params: Sequence[Tuple[str, nn.Parameter]], lr: float) -> List[Dict[str, Any]]:
            decay, no_decay = [], []
            for name, p in params:
                if not p.requires_grad:
                    continue
                if p.ndim <= 1 or name.endswith(no_decay_suffixes):
                    no_decay.append(p)
                else:
                    decay.append(p)
            groups = []
            if decay:
                groups.append({"params": decay, "lr": lr, "weight_decay": train_cfg.weight_decay})
            if no_decay:
                groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
            return groups

        named = list(self.named_parameters())
        encoder_prefix = "feature_extractor.ecg_fm_model."
        head_named = [(n, p) for n, p in named if not n.startswith(encoder_prefix)]

        if self.stage == 1 or not train_cfg.two_stage:
            return split(head_named, train_cfg.stage1_lr)

        groups = split(head_named, train_cfg.stage2_head_lr)
        layers = self.feature_extractor.transformer_layers
        n_layers = len(layers)
        for k, layer in enumerate(layers, start=1):
            lr = train_cfg.stage2_lr * (train_cfg.llrd ** (n_layers - k))
            groups.extend(split(list(layer.named_parameters()), lr))
        # Anything else inside the backbone that was explicitly unfrozen (the conv
        # frontend under train_conv_frontend) gets the lowest layer's rate.
        layer_ids = {id(p) for layer in layers for p in layer.parameters()}
        rest = [
            (n, p)
            for n, p in named
            if n.startswith(encoder_prefix) and id(p) not in layer_ids
        ]
        groups.extend(split(rest, train_cfg.stage2_lr * (train_cfg.llrd ** n_layers)))
        return groups

    def configure_optimizers(self) -> Dict[str, Any]:
        train_cfg = self.cfg.train
        groups = self.param_groups()
        if not groups:
            raise RuntimeError("no trainable parameters; check freeze/unfreeze state")
        optimizer = torch.optim.AdamW(groups, lr=train_cfg.stage1_lr)

        total_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0) or 0)
        if total_steps <= 1:
            return {"optimizer": optimizer}
        warmup = max(1, int(round(train_cfg.warmup_frac * total_steps)))
        warmup = min(warmup, max(1, total_steps - 1))

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return float(step + 1) / float(warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
