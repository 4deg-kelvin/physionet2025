import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import scipy.io
from scipy.signal import resample
import os
import math
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy, AUROC
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
import neurokit2 as nk
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

# import model_checkpoint
import time
from sklearn.model_selection import train_test_split, StratifiedKFold

# --- Standard Imports ---
import helper_code
import utils
from dataloader import ECGDataset, collate_fn_skip_none, ECGDataModule
from torch.utils.data import DataLoader
import torch.optim as optim
import custom_helper_code

class TimeLimitCallback(pl.Callback):
    """Stop training when wall-clock time approaches a limit."""
    def __init__(self, max_hours=71.0):
        super().__init__()
        self.max_seconds = max_hours * 3600
        self.start_time = None

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        print(f"[TimeLimitCallback] Training started. Will stop after {self.max_seconds/3600:.1f} hours.")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        elapsed = time.time() - self.start_time
        if elapsed >= self.max_seconds:
            hours = elapsed / 3600
            print(f"\n[TimeLimitCallback] Time limit reached ({hours:.2f}h). Stopping training.")
            trainer.should_stop = True

# --- 1. Model Components ---

import torch
import torch.nn as nn
from fairseq_signals.models import build_model_from_checkpoint
from fairseq_signals.models.classification.ecg_transformer_classifier import ECGTransformerClassificationModel
import pytorch_lightning as pl
import shutil

# class TPRMaxLoss(nn.Module):
#     def __init__(self, fraction_capacity=0.05, temperature=1.0):
#         super().__init__()
#         self.fraction_capacity = fraction_capacity
#         self.temperature = temperature
        
#     def forward(self, outputs, labels):
#         batch_size = outputs.size(0)
#         k = max(1, int(self.fraction_capacity * batch_size))
        
#         # Soft top-k selection using temperature-scaled softmax
#         scaled_outputs = outputs.squeeze() / self.temperature
#         weights = torch.softmax(scaled_outputs, dim=0)
        
#         # Approximate TPR: weighted sum of positive labels
#         total_positives = torch.sum(labels)
#         if total_positives == 0:
#             return torch.tensor(0.0, requires_grad=True)
        
#         # Weight the top predictions more heavily
#         top_k_weights = torch.zeros_like(weights)
#         _, top_k_indices = torch.topk(outputs.squeeze(), k)
#         top_k_weights[top_k_indices] = 1.0
        
#         captured_positives = torch.sum(top_k_weights * labels)
#         tpr = captured_positives / total_positives
        
#         # Maximize TPR (minimize negative TPR)
#         return -tpr
    
# class TopKFocalLoss(nn.Module):
#     def __init__(self, fraction_capacity=0.05, alpha=0.25, gamma=2.0, top_k_weight=5.0):
#         super().__init__()
#         self.fraction_capacity = fraction_capacity
#         self.alpha = alpha
#         self.gamma = gamma
#         self.top_k_weight = top_k_weight
        
#     def forward(self, outputs, labels):
#         batch_size = outputs.size(0)
#         k = max(1, int(self.fraction_capacity * batch_size))
        
#         # Standard focal loss
#         ce_loss = nn.functional.binary_cross_entropy_with_logits(
#             outputs.squeeze(), labels.float(), reduction='none'
#         )
#         pt = torch.exp(-ce_loss)
#         focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
#         # Get top-k predictions and weight them more heavily
#         _, top_k_indices = torch.topk(outputs.squeeze(), k)
#         weights = torch.ones_like(focal_loss)
#         weights[top_k_indices] *= self.top_k_weight
        
#         return torch.mean(focal_loss * weights)
    

# class PairwiseHingeLoss(nn.Module):
#     def __init__(self, margin=1.0):
#         """
#         Initializes the Pairwise Hinge Loss module.
        
#         Args:
#             margin (float): The desired gap between positive and negative scores.
#         """
#         super(PairwiseHingeLoss, self).__init__()
#         self.margin = margin

#     def forward(self, y_pred, y_true):
#         """
#         Calculates the pairwise ranking loss.

#         Args:
#             y_pred (torch.Tensor): Model predictions/scores. Shape: (batch_size,)
#             y_true (torch.Tensor): Ground truth labels (0s and 1s). Shape: (batch_size,)

#         Returns:
#             torch.Tensor: A scalar loss value.
#         """
#         # Find indices of positive (1) and negative (0) samples
#         positive_indices = torch.where(y_true == 1)[0]
#         negative_indices = torch.where(y_true == 0)[0]
        
#         # If there are no positive or no negative samples in the batch, loss is 0
#         if len(positive_indices) == 0 or len(negative_indices) == 0:
#             return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

#         # Randomly sample one positive and one negative index
#         # This makes the process stochastic and efficient
#         rand_pos_idx = positive_indices[torch.randint(len(positive_indices), (1,))]
#         rand_neg_idx = negative_indices[torch.randint(len(negative_indices), (1,))]
        
#         # Get the scores for the sampled pair
#         score_positive = y_pred[rand_pos_idx]
#         score_negative = y_pred[rand_neg_idx]
        
#         # Calculate the hinge loss for the pair
#         loss = torch.clamp(self.margin - (score_positive - score_negative), min=0.0)
        
#         return loss


# class ListNetLoss(nn.Module):
#     def __init__(self):
#         super(ListNetLoss, self).__init__()

#     def forward(self, y_pred, y_true):
#         """
#         Calculates the ListNet loss.

#         Args:
#             y_pred (torch.Tensor): Model predictions/scores. Shape: (batch_size,)
#             y_true (torch.Tensor): Ground truth labels (0s and 1s). Shape: (batch_size,)

#         Returns:
#             torch.Tensor: A scalar loss value.
#         """
#         # Create probability distributions from scores and labels using softmax
#         pred_probs = F.softmax(y_pred, dim=0)
#         true_probs = F.softmax(y_true, dim=0)
        
#         # Add a small epsilon to true_probs to avoid log(0) which is -inf
#         true_probs = true_probs + 1e-9
        
#         # Compute the KL Divergence between the two distributions
#         # This is equivalent to cross-entropy: -sum(P_true * log(P_pred))
#         loss = -torch.sum(true_probs * torch.log(pred_probs))
        
#         return loss
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

# FINAL CORRECTED GradCAM CLASS
class GradCAM:
    """
    Grad-CAM for visualizing model attention on sequential data like ECGs.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            # ✅ CORRECTED: Do NOT detach the activations.
            # This preserves the computation graph for the backward pass.
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, info_tensor=None):
        """
        Generates the Class Activation Map, temporarily enabling gradients.
        """
        is_training = self.model.training
        feature_extractor_frozen = self.model.feature_extractor.freeze_encoder
        
        self.model.eval()
        self.model.feature_extractor.unfreeze()
        for param in self.model.feature_extractor.parameters():
            param.requires_grad = True
        
        self.model.zero_grad()

        try:
            logits = self.model(input_tensor, info_tensor)
            logits.backward()

            if self.activations is None or self.gradients is None:
                raise RuntimeError("Failed to capture activations or gradients.")

            pooled_gradients = torch.mean(self.gradients, dim=1, keepdim=True)
            self.activations = self.activations.detach() # Detach here, after gradients are calculated
            weighted_activations = self.activations * pooled_gradients
            
            cam = torch.mean(weighted_activations, dim=2).squeeze(0)
            cam = F.relu(cam)
            
            cam = F.interpolate(
                cam.unsqueeze(0).unsqueeze(0),
                size=input_tensor.shape[2],
                mode='linear',
                align_corners=False
            ).squeeze().cpu().numpy()
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
            
            return cam

        finally:
            self.model.zero_grad()
            if feature_extractor_frozen:
                self.model.feature_extractor.freeze_encoder = True
                for param in self.model.feature_extractor.parameters():
                    param.requires_grad = False
            self.model.train(is_training)

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()
def plot_grad_cam(ecg_signal, cam, prediction_score, lead_names=None):
    """
    Plots the 12-lead ECG with Grad-CAM overlay.
    
    Args:
        ecg_signal (np.array): The ECG signal of shape (12, length).
        cam (np.array): The generated CAM of shape (length,).
        prediction_score (float): The model's output score for this ECG.
        lead_names (list, optional): List of names for the 12 leads.
    """
    if lead_names is None:
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    num_leads, signal_len = ecg_signal.shape
    time_axis = np.arange(signal_len)
    
    fig, axes = plt.subplots(num_leads, 1, figsize=(18, 12), sharex=True)
    fig.suptitle(f'Grad-CAM Visualization (Prediction Score: {prediction_score:.4f})', fontsize=16)
    
    # Create a colormap
    cmap = plt.get_cmap('jet')
    norm = mcolors.Normalize(vmin=0, vmax=1)
    
    for i in range(num_leads):
        axes[i].plot(time_axis, ecg_signal[i], color='black', linewidth=0.8)
        axes[i].set_ylabel(lead_names[i], rotation=0, labelpad=20, va='center')
        axes[i].tick_params(axis='y', left=False, labelleft=False)
        axes[i].grid(True, linestyle='--', alpha=0.5)
        
        # Overlay the heatmap
        im = axes[i].imshow(
            cam[np.newaxis, :],
            aspect='auto',
            cmap=cmap,
            norm=norm,
            extent=(0, signal_len, np.min(ecg_signal[i]), np.max(ecg_signal[i]))
        )
    
    # Add a shared colorbar
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation='vertical', label='Attention Intensity')
    
    axes[-1].set_xlabel('Time Steps')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.savefig('grad_cam_ecg.png', dpi=300)
    
class PercentileRankingLoss(nn.Module):
    """
    Maintains running percentile estimates using exponential moving average.
    More memory efficient than GlobalTopKLoss.
    """
    def __init__(self, top_percent=0.05, margin=1.0, momentum=0.99, warmup_steps=100):
        super().__init__()
        self.top_percent = top_percent
        self.margin = margin
        self.momentum = momentum
        self.warmup_steps = warmup_steps
        
        # Running percentile estimates
        self.register_buffer('running_threshold', torch.tensor(0.0))
        self.register_buffer('running_pos_mean', torch.tensor(0.0))
        self.register_buffer('running_neg_mean', torch.tensor(0.0))
        self.register_buffer('step', torch.tensor(0))
        
        # Percentile tracker (simple histogram approach)
        self.register_buffer('score_bins', torch.linspace(-10, 10, 100))
        self.register_buffer('score_counts', torch.zeros(100))
        
    def update_statistics(self, scores):
        """Update running estimates of score distribution"""
        with torch.no_grad():
            self.step += 1
            
            # Update histogram
            hist = torch.histc(scores, bins=100, min=-10, max=10)
            if self.step < self.warmup_steps:
                # During warmup, average the histograms
                self.score_counts = ((self.step - 1) * self.score_counts + hist) / self.step
            else:
                # After warmup, use EMA
                self.score_counts = self.momentum * self.score_counts + (1 - self.momentum) * hist
            
            # Compute percentile from histogram
            cumsum = torch.cumsum(self.score_counts, dim=0)
            total = cumsum[-1]
            if total > 0:
                percentile_idx = (1 - self.top_percent) * total
                idx = torch.searchsorted(cumsum, percentile_idx)
                idx = min(idx, len(self.score_bins) - 1)
                self.running_threshold = self.score_bins[idx]
    
    def forward(self, logits, labels):
        logits = logits.squeeze(-1)
        labels = labels.squeeze(-1)
        
        # Update distribution statistics
        self.update_statistics(logits.detach())
        
        pos_mask = labels == 1
        neg_mask = labels == 0
        
        if not pos_mask.any() or not neg_mask.any():
            return F.binary_cross_entropy_with_logits(logits, labels.float())
        
        pos_scores = logits[pos_mask]
        neg_scores = logits[neg_mask]
        
        # Use running threshold after warmup
        if self.step > self.warmup_steps and self.running_threshold != 0:
            threshold = self.running_threshold
        else:
            # During warmup, use batch statistics
            k = max(1, int(len(logits) * self.top_percent))
            sorted_scores, _ = torch.sort(logits, descending=True)
            threshold = sorted_scores[min(k-1, len(logits)-1)]
        
        # Ranking losses
        pos_loss = F.relu(threshold + self.margin - pos_scores).mean()
        neg_loss = F.relu(neg_scores - threshold + self.margin).mean()
        
        # Additional: maximize gap between pos and neg means
        gap_loss = F.relu(self.margin - (pos_scores.mean() - neg_scores.mean()))
        
        total_loss = pos_loss + neg_loss + 0.1 * gap_loss
        
        # BCE for stability
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        
        return 0.5 * total_loss + 0.5 * bce_loss
class TopKRankingLoss(nn.Module):
    def __init__(self, fraction_capacity=0.05, margin=1.0):
        super().__init__()
        self.fraction_capacity = fraction_capacity
        self.margin = margin
        
    def forward(self, outputs, labels):
        batch_size = outputs.size(0)
        k = max(1, int(self.fraction_capacity * batch_size))
        
        _, top_k_indices = torch.topk(outputs.squeeze(), k)
        
        loss = 0
        for i in range(batch_size):
            for j in range(batch_size):
                if labels[i] == 1 and labels[j] == 0:  
                    loss += torch.relu(self.margin - (outputs[i] - outputs[j]))
        
        # Additional penalty for missing positives in top-k
        top_k_labels = labels[top_k_indices]
        missed_positives = torch.sum(labels) - torch.sum(top_k_labels)
        loss += missed_positives * self.margin
        
        return loss / (batch_size * batch_size)

class ChallengeScoreLoss(nn.Module):
    def __init__(self, fraction_capacity=0.05, bce_weight=0.3, ranking_weight=0.7):
        super().__init__()
        self.fraction_capacity = fraction_capacity
        self.bce_weight = bce_weight
        self.ranking_weight = ranking_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ranking_loss = TopKRankingLoss(fraction_capacity)
        
    def forward(self, outputs, labels):
        bce = self.bce_loss(outputs, labels.float())
        ranking = self.ranking_loss(outputs, labels)
        return self.bce_weight * bce + self.ranking_weight * ranking

class DAM_Momentum(nn.Module):
    """
    DAM with momentum updates for primal variables
    """
    def __init__(self, margin=1.0, momentum=0.9):
        super().__init__()
        self.margin = margin
        self.momentum = momentum
        
        # Initialize primal variables
        self.register_buffer('a', torch.zeros(1))
        self.register_buffer('b', torch.zeros(1)) 
        self.register_buffer('alpha', torch.zeros(1))
        
        # Momentum buffers
        self.register_buffer('a_momentum', torch.zeros(1))
        self.register_buffer('b_momentum', torch.zeros(1))
        self.register_buffer('alpha_momentum', torch.zeros(1))
        
    def forward(self, logits, labels, lr=0.1):
        p = labels.float().mean()
        
        pos_mask = labels == 1
        neg_mask = labels == 0
        
        # Compute gradients w.r.t. primal variables
        if pos_mask.any():
            pos_preds = logits[pos_mask]
            grad_a = 2 * (1 - p) * (self.a - pos_preds.mean().detach())
            
            # Momentum update for a
            self.a_momentum = self.momentum * self.a_momentum + (1 - self.momentum) * grad_a
            self.a = self.a - lr * self.a_momentum
            
            pos_loss = (1 - p) * ((pos_preds - self.a) ** 2).mean()
        else:
            pos_loss = 0
            
        if neg_mask.any():
            neg_preds = logits[neg_mask]
            grad_b = 2 * p * (self.b - neg_preds.mean().detach())
            
            # Momentum update for b
            self.b_momentum = self.momentum * self.b_momentum + (1 - self.momentum) * grad_b
            self.b = self.b - lr * self.b_momentum
            
            neg_loss = p * ((neg_preds - self.b) ** 2).mean()
        else:
            neg_loss = 0
        
        # Update alpha (threshold)
        grad_alpha = 2 * ((1 - p) * self.a - p * self.b).detach()
        self.alpha_momentum = self.momentum * self.alpha_momentum + (1 - self.momentum) * grad_alpha
        self.alpha = self.alpha - lr * self.alpha_momentum
        
        # Compute final loss
        loss = pos_loss + neg_loss - p * (1 - p) * self.margin ** 2
        
        return loss

class ECGFMFeatureExtractor(nn.Module):
    """
    Wrapper around ECG-FM model to extract features for downstream tasks.
    Uses layer aggregation: learnable weighted sum of all 12 transformer
    layer outputs, followed by masked mean pooling over the time dimension.
    """
    def __init__(self, checkpoint_path, freeze_encoder=True):
        super().__init__()
        self.ecg_fm_model = build_model_from_checkpoint(checkpoint_path)
        self.freeze_encoder = freeze_encoder

        # Access the 12-layer TransformerEncoder stack
        # Hierarchy: ecg_fm_model (ClassificationModel extends FinetuningModel)
        #   .encoder (ECGTransformerModel)
        #     .encoder (TransformerEncoder)
        #       .layers (nn.ModuleList of TransformerEncoderLayer)
        # NB: exposed as a property, not an attribute. Assigning an already-registered
        # submodule to a second attribute name makes state_dict() emit the entire
        # ~85M-parameter transformer stack twice (state_dict does not dedupe modules
        # the way named_parameters does), roughly doubling the saved checkpoint.
        num_layers = len(self.transformer_encoder.layers)

        # Learnable layer weights (initialized to zeros → uniform softmax)
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))

        # Storage for hook outputs (populated during forward pass), keyed by layer index
        # rather than appended, so a skipped layer cannot silently shift the mapping
        # between layer_weights[i] and the layer it is supposed to weight.
        self._layer_outputs = {}

        # Register forward hooks on each TransformerEncoderLayer
        self._hooks = []
        for i, layer in enumerate(self.transformer_encoder.layers):
            hook = layer.register_forward_hook(self._make_hook(i))
            self._hooks.append(hook)

        if self.freeze_encoder:
            self.ecg_fm_model.eval()
            for p in self.ecg_fm_model.parameters():
                p.requires_grad = False

    @property
    def transformer_encoder(self):
        """The 12-layer TransformerEncoder stack, reached through its real owner."""
        return self.ecg_fm_model.encoder.encoder

    def _make_hook(self, layer_idx):
        """Create a forward hook that captures the output of a transformer layer."""
        def hook_fn(module, input, output):
            # Each TransformerEncoderLayer returns (x, (attn, layer_result))
            # x is (T, B, C)
            x = output[0]
            self._layer_outputs[layer_idx] = x
        return hook_fn

    def train(self, mode=True):
        """Keep a frozen encoder in eval mode.

        nn.Module.train() is recursive, so Lightning calling model.train() each epoch
        would otherwise undo the .eval() set in __init__ and re-enable the encoder's
        dropout (and layerdrop, which skips whole layers and breaks the layer
        aggregation above). That would make training see noisy embeddings while
        inference sees clean ones.
        """
        super().train(mode)
        if self.freeze_encoder:
            self.ecg_fm_model.eval()
        return self

    def unfreeze(self):
        if self.freeze_encoder:
            for p in self.ecg_fm_model.parameters():
                p.requires_grad = True
            self.ecg_fm_model.train()
            self.freeze_encoder = False
            print("[ECGFMFeatureExtractor] Encoder unfrozen for fine-tuning.")

    def forward(self, x):
        """
        Args:
            x: (batch_size, 12, seq_length)
        Returns:
            (batch_size, embed_dim)
        """
        # Clear captured outputs from previous forward pass
        self._layer_outputs = {}

        # Call the base encoder directly (ECGTransformerModel), bypassing the
        # classifier's forward() which applies .detach() to encoder_out.
        # ECGTransformerModel.forward returns {"x": (B,T,C), "padding_mask": (B,T), "saliency": ...}
        base_encoder = self.ecg_fm_model.encoder  # ECGTransformerModel
        if self.freeze_encoder:
            with torch.no_grad():
                out = base_encoder(source=x)
        else:
            out = base_encoder(source=x)

        padding_mask = out.get("padding_mask", None)  # (B, T) bool or None

        # Layer aggregation: weighted sum of all layer outputs
        # Each layer output is (T, B, C) — transpose to (B, T, C)
        num_layers = self.layer_weights.numel()
        if len(self._layer_outputs) != num_layers:
            # Only reachable if a layer was skipped (encoder layerdrop while in train
            # mode). Previously this silently mis-paired weights with layers via zip().
            raise RuntimeError(
                f"Captured {len(self._layer_outputs)} layer outputs but expected "
                f"{num_layers}. The encoder is in train mode with layerdrop active; "
                f"layer aggregation requires every layer to run."
            )
        weights = torch.softmax(self.layer_weights, dim=0)
        aggregated = torch.zeros_like(self._layer_outputs[0])  # (T, B, C)
        for i in range(num_layers):
            aggregated = aggregated + weights[i] * self._layer_outputs[i]
        aggregated = aggregated.transpose(0, 1)  # (B, T, C)

        # Drop the captured references now that they are folded into `aggregated`;
        # otherwise 12 activation tensors stay alive between steps.
        self._layer_outputs = {}

        # Masked mean pooling using proper padding_mask
        if padding_mask is not None and padding_mask.any():
            # padding_mask is True for padded positions — invert for valid positions
            valid_mask = ~padding_mask  # (B, T)
            valid_mask_expanded = valid_mask.unsqueeze(-1)  # (B, T, 1)
            aggregated = aggregated * valid_mask_expanded
            lengths = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
            pooled_features = aggregated.sum(dim=1) / lengths  # (B, C)
        else:
            # No padding — simple mean over time
            pooled_features = aggregated.mean(dim=1)  # (B, C)

        return pooled_features

# class FMChagasClassifier(pl.LightningModule):
#     """
#     Chagas disease classifier using ECG-FM pretrained features.
#     """
#     def __init__(self, ecg_fm_checkpoint_path, num_classes=1, lr=1e-4, freeze_encoder=True):
#         super().__init__()
#         self.save_hyperparameters()
        
#         # ECG-FM feature extractor
#         self.feature_extractor = ECGFMFeatureExtractor(
#             ecg_fm_checkpoint_path, 
#             freeze_encoder=freeze_encoder
#         )
        
#         # Get feature dimension from ECG-FM (typically 768)
#         self.feature_dim = 768  # ECG-FM embedding dimension
        
#         # Classification head
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.1),
#             nn.Linear(self.feature_dim, 256),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(256, num_classes)
#         )
        
#         self.criterion = nn.BCEWithLogitsLoss()
        
#     def forward(self, x):
#         # Extract features using ECG-FM
#         features = self.feature_extractor(x)
#         # Classify
#         logits = self.classifier(features)
#         return logits
    
#     def _common_step(self, batch, batch_idx):
#         signals, labels = batch[0], batch[1]
#         logits = self(signals)
#         loss = self.criterion(logits, labels.float())
#         probs = torch.sigmoid(logits)
#         return loss, probs, labels
    
#     def training_step(self, batch, batch_idx):
#         loss, probs, labels = self._common_step(batch, batch_idx)
#         self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
#         return loss
    
#     def validation_step(self, batch, batch_idx):
#         loss, probs, labels = self._common_step(batch, batch_idx)
#         self.log('val_loss', loss, on_epoch=True, prog_bar=True)
#         return {'val_loss': loss, 'probs': probs, 'labels': labels}
    
#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
#         return [optimizer], [scheduler]


class FMChagasClassifier(pl.LightningModule):
    def __init__(self, ecg_fm_checkpoint_path, freeze_encoder, optimizer_hparams, info_features=0,
                 loss_type='bce', loss_top_percent=0.05, loss_margin=1.0, loss_momentum=0.99,
                 loss_warmup_steps=100, unfreeze_after_epoch=None):
        super().__init__()
        self.save_hyperparameters()
        self.feature_extractor = ECGFMFeatureExtractor(
            ecg_fm_checkpoint_path,
            freeze_encoder=freeze_encoder
        )
        
        # Get feature dimension from ECG-FM (typically 768)
        self.feature_dim = 768  # ECG-FM embedding dimension
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim + self.hparams.info_features, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        if self.hparams.loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.hparams.loss_type == 'percentile':
            self.criterion = PercentileRankingLoss(
                top_percent=self.hparams.loss_top_percent,
                margin=self.hparams.loss_margin,
                momentum=self.hparams.loss_momentum,
                warmup_steps=self.hparams.loss_warmup_steps
            )
        else:
            raise ValueError(f"Unsupported loss type: {self.hparams.loss_type}")
        
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.test_auroc = AUROC(task="binary")

        self.validation_step_outputs = []
        self.validation_step_labels = []
        self.test_step_outputs = []
        self.test_step_labels = []

        self.unfreeze_after_epoch = unfreeze_after_epoch
        self._encoder_unfrozen = False

    def forward(self, x, info=None):
        # Extract features using ECG-FM
        features = self.feature_extractor(x)
        
        if self.hparams.info_features > 0 and info is not None:
            features = torch.cat((features, info), dim=1)

        # Classify
        logits = self.classifier(features)
        return logits

    def _common_step(self, batch, batch_idx):
        # collate_fn_skip_none returns None when every record in the batch failed to
        # load. Must be checked before unpacking, or this raises TypeError and aborts
        # the whole run on a single bad batch.
        if batch is None:
            return None, None, None

        # Unpack batch, which may or may not have info features
        if self.hparams.info_features > 0:
            signals, info, labels = batch
        else:
            signals, labels = batch
            info = None
            
        # Handle cases where the batch is empty after filtering
        if signals.numel() == 0:
            return None, None, None

        logits = self(signals, info)
        
        # The loss function handles squeezing internally.
        loss = self.criterion(logits, labels)
        
        return loss, logits.squeeze(-1), labels.squeeze(-1)

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._common_step(batch, batch_idx)
        # Skip step if the batch was empty
        if loss is None:
            return None
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_acc(preds, labels.int())
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._common_step(batch, batch_idx)
        if loss is None:
            return None

        self.log('val_loss', loss, prog_bar=True)
        self.val_acc(preds, labels.int())
        self.val_auroc(preds, labels)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=True)
        self.log('val_auroc', self.val_auroc, on_epoch=True, prog_bar=True)

        self.validation_step_outputs.append(preds)
        self.validation_step_labels.append(labels)
    
    def on_validation_epoch_start(self):
        self.validation_step_outputs.clear()
        self.validation_step_labels.clear()

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking or not self.validation_step_outputs:
            return
            
        all_preds = torch.cat(self.validation_step_outputs).squeeze().cpu().numpy()
        all_labels = torch.cat(self.validation_step_labels).squeeze().cpu().numpy()

        challenge_score = utils.compute_challenge_score(all_labels, all_preds)
        self.log('val_challenge_score', challenge_score, prog_bar=True)
        
        print(f"\nEpoch {self.current_epoch}: Validation Challenge Score = {challenge_score:.4f}")

        self.validation_step_outputs.clear()
        self.validation_step_labels.clear()

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self._common_step(batch, batch_idx)
        if loss is None:
            return None

        self.log('test_loss', loss)
        self.test_acc(preds, labels.int())
        self.test_auroc(preds, labels)
        self.log('test_acc', self.test_acc, on_epoch=True, prog_bar=True)
        self.log('test_auroc', self.test_auroc, on_epoch=True, prog_bar=True)

        self.test_step_outputs.append(preds)
        self.test_step_labels.append(labels)

    def on_test_epoch_start(self):
        self.test_step_outputs.clear()
        self.test_step_labels.clear()

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            print("No test outputs were generated, skipping score calculation.")
            return
        all_preds = torch.cat(self.test_step_outputs).squeeze().cpu().numpy()
        all_labels = torch.cat(self.test_step_labels).squeeze().cpu().numpy()

        challenge_score = utils.compute_challenge_score(all_labels, all_preds)
        self.log('test_challenge_score', challenge_score, prog_bar=True)

        self.test_step_outputs.clear()
        self.test_step_labels.clear()

    def unfreeze_encoder(self):
        if not self._encoder_unfrozen:
            self.feature_extractor.unfreeze()
            self._encoder_unfrozen = True
            # Optionally adjust learning rates (keep simple here)
            print(f"[FMChagasClassifier] Encoder unfrozen at epoch {self.current_epoch}.")

    def on_train_epoch_start(self):
        # Trigger unfreeze exactly once when epoch matches.
        # A negative value means "never unfreeze" -- without the >= 0 guard, the
        # documented -1 sentinel satisfies `current_epoch >= -1` on epoch 0 and
        # unfreezes immediately, the exact opposite of what it means.
        if (self.unfreeze_after_epoch is not None and
            self.unfreeze_after_epoch >= 0 and
            self.current_epoch >= self.unfreeze_after_epoch and
            not self._encoder_unfrozen):
            self.unfreeze_encoder()

    def configure_optimizers(self):
        # Split into decay / no-decay groups. AdamW's default weight_decay=0.01 would
        # otherwise apply to layer_weights, which is initialised to zeros (= uniform
        # softmax over layers) -- decay actively pulls it back there and the learnable
        # layer aggregation never goes anywhere. Biases and norms are excluded by the
        # usual convention.
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.endswith('layer_weights') or name.endswith('.bias') or p.ndim <= 1:
                no_decay.append(p)
            else:
                decay.append(p)
        optimizer = torch.optim.AdamW(
            [
                {'params': decay, 'weight_decay': 0.01},
                {'params': no_decay, 'weight_decay': 0.0},
            ],
            lr=self.hparams.optimizer_hparams['lr']
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self.hparams.optimizer_hparams.get('warmup_steps', 0)
        lr_end = self.hparams.optimizer_hparams.get('lr_end', 1e-7)

        if warmup_steps <= 0:
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps, eta_min=lr_end
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": main_scheduler, "interval": "step"},
            }

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps
        )   
        
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=lr_end
        )

        sequential_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sequential_scheduler,
                "interval": "step",
            },
        }
# --- 4. Training Script ---

# set default loss type


if __name__ == '__main__':
    
    # add args parser for debug mode
    import argparse
    parser = argparse.ArgumentParser(description="Train ECG-FM for Chagas classification")
    parser.add_argument('--debug', action='store_true', help="Run in debug mode with a smaller dataset")
    parser.add_argument('--unfreeze_after_epoch', type=int, default=-1,
                        help="Epoch after which to unfreeze ECG-FM encoder (-1 to keep frozen).")
    parser.add_argument('--grad-cam', action='store_true', help="Run Grad-CAM visualization after training")
    # ADDED: evaluation mode args
    parser.add_argument('--evaluate', action='store_true',
                        help='Train then export validation predictions with best checkpoint; skip test phase.')
    parser.add_argument('--predictions-output', type=str, default='validation_predictions_fm.csv',
                        help='CSV path for saved validation predictions when using --evaluate.')

    args = parser.parse_args()

    if args.grad_cam:
        DATA_DIR = "../training_data/"
        MODEL_CKPT_PATH = "/sailhome/kelvinkn/scr2_juice/other_work/edwards/physionet2025/fm/foundation_model/n5v9hf1p/checkpoints/best-challenge-epoch=09-val_challenge_score=0.4371.ckpt.ckpt" # <--- UPDATE THIS PATH
        SEQ_LENGTH = utils.UNIFIED_FREQUENCY * 10
        
        # --- 2. LOAD YOUR TRAINED MODEL ---
        print(f"Loading model from {MODEL_CKPT_PATH}...")
        # NOTE: You might need to provide the hparams your model was saved with
        # if they are not automatically loaded.
        optimizer_hparams = {'lr': 1e-4, 'warmup_steps': 700, 'lr_end': 1e-7}
        model = FMChagasClassifier.load_from_checkpoint(
            checkpoint_path=MODEL_CKPT_PATH,
            ecg_fm_checkpoint_path='./ckpts/mimic_iv_ecg_finetuned.pt',
            optimizer_hparams=optimizer_hparams, 
            strict=False, 
            freeze_encoder=False
        )
        model.eval() # Set to evaluation mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print("Model loaded successfully.")

        # --- 3. GET A SAMPLE ECG ---
        # We'll grab one positive and one negative example for comparison
        all_records = custom_helper_code.find_records_abs(DATA_DIR)
        if args.debug:
            print("--- DEBUG MODE: Using a random subset of 1000 records. ---")
            np.random.shuffle(all_records)
            all_records = all_records[:10000]
        records_meta = utils.prepare_stratification(all_records)
        df = pd.DataFrame(records_meta)

        # Find a positive and a negative sample
        positive_sample_path = df[df['label'] == 1]['record'].iloc[0]
        negative_sample_path = df[df['label'] == 0]['record'].iloc[10]

        # Create a temporary dataset to load the data easily
        temp_dataset = ECGDataset(
            [positive_sample_path, negative_sample_path], 
            data_dir=DATA_DIR, 
            is_training=False,
            seq_len=SEQ_LENGTH, 
            windowing_method='entire_recording', 
            include_wide_feats=True
        )
        
        # --- 4. GENERATE & PLOT GRAD-CAM FOR EACH SAMPLE ---
    # --- 4. GENERATE & PLOT GRAD-CAM FOR EACH SAMPLE ---
        for i in range(len(temp_dataset)):
            signal, info, label = temp_dataset[i]
            
            signal_tensor = signal.unsqueeze(0).to(device)
            info_tensor = info.unsqueeze(0).to(device)

            # --- 5. IDENTIFY THE TARGET LAYER ---
            # ✅ CONFIRMED FROM YOUR PRINTOUT
            # The correct path is model -> feature_extractor -> ecg_fm_model -> encoder -> layers -> last layer
            # Print all attributes of the encoder
            print(dir(model.feature_extractor.ecg_fm_model.encoder))
            target_layer = model.feature_extractor.ecg_fm_model.encoder.encoder.layers[-1]
            
            # --- 6. RUN GRAD-CAM ---
            grad_cam = GradCAM(model=model, target_layer=target_layer)
            cam = grad_cam.generate_cam(signal_tensor, info_tensor)
            grad_cam.remove_hooks() 

            # --- 7. VISUALIZE ---
            with torch.no_grad():
                score = torch.sigmoid(model(signal_tensor, info_tensor)).item()
            
            print(f"\n--- Visualizing Sample {i+1} ---")
            print(f"Record Path: {temp_dataset.records_list[i]}")  # changed from record_paths -> records_list
            print(f"True Label: {label.item()}, Predicted Score: {score:.4f}")
            
            plot_grad_cam(signal.cpu().numpy(), cam, score)

            exit(0)

    data_folder='../training_data'
    model_folder='./ckpts'
    checkpoint_path='./ckpts/mimic_iv_ecg_finetuned.pt'
    verbose=True

    # --- Hyperparameters ---
    DATA_DIR = "../training_data/" 
    BATCH_SIZE = 32
    # SEQ_LEN = utils.WINDOW_SIZE
    NUM_LEADS = 12
    WINDOWING_METHOD = 'random'
    NUM_EPOCHS = 16
    CHECKPOINT_MONITOR_METRIC = 'val_challenge_score'
    SEQ_LENGTH = utils.UNIFIED_FREQUENCY * 10  
    LR = 1e-4
    HAS_CODE15 = True
    LOSS_TYPE = 'bce'
    UNFREEZE = -1 # 0 for immediate unfreeze, -1 to keep frozen
    N_FOLDS = 5

    # Add these values into the config dictionary
    config = {
        "data_dir": DATA_DIR,
        "batch_size": BATCH_SIZE,
        "seq_length": SEQ_LENGTH,
        "num_leads": NUM_LEADS,
        "windowing_method": WINDOWING_METHOD,
        "num_epochs": NUM_EPOCHS,
        "checkpoint_monitor_metric": CHECKPOINT_MONITOR_METRIC,
        # "precision": PRECISION,
        "lr": LR, 
        'has_code15': HAS_CODE15,
        'loss_type': LOSS_TYPE,
        'unfreeze_after_epoch': UNFREEZE,
        'n_folds': N_FOLDS
    }

    try:
        if torch.cuda.is_available():
            if torch.cuda.get_device_capability()[0] >= 8:
                print("Setting TF32 matmul precision for Ampere GPUs")
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.set_float32_matmul_precision('high')
            else:
                print("Warning: Not an Ampere GPU. Some precision settings may not be optimal.")
    except AttributeError:
        print("Warning: TF32 matmul precision setting not available. Skipping this step.")

    if not os.path.isdir(DATA_DIR):
        print(f"Error: Data directory not found at '{DATA_DIR}'")
        print("Please update the DATA_DIR variable to point to your dataset.")
        sys.exit(1)

# 2. Prepare data for fine-tuning
    all_records = custom_helper_code.find_records_abs(DATA_DIR)
    if args.debug:
        print("--- DEBUG MODE: Using a random subset of 1000 records. ---")
        np.random.shuffle(all_records)
        all_records = all_records[:10000]
    print(f"{len(all_records)} records found in the dataset.")
    records_meta = utils.prepare_stratification(all_records)
    df = pd.DataFrame(records_meta)

    # Get the percentage of the dataset from each source
    source_counts = df['source'].value_counts(normalize=True) * 100
    print("Dataset source distribution (%):")
    for source, percent in source_counts.items():
        print(f"  {source}: {percent:.2f}%")    

    
    
    # Exclude records from source 'CODE-15%'
    # df = df[df['source'] != 'CODE-15%'] 
    # --- Exclude records from code15_label_issues.csv ---
    # try:
    #     issues_df = pd.read_csv('code15_label_issues.csv')
    #     # Extract stem (filename without extension) from the full path for exclusion list
    #     stems_to_exclude = set(issues_df['record_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0]))
        
    #     # Create a new 'base_record' column with the stem for matching
    #     df['base_record'] = df['record'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
        
    #     initial_count = len(df)
    #     # Filter out the records using the new column
    #     df = df[~df['base_record'].isin(stems_to_exclude)]
    #     final_count = len(df)
        
    #     print(f"Excluded {initial_count - final_count} records based on 'code15_label_issues.csv'.")
    #     # Drop the temporary 'base_record' column
    #     df = df.drop(columns=['base_record'])
        
    # except FileNotFoundError:
    #     print("Warning: 'code15_label_issues.csv' not found. No records will be excluded.")
    
    # --- Stratified Data Splitting ---
    # Split the data into training (80%), validation (10%), and test (10%) sets.
    # The splits are stratified by the 'label' column to maintain class distribution.
    
    # Exclude records from source 'CODE-15%'
    if not HAS_CODE15:
        df = df[df['source'] != 'CODE-15%'].reset_index(drop=True)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # First, split into training and a temporary set (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,  # 20% for val and test
        random_state=42,
        stratify=df['label']
    )
    
    # Next, split the temporary set into validation and test sets
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,  # 50% of temp_df (which is 10% of original)
        random_state=42,
        stratify=temp_df['label']
    )

    # Convert to lists of record paths
    train_records = train_df['record'].tolist()
    val_records = val_df['record'].tolist()
    test_records = test_df['record'].tolist()
    
#     # 3. Set up K-Fold Cross-Validation on the Training Set
#     y_train = train_df['label'].values

#     # --- Cross-Validation Loop ---
#     fold_results = []
#     best_global_score = float('-inf')
#     best_global_ckpt = None
#     model_folder = os.path.join("kfold")
#     os.makedirs(model_folder, exist_ok=True)
#     unified_ckpt_path = os.path.join(model_folder, 'foundation_model_finetuned.ckpt')

#     train_df = train_df.reset_index(drop=True)
#     test_df = test_df.reset_index(drop=True)
#     # The loop now iterates over splits of the train_df
#     for fold, (train_idx, val_idx) in enumerate(skf.split(train_df['record'], y_train), start=1):
#         print(f"\n===== Fold {fold}/{N_FOLDS} =====")
#         pl.seed_everything(42 + fold)

#         # Get records for this fold's train and validation sets from train_df
#         train_records = train_df.loc[train_idx, 'record'].tolist()
#         val_records   = train_df.loc[val_idx, 'record'].tolist()

#         print(f"Fold {fold}: {len(train_records)} train / {len(val_records)} val")
#         pos_ratio_train = train_df.loc[train_idx, 'label'].mean()
#         pos_ratio_val   = train_df.loc[val_idx, 'label'].mean()
#         print(f"  Train positive ratio: {pos_ratio_train:.2%}")
#         print(f"  Val   positive ratio: {pos_ratio_val:.2%}")

#         # DataModule per fold
#         data_module = ECGDataModule(
#             data_dir=DATA_DIR,
#             batch_size=BATCH_SIZE,
#             seq_len=SEQ_LENGTH,
#             windowing_method='entire_recording',
#         )
#         # The datasets are now created from the fold's specific record lists
#         data_module.train_dataset = ECGDataset(
#             train_records, DATA_DIR, is_training=True,
#             seq_len=SEQ_LENGTH, windowing_method='entire_recording',
#             include_wide_feats=True
#         )
#         data_module.val_dataset = ECGDataset(
#             val_records, DATA_DIR, is_training=False,
#             seq_len=SEQ_LENGTH, windowing_method='entire_recording',
#             include_wide_feats=True
#         )
#         data_module.test_dataset = None  # No test set during CV

#         optimizer_hparams = {
#             'lr': LR,
#             'warmup_steps': 700,
#             'lr_end': 1e-7
#         }

#         model = FMChagasClassifier(
#             ecg_fm_checkpoint_path=checkpoint_path,
#             freeze_encoder=True,
#             optimizer_hparams=optimizer_hparams,
#             info_features=2,
#             loss_type=LOSS_TYPE
#         )

#         ckpt_callback = pl.callbacks.ModelCheckpoint(
#             dirpath=model_folder,
#             filename=f'foundation_model_finetuned_fold{fold}',
#             monitor=CHECKPOINT_MONITOR_METRIC,
#             mode='max',
#             save_top_k=1,
#             save_last=False,
#             save_weights_only=True,
#         )
#         wandb_logger = WandbLogger(
#             project="foundation_model",
#             entity="edwards_physionet",
#             name=f"fold_{fold}_of_{N_FOLDS}",  # Unique name for this specific run
#             group="cv_with_test",              # Shared group name for the CV experiment
#             job_type='cv_no_code15'             # Optional: helps organize runs
#         )
#         trainer = pl.Trainer(
#             max_epochs=NUM_EPOCHS,
#             accelerator="auto",
#             devices=1,
#             callbacks=[ckpt_callback],
#             logger=wandb_logger,
#             limit_test_batches=0
#         )

#         print(f"\n--- Training Fold {fold} ---")
#         trainer.fit(model, datamodule=data_module)

#         best_ckpt = ckpt_callback.best_model_path
#         if best_ckpt:
#             print(f"Best checkpoint (fold {fold}): {best_ckpt}")
#             # Re-run validation on best checkpoint to capture logged metrics
#             val_metrics = trainer.validate(model=model, datamodule=data_module, ckpt_path=best_ckpt, verbose=False)
#             fold_score = val_metrics[0].get('val_challenge_score', None) if val_metrics else None
#         else:
#             print(f"No best checkpoint saved for fold {fold}.")
#             fold_score = None

#         if best_ckpt and fold_score is not None:
#             if fold_score > best_global_score:
#                 best_global_score = fold_score
#                 best_global_ckpt = best_ckpt
#                 try:
#                     shutil.copyfile(best_ckpt, unified_ckpt_path)
#                     print(f"Updated best global checkpoint (fold {fold}, score {fold_score:.4f}) -> {unified_ckpt_path}")
#                 except Exception as e:
#                     print(f"Warning: failed to copy best checkpoint for fold {fold}: {e}")

#         fold_results.append({
#             'fold': fold,
#             'best_checkpoint': best_ckpt,
#             'val_challenge_score': fold_score,
#             'train_pos_ratio': pos_ratio_train,
#             'val_pos_ratio': pos_ratio_val
#         })

#     print("\n===== Cross-Validation Complete =====")
#     if best_global_ckpt is not None:
#         print(f"Best overall fold checkpoint: {best_global_ckpt}")
#         print(f"Best overall challenge score: {best_global_score:.4f}")
#         print(f"Unified saved checkpoint for final evaluation: {unified_ckpt_path}")
#     else:
#         print("No valid fold produced a checkpoint with a challenge score.")

# # --- Final Evaluation on the Held-Out Test Set ---
# # After the CV loop, you would use the 'best_global_ckpt' (or the unified copy)
# # to make predictions on the 'test_df' to get a final, unbiased estimate of
# # your model's performance on unseen data.

# # Example (pseudo-code):

#     # Only proceed if a best model was found and saved during cross-validation
# # --- FINAL EVALUATION & W&B SUMMARY ---
#     test_results = None

#     # 1. Evaluate on the held-out test set
#     if best_global_ckpt is not None and os.path.exists(unified_ckpt_path):
#         print("\n===== Final Evaluation on Test Set =====")
#         print(f"Loading best model from: {unified_ckpt_path}")

#         final_model = FMChagasClassifier.load_from_checkpoint(unified_ckpt_path)
#         final_model.eval()

#         test_records = test_df['record'].tolist()
#         test_data_module = ECGDataModule(
#             data_dir=DATA_DIR,
#             batch_size=BATCH_SIZE * 2,
#             seq_len=SEQ_LENGTH
#         )
#         test_data_module.test_dataset = ECGDataset(
#             test_records, DATA_DIR, is_training=False, seq_len=SEQ_LENGTH,
#             include_wide_feats=True
#         )

#         test_trainer = pl.Trainer(accelerator="auto", devices=1, logger=False)

#         print(f"Evaluating on {len(test_records)} test records...")
#         test_results_list = test_trainer.test(model=final_model, datamodule=test_data_module)
#         if test_results_list:
#             test_results = test_results_list[0] # Extract the metrics dictionary

#         print("\n--- Test Set Performance ---")
#         if test_results:
#             for metric, value in test_results.items():
#                 print(f"{metric}: {value:.4f}")
#         else:
#             print("Testing did not produce any results.")
#     else:
#         print("\nSkipping final evaluation because no best model checkpoint was found.")


#     # 2. Log Final Summary to Weights & Biases
#     print("\n===== Logging Summary to W&B =====")
#     results_df = pd.DataFrame(fold_results)
#     summary_logger = WandbLogger(
#         project='foundation_model',
#         entity='edwards_physionet',
#         name="cv_summary",
#         group='cv_with_test', # Use the same group to associate it
#         job_type='cv_no_code15'
#     )
#     # Log aggregated CV metrics
#     mean_score = results_df['val_challenge_score'].mean()
#     std_score = results_df['val_challenge_score'].std()
#     summary_logger.log_metrics({
#         "cv_mean_challenge_score": mean_score,
#         "cv_std_challenge_score": std_score
#     })

#     # ✅ Add the final test score to the same summary run
#     if test_results:
#         # Prefix test metrics for clarity in the W&B dashboard
#         final_test_metrics = {f"final_{k}": v for k, v in test_results.items()}
#         summary_logger.log_metrics(final_test_metrics)
#         print("Logged final test metrics to W&B.")

#     # Log the detailed CV results as a W&B Table
#     summary_logger.log_table(key="cv_results_summary", dataframe=results_df)

#     # Finalize the summary run
#     summary_logger.finalize("success")

#     print(f"\n✅ Final summary and test scores logged to W&B under group: 'cv_with_test'")



    print(f"Training on {len(train_records)} records.")
    print(f"Final training set positive ratio: {train_df['label'].value_counts(normalize=True).get(1, 0):.2%}")
    # print(f"Validating on {len(val_records)} records.")
    # print(f"Validation set positive ratio: {val_df['label'].value_counts(normalize=True).get(1, 0):.2%}")
    print(f"Testing on {len(test_records)} records.")
    print(f"Test set positive ratio: {test_df['label'].value_counts(normalize=True).get(1, 0):.2%}")

    # 3. Create DataModule
    data_module = ECGDataModule(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LENGTH,
        windowing_method='entire_recording',

    )
    data_module.train_dataset = ECGDataset(train_records, DATA_DIR, is_training=True, seq_len=config["seq_length"], windowing_method='entire_recording', include_wide_feats=True)
    data_module.val_dataset   = ECGDataset(val_records,   DATA_DIR, is_training=False, seq_len=config["seq_length"], windowing_method='entire_recording', include_wide_feats=True)
    data_module.test_dataset  = ECGDataset(test_records,  DATA_DIR, is_training=False, seq_len=config["seq_length"], windowing_method='entire_recording', include_wide_feats=True)

    optimizer_hparams = {
        'lr': 1e-4,
        'warmup_steps': 700,
        'lr_end': 1e-7
    }

    model = FMChagasClassifier(
        ecg_fm_checkpoint_path=checkpoint_path,
        freeze_encoder=True,
        optimizer_hparams=optimizer_hparams,
        info_features=2,  # Age and Sex
        loss_type=LOSS_TYPE,
        unfreeze_after_epoch=config['unfreeze_after_epoch']
    )
    
    print("\n--- Model Summary ---")
    print(model)
    
    logger_name = "foundation_model"
    if HAS_CODE15:
        logger_name += "_all_data"
    else:
        logger_name += "_no_code15"
    logger_name += f"_{LOSS_TYPE}"

    # BEFORE creating trainer: define checkpoint callback so we can access best path later
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor=CHECKPOINT_MONITOR_METRIC,
        mode='max',
        filename='best-challenge-{epoch:02d}-{val_challenge_score:.4f}.ckpt',
        dirpath=model_folder
    )

    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="auto",
        devices=1,
        logger=WandbLogger(project="foundation_model", name=logger_name, entity="edwards_physionet"),
        callbacks=[
            ckpt_callback,
            pl.callbacks.DeviceStatsMonitor()
        ]
    )

    # --- Run Training and Testing ---
    print(f"\n--- Starting Training with {len(train_records)} records from '{DATA_DIR}' ---")
    trainer.fit(model, datamodule=data_module)
    print("\n--- Training Finished ---")

    # ADDED: evaluation mode block
    if args.evaluate:
        if data_module.val_dataset is None:
            print("No validation dataset available; cannot run --evaluate mode.")
            sys.exit(0)

        best_ckpt = ckpt_callback.best_model_path
        if best_ckpt and os.path.isfile(best_ckpt):
            print(f"Loading best checkpoint for inference: {best_ckpt}")
            best_model = FMChagasClassifier.load_from_checkpoint(
                best_ckpt,
                ecg_fm_checkpoint_path=checkpoint_path,
                freeze_encoder=True,
                optimizer_hparams=optimizer_hparams,
                info_features=2,
                loss_type=LOSS_TYPE,
                unfreeze_after_epoch=config['unfreeze_after_epoch']
            )
        else:
            print("Best checkpoint not found; using in-memory model.")
            best_model = model

        best_model.eval()
        device = trainer.strategy.root_device if hasattr(trainer, 'strategy') else (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        best_model.to(device)

        val_loader = data_module.val_dataloader()
        val_dataset = data_module.val_dataset

        # Get the ordered list of validation record paths (primary source of record_id)
        val_record_paths = getattr(val_dataset, 'records_list', None)
        if val_record_paths is None:
            # fallback to earlier split list
            try:
                val_record_paths = val_records
                print("Using fallback val_records list for record_id mapping.")
            except NameError:
                val_record_paths = None
                print("Warning: Could not resolve validation record paths; record_id will be None.")

        if val_record_paths is not None:
            val_record_paths = list(val_record_paths)

        all_record_ids, all_labels, all_preds = [], [], []
        idx_counter = 0  # how many samples processed so far

        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue

                # Original dataloader returns (signals, info, labels) with include_wide_feats=True
                # or (signals, labels) otherwise. We avoid relying on record_ids inside the batch.
                if isinstance(batch, (list, tuple)):
                    if len(batch) == 3:
                        signals, info, labels = batch
                    elif len(batch) == 2:
                        signals, labels = batch
                        info = None
                    else:
                        raise ValueError("Unexpected batch structure in validation loader.")
                else:
                    raise ValueError("Validation batch is not a tuple/list.")

                if signals.numel() == 0:
                    continue

                bsz = signals.size(0)
                signals = signals.to(device)
                if info is not None:
                    info = info.to(device)

                logits = best_model(signals, info)
                probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
                labels_np = labels.squeeze(-1).cpu().numpy()

                if val_record_paths is not None:
                    batch_paths = val_record_paths[idx_counter:idx_counter + bsz]
                    if len(batch_paths) != bsz:
                        # Safety adjustment if something became misaligned
                        print(f"Warning: slice length {len(batch_paths)} != batch size {bsz}; padding/truncating.")
                        if len(batch_paths) < bsz:
                            batch_paths += [None] * (bsz - len(batch_paths))
                        else:
                            batch_paths = batch_paths[:bsz]
                    batch_record_ids = [
                        (os.path.splitext(os.path.basename(p))[0] if p is not None else None)
                        for p in batch_paths
                    ]
                else:
                    batch_record_ids = [None] * bsz

                all_record_ids.extend(batch_record_ids)
                all_labels.extend(labels_np.tolist())
                all_preds.extend(probs.tolist())

                idx_counter += bsz

        out_df = pd.DataFrame({
            'record_id': all_record_ids,
            'label': all_labels,
            'prediction': all_preds
        })
        out_df.to_csv(args.predictions_output, index=False)
        print(f"Validation predictions saved to {args.predictions_output}")
        sys.exit(0)

    # ORIGINAL test phase kept when not evaluating
    print("\n--- Starting Testing ---")
    trainer.test(model, datamodule=data_module)
    print("\n--- Testing Finished ---")
    
def train_model(data_folder, model_folder):
    checkpoint_path='mimic_iv_ecg_finetuned.pt'

    # --- Hyperparameters ---
    DATA_DIR = data_folder
    BATCH_SIZE = 16
    # SEQ_LEN = utils.WINDOW_SIZE
    NUM_LEADS = 12
    WINDOWING_METHOD = 'entire_recording'
    NUM_EPOCHS = 16
    SEQ_LENGTH = utils.UNIFIED_FREQUENCY * 10  
    LR = 1e-4
    LOSS_TYPE = 'bce'

    # Add these values into the config dictionary
    config = {
        "data_dir": DATA_DIR,
        "batch_size": BATCH_SIZE,
        "seq_length": SEQ_LENGTH,
        "num_leads": NUM_LEADS,
        "windowing_method": WINDOWING_METHOD,
        "num_epochs": NUM_EPOCHS,
        "lr": LR,
        "loss_type": LOSS_TYPE
    }

    try:
        if torch.cuda.is_available():
            if torch.cuda.get_device_capability()[0] >= 8:
                print("Setting TF32 matmul precision for Ampere GPUs")
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.set_float32_matmul_precision('high')
            else:
                print("Warning: Not an Ampere GPU. Some precision settings may not be optimal.")
    except AttributeError:
        print("Warning: TF32 matmul precision setting not available. Skipping this step.")

    if not os.path.isdir(DATA_DIR):
        print(f"Error: Data directory not found at '{DATA_DIR}'")
        print("Please update the DATA_DIR variable to point to your dataset.")
        sys.exit(1)

# 2. Prepare data for fine-tuning
    all_records = custom_helper_code.find_records_abs(DATA_DIR)
    print(f"{len(all_records)} records found in the dataset.")
    records_meta = utils.prepare_stratification(all_records)
    df = pd.DataFrame(records_meta)
    
    # # --- Exclude records from code15_label_issues.csv ---
    # try:
    #     issues_df = pd.read_csv('code15_label_issues.csv')
    #     # Extract stem (filename without extension) from the full path for exclusion list
    #     stems_to_exclude = set(issues_df['record_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0]))
        
    #     # Create a new 'base_record' column with the stem for matching
    #     df['base_record'] = df['record'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
        
    #     initial_count = len(df)
    #     # Filter out the records using the new column
    #     df = df[~df['base_record'].isin(stems_to_exclude)]
    #     final_count = len(df)
        
    #     print(f"Excluded {initial_count - final_count} records based on 'code15_label_issues.csv'.")
    #     # Drop the temporary 'base_record' column
    #     df = df.drop(columns=['base_record'])
        
    # except FileNotFoundError:
    #     print("Warning: 'code15_label_issues.csv' not found. No records will be excluded.")
    
    # --- Stratified Data Splitting ---
    # Split the data into training (80%), validation (10%), and test (10%) sets.
    # The splits are stratified by the 'label' column to maintain class distribution.
    
    # First, split into training and a temporary set (val + test)
    train_df = df
    # prepare_stratification returns a malformed entry for any record whose label or
    # source could not be read; those become NaN rows here. Drop them so the sample
    # weights below stay aligned with train_records (they would fail to load anyway).
    before = len(train_df)
    train_df = train_df.dropna(subset=['record', 'label'])
    if len(train_df) < before:
        print(f"Dropped {before - len(train_df)} records with unreadable label/source.")
    # Convert to lists of record paths
    train_records = train_df['record'].tolist()


    print(f"Training on {len(train_records)} records.")
    print(f"Final training set positive ratio: {train_df['label'].value_counts(normalize=True).get(1, 0):.2%}")

    # Inverse-frequency sample weights. At ~2% prevalence with batch_size=16 the median
    # unweighted batch contains zero positives, so the head just learns the base rate.
    labels = train_df['label'].astype(int).to_numpy()
    class_counts = np.bincount(labels, minlength=2).astype(np.float64)
    class_counts[class_counts == 0] = 1.0  # avoid div-by-zero on a single-class set
    sample_weights = (1.0 / class_counts)[labels]
    print(f"Sampling weights: {int(class_counts[0])} negatives, {int(class_counts[1])} positives.")

    # 3. Create DataModule
    data_module = ECGDataModule(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LENGTH,
        windowing_method='entire_recording',
        sample_weights=sample_weights,
    )
    data_module.train_dataset = ECGDataset(train_records, DATA_DIR, is_training=True, seq_len=config["seq_length"], windowing_method='entire_recording', include_wide_feats=True,)
    data_module.val_dataset   = None
    data_module.test_dataset  = None

    optimizer_hparams = {
        'lr': 1e-4,
        'warmup_steps': 700,
        'lr_end': 1e-7
    }

    model = FMChagasClassifier(
        ecg_fm_checkpoint_path=checkpoint_path,
        freeze_encoder=True,
        optimizer_hparams=optimizer_hparams,
        info_features=2 # Age and Sex
    )
    
    
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="auto",
        devices=1,
        callbacks=[TimeLimitCallback(max_hours=71.0)],
        logger=False,
        num_sanity_val_steps=0,
        # This path trains on all records with no validation split. num_sanity_val_steps
        # only skips the sanity check; without this, Lightning still calls
        # val_dataloader() at the end of epoch 1 and gets DataLoader(None).
        limit_val_batches=0,
    )

    # --- Run Training and Testing ---
    print(f"\n--- Starting Training with {len(train_records)} records from '{DATA_DIR}' ---")
    trainer.fit(model, datamodule=data_module)
    print("\n--- Training Finished ---")

    # Save the model to model_folder
    final_checkpoint_path = os.path.join(model_folder, "foundation_model_finetuned.ckpt")
    trainer.save_checkpoint(final_checkpoint_path, weights_only=True)
    print(f"Model weights saved to {final_checkpoint_path}")
    

# def train_model(data_folder, model_folder, checkpoint_path, verbose):
#     pl.seed_everything(42)  # For reproducibility
#     # Hyperparameters
#     SEQ_LENGTH = utils.UNIFIED_FREQUENCY * 5  
#     BATCH_SIZE = 16  # Smaller batch size due to ECG-FM memory requirements
#     EPOCHS = 50
#     LR = 1e-4
#     NO_LABELS = False  # We need labels for Chagas classification

#     # Instantiate Lightning module for MAE with SE flag
#     model = FMChagasClassifier(
#         ecg_fm_checkpoint_path=checkpoint_path,
#         num_classes=1,  # Binary classification for Chagas
#         lr=LR,
#         freeze_encoder=True  # Start with frozen encoder, can fine-tune later
#     )
#         # Data directories
#     DATA_DIR = data_folder

#     # --- Corrected Data Loading and Splitting ---
#     # 1. Load ALL records, for pretraining (since this is official code)
#     records_meta = custom_helper_code.find_records(DATA_DIR)
    
#     # 2. Combine all records into a single list
#     print("WARNING: EXCLUDING UNLABELED RECORDS, THIS IS FOR COMPETITION MODEL TRAINING")
#     # all_records = labeled_records + unlabeled_records
#     all_records = records_meta
#     print(f"Total records found: {len(all_records)}")
#     if len(all_records) == 0:
#         raise ValueError("No records found in the specified directories. Please check the paths.")

#     np.random.shuffle(all_records)
#     train_records = all_records  # use all data for training

#     # Wire up ECGDataModule
#     data_module = ECGDataModule(
#         data_dir=DATA_DIR, # Base directory, not strictly needed since paths are absolute
#         batch_size=BATCH_SIZE,
#         seq_len=SEQ_LENGTH,
#         windowing_method='entire_recording'
#     )

#     data_module.train_dataset = ECGDataset(train_records, DATA_DIR, is_training=True, no_labels=NO_LABELS,
#                                            seq_len=SEQ_LENGTH, windowing_method='entire_recording')

#     trainer = pl.Trainer(
#         max_epochs=EPOCHS,
#         accelerator='auto',
#         devices=1,
#         callbacks=[],
#         gradient_clip_val=1.0,  # Added gradient clipping
#         num_sanity_val_steps=0, 
#         enable_checkpointing=False
#     )
    
#     # fit and validate, starting a new training run without ckpt_path
#     trainer.fit(model, datamodule=data_module)
    
#     # Save the model weights only to the checkpoint dir
#     final_checkpoint_path = os.path.join(model_folder, "mae_encoder_pretrained.ckpt")
#     trainer.save_checkpoint(final_checkpoint_path, weights_only=True)
#     print(f"Model weights saved to {final_checkpoint_path}")

def load_finetuned_model(ckpt_path, map_location=None, strict=True, override_hparams=None):
    """
    Load a fine-tuned FMChagasClassifier from a Lightning .ckpt file.

    Args:
        ckpt_path (str): Path to the .ckpt file (e.g., 'foundation_model_finetuned.ckpt').
        map_location (str or torch.device, optional): Device mapping for loading.
            Defaults to 'cuda' if available, else 'cpu'.
        strict (bool): Strict state_dict loading. Default True.
        override_hparams (dict, optional): Optional kwargs to override saved hparams,
            e.g., {'loss_type': 'bce'}.

    Returns:
        FMChagasClassifier: Model loaded with weights, set to eval() mode.
    """
    if map_location is None:
        map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    override_hparams = override_hparams or {}
    model = FMChagasClassifier.load_from_checkpoint(
        ckpt_path,
        ecg_fm_checkpoint_path="mimic_iv_ecg_finetuned.pt",
        map_location=map_location,
        strict=strict,
        **override_hparams
    )

    # load_from_checkpoint constructs the module first (randomly initialising the
    # classification head) and only then loads the state dict. With strict=False a
    # checkpoint missing those keys loads "successfully" and the model emits confident
    # noise. Fail loudly instead -- a silent random head wastes a scarce submission.
    if not strict:
        ckpt_keys = set(torch.load(
            ckpt_path, map_location='cpu', weights_only=False
        ).get('state_dict', {}).keys())
        critical = {
            k for k in model.state_dict()
            if k.startswith('classifier.') or k.endswith('layer_weights')
        }
        missing = sorted(critical - ckpt_keys)
        if missing:
            raise RuntimeError(
                f"Checkpoint {ckpt_path} is missing trained parameters {missing}; "
                f"they would be left randomly initialised. Refusing to run inference."
            )

    model.eval()
    return model