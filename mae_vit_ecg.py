"""
MAE-based Vision Transformer for ECG analysis
- ViT backbone adapted for 1D ECG signals
- Masked autoencoder logic
- Pretraining and fine-tuning interfaces
"""
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import numpy as np

from torchmetrics.classification import Accuracy, AUROC

import pytorch_lightning as pl
from torchmetrics import AUROC, Accuracy
from typing import List, Tuple
import sys
import os
from typing import Optional, Union
from pytorch_lightning.callbacks import EarlyStopping   

try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)
except NameError:
    print("Warning: __file__ not defined. Assuming 'helper_code.py' and 'utils.py' are in the Python path.")
    # In an interactive environment (like a notebook), ensure the parent directory
    # containing your modules is in the path. You might add it manually:
    # sys.path.append('path/to/your/project/root')
    pass

import utils
from utils import compute_challenge_score

class PatchEmbed1D(nn.Module):
    """1D Patch Embedding for ECG signals"""
    def __init__(self, in_channels=12, seq_len=5000, patch_size=250, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (batch, channels, seq_len)
        x = self.proj(x)  # (batch, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch, num_patches, embed_dim)
        return x

class MAEViTECG(nn.Module):
    def __init__(self, in_channels=12, seq_len=5000, patch_size=250, embed_dim=768, mask_ratio=0.75):
        super().__init__()
        self.patch_embed = PatchEmbed1D(in_channels, seq_len, patch_size, embed_dim)
        
        # Option 1: Use PyTorch's native implementation (torchvision)
        import torchvision.models as models
        vit = models.vit_b_16(weights=None)  # equivalent to vit_base_patch16_224
        self.transformer_encoder_blocks = vit.encoder.layers
        
        # Option 2: Use HuggingFace Transformers
        # from transformers import ViTModel, ViTConfig
        # config = ViTConfig(hidden_size=embed_dim, num_hidden_layers=12, 
        #                    num_attention_heads=12, intermediate_size=embed_dim*4)
        # vit = ViTModel(config)
        # self.transformer_encoder_blocks = vit.encoder.layer
        
        # Option 3: Create a custom implementation
        # num_layers = 12
        # self.transformer_encoder_blocks = nn.ModuleList([
        #     TransformerEncoderBlock(embed_dim=embed_dim, num_heads=12) 
        #     for _ in range(num_layers)
        # ])
        
        self.mask_ratio = mask_ratio
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, patch_size * in_channels),
            nn.Tanh()
        )

    def random_masking(self, x):
        # x: (batch, num_patches, embed_dim)
        batch, num_patches, _ = x.shape
        num_mask = int(self.mask_ratio * num_patches)
        mask = torch.zeros((batch, num_patches), dtype=torch.bool)
        for i in range(batch):
            idx = np.random.choice(num_patches, num_mask, replace=False)
            mask[i, idx] = True
        return mask

    def forward(self, x):
        patches = self.patch_embed(x)  # (batch, num_patches, embed_dim)
        mask = self.random_masking(patches)
        # Masked input: set masked patches to zero
        masked_patches = patches.clone()
        masked_patches[mask] = 0
        # Feed patch embeddings into ViT encoder blocks
        x = masked_patches
        for block in self.transformer_encoder_blocks:
            x = block(x)
        encoded = x
        # recon: (batch, num_patches, patch_size * in_channels)
        batch_size, num_patches, _ = encoded.shape
        recon = self.decoder(encoded)  # (batch, num_patches, patch_size * in_channels)
        # Reshape to (batch, in_channels, num_patches * patch_size)
        recon = recon.view(batch_size, num_patches, self.patch_embed.patch_size, self.patch_embed.proj.in_channels)
        recon = recon.permute(0, 3, 1, 2).contiguous()  # (batch, channels, num_patches, patch_size)
        recon = recon.view(batch_size, self.patch_embed.proj.in_channels, num_patches * self.patch_embed.patch_size)
        return recon, mask

# If using Option 3 (custom implementation), add this class:
# class TransformerEncoderBlock(nn.Module):
#     def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.0):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
#         self.norm2 = nn.LayerNorm(embed_dim)
#         mlp_hidden_dim = int(embed_dim * mlp_ratio)
#         self.mlp = nn.Sequential(
#             nn.Linear(embed_dim, mlp_hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(mlp_hidden_dim, embed_dim),
#             nn.Dropout(dropout)
#         )
#     
#     def forward(self, x):
#         x_norm = self.norm1(x)
#         x_attn, _ = self.attn(x_norm, x_norm, x_norm)
#         x = x + x_attn
#         x = x + self.mlp(self.norm2(x))
#         return x

# Add training and fine-tuning routines in separate scripts
# =============================================================================
# 2. PyTorch Lightning Module
# =============================================================================

class MAEViTECG_Lightning(pl.LightningModule):
    """
    PyTorch Lightning wrapper for the MAEViTECG model.
    Handles pre-training and fine-tuning with detailed metric logging.
    """
    def __init__(self, 
                 in_channels: int = 12, 
                 seq_len: int = 5000, 
                 patch_size: int = 250, 
                 embed_dim: int = 768, 
                 mask_ratio: float = 0.75, 
                 lr: float = 1e-4,
                 warmup_epochs: int = 5,
                 weight_decay: float = 0.05,
                 finetune: bool = False,
                 num_classes: int = 1):
        super().__init__()
        self.save_hyperparameters()
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # Fix: Remove finetune and num_classes parameters from MAEViTECG initialization
        self.model = MAEViTECG(
            in_channels=in_channels, 
            seq_len=seq_len, 
            patch_size=patch_size,
            embed_dim=embed_dim, 
            mask_ratio=mask_ratio
        )
        


        if self.hparams.finetune:
            self.criterion = nn.BCEWithLogitsLoss()
            # Initialize metrics from torchmetrics
            self.val_auroc = AUROC(task="binary")
            self.val_accuracy = Accuracy(task="binary")
            self.test_auroc = AUROC(task="binary")
            self.test_accuracy = Accuracy(task='binary')
        else:
            self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def _pretrain_step(self, batch):
        # Fix: Correctly handle the pretraining step
        if isinstance(batch, list) or isinstance(batch, tuple):
            signals = batch[0]
        else:
            signals = batch
            
        # Get reconstruction and mask from the model
        recon, mask = self(signals)
        target = signals
        
        # Reshape for computing loss on patches
        patch_size = self.model.patch_embed.patch_size
        channels = self.model.patch_embed.proj.in_channels
        num_patches = recon.shape[2] // patch_size
        
        # Reshape to (batch, num_patches, patch_size, channels)
        recon_patches = recon.view(recon.shape[0], channels, num_patches, patch_size)
        recon_patches = recon_patches.permute(0, 2, 3, 1)
        target_patches = target.view(target.shape[0], channels, num_patches, patch_size)
        target_patches = target_patches.permute(0, 2, 3, 1)
        
        # Compute loss only on masked patches
        losses = []
        for b in range(recon_patches.shape[0]):
            masked_idx = mask[b]
            if masked_idx.any():
                # Extract masked patches and compute MSE
                diff = recon_patches[b][masked_idx] - target_patches[b][masked_idx]
                losses.append(diff.pow(2).mean())
        
        # Average losses across batch
        if losses:
            loss = torch.stack(losses).mean()
        else:
            loss = torch.tensor(0.0, device=self.device)
        
        return loss

    def _finetune_step(self, batch):
        # Fix: Handle finetune step with MAEViTECG's output format
        signals, labels = batch
        
        # For finetuning, we need a classifier head that's not implemented
        # in the current MAEViTECG. This is a placeholder assuming such implementation.
        # In a real scenario, you would need to extend MAEViTECG or create a new model
        # that uses the encoder and adds a classification head.
        recon, _ = self(signals)
        
        # In a real implementation, you'd use a classifier head here
        # This is just a placeholder for the expected behavior
        raise NotImplementedError(
            "Finetuning not implemented in MAEViTECG. " +
            "Create a separate model that uses the encoder and adds a classification head."
        )

    def training_step(self, batch, batch_idx):
        if self.hparams.finetune:
            # Finetune mode not fully implemented - would need a classifier head
            loss, _, _ = self._finetune_step(batch)
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        else:
            # Pretraining mode - working correctly
            loss = self._pretrain_step(batch)
            self.log('pretrain_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        self.log('learning_rate', self.optimizers().param_groups[0]['lr'], prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        if not self.hparams.finetune: return
        loss, preds, labels = self._finetune_step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        # Update metrics
        self.val_auroc.update(preds, labels)
        self.val_accuracy.update(preds, labels)
        # Store outputs for epoch-end calculation
        self.validation_step_outputs.append({'preds': preds.detach(), 'labels': labels.detach()})

    def on_validation_epoch_end(self):
        if not self.hparams.finetune or not self.validation_step_outputs: return
        # Compute and log epoch-level metrics
        self.log('val_auroc', self.val_auroc.compute(), prog_bar=True)
        self.log('val_accuracy', self.val_accuracy.compute(), prog_bar=True)
        
        # Aggregate all predictions and labels for challenge score
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs]).cpu().numpy()
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs]).cpu().numpy()
        challenge_score = compute_challenge_score(all_labels, all_preds)
        self.log('val_challenge_score', challenge_score, prog_bar=True)
        
        self.validation_step_outputs.clear() # free memory

    def test_step(self, batch, batch_idx):
        if not self.hparams.finetune: return
        loss, preds, labels = self._finetune_step(batch)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.test_auroc.update(preds, labels)
        self.test_accuracy.update(preds, labels)
        self.test_step_outputs.append({'preds': preds.detach(), 'labels': labels.detach()})

    def on_test_epoch_end(self):
        if not self.hparams.finetune or not self.test_step_outputs: return
        self.log('test_auroc', self.test_auroc.compute(), prog_bar=True)
        self.log('test_accuracy', self.test_accuracy.compute(), prog_bar=True)
        
        all_preds = torch.cat([x['preds'] for x in self.test_step_outputs]).cpu().numpy()
        all_labels = torch.cat([x['labels'] for x in self.test_step_outputs]).cpu().numpy()
        challenge_score = compute_challenge_score(all_labels, all_preds)
        self.log('test_challenge_score', challenge_score, prog_bar=True)
        
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        def lr_lambda(epoch):
            if epoch < self.hparams.warmup_epochs:
                return float(epoch) / float(max(1, self.hparams.warmup_epochs))
            progress = float(epoch - self.hparams.warmup_epochs) / float(max(1, self.trainer.max_epochs - self.hparams.warmup_epochs))
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1}}
class MAEViTECGFinetune(pl.LightningModule):
    """
    PyTorch Lightning module for finetuning MAE-ViT ECG model.
    Supports layer freezing and various training strategies.
    """
    def __init__(self,
                 pretrained_checkpoint_path: str,
                 num_classes: int = 1,
                 freeze_layers: Optional[Union[int, List[int], str]] = None,
                 lr: float = 1e-4,
                 weight_decay: float = 0.01,
                 warmup_epochs: int = 5,
                 dropout: float = 0.1,
                 use_cls_token: bool = True,
                 lr_schedule: str = 'cosine',  # 'cosine', 'linear', 'constant'
                 min_lr: float = 1e-6,
                 num_wide_feats: int = 0):
        super().__init__()
        self.save_hyperparameters()
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        # Load pretrained MAE model
        self._load_pretrained_model(pretrained_checkpoint_path)
        
        # Create classifier model
        self.model = MAEViTECGClassifier(
            mae_model=self.mae_model,
            num_classes=num_classes,
            dropout=dropout,
            use_cls_token=use_cls_token,
            num_wide_feats=num_wide_feats
        )
        
        # Freeze layers if specified
        self._freeze_layers(freeze_layers)
        
        # Loss and metrics
        self.criterion = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
        
        # Metrics
        if num_classes == 1:
            self.train_auroc = AUROC(task="binary")
            self.val_auroc = AUROC(task="binary")
            self.test_auroc = AUROC(task="binary")
            self.train_acc = Accuracy(task="binary")
            self.val_acc = Accuracy(task="binary")
            self.test_acc = Accuracy(task="binary")
        else:
            self.train_auroc = AUROC(task="multiclass", num_classes=num_classes)
            self.val_auroc = AUROC(task="multiclass", num_classes=num_classes)
            self.test_auroc = AUROC(task="multiclass", num_classes=num_classes)
            self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
            self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
            self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
    
    def _load_pretrained_model(self, checkpoint_path: str):
        """Load pretrained MAE model from checkpoint"""
        # Load the pretrained weights
        checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)
        
        # Extract model configuration from checkpoint
        hparams = checkpoint.get('hyper_parameters', {})
        
        # Create base MAE model
        self.mae_model = MAEViTECG(
            in_channels=hparams.get('in_channels', 12),
            seq_len=hparams.get('seq_len', 5000),
            patch_size=hparams.get('patch_size', 250),
            embed_dim=hparams.get('embed_dim', 768),
            mask_ratio=0.0  # No masking during finetuning
        )
        
        # Load state dict
        if 'state_dict' in checkpoint:
            # Filter out only the model weights (remove 'model.' prefix if present)
            model_state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('model.'):
                    model_state_dict[k[6:]] = v  # Remove 'model.' prefix
                else:
                    model_state_dict[k] = v
            
            # Load weights
            self.mae_model.load_state_dict(model_state_dict, strict=False)
            print(f"Loaded pretrained weights from {checkpoint_path}")
        else:
            raise ValueError(f"No state_dict found in checkpoint {checkpoint_path}")
    
    def _freeze_layers(self, freeze_layers: Optional[Union[int, List[int], str]]):
        """
        Freeze specified layers of the model.
        
        Args:
            freeze_layers: Can be:
                - None: Don't freeze any layers
                - int: Freeze first n transformer blocks
                - List[int]: Freeze specific transformer blocks by index
                - 'patch_embed': Freeze only patch embedding
                - 'all_but_head': Freeze everything (patch embedding + all transformer blocks) except classifier
                - 'all_but_last_n': Freeze all but last n transformer blocks (e.g., 'all_but_last_3')
        """
        if freeze_layers is None:
            print("No layers frozen - training all parameters")
            return
        
        # Get total number of transformer blocks
        num_blocks = len(self.mae_model.transformer_encoder_blocks)
        
        # Handle patch embedding freezing first
        if freeze_layers == 'patch_embed' or freeze_layers == 'all_but_head':
            for param in self.mae_model.patch_embed.parameters():
                param.requires_grad = False
            print("Froze patch embedding layers")
        
        # Determine which transformer blocks to freeze
        blocks_to_freeze = []
        
        if isinstance(freeze_layers, int):
            # Freeze first n blocks
            if freeze_layers > num_blocks:
                print(f"Warning: Requested to freeze {freeze_layers} blocks, but only {num_blocks} exist. Freezing all blocks.")
                blocks_to_freeze = list(range(num_blocks))
            else:
                blocks_to_freeze = list(range(freeze_layers))
                
        elif isinstance(freeze_layers, list):
            # Freeze specific blocks
            valid_blocks = [idx for idx in freeze_layers if 0 <= idx < num_blocks]
            if len(valid_blocks) != len(freeze_layers):
                print(f"Warning: Some block indices were out of range (0-{num_blocks-1}) and were ignored.")
            blocks_to_freeze = valid_blocks
            
        elif freeze_layers == 'all_but_head':
            # Freeze all transformer blocks
            blocks_to_freeze = list(range(num_blocks))
            
        elif isinstance(freeze_layers, str) and freeze_layers.startswith('all_but_last_'):
            # Freeze all but last n blocks
            try:
                n = int(freeze_layers.split('_')[-1])
                if n >= num_blocks:
                    print(f"Warning: Requested to keep last {n} blocks unfrozen, but only {num_blocks} exist. No blocks will be frozen.")
                else:
                    blocks_to_freeze = list(range(num_blocks - n))
            except ValueError:
                print(f"Error parsing '{freeze_layers}'. Format should be 'all_but_last_n' where n is an integer.")
        
        # Apply freezing to transformer blocks
        for idx in blocks_to_freeze:
            if 0 <= idx < num_blocks:
                for param in self.mae_model.transformer_encoder_blocks[idx].parameters():
                    param.requires_grad = False
                print(f"Froze transformer block {idx}")
        
        # Count trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    def forward(self, x, wide_feats=None):
        return self.model(x, wide_feats=wide_feats)
    
    def training_step(self, batch, batch_idx):
        wide_feats = None
        if self.hparams.num_wide_feats > 0:
            signals, wide_feats, labels = batch
        else:
            signals, labels = batch
        
        logits = self(signals, wide_feats=wide_feats)
        
        if self.hparams.num_classes == 1:
            logits = logits.squeeze(-1)
            labels = labels.float().squeeze(-1)  # Add squeeze to match logits dimension
            preds = torch.sigmoid(logits)
        else:
            preds = torch.softmax(logits, dim=-1)
        
        loss = self.criterion(logits, labels)
        
        # Update metrics
        self.train_auroc.update(preds, labels.int())
        self.train_acc.update(preds, labels.int())
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_auroc', self.train_auroc, on_step=False, on_epoch=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'], on_step=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        wide_feats = None
        if self.hparams.num_wide_feats > 0:
            signals, wide_feats, labels = batch
        else:
            signals, labels = batch

        logits = self(signals, wide_feats=wide_feats)
        
        if self.hparams.num_classes == 1:
            logits = logits.squeeze(-1)
            labels = labels.float().squeeze(-1)  # Add squeeze to match logits dimension
            preds = torch.sigmoid(logits)
        else:
            preds = torch.softmax(logits, dim=-1)
        
        loss = self.criterion(logits, labels)
        
        # Update metrics
        self.val_auroc.update(preds, labels.int())
        self.val_acc.update(preds, labels.int())
        
        # Store for epoch end
        self.validation_step_outputs.append({
            'preds': preds.detach(),
            'labels': labels.detach()
        })
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
        # Log metrics
        self.log('val_auroc', self.val_auroc.compute(), prog_bar=True)
        self.log('val_acc', self.val_acc.compute(), prog_bar=True)
        
        # Calculate challenge score if applicable
        if self.validation_step_outputs and hasattr(utils, 'compute_challenge_score'):
            all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs]).cpu().numpy()
            all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs]).cpu().numpy()
            
            if self.hparams.num_classes == 1:
                challenge_score = utils.compute_challenge_score(all_labels, all_preds)
                self.log('val_challenge_score', challenge_score, prog_bar=True)
        
        self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        wide_feats = None
        if self.hparams.num_wide_feats > 0:
            signals, wide_feats, labels = batch
        else:
            signals, labels = batch
            
        logits = self(signals, wide_feats=wide_feats)
        
        if self.hparams.num_classes == 1:
            logits = logits.squeeze(-1)
            labels = labels.float().squeeze(-1)  # Add squeeze to match logits dimension
            preds = torch.sigmoid(logits)
        else:
            preds = torch.softmax(logits, dim=-1)
        
        loss = self.criterion(logits, labels)
        
        # Update metrics
        self.test_auroc.update(preds, labels.int())
        self.test_acc.update(preds, labels.int())
        
        # Store for epoch end
        self.test_step_outputs.append({
            'preds': preds.detach(),
            'labels': labels.detach()
        })
        
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def on_test_epoch_end(self):
        # Log metrics
        self.log('test_auroc', self.test_auroc.compute(), prog_bar=True)
        self.log('test_acc', self.test_acc.compute(), prog_bar=True)
        
        # Calculate challenge score if applicable
        if self.test_step_outputs and hasattr(utils, 'compute_challenge_score'):
            all_preds = torch.cat([x['preds'] for x in self.test_step_outputs]).cpu().numpy()
            all_labels = torch.cat([x['labels'] for x in self.test_step_outputs]).cpu().numpy()
            
            if self.hparams.num_classes == 1:
                challenge_score = utils.compute_challenge_score(all_labels, all_preds)
                self.log('test_challenge_score', challenge_score, prog_bar=True)
        
        self.test_step_outputs.clear()
    
    def configure_optimizers(self):
        # Only optimize parameters that require gradients
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        
        # Learning rate scheduler
        if self.hparams.lr_schedule == 'cosine':
            def lr_lambda(epoch):
                if epoch < self.hparams.warmup_epochs:
                    return float(epoch) / float(max(1, self.hparams.warmup_epochs))
                progress = float(epoch - self.hparams.warmup_epochs) / float(max(1, self.trainer.max_epochs - self.hparams.warmup_epochs))
                return max(self.hparams.min_lr / self.hparams.lr,
                          0.5 * (1.0 + np.cos(np.pi * progress)))
            
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        elif self.hparams.lr_schedule == 'linear':
            scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=self.hparams.min_lr / self.hparams.lr,
                total_iters=self.trainer.max_epochs
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        else:  # constant
            return optimizer
            
    @classmethod
    def resume_from_checkpoint(cls, checkpoint_path: str, reset_params: dict = None, **kwargs):
        """
        Resume training from a checkpoint but reset optimizer state and epoch counter.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            reset_params: Dictionary of parameters to override from the checkpoint
            **kwargs: Additional parameters to pass to the model constructor
            
        Returns:
            A new model instance with weights from the checkpoint but fresh optimizer/training state
        """
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract hyperparameters from the checkpoint
        if 'hyper_parameters' in checkpoint:
            hparams = checkpoint['hyper_parameters']
            
            # Override with reset_params if provided
            if reset_params is not None:
                for key, value in reset_params.items():
                    if key in hparams:
                        hparams[key] = value
                    else:
                        print(f"Warning: Parameter {key} not found in checkpoint hyperparameters")
            
            # Create a new model with the hyperparameters
            model = cls(
                pretrained_checkpoint_path=hparams.get('pretrained_checkpoint_path', kwargs.get('pretrained_checkpoint_path')),
                num_classes=hparams.get('num_classes', kwargs.get('num_classes', 1)),
                freeze_layers=hparams.get('freeze_layers', kwargs.get('freeze_layers', None)),
                lr=hparams.get('lr', kwargs.get('lr', 1e-4)),
                weight_decay=hparams.get('weight_decay', kwargs.get('weight_decay', 0.01)),
                warmup_epochs=hparams.get('warmup_epochs', kwargs.get('warmup_epochs', 5)),
                dropout=hparams.get('dropout', kwargs.get('dropout', 0.1)),
                use_cls_token=hparams.get('use_cls_token', kwargs.get('use_cls_token', True)),
                lr_schedule=hparams.get('lr_schedule', kwargs.get('lr_schedule', 'cosine')),
                min_lr=hparams.get('min_lr', kwargs.get('min_lr', 1e-6)),
            )
            
            # Extract only model state dict (no optimizer, scheduler, etc.)
            model_state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                model_state_dict[k] = v
                
            # Load the model weights
            model.load_state_dict(model_state_dict)
            print(f"Resumed model from {checkpoint_path} with reset training state")
            
            return model
        else:
            raise ValueError(f"No hyperparameters found in checkpoint {checkpoint_path}")
class MAEViTECGClassifier(nn.Module):
    """
    Extended MAE-ViT model with classification head for finetuning.
    Uses the pretrained encoder and adds a classification head.
    """
    def __init__(self, 
                 mae_model: MAEViTECG,
                 num_classes: int = 1,
                 dropout: float = 0.1,
                 use_cls_token: bool = True,
                 num_wide_feats: int = 0):
        super().__init__()
        self.mae_model = mae_model
        self.embed_dim = mae_model.patch_embed.embed_dim
        self.use_cls_token = use_cls_token
        self.num_wide_feats = num_wide_feats
        
        # Add CLS token if using
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Classification head
        classifier_input_dim = self.embed_dim + self.num_wide_feats
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim // 2, num_classes)
        )
        
    def forward(self, x, wide_feats=None, return_features=False):
        # Get patch embeddings
        patches = self.mae_model.patch_embed(x)  # (batch, num_patches, embed_dim)
        
        # Add CLS token
        if self.use_cls_token:
            batch_size = patches.shape[0]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            patches = torch.cat([cls_tokens, patches], dim=1)
        
        # Pass through transformer encoder
        for block in self.mae_model.transformer_encoder_blocks:
            patches = block(patches)
        
        # Use CLS token or global average pooling
        if self.use_cls_token:
            features = patches[:, 0]  # CLS token
        else:
            features = patches.mean(dim=1)  # Global average pooling
        
        # Concatenate wide features if they exist
        if self.num_wide_feats > 0 and wide_feats is not None:
            combined_features = torch.cat([features, wide_feats], dim=1)
        else:
            combined_features = features

        # Classification
        logits = self.classifier(combined_features)
        
        if return_features:
            return logits, combined_features
        return logits

