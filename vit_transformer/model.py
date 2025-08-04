import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import random
from typing import Tuple, Optional, Dict, List
from torchmetrics import Accuracy, AUROC, F1Score, ConfusionMatrix
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

import pytorch_lightning as pl  
from typing import Dict, Any
import utils

class PatchEmbedding(nn.Module):
    """Convert ECG image into patches and embed them."""
    def __init__(self, img_size=224, patch_size=16, in_channels=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, n_patches_h, n_patches_w)
        x = rearrange(x, 'b e h w -> b (h w) e')  # (B, n_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Self Attention module."""
    def __init__(self, embed_dim=768, n_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP."""
    def __init__(self, embed_dim=768, n_heads=12, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with Vision Transformer backbone for ECG analysis.
    
    Following HeartBEiT implementation:
    - 12 layers, 768 hidden dim, 12 attention heads
    - 16x16 patches on 224x224 images
    - 40% masking ratio during pretraining
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=1, 
                 embed_dim=768, depth=12, n_heads=12, mlp_ratio=4., 
                 decoder_embed_dim=512, decoder_depth=2, decoder_n_heads=8,
                 norm_pix_loss=False):
        super().__init__()
        
        # Encoder
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio) 
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim))
        
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_n_heads, mlp_ratio)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, 
                                     patch_size**2 * in_channels, bias=True)
        
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()
        
    def initialize_weights(self):
        # Initialize position embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], 
            int(self.patch_embed.n_patches**.5), 
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], 
            int(self.patch_embed.n_patches**.5), 
            cls_token=True
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        
        # Initialize patch embedding
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # Initialize other parameters
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        
        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def random_masking(self, x, mask_ratio):
        """Random masking following HeartBEiT: mask 40% of patches."""
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, 
                               index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x, mask_ratio):
        # Embed patches
        x = self.patch_embed(x)
        
        # Add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        
        # Masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        # Embed tokens
        x = self.decoder_embed(x)
        
        # Append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, 
                         index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        
        # Add pos embed
        x = x + self.decoder_pos_embed
        
        # Apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # Predictor projection
        x = self.decoder_pred(x)
        
        # Remove cls token
        x = x[:, 1:, :]
        
        return x
    
    def patchify(self, imgs):
        """Convert images to patches."""
        p = self.patch_embed.patch_size
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], 1, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p**2 * 1)
        return x
    
    def unpatchify(self, x):
        """Convert patches back to images."""
        p = self.patch_embed.patch_size
        h = w = int(x.shape[1]**.5)
        x = x.reshape(x.shape[0], h, w, p, p, 1)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], 1, h * p, h * p)
        return imgs
    
    def forward_loss(self, imgs, pred, mask):
        """Compute reconstruction loss."""
        target = self.patchify(imgs)
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
            
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        
        loss = (loss * mask).sum() / mask.sum()
        return loss
    
    def forward(self, imgs, mask_ratio=0.4):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


class HeartBEiT(nn.Module):
    """HeartBEiT model for ECG classification.
    
    Uses pretrained MaskedAutoencoderViT encoder with a classification head.
    """
    def __init__(self, pretrained_mae, num_classes=2, embed_dim=768):
        super().__init__()
        
        # Use pretrained encoder
        self.patch_embed = pretrained_mae.patch_embed
        self.cls_token = pretrained_mae.cls_token
        self.pos_embed = pretrained_mae.pos_embed
        self.blocks = pretrained_mae.blocks
        self.norm = pretrained_mae.norm
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        # Embed patches
        x = self.patch_embed(x)
        
        # Add pos embed
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        # Apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        # Use cls token for classification
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)
        
        return logits


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """Generate 2D sine-cosine positional embeddings."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Generate positional embeddings from grid."""
    H, W = grid.shape[2:]
    
    grid = grid.reshape([2, -1]).T
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[:, 0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[:, 1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Generate 1D sine-cosine positional embeddings."""
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


class ECGDataset(Dataset):
    """Dataset for 12-lead ECG images."""
    def __init__(self, ecg_signals, labels=None, sampling_rate=500, 
                 duration=10, img_size=224, transform=None):
        """
        Args:
            ecg_signals: numpy array of shape (n_samples, n_leads, n_timesteps)
            labels: numpy array of shape (n_samples,) for supervised tasks
            sampling_rate: ECG sampling rate in Hz
            duration: ECG duration in seconds
            img_size: size of output image
            transform: optional transform to apply
        """
        self.ecg_signals = ecg_signals
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.img_size = img_size
        self.transform = transform
        
    def __len__(self):
        return len(self.ecg_signals)
    
    def __getitem__(self, idx):
        ecg = self.ecg_signals[idx]
        
        # Convert ECG signal to image
        img = self.ecg_to_image(ecg)
        
        if self.transform:
            img = self.transform(img)
            
        if self.labels is not None:
            return img, self.labels[idx]
        return img
    
    def ecg_to_image(self, ecg):
        """Convert 12-lead ECG signal to image format.
        
        Following HeartBEiT approach:
        - Create a 224x224 grayscale image
        - Plot 12 leads in standard ECG format
        """
        fig, axes = plt.subplots(12, 1, figsize=(10, 12), dpi=self.img_size/10)
        fig.patch.set_facecolor('white')
        
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
                     'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        for i, (ax, lead_name) in enumerate(zip(axes, lead_names)):
            ax.plot(ecg[i], 'k-', linewidth=0.5)
            ax.set_ylabel(lead_name, fontsize=8)
            ax.set_xlim(0, len(ecg[i]))
            ax.grid(True, alpha=0.3)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            
        plt.tight_layout()
        
        # Convert to numpy array
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        # Convert to grayscale
        img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
        
        # Resize to target size
        from PIL import Image
        img = Image.fromarray(img.astype(np.uint8))
        img = img.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        img = np.array(img).astype(np.float32) / 255.0
        
        # Add channel dimension
        img = np.expand_dims(img, axis=0)
        
        return torch.from_numpy(img).float()


def pretrain_mae(model, train_loader, epochs=100, lr=5e-4, device='cuda'):
    """Pretrain the masked autoencoder following HeartBEiT settings."""
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, imgs in enumerate(train_loader):
            imgs = imgs.to(device)
            
            # Forward pass with 40% masking ratio
            loss, pred, mask = model(imgs, mask_ratio=0.4)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    
    return model


def finetune_classifier(pretrained_mae, train_loader, val_loader, 
                       num_classes=2, epochs=50, lr=1e-4, device='cuda'):
    """Fine-tune for classification following HeartBEiT approach."""
    # Create classifier model
    model = HeartBEiT(pretrained_mae, num_classes)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'heartbeit_best.pth')
            
    return model


# Example usage
if __name__ == "__main__":
    # Create dummy ECG data for demonstration
    # In practice, load real 12-lead ECG data
    n_samples = 1000
    n_leads = 12
    n_timesteps = 5000  # 10 seconds at 500Hz
    
    # Generate synthetic ECG-like signals
    ecg_signals = np.random.randn(n_samples, n_leads, n_timesteps) * 0.1
    labels = np.random.randint(0, 2, n_samples)  # Binary classification
    
    # Create datasets
    train_dataset = ECGDataset(ecg_signals[:800], labels[:800])
    val_dataset = ECGDataset(ecg_signals[800:], labels[800:])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create MAE model
    mae_model = MaskedAutoencoderViT()
    
    # Pretrain
    print("Pretraining MAE...")
    mae_model = pretrain_mae(mae_model, train_loader, epochs=10)
    
    # Fine-tune for classification
    print("\nFine-tuning for classification...")
    classifier = finetune_classifier(mae_model, train_loader, val_loader, epochs=10)
    
    print("\nTraining complete!")
class HeartBEiTMAELightning(pl.LightningModule):
    """PyTorch Lightning module for HeartBEiT Masked Autoencoder pretraining."""
    
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        
        # Create MAE model
        self.mae = MaskedAutoencoderViT(
            img_size=config.get('img_size', 224),
            patch_size=config.get('patch_size', 16),
            in_channels=1,
            embed_dim=config.get('embed_dim', 768),
            depth=config.get('depth', 12),
            n_heads=config.get('n_heads', 12),
            mlp_ratio=config.get('mlp_ratio', 4.),
            decoder_embed_dim=config.get('decoder_embed_dim', 512),
            decoder_depth=config.get('decoder_depth', 2),
            decoder_n_heads=config.get('decoder_n_heads', 8),
            norm_pix_loss=config.get('norm_pix_loss', False)
        )
        
        # Masking ratio
        self.mask_ratio = config.get('mask_ratio', 0.4)
        
        # For visualization
        self.example_input_array = torch.randn(1, 1, 224, 224)
        
    def forward(self, x):
        return self.mae(x, self.mask_ratio)
    
    def training_step(self, batch, batch_idx):
        # The dataloader returns (imgs, wide_feats, labels)
        # For pretraining, we only need the images.
        imgs, _, _ = batch
            
        # Forward pass
        loss, pred, mask = self.mae(imgs, self.mask_ratio)
        
        # Log metrics
        self.log('train/loss', loss, prog_bar=True, logger=True)
        self.log('train/mask_ratio', self.mask_ratio, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # The dataloader returns (imgs, wide_feats, labels)
        imgs, _, _ = batch
            
        # Forward pass
        loss, pred, mask = self.mae(imgs, self.mask_ratio)
        
        # Log metrics
        self.log('val/loss', loss, prog_bar=True, logger=True)
        
        return loss
    
    def configure_optimizers(self):
        # Optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.get('learning_rate', 5e-4),
            weight_decay=self.hparams.get('weight_decay', 0.05),
            betas=(0.9, 0.95)  # Following BERT/MAE settings
        )
        
        # Scheduler with linear warmup and cosine decay
        warmup_epochs = self.hparams.get('warmup_epochs', 5)
        max_epochs = self.hparams.get('max_epochs', 100)

        def lr_lambda(current_epoch):
            if current_epoch < warmup_epochs:
                return float(current_epoch) / float(max(1, warmup_epochs))
            return max(
                0.0, float(max_epochs - current_epoch) / float(max(1, max_epochs - warmup_epochs))
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def on_train_epoch_end(self):
        """Log epoch-level metrics."""
        # Log current learning rate
        lr = self.optimizers().param_groups[0]['lr']
        self.log('train/learning_rate', lr, logger=True)


class HeartBEiTClassifierLightning(pl.LightningModule):
    """PyTorch Lightning module for HeartBEiT classification fine-tuning."""
    
    def __init__(self, config: Dict[str, Any], pretrained_mae: Optional[MaskedAutoencoderViT] = None):
        super().__init__()
        self.save_hyperparameters(config)
        
        # Load pretrained MAE if not provided
        if pretrained_mae is None:
            pretrained_mae = self._load_pretrained_mae()
            
        # Create classifier
        self.classifier = HeartBEiT(
            pretrained_mae,
            num_classes=1,  # Hardcoded for binary classification
            embed_dim=config.get('embed_dim', 768)
        )
        
        # Metrics for binary classification
        self.train_acc = Accuracy(task='binary')
        self.val_acc = Accuracy(task='binary')
        self.test_acc = Accuracy(task='binary')
        self.train_auc = AUROC(task='binary')
        self.val_auc = AUROC(task='binary')
        self.test_auc = AUROC(task='binary')
        self.train_f1 = F1Score(task='binary')
        self.val_f1 = F1Score(task='binary')
        self.test_f1 = F1Score(task='binary')
        self.confusion_matrix = ConfusionMatrix(task='binary')
        
        # Loss function for binary classification
        self.criterion = nn.BCEWithLogitsLoss()

        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        # For visualization
        self.example_input_array = torch.randn(1, 1, 224, 224)
        
    def _load_pretrained_mae(self):
        """Load pretrained MAE from checkpoint."""
        mae = MaskedAutoencoderViT(
            img_size=self.hparams.get('img_size', 224),
            patch_size=self.hparams.get('patch_size', 16),
            in_channels=1,
            embed_dim=self.hparams.get('embed_dim', 768),
            depth=self.hparams.get('depth', 12),
            n_heads=self.hparams.get('n_heads', 12)
        )
        
        if self.hparams.get('mae_checkpoint'):
            checkpoint = torch.load(self.hparams['mae_checkpoint'])
            mae.load_state_dict(checkpoint['state_dict'])
            print(f"Loaded MAE from {self.hparams['mae_checkpoint']}")
            
        return mae
    
    def forward(self, x):
        return self.classifier(x)
    
    def training_step(self, batch, batch_idx):
        imgs, wide_feats, labels = batch
        
        # Forward pass
        logits = self.classifier(imgs)
        loss = self.criterion(logits.squeeze(1), labels.float())
        
        # Calculate metrics
        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs > 0.5).int()
        
        # Update metrics
        self.train_acc(preds, labels)
        self.train_auc(probs, labels)
        self.train_f1(preds, labels)
        
        # Log metrics
        self.log('train/loss', loss, prog_bar=True, logger=True)
        self.log('train/acc', self.train_acc, prog_bar=True, logger=True)
        self.log('train/auc', self.train_auc, logger=True)
        self.log('train/f1', self.train_f1, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        imgs, wide_feats, labels = batch
        
        # Forward pass
        logits = self.classifier(imgs)
        loss = self.criterion(logits.squeeze(1), labels.float())
        
        # Calculate metrics
        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs > 0.5).int()
        
        # Update metrics
        self.val_acc(preds, labels)
        self.val_auc(probs, labels)
        self.val_f1(preds, labels)
        
        # Log metrics
        self.log('val/loss', loss, prog_bar=True, logger=True)
        self.log('val/acc', self.val_acc, prog_bar=True, logger=True)
        self.log('val/auc', self.val_auc, logger=True)
        self.log('val/f1', self.val_f1, logger=True)
        
        output = {'loss': loss, 'preds': preds, 'labels': labels, 'probs': probs}
        self.validation_step_outputs.append(output)
        return output
    
    def test_step(self, batch, batch_idx):
        imgs, wide_feats, labels = batch
        
        # Forward pass
        logits = self.classifier(imgs)
        loss = self.criterion(logits.squeeze(1), labels.float())
        
        # Calculate metrics
        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs > 0.5).int()
        
        # Update metrics
        self.test_acc(preds, labels)
        self.test_auc(probs, labels)
        self.test_f1(preds, labels)
        self.confusion_matrix(preds, labels)
        
        # Log metrics
        self.log('test/loss', loss, logger=True)
        self.log('test/acc', self.test_acc, logger=True)
        self.log('test/auc', self.test_auc, logger=True)
        self.log('test/f1', self.test_f1, logger=True)
        
        output = {'loss': loss, 'preds': preds, 'labels': labels, 'probs': probs}
        self.test_step_outputs.append(output)
        return output
    
    def on_validation_epoch_end(self):
        """Log confusion matrix and other visualizations."""
        # Get confusion matrix
        cm = self.confusion_matrix.compute()
        self.confusion_matrix.reset()
        
        # Calculate and log challenge score
        all_probs = torch.cat([x['probs'] for x in self.validation_step_outputs]).cpu().numpy()
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs]).cpu().numpy()
        challenge_score = utils.compute_challenge_score(all_labels, all_probs)
        self.log('val/challenge_score', challenge_score, prog_bar=True)
        
        self.validation_step_outputs.clear()
        
    def on_test_epoch_end(self):
        """Log final test metrics and visualizations."""
        # Get confusion matrix
        cm = self.confusion_matrix.compute()
        
        # Calculate and log challenge score
        all_probs = torch.cat([x['probs'] for x in self.test_step_outputs]).cpu().numpy()
        all_labels = torch.cat([x['labels'] for x in self.test_step_outputs]).cpu().numpy()
        challenge_score = utils.compute_challenge_score(all_labels, all_probs)
        self.log('test/challenge_score', challenge_score)
        print(f"  Challenge Score: {challenge_score:.4f}")

        self.test_step_outputs.clear()
        
        # Log final metrics
        print(f"\nTest Results:")
        print(f"  Accuracy: {self.test_acc.compute():.4f}")
        print(f"  AUC: {self.test_auc.compute():.4f}")
        print(f"  F1 Score: {self.test_f1.compute():.4f}")
        
    def configure_optimizers(self):
        # Different parameter groups for pretrained encoder and new head
        encoder_params = []
        head_params = []
        
        for name, param in self.classifier.named_parameters():
            if 'head' in name:
                head_params.append(param)
            else:
                encoder_params.append(param)
                
        # Optimizer with different learning rates
        optimizer = AdamW([
            {'params': encoder_params, 'lr': self.hparams.get('encoder_lr', 1e-5)},
            {'params': head_params, 'lr': self.hparams.get('head_lr', 1e-4)}
        ], weight_decay=self.hparams.get('weight_decay', 0.01))
        
        # Scheduler with linear warmup and cosine decay
        warmup_epochs = self.hparams.get('warmup_epochs', 5)
        max_epochs = self.hparams.get('max_epochs', 50)

        def lr_lambda(current_epoch):
            if current_epoch < warmup_epochs:
                return float(current_epoch) / float(max(1, warmup_epochs))
            return max(
                0.0, float(max_epochs - current_epoch) / float(max(1, max_epochs - warmup_epochs))
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def on_train_epoch_end(self):
        """Log learning rates."""
        # Log current learning rates
        for i, param_group in enumerate(self.optimizers().param_groups):
            self.log(f'train/lr_group_{i}', param_group['lr'], logger=True)
        """Log learning rates."""
        # Log current learning rates
        for i, param_group in enumerate(self.optimizers().param_groups):
            self.log(f'train/lr_group_{i}', param_group['lr'], logger=True)
