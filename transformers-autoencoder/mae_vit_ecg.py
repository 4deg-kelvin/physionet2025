"""
MAE-based Vision Transformer for ECG analysis
- ViT backbone adapted for 1D ECG signals
- Masked autoencoder logic
- Pretraining and fine-tuning interfaces
"""
import torch
import torch.nn as nn
import timm
import numpy as np

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
        # Use transformer encoder blocks from a prebuilt ViT model
        vit = timm.create_model('vit_base_patch16_224', pretrained=False)
        self.transformer_encoder_blocks = vit.blocks  # nn.Sequential of transformer layers
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

# Add training and fine-tuning routines in separate scripts
