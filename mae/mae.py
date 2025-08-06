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
import pywt # Added for Daubechies wavelet generation

# import model_checkpoint
import time
from sklearn.model_selection import train_test_split

# Import parent directoyr to access helper_code and utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Standard Imports ---
import helper_code
import utils
from dataloader import ECGDataset, collate_fn_skip_none, ECGDataModule
from torch.utils.data import DataLoader
import torch.optim as optim


# --- Wavelet Convolution Layer (for 'wavelnet' patching type) ---

def scale_wavelet(mother_wavelet, a, len_wavelet):
    wavelet_scaled = torch.zeros(a.size(dim=0), len_wavelet)
    wavelet_scaled = wavelet_scaled.to(mother_wavelet.device)
    # The mother wavelet is reshaped to its own length for interpolation
    mother_wavelet_3d = (mother_wavelet).view(1, 1, -1)
    for i in range(a.size(dim=0)):
        wavelet_scaled_i = F.interpolate(mother_wavelet_3d, scale_factor=a[i].item(), mode='linear', align_corners=False, recompute_scale_factor=True)
        wavelet_scaled_i = wavelet_scaled_i.reshape(-1)
        len_scaled = len(wavelet_scaled_i)
        offset = (len_wavelet - len_scaled) // 2
        if offset >= 0:
            wavelet_scaled[i][offset:offset+len_scaled] = torch.div(wavelet_scaled_i, torch.sqrt(a[i]))

    return wavelet_scaled

class WaveletConv(nn.Module):
        """
        Wavelet-based convolution

        Parameters
        ----------
        n_channels_in : 'int'
            Number of input channels. Must be 1.
        n_channels_out : 'int'
            Number of filters.
        size_kernel : 'int'
            Filter length. Must be an odd number.
        f_sampling : 'int'
            Sampling frequency of input signal.
        f_cufoff_min: 'int'
            Minimal cut-off frequency of filter.
        f_cufoff_min: 'int'
            Maximal cut-off frequency of filter.

        Usage
        -----
        See `torch.nn.Conv1d`
        m = SincConv_NHK(n_channels_in=720,
                        n_channels_out=64,
                        size_kernel=360,
                        f_sampling=360,
                        f_cutoff_min=1.5,
                        f_cutoff_max=64)
        input = torch.randn(50,1,720)
        features = m(input)

        Reference
        ---------
        Mirco Ravanelli, Yoshua Bengio,
        "Speaker Recognition from raw waveform with SincNet".
        https://arxiv.org/abs/1808.00158
        """

        def __init__(self, in_channels, out_channels, kernel_size, fs, mother_wavelet, fs_wavelet, fc_wavelet, a_min=None,
                    stride=1, dilation=1, bias=None, groups=1):
            super(WaveletConv, self).__init__()

            if in_channels != 1:
                msg = 'WaveletConv only supports one input channel (here, n_channels_input = {%i})'\
                    % (in_channels)
                raise ValueError(msg)
            if kernel_size % 2 != 1:
                msg = 'WaveletConv only supports an odd number as the size of kernel (here, size_kernel = {%i})'\
                    % (kernel_size)
                raise ValueError(msg)
            if bias:
                raise ValueError('WaveletConv does not support bias.')
            if groups > 1:
                raise ValueError('WaveletConv does not support groups.')

            if np.abs(np.sum(mother_wavelet)) > 0.01:
                print(np.abs(np.sum(mother_wavelet)))
                raise ValueError('Mother wavelet does not satisfy zero mean condition.')
            if np.abs(np.sum(mother_wavelet**2) - 1) > 0.01:
                print(np.abs(np.sum(mother_wavelet**2) - 1))
                raise ValueError('Mother wavelet does not satisfy square norm one condition.')
            if not a_min is None:
                if a_min >= 1 or a_min <= 0:
                    raise ValueError('Minimum scale parameter should be larger than 0 and smaller 1.')
            if a_min is None:
                # determine the lower bound of scale parameter
                a_min = max(fc_wavelet / (fs/2), 11/kernel_size)

            self.n_channels_in = in_channels
            self.n_channels_out = out_channels
            self.size_kernel = kernel_size
            self.mother_wavelet = mother_wavelet
            self.a_min = a_min
            self.stride = stride
            self.padding = int(np.floor(self.size_kernel/2))
            self.dilation = dilation
            self.bias = bias
            self.groups = groups

            # initialize scale parameters of wavelets (which would be used as kernels)
            # they are evenly spaced
            a_init = np.linspace(self.a_min, 1, self.n_channels_out)

            # set scale parameters to be learnable
            # a : (n_channels_out, 1)
            self.a = torch.nn.Parameter(torch.Tensor(a_init).view(-1, 1))

            self.mother_wavelet = torch.tensor(self.mother_wavelet, dtype=torch.float32)

            self.wavelet_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1),
            nn.Sigmoid()
            )
            self.wavelet_mixer = nn.Conv1d(out_channels, out_channels, 1)
        def forward(self, input):
            """
            Wavelet-based convolution
            Optimized by Namho Kim

            Parameters
            ----------
            input : 'torch.Tensor' (batch_size, 1, n_samples)
                Batch of input signals.

            Returns
            -----
            features : 'torch.Tensor' (batch_size, n_channels_out, n_samples)
                Batch of wavelet filters activations.
            """
            self.mother_wavelet = self.mother_wavelet.to(input.device)

            # Generate wavelets as kernels
            a_prac = torch.clamp(input=self.a, min=self.a_min, max=1)
            a_prac = a_prac.to(input.device)
            wavelet_kernels = scale_wavelet(self.mother_wavelet, a_prac, self.size_kernel)

            self.wavelet_kernels_final = (wavelet_kernels).view(self.n_channels_out, 1, self.size_kernel)

            x =  nn.functional.conv1d(input=input,
                                        weight=self.wavelet_kernels_final,
                                        stride=self.stride,
                                        padding=self.padding,
                                        dilation=self.dilation,
                                        bias=self.bias,
                                        groups=self.groups)
            # Apply wavelet gate and mixer
            gate = self.wavelet_gate(x)
            x = x * gate
            x = self.wavelet_mixer(x)
            return x


# --- 1. Model Components ---

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    """
    def __init__(self, num_channels, reduction_ratio=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False),
            nn.Sigmoid()
        )
        
        # Initialize to near-identity function
        with torch.no_grad():
            # Last layer weights initialized to create outputs near 1.0 after sigmoid
            self.excitation[-2].weight.data.fill_(0.1)
    
    def forward(self, x):
        # Handle different input formats based on dimensions
        if x.dim() == 4:
            # x shape: (batch, num_patches, num_channels, channel_size)
            B, N, C, S = x.shape
            
            # Squeeze: Global average pooling across channel dimension
            x_squeezed = x.mean(dim=-1)  # (B, N, C)
            
            # Process each patch's channel weights
            x_flat = x_squeezed.view(B * N, C)
            weights = self.excitation(x_flat)  # (B*N, C)
            weights = weights.view(B, N, C, 1)  # (B, N, C, 1)
            
            # Excitation: Scale channels by weights
            return x * weights
        else:
            # x shape: (batch, channels, length)
            b, c, _ = x.shape
            # Squeeze: Global average pooling across length dimension
            y = self.squeeze(x).view(b, c)
            # Excitation:
            y = self.excitation(y).view(b, c, 1)
            # Scale channels by weights
            return x * y.expand_as(x)

class Patching(nn.Module):
    """
    Converts a 12-lead ECG signal into patch embeddings.
    Supports 'linear' (standard ViT) and 'wavelnet' (convolutional stem) patching.
    
    Changes:
    - Moved the SE block to operate on the feature maps after the WaveletConv layers,
      allowing it to recalibrate based on extracted features rather than raw input.
    """
    def __init__(self, patch_size, overlap_ratio=0.0, use_se=True, num_leads=12,
                 patching_type='linear', embed_dim=768, wavelnet_out_channels=32):
        super().__init__()
        self.patch_size = patch_size
        self.stride = int(patch_size * (1 - overlap_ratio))
        self.use_se = use_se
        self.num_leads = num_leads
        self.patching_type = patching_type
        self.se_block = None # Initialize as None

        if self.patching_type == 'wavelnet':
            # Create a Daubechies mother wavelet
            db_wavelet = pywt.Wavelet('db6')
            mother_wavelet_func = db_wavelet.wavefun(level=8)[1] # Use the wavelet function psi
            mother_wavelet = mother_wavelet_func - np.mean(mother_wavelet_func)
            mother_wavelet /= np.sqrt(np.sum(mother_wavelet**2))
            
            # Define wavelet properties
            fs_wavelet = 1.0 # Normalized frequency for pywt's internal representation
            fc_wavelet = pywt.central_frequency('db6', precision=8) # Center frequency of db6
            
            # Create a WaveletConv for each lead
            self.wavelnet_convs = nn.ModuleList([
                WaveletConv(
                    in_channels=1, 
                    out_channels=wavelnet_out_channels, 
                    kernel_size=129, 
                    fs=utils.UNIFIED_FREQUENCY,  # Uncommented to provide the required fs parameter
                    mother_wavelet=mother_wavelet,
                    fs_wavelet=fs_wavelet,
                    fc_wavelet=fc_wavelet
                )
                for _ in range(num_leads)
            ])
            
            if self.use_se:
                # MODIFIED: SE block is now defined on the output channels of the wavelnet
                num_se_channels = num_leads * wavelnet_out_channels
                self.se_block = SEBlock(num_channels=num_se_channels, reduction_ratio=16) # Increased reduction for more channels

            patch_dim = 2 * num_leads * wavelnet_out_channels
            self.patch_embedding = nn.Linear(patch_dim, embed_dim)
        
        elif self.patching_type == 'linear':
            if self.use_se:
                # In the linear case, the SE block operates on the original leads
                self.se_block = SEBlock(num_channels=num_leads, reduction_ratio=4)

            patch_dim = 2 * num_leads
            self.patch_embedding = nn.Linear(patch_dim, embed_dim)
        else:
            raise ValueError(f"Unknown patching_type: {patching_type}")

    def forward(self, x):
        # Input x shape: (batch_size, num_leads, seq_length)
        
        if self.patching_type == 'wavelnet':
            lead_outputs = []
            for i in range(self.num_leads):
                lead_input = x[:, i:i+1, :]
                lead_output = F.relu(self.wavelnet_convs[i](lead_input))
                lead_outputs.append(lead_output)
            x = torch.cat(lead_outputs, dim=1) # (B, num_leads * wavelnet_out_channels, L)
            
            # MODIFIED: Apply SE block AFTER feature extraction for more meaningful recalibration
            if self.use_se:
                x = self.se_block(x)

        elif self.patching_type == 'linear':
             # MODIFIED: In the linear case, apply SE block to the raw input
             if self.use_se:
                x = self.se_block(x)

        # Unfold the signal into patches.
        # Shape: (B, C, L) -> (B, C, num_patches, patch_size)
        x = x.unfold(2, self.patch_size, self.stride)
        
        # Permute to bring num_patches forward.
        # Shape: (B, C, num_patches, patch_size) -> (B, num_patches, C, patch_size)
        x = x.permute(0, 2, 1, 3)

        # Perform mixed pooling over the patch_size dimension (dim=3).
        x_mean = x.mean(dim=3)
        x_max = x.max(dim=3)[0] # .max() returns (values, indices), we only need values
        x = torch.cat((x_mean, x_max), dim=2) # Concatenate along the channel dimension
        # Shape: (B, num_patches, 2 * C)
        
        # Project to the embedding dimension.
        x = self.patch_embedding(x)
        # Final shape: (batch_size, num_patches, embed_dim)
        return x

class TransformerEncoderLayerWithSE(nn.TransformerEncoderLayer):
    """SE block applied to the hidden feedforward dimension"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=F.relu, layer_norm_eps=1e-5, batch_first=False,
                 norm_first=False, device=None, dtype=None):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                         layer_norm_eps, batch_first, norm_first, device, dtype)
        
        # SE block for the feedforward hidden dimension
        self.se_block = SEBlock(num_channels=dim_feedforward, reduction_ratio=16)

    def _ff_block(self, x):
        # Apply SE in the middle of FFN
        x = self.linear1(x)  # (B, N, d_model) -> (B, N, dim_feedforward)
        x = self.activation(x)
        
        # Apply SE block on the expanded dimension
        x = x.permute(0, 2, 1)  # (B, dim_feedforward, N)
        x = self.se_block(x)
        x = x.permute(0, 2, 1)  # (B, N, dim_feedforward)
        
        x = self.dropout(x)
        x = self.linear2(x)  # (B, N, dim_feedforward) -> (B, N, d_model)
        return self.dropout2(x)

class MAEEncoder(nn.Module):
    """
    Transformer-based encoder for the Masked Autoencoder.
    """
    def __init__(self, num_patches, patch_dim, embed_dim, num_heads, num_layers):
        super().__init__()
        # self.patch_embedding = nn.Linear(patch_dim, embed_dim) # This is now handled in Patching
        
        # Initialize positional embeddings with proper initialization
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            batch_first=True, 
            dim_feedforward=4*embed_dim,  # Reduced from 6x to 4x for efficiency
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Add layer normalization
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # apply patch embedding
        # x_embed = self.patch_embedding(x) # Patches are already embedded
        
        # add positional embeddings, slicing if input is shorter than initialized
        seq_len = x.size(1)
        pos = self.pos_embedding
        if pos.size(1) != seq_len:
            pos = pos[:, :seq_len, :]
        x = x + pos
        
        # encode
        x = self.transformer_encoder(x)
        
        # normalize
        x = self.norm(x)
        
        return x

class MAEDecoder(nn.Module):
    """
    Lightweight decoder for the Masked Autoencoder.
    """
    def __init__(self, num_patches, patch_dim, embed_dim, num_heads, num_layers):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # Initialize positional embeddings properly
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        
        # Decoder layers with dropout
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dropout=0.1, 
            batch_first=True,
            dim_feedforward=4*embed_dim
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.decoder_pred = nn.Linear(embed_dim, patch_dim)
        
        # Layer norm before output
        self.norm = nn.LayerNorm(embed_dim)
        
        # Dropout after decoding
        self.decoder_dropout = nn.Dropout(0.1)

    def forward(self, encoded_patches, restore_indices):
        B, U, D = encoded_patches.shape
        N = restore_indices.shape[1]
        M = N - U

        # build mask tokens
        mask_tokens = self.mask_token.repeat(B, M, 1)

        # reconstruct full sequence (unshuffled)
        x_shuffled = torch.cat([encoded_patches, mask_tokens], dim=1)  # (B, U+M, D)
        full_seq = torch.gather(
            x_shuffled, 
            1,
            restore_indices.unsqueeze(-1).expand(-1, -1, D)
        )  # (B, N, D)
        
        # Add positional embeddings
        full_pos = self.pos_embedding[:, :N, :]  # Handle case where N < num_patches
        full_seq = full_seq + full_pos

        # Find indices for masked and unmasked positions in the restored sequence
        # restore_indices tells us where each position in x_shuffled goes
        # Positions 0:U in x_shuffled are unmasked, U:N are masked
        arange = torch.arange(N, device=encoded_patches.device)
        
        # Get the positions where unmasked and masked patches ended up
        unmasked_positions = restore_indices[:, :U]  # Where unmasked patches went
        masked_positions = restore_indices[:, U:]    # Where masked patches went
        
        # Gather memory (unmasked) and target (masked) sequences
        memory_indices = unmasked_positions.unsqueeze(-1).expand(-1, -1, D)
        tgt_indices = masked_positions.unsqueeze(-1).expand(-1, -1, D)
        
        memory = torch.gather(full_seq, 1, memory_indices)  # (B, U, D)
        tgt = torch.gather(full_seq, 1, tgt_indices)      # (B, M, D)

        # Decode only the masked positions using cross-attention to memory
        decoded_tgt = self.transformer_decoder(tgt, memory)  # (B, M, D)
        decoded_tgt = self.norm(decoded_tgt)
        decoded_tgt = self.decoder_dropout(decoded_tgt)
        pred_tgt = self.decoder_pred(decoded_tgt)  # (B, M, patch_dim)

        # Create output tensor and place predictions at masked positions
        recon = torch.zeros(B, N, pred_tgt.size(-1), device=pred_tgt.device)
        
        # Scatter predictions back to their original positions
        recon.scatter_(1, 
                      masked_positions.unsqueeze(-1).expand(-1, -1, pred_tgt.size(-1)), 
                      pred_tgt)
        
        return recon

class MaskedAutoencoder(nn.Module):
    """
    Masked Autoencoder for 12-lead ECG with Contiguous Block Masking.
    """
    def __init__(self, patch_size, num_patches, patch_dim, embed_dim, num_heads, 
                 encoder_layers, decoder_layers, masking_ratio=0.75, overlap_ratio=0.5, use_se=True,
                 masking_type='contiguous', patching_type='linear', wavelnet_out_channels=32):
        super().__init__()
        self.patching = Patching(
            patch_size=patch_size, 
            overlap_ratio=overlap_ratio, 
            use_se=use_se,
            patching_type=patching_type,
            embed_dim=embed_dim,
            wavelnet_out_channels=wavelnet_out_channels
        )
        self.encoder = MAEEncoder(num_patches, patch_dim, embed_dim, num_heads, encoder_layers)
        self.decoder = MAEDecoder(num_patches, patch_dim, embed_dim, num_heads, decoder_layers)
        self.masking_ratio = masking_ratio
        self.masking_type = masking_type

    def forward(self, x):
        # Patching and Embedding
        embedded_patches = self.patching(x)
        batch_size, num_patches, embed_dim = embedded_patches.shape
        device = embedded_patches.device

        # --- Contiguous Temporal Block Masking ---
        num_masked = int(self.masking_ratio * num_patches)
        
        if self.masking_type == 'contiguous':
            # --- Contiguous Temporal Block Masking ---
            num_unmasked = num_patches - num_masked

            # 1. For each sample, find a random start index for the mask block.
            start_idx = torch.randint(0, num_patches, (batch_size, 1), device=device)
            
            # 2. Create the indices for the contiguous block to be masked.
            mask_range = torch.arange(num_masked, device=device).unsqueeze(0)
            masked_indices = (start_idx + mask_range) % num_patches
            
            # 3. Determine the unmasked indices.
            full_indices = torch.arange(num_patches, device=device).unsqueeze(0).repeat(batch_size, 1)
            bool_mask = torch.ones_like(full_indices, dtype=torch.bool)
            bool_mask.scatter_(1, masked_indices, False)
            unmasked_indices = full_indices[bool_mask].reshape(batch_size, -1)
        
        elif self.masking_type == 'random':
            # --- Random Point Masking ---
            # 1. Generate a random permutation of indices for each sample.
            noise = torch.rand(batch_size, num_patches, device=device)
            ids_shuffle = torch.argsort(noise, dim=1)
            
            # 2. Split into masked and unmasked indices.
            masked_indices = ids_shuffle[:, :num_masked]
            unmasked_indices = ids_shuffle[:, num_masked:]
            
            # 3. Sort unmasked indices to maintain some temporal order (optional but common).
            unmasked_indices = torch.sort(unmasked_indices, dim=1)[0]
        else:
            raise ValueError(f"Unknown masking type: {self.masking_type}")

        # 4. Create the permutation that restores the original patch order for the decoder.
        shuffled_indices = torch.cat([unmasked_indices, masked_indices], dim=1)
        restore_indices = shuffled_indices.argsort(dim=-1)

        # 5. Encode only the unmasked patches.
        unmasked_patches = torch.gather(embedded_patches, 1, 
                                      unmasked_indices.unsqueeze(-1).expand(-1, -1, embed_dim))
        encoded_patches = self.encoder(unmasked_patches)

        # 6. Decode the full sequence of patches.
        decoded_patches = self.decoder(encoded_patches, restore_indices)

        # We need the original, non-embedded patches for the loss function
        original_patches = x.unfold(2, self.patching.patch_size, self.patching.stride).permute(0, 2, 1, 3).flatten(2)

        return decoded_patches, original_patches, masked_indices

class MAELightningModule(pl.LightningModule):
    def __init__(self, patch_size, num_patches, patch_dim, embed_dim, num_heads, 
                 encoder_layers, decoder_layers, masking_ratio=0.75, lr=1e-3, 
                 overlap_ratio=0.0, use_se=True, log_reconstructions=True,
                 masking_type='contiguous', patching_type='linear', wavelnet_out_channels=32):
        super().__init__()
        # Save hyperparameters (now includes use_se)
        self.save_hyperparameters()

        # Define the model
        self.model = MaskedAutoencoder(
            patch_size=patch_size,
            num_patches=num_patches,
            patch_dim=patch_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            masking_ratio=masking_ratio,
            overlap_ratio=overlap_ratio,
            use_se=use_se,
            masking_type=masking_type,
            patching_type=patching_type,
            wavelnet_out_channels=wavelnet_out_channels
        )
        self.criterion = nn.MSELoss(reduction='none')  # Changed to 'none' for efficient loss computation

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx):
        # batch[0] is the ECG signal tensor
        signals = batch[0]
        
        # forward
        reconstructed, original, masked_indices = self(signals)
        
        # Efficient loss computation only on masked patches
        B = signals.shape[0]
        total_loss = 0
        
        for i in range(B):
            masked_idx = masked_indices[i]
            # Compute loss only for masked patches
            patch_loss = self.criterion(
                reconstructed[i, masked_idx], 
                original[i, masked_idx]
            )
            total_loss += patch_loss.mean()
        
        loss = total_loss / B
        
        return loss
    


    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('learning_rate', self.optimizers().param_groups[0]['lr'], prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Use AdamW optimizer
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.05)
        
        # Cosine annealing with warmup
        def lr_lambda(epoch):
            warmup_epochs = 5
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (self.trainer.max_epochs - warmup_epochs)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
if __name__ == "__main__":
    pl.seed_everything(42)  # For reproducibility
    # Hyperparameters
    SEQ_LENGTH = utils.UNIFIED_FREQUENCY * 10
    PATCH_SIZE = 125
    OVERLAP_RATIO = 0.5  # 50% overlap between patches
    NUM_LEADS = 12
    
    # Calculate number of patches with overlap
    stride = int(PATCH_SIZE * (1 - OVERLAP_RATIO))
    NUM_PATCHES = (SEQ_LENGTH - PATCH_SIZE) // stride + 1
    
    PATCH_DIM = NUM_LEADS * PATCH_SIZE
    EMBED_DIM = 768
    NUM_HEADS = 12
    ENCODER_LAYERS = 8
    DECODER_LAYERS = 2  # Increased from 1 for better reconstruction
    MASKING_RATIO = 0.75  # Increased back to standard MAE ratio
    BATCH_SIZE = 32  # Reduced slightly for memory with overlap
    EPOCHS = 20  # Increased epochs
    NO_LABELS = True  # No labels for pre-training
    # Learning rate with linear scaling: base_lr * (batch_size / 256)
    LR = 1.5e-4
    USE_SE = True  # Whether to use Squeeze-and-Excitation blocks
    MASKING_TYPE = 'contiguous' # 'contiguous' or 'random'
    PATCHING_TYPE = 'wavelnet' # 'linear' or 'wavelnet'
    WAVELNET_OUT_CHANNELS = 32 # Only used if PATCHING_TYPE is 'wavelnet'

    # --- WandB and Config Setup ---
    config = {
        "seq_length": SEQ_LENGTH, "patch_size": PATCH_SIZE, "overlap_ratio": OVERLAP_RATIO,
        "num_leads": NUM_LEADS, "num_patches": NUM_PATCHES, "patch_dim": PATCH_DIM,
        "embed_dim": EMBED_DIM, "num_heads": NUM_HEADS, "encoder_layers": ENCODER_LAYERS,
        "decoder_layers": DECODER_LAYERS, "masking_ratio": MASKING_RATIO,
        "batch_size": BATCH_SIZE, "epochs": EPOCHS, "lr": LR, "use_se": USE_SE,
        "masking_type": MASKING_TYPE,
        "patching_type": PATCHING_TYPE,
        "wavelnet_out_channels": WAVELNET_OUT_CHANNELS
    }

    wandb_logger = WandbLogger(
        entity="edwards_physionet",
        project="ecg_mae",
        name=f"ecg_mae_{MASKING_TYPE}_{PATCHING_TYPE}",
        config=config
    )

    if NO_LABELS:
        print("WARNING: Pre-training without labels. This is intended for unsupervised pre-training only.")

    # Instantiate Lightning module for MAE with SE flag
    mae_module = MAELightningModule(
        patch_size=PATCH_SIZE,
        num_patches=NUM_PATCHES,
        patch_dim=PATCH_DIM,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        encoder_layers=ENCODER_LAYERS,
        decoder_layers=DECODER_LAYERS,
        masking_ratio=MASKING_RATIO,
        lr=LR,
        overlap_ratio=OVERLAP_RATIO,
        use_se=USE_SE,
        log_reconstructions=False,
        masking_type=MASKING_TYPE,
        patching_type=PATCHING_TYPE,
        wavelnet_out_channels=WAVELNET_OUT_CHANNELS
    )
        # Data directories
    DATA_DIR = "../training_data"
    ADDIT_DIRS = [
        r"/juice2/scr2/kelvinkn/other_work/edwards/prna_2020_pooled_inputs",
        r"/juice2/scr2/kelvinkn/other_work/edwards/physionet2021_data",
    ]


    # --- Corrected Data Loading and Splitting ---
    # 1. Load ALL records, for pretraining (since this is official code)
    records_meta = utils.prepare_stratification(helper_code.find_records_abs(DATA_DIR))
    
    # Exclude PTB-XL and SaMi-Trop
    records_meta = [rec['record'] for rec in records_meta if rec['source'] not in ['PTB-XL', 'SaMi-Trop']]

    addit_data = []
    # Also load the addit_dirs 
    for addit_dir in ADDIT_DIRS:
        addit_data += helper_code.find_records_abs(addit_dir)
    

    # 2. Combine all records into a single list
    # print("WARNING: EXCLUDING UNLABELED RECORDS, THIS IS FOR COMPETITION MODEL TRAINING")
    # all_records = labeled_records + unlabeled_records
    all_records = records_meta + addit_data     
    print(f"Total records found: {len(all_records)}")
    if len(all_records) == 0:
        raise ValueError("No records found in the specified directories. Please check the paths.")

    np.random.shuffle(all_records)
    
    # Split records into train, validation, and test sets (80/10/10)
    total_size = len(all_records)
    val_test_size = int(0.2 * total_size)
    val_size = val_test_size // 2
    
    train_records = all_records[:-val_test_size]
    val_records = all_records[-val_test_size:-val_size]
    test_records = all_records[-val_size:]

    print(f"Training records: {len(train_records)}")
    print(f"Validation records: {len(val_records)}")
    print(f"Test records: {len(test_records)}")

    # Wire up ECGDataModule
    data_module = ECGDataModule(
        data_dir=DATA_DIR, # Base directory, not strictly needed since paths are absolute
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LENGTH,
        windowing_method='entire_recording'
    )
    data_module.train_dataset = ECGDataset(train_records, DATA_DIR, is_training=True, no_labels=NO_LABELS,
                                            seq_len=SEQ_LENGTH, windowing_method='entire_recording')
    data_module.val_dataset = ECGDataset(val_records, DATA_DIR, is_training=False, no_labels=NO_LABELS,
                                            seq_len=SEQ_LENGTH, windowing_method='entire_recording')
    data_module.test_dataset = ECGDataset(test_records, DATA_DIR, is_training=False, no_labels=NO_LABELS,
                                            seq_len=SEQ_LENGTH, windowing_method='entire_recording')
    
    # Train MAE via Lightning with validation monitoring
    checkpoint_cb = ModelCheckpoint(
        filename='mae_encoder_pretrained',       
        save_last=True,
        monitor='val_loss',
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator='auto',
        devices=1,
        callbacks=[checkpoint_cb],
        gradient_clip_val=1.0,  # Added gradient clipping
        logger=wandb_logger
    )
    
    # fit and validate, starting a new training run without ckpt_path
    trainer.fit(mae_module, datamodule=data_module)
def train_model(data_folder, model_folder, verbose):
    pl.seed_everything(42)  # For reproducibility
    # Hyperparameters
    SEQ_LENGTH = utils.UNIFIED_FREQUENCY * 10
    PATCH_SIZE = 125
    OVERLAP_RATIO = 0.5  # 50% overlap between patches
    NUM_LEADS = 12
    
    # Calculate number of patches with overlap
    stride = int(PATCH_SIZE * (1 - OVERLAP_RATIO))
    NUM_PATCHES = (SEQ_LENGTH - PATCH_SIZE) // stride + 1
    
    PATCH_DIM = NUM_LEADS * PATCH_SIZE
    EMBED_DIM = 768
    NUM_HEADS = 12
    ENCODER_LAYERS = 8
    DECODER_LAYERS = 2  # Increased from 1 for better reconstruction
    MASKING_RATIO = 0.75  # Increased back to standard MAE ratio
    BATCH_SIZE = 32  # Reduced slightly for memory with overlap
    EPOCHS = 20  # Increased epochs
    NO_LABELS = True  # No labels for pre-training
    # Learning rate with linear scaling: base_lr * (batch_size / 256)
    LR = 1.5e-4
    USE_SE = True  # Whether to use Squeeze-and-Excitation blocks
    MASKING_TYPE = 'contiguous' # 'contiguous' or 'random'
    PATCHING_TYPE = 'wavelnet' # 'linear' or 'wavelnet'
    WAVELNET_OUT_CHANNELS = 8 # Only used if PATCHING_TYPE is 'wavelnet'

    if NO_LABELS:
        print("WARNING: Pre-training without labels. This is intended for unsupervised pre-training only.")

    # Instantiate Lightning module for MAE with SE flag
    mae_module = MAELightningModule(
        patch_size=PATCH_SIZE,
        num_patches=NUM_PATCHES,
        patch_dim=PATCH_DIM,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        encoder_layers=ENCODER_LAYERS,
        decoder_layers=DECODER_LAYERS,
        masking_ratio=MASKING_RATIO,
        lr=LR,
        overlap_ratio=OVERLAP_RATIO,
        use_se=USE_SE,
        log_reconstructions=False,
        masking_type=MASKING_TYPE,
        patching_type=PATCHING_TYPE,
        wavelnet_out_channels=WAVELNET_OUT_CHANNELS
    )
        # Data directories
    DATA_DIR = data_folder
    ADDIT_DIRS = [
        r"/juice2/scr2/kelvinkn/other_work/edwards/prna_2020_pooled_inputs",
        r"/juice2/scr2/kelvinkn/other_work/edwards/physionet2021_data",
    ]


    print("Preparing stratification and loading records...")
    # --- Corrected Data Loading and Splitting ---
    # 1. Load ALL records, for pretraining (since this is official code)
    records_meta = utils.prepare_stratification(helper_code.find_records_abs(DATA_DIR))
    
    # Exclude PTB-XL and SaMi-Trop
    records_meta = [rec['record'] for rec in records_meta if rec['source'] not in ['PTB-XL', 'SaMi-Trop']]

    addit_data = []
    # Also load the addit_dirs 
    for addit_dir in ADDIT_DIRS:
        addit_data += helper_code.find_records_abs(addit_dir)
    

    # 2. Combine all records into a single list
    # print("WARNING: EXCLUDING UNLABELED RECORDS, THIS IS FOR COMPETITION MODEL TRAINING")
    # all_records = labeled_records + unlabeled_records
    all_records = records_meta + addit_data     
    print(f"Total records found: {len(all_records)}")
    if len(all_records) == 0:
        raise ValueError("No records found in the specified directories. Please check the paths.")

    np.random.shuffle(all_records)
    train_records = all_records  # use all data for training

    # Wire up ECGDataModule
    data_module = ECGDataModule(
        data_dir=DATA_DIR, # Base directory, not strictly needed since paths are absolute
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LENGTH,
        windowing_method='entire_recording'
    )
    data_module.train_dataset = ECGDataset(train_records, DATA_DIR, is_training=True, no_labels=NO_LABELS,
                                           seq_len=SEQ_LENGTH, windowing_method='entire_recording')
    # No validation or test sets for pre-training
    
    # Train MAE via Lightning with validation monitoring
    checkpoint_cb = ModelCheckpoint(
        dirpath=model_folder,      # save to the user-specified model_folder
        filename='mae_encoder_pretrained',       
        save_last=True
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator='auto',
        devices=1,
        callbacks=[checkpoint_cb],
        gradient_clip_val=1.0,  # Added gradient clipping
    )
    
    # Load weights from a partial checkpoint if it exists, but start a fresh training session.
    partial_ckpt_path = os.path.join(model_folder, 'mae_encoder_partially_pretrained.ckpt')
    if os.path.isfile(partial_ckpt_path):
        if verbose:
            print(f"Loading weights from partial checkpoint: {partial_ckpt_path}")
        # Load the state dict from the checkpoint file.
        # Using strict=False allows us to load only the model weights
        # and ignore other parts of the checkpoint like optimizer states.
        state_dict = torch.load(partial_ckpt_path, map_location='cpu')['state_dict']
        mae_module.load_state_dict(state_dict, strict=False)
    elif verbose:
        print("Partial checkpoint not found. Starting pre-training from scratch.")

    # fit and validate, starting a new training run without ckpt_path
    trainer.fit(mae_module, datamodule=data_module)
    
    # # Get best checkpoint path
    # best_model_path = checkpoint_cb.best_model_path
    # print(f"Best model checkpoint path: {best_model_path}")

    # # Test the best model
    # trainer.test(datamodule=data_module, ckpt_path=best_model_path)
    
    # # Load the best model to save its encoder
    # best_mae_module = MAELightningModule.load_from_checkpoint(best_model_path)

    # # Save the pre-trained encoder weights from the best model
    # encoder_save_path = 'mae_encoder_pretrained_improved.pth'
    # torch.save(best_mae_module.model.encoder.state_dict(), encoder_save_path)
    # print(f"Pre-trained encoder from best checkpoint saved to {encoder_save_path}")

    # # Log the source checkpoint to wandb summary
    # if wandb.run:
    #     wandb.summary['best_checkpoint_path'] = best_model_path
    #     wandb.summary['encoder_save_path'] = encoder_save_path