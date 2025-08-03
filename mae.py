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

# import model_checkpoint
import time
from sklearn.model_selection import train_test_split

# --- Standard Imports ---
import helper_code
import utils
from dataloader import ECGDataset, collate_fn_skip_none, ECGDataModule
from torch.utils.data import DataLoader
import torch.optim as optim
import custom_helper_code

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


class Patching(nn.Module):
    """
    Converts a 12-lead ECG signal into patches with optional overlap and SE-based lead attention.
    """
    def __init__(self, patch_size, overlap_ratio=0.0, use_se=True, num_leads=12):
        super().__init__()
        self.patch_size = patch_size
        self.stride = int(patch_size * (1 - overlap_ratio))
        self.use_se = use_se
        self.num_leads = num_leads
        
        if use_se:
            self.se_block = SEBlock(num_channels=num_leads, reduction_ratio=4)
        
    def forward(self, x):
        # x shape: (batch_size, 12, seq_length)
        x = x.unfold(2, self.patch_size, self.stride)
        # x shape: (batch_size, 12, num_patches, patch_size)
        x = x.permute(0, 2, 1, 3)
        # x shape: (batch_size, num_patches, 12, patch_size)
        
        # Apply SE block for lead-wise attention
        if self.use_se:
            x = self.se_block(x)
        
        # Flatten leads and patch_size dimensions
        x = x.flatten(2)
        # x shape: (batch_size, num_patches, 12 * patch_size)
        return x

class MAEEncoder(nn.Module):
    """
    Transformer-based encoder for the Masked Autoencoder.
    """
    def __init__(self, num_patches, patch_dim, embed_dim, num_heads, num_layers):
        super().__init__()
        self.patch_embedding = nn.Linear(patch_dim, embed_dim)
        
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
        x_embed = self.patch_embedding(x)
        
        # add positional embeddings, slicing if input is shorter than initialized
        seq_len = x_embed.size(1)
        pos = self.pos_embedding
        if pos.size(1) != seq_len:
            pos = pos[:, :seq_len, :]
        x = x_embed + pos
        
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
                 encoder_layers, decoder_layers, masking_ratio=0.75, overlap_ratio=0.5, use_se=True):
        super().__init__()
        self.patching = Patching(patch_size, overlap_ratio, use_se, )
        self.encoder = MAEEncoder(num_patches, patch_dim, embed_dim, num_heads, encoder_layers)
        self.decoder = MAEDecoder(num_patches, patch_dim, embed_dim, num_heads, decoder_layers)
        self.masking_ratio = masking_ratio

    def forward(self, x):
        # Patching
        patches = self.patching(x)
        batch_size, num_patches, patch_dim = patches.shape
        device = patches.device

        # --- Contiguous Temporal Block Masking ---
        num_masked = int(self.masking_ratio * num_patches)
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

        # 4. Create the permutation that restores the original patch order for the decoder.
        shuffled_indices = torch.cat([unmasked_indices, masked_indices], dim=1)
        restore_indices = shuffled_indices.argsort(dim=-1)

        # 5. Encode only the unmasked patches.
        unmasked_patches = torch.gather(patches, 1, 
                                      unmasked_indices.unsqueeze(-1).expand(-1, -1, patch_dim))
        encoded_patches = self.encoder(unmasked_patches)

        # 6. Decode the full sequence of patches.
        decoded_patches = self.decoder(encoded_patches, restore_indices)

        return decoded_patches, patches, masked_indices

class MAELightningModule(pl.LightningModule):
    def __init__(self, patch_size, num_patches, patch_dim, embed_dim, num_heads, 
                 encoder_layers, decoder_layers, masking_ratio=0.75, lr=1e-3, 
                 overlap_ratio=0.0, use_se=True, log_reconstructions=True):
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
            use_se=use_se
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
    EPOCHS = 22  # Increased epochs
    NO_LABELS = True  # No labels for pre-training
    # Learning rate with linear scaling: base_lr * (batch_size / 256)
    LR = 1.5e-4
    USE_SE = True  # Whether to use Squeeze-and-Excitation blocks

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
        log_reconstructions=True
    )
        # Data directories
    DATA_DIR = data_folder
    ADDIT_DIRS = [
        # r"/juice2/scr2/kelvinkn/other_work/edwards/prna_2020_pooled_inputs",
        # r"/juice2/scr2/kelvinkn/other_work/edwards/physionet2021_data",
    ]


    # --- Corrected Data Loading and Splitting ---
    # 1. Load ALL records, for pretraining (since this is official code)
    records_meta = custom_helper_code.find_records(DATA_DIR)
    

    # 2. Combine all records into a single list
    print("WARNING: EXCLUDING UNLABELED RECORDS, THIS IS FOR COMPETITION MODEL TRAINING")
    # all_records = labeled_records + unlabeled_records
    all_records = records_meta
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
    
    # # Train MAE via Lightning with validation monitoring
    # checkpoint_cb = ModelCheckpoint(
    #     dirpath=model_folder,      # save to the user-specified model_folder
    #     filename='mae_encoder_pretrained',       
    #     save_last=True
    # )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator='auto',
        devices=1,
        callbacks=[],
        gradient_clip_val=1.0,  # Added gradient clipping
        num_sanity_val_steps=0, 
        enable_checkpointing=False
    )
    
    # Load weights from a partial checkpoint if it exists, but start a fresh training session.
    partial_ckpt_path = 'mae_encoder_partially_pretrained.ckpt'
    if os.path.isfile(partial_ckpt_path):
        if verbose:
            print(f"Loading weights from partial checkpoint: {partial_ckpt_path}")
        
        # Load the checkpoint dictionary
        checkpoint = torch.load(partial_ckpt_path, map_location='cpu', weights_only=False)
        
        # Check if the checkpoint is a full Lightning checkpoint or just a state_dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Load the state dict into the model, ignoring non-matching keys
        mae_module.load_state_dict(state_dict, strict=False)
    # Search in model folder for a partial checkpoint
    elif os.path.isfile(os.path.join(model_folder, partial_ckpt_path)):
        if verbose:
            print(f"Loading weights from partial checkpoint in model folder: {os.path.join(model_folder, partial_ckpt_path)}")  
        # Load the checkpoint dictionary
        checkpoint = torch.load(os.path.join(model_folder, partial_ckpt_path), map_location='cpu', weights_only=False)
        
        # Check if the checkpoint is a full Lightning checkpoint or just a state_dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Load the state dict into the model, ignoring non-matching keys
        mae_module.load_state_dict(state_dict, strict=False)
    else:
        raise ValueError(f"Partial checkpoint not found at {partial_ckpt_path}. ")


    # fit and validate, starting a new training run without ckpt_path
    trainer.fit(mae_module, datamodule=data_module)
    
    # Save the model weights only to the checkpoint dir
    final_checkpoint_path = os.path.join(model_folder, "mae_encoder_pretrained.ckpt")
    trainer.save_checkpoint(final_checkpoint_path, weights_only=True)
    print(f"Model weights saved to {final_checkpoint_path}")

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
