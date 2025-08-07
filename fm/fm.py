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

import torch
import torch.nn as nn
from fairseq_signals.models import build_model_from_checkpoint
from fairseq_signals.models.classification.ecg_transformer_classifier import ECGTransformerClassificationModel
import pytorch_lightning as pl

class ECGFMFeatureExtractor(nn.Module):
    """
    Wrapper around ECG-FM model to extract features for downstream tasks.
    """
    def __init__(self, checkpoint_path, freeze_encoder=True):
        super().__init__()
        # Load pretrained ECG-FM model
        self.ecg_fm_model = build_model_from_checkpoint(checkpoint_path)
        self.ecg_fm_model.eval()
        
        # Freeze encoder weights if specified
        if freeze_encoder:
            for param in self.ecg_fm_model.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """
        Extract features from ECG-FM encoder
        Args:
            x: (batch_size, 12, seq_length) ECG signals
        Returns:
            features: (batch_size, embed_dim) pooled features
        """
        # ECG-FM expects (batch_size, 12, 2500) for 5-second segments at 500Hz
        # You may need to segment your 10-second signals
        batch_size, num_leads, seq_length = x.shape
        
        if seq_length != 2500:  # ECG-FM expects 5-second segments
            # Segment into overlapping 5-second windows
            segment_length = 2500
            stride = 1250  # 50% overlap
            segments = []
            
            for i in range(0, seq_length - segment_length + 1, stride):
                segment = x[:, :, i:i + segment_length]
                segments.append(segment)
            
            # Process all segments
            all_features = []
            for segment in segments:
                with torch.no_grad():
                    out = self.ecg_fm_model(source=segment)
                    # Extract encoder output and pool
                    encoder_out = out['encoder_out']  # (batch, seq_len, embed_dim)
                    # Global average pooling
                    features = torch.div(encoder_out.sum(dim=1), (encoder_out != 0).sum(dim=1))
                    all_features.append(features)
            
            # Average features across segments
            pooled_features = torch.stack(all_features, dim=1).mean(dim=1)
        else:
            # Single 5-second segment
            with torch.no_grad():
                out = self.ecg_fm_model(source=x)
                encoder_out = out['encoder_out']
                pooled_features = torch.div(encoder_out.sum(dim=1), (encoder_out != 0).sum(dim=1))
        
        return pooled_features

class FMChagasClassifier(pl.LightningModule):
    """
    Chagas disease classifier using ECG-FM pretrained features.
    """
    def __init__(self, ecg_fm_checkpoint_path, num_classes=1, lr=1e-4, freeze_encoder=True):
        super().__init__()
        self.save_hyperparameters()
        
        # ECG-FM feature extractor
        self.feature_extractor = ECGFMFeatureExtractor(
            ecg_fm_checkpoint_path, 
            freeze_encoder=freeze_encoder
        )
        
        # Get feature dimension from ECG-FM (typically 768)
        self.feature_dim = 768  # ECG-FM embedding dimension
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        self.criterion = nn.BCEWithLogitsLoss()
        
    def forward(self, x):
        # Extract features using ECG-FM
        features = self.feature_extractor(x)
        # Classify
        logits = self.classifier(features)
        return logits
    
    def _common_step(self, batch, batch_idx):
        signals, labels = batch[0], batch[1]
        logits = self(signals)
        loss = self.criterion(logits, labels.float())
        probs = torch.sigmoid(logits)
        return loss, probs, labels
    
    def training_step(self, batch, batch_idx):
        loss, probs, labels = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, probs, labels = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return {'val_loss': loss, 'probs': probs, 'labels': labels}
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [scheduler]

def train_model(data_folder, model_folder, checkpoint_path, verbose):
    pl.seed_everything(42)  # For reproducibility
    # Hyperparameters
    SEQ_LENGTH = utils.UNIFIED_FREQUENCY * 5  
    BATCH_SIZE = 16  # Smaller batch size due to ECG-FM memory requirements
    EPOCHS = 50
    LR = 1e-4
    NO_LABELS = False  # We need labels for Chagas classification

    # Instantiate Lightning module for MAE with SE flag
    model = FMChagasClassifier(
        ecg_fm_checkpoint_path=checkpoint_path,
        num_classes=1,  # Binary classification for Chagas
        lr=LR,
        freeze_encoder=True  # Start with frozen encoder, can fine-tune later
    )
        # Data directories
    DATA_DIR = data_folder

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

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator='auto',
        devices=1,
        callbacks=[],
        gradient_clip_val=1.0,  # Added gradient clipping
        num_sanity_val_steps=0, 
        enable_checkpointing=False
    )
    
    # fit and validate, starting a new training run without ckpt_path
    trainer.fit(model, datamodule=data_module)
    
    # Save the model weights only to the checkpoint dir
    final_checkpoint_path = os.path.join(model_folder, "mae_encoder_pretrained.ckpt")
    trainer.save_checkpoint(final_checkpoint_path, weights_only=True)
    print(f"Model weights saved to {final_checkpoint_path}")

train_model(
    data_folder='../training_data',
    model_folder='./ckpts',
    checkpoint_path='./ckpts/mimic_iv_ecg_finetuned.pt',
    verbose=True
)