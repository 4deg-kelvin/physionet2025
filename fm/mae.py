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

# ...existing code...

def train_model_with_ecgfm(data_folder, model_folder, ecg_fm_checkpoint_path, verbose):
    pl.seed_everything(42)
    
    # Hyperparameters
    SEQ_LENGTH = utils.UNIFIED_FREQUENCY * 10  # 10 seconds
    BATCH_SIZE = 16  # Smaller batch size due to ECG-FM memory requirements
    EPOCHS = 50
    LR = 1e-4
    NO_LABELS = False  # We need labels for Chagas classification
    
    # Initialize Chagas classifier with ECG-FM backbone
    model = FMChagasClassifier(
        ecg_fm_checkpoint_path=ecg_fm_checkpoint_path,
        num_classes=1,  # Binary classification for Chagas
        lr=LR,
        freeze_encoder=True  # Start with frozen encoder, can fine-tune later
    )
    
    # Data preparation (same as before but with labels)
    DATA_DIR = data_folder
    print("Preparing stratification and loading records...")
    
    records_meta = utils.prepare_stratification(helper_code.find_records_abs(DATA_DIR))
    records_meta = [rec['record'] for rec in records_meta if rec['source'] not in ['PTB-XL', 'SaMi-Trop']]
    
    # Filter for labeled records only (for Chagas classification)
    labeled_records = [rec for rec in records_meta if helper_code.get_labels(rec)]
    
    print(f"Total labeled records found: {len(labeled_records)}")
    if len(labeled_records) == 0:
        raise ValueError("No labeled records found for classification.")
    
    np.random.shuffle(labeled_records)
    
    # Split data
    total_size = len(labeled_records)
    val_test_size = int(0.2 * total_size)
    val_size = val_test_size // 2
    
    train_records = labeled_records[:-val_test_size]
    val_records = labeled_records[-val_test_size:-val_size]
    test_records = labeled_records[-val_size:]
    
    print(f"Training records: {len(train_records)}")
    print(f"Validation records: {len(val_records)}")
    print(f"Test records: {len(test_records)}")
    
    # Create data module
    data_module = ECGDataModule(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LENGTH,
        windowing_method='entire_recording'
    )
    
    data_module.train_dataset = ECGDataset(
        train_records, DATA_DIR, is_training=True, no_labels=NO_LABELS,
        seq_len=SEQ_LENGTH, windowing_method='entire_recording'
    )
    data_module.val_dataset = ECGDataset(
        val_records, DATA_DIR, is_training=False, no_labels=NO_LABELS,
        seq_len=SEQ_LENGTH, windowing_method='entire_recording'
    )
    data_module.test_dataset = ECGDataset(
        test_records, DATA_DIR, is_training=False, no_labels=NO_LABELS,
        seq_len=SEQ_LENGTH, windowing_method='entire_recording'
    )
    
    # Training setup
    checkpoint_cb = ModelCheckpoint(
        dirpath=model_folder,
        filename='chagas_ecgfm_classifier',
        save_last=True,
        monitor='val_loss',
        mode='min'
    )
    
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator='auto',
        devices=1,
        callbacks=[checkpoint_cb],
        gradient_clip_val=1.0,
    )
    
    # Train the model
    trainer.fit(model, datamodule=data_module)
    
    # Test the best model
    trainer.test(datamodule=data_module, ckpt_path=checkpoint_cb.best_model_path)