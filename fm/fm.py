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
        # Single 5-second segment
        with torch.no_grad():
            out = self.ecg_fm_model(source=x)
            encoder_out = out['encoder_out']
            pooled_features = torch.div(encoder_out.sum(dim=1), (encoder_out != 0).sum(dim=1))
        
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
    def __init__(self, ecg_fm_checkpoint_path, freeze_encoder, optimizer_hparams):
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
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([utils.POS_WEIGHT]))
        
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.test_auroc = AUROC(task="binary")

        self.validation_step_outputs = []
        self.validation_step_labels = []
        self.test_step_outputs = []
        self.test_step_labels = []

    def forward(self, x):
        # Extract features using ECG-FM
        features = self.feature_extractor(x)
        # Classify
        logits = self.classifier(features)
        return logits

    def _common_step(self, batch, batch_idx):
        signals, labels = batch
        # Handle cases where the batch is empty after filtering
        if signals.numel() == 0:
            return None, None, None

        logits = self(signals)
        loss = self.criterion(logits, labels)
        preds = torch.sigmoid(logits)
        return loss, preds, labels

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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.optimizer_hparams['lr'])

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

if __name__ == '__main__':
    
    data_folder='../training_data'
    model_folder='./ckpts'
    checkpoint_path='./ckpts/mimic_iv_ecg_finetuned.pt'
    verbose=True

    # --- Hyperparameters ---
    DATA_DIR = "../training_data/" 
    BATCH_SIZE = 16
    # SEQ_LEN = utils.WINDOW_SIZE
    NUM_LEADS = 12
    SPLIT_FILE = "../train_val_test_sets.csv"
    WINDOWING_METHOD = 'random'
    NUM_EPOCHS = 16
    CHECKPOINT_MONITOR_METRIC = 'val_challenge_score'
    PRECISION = "16-mixed"
    SEQ_LENGTH = utils.UNIFIED_FREQUENCY * 5  
    BATCH_SIZE = 16  # Smaller batch size due to ECG-FM memory requirements
    EPOCHS = 50
    LR = 1e-4

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

    print("\n--- Starting Training Script ---")
    print(f"Using data directory: {DATA_DIR}")
    # --- Setup Data and Model ---
    record_files = helper_code.find_records_abs(DATA_DIR)
    
    print(f"Found {len(record_files)} records in '{DATA_DIR}'")
    print(f"Using split file: {SPLIT_FILE}")

    optimizer_hparams = {
        'lr': 2e-5,
        'warmup_steps': 700,
        'lr_end': 1e-7
    }

    data_module = ECGDataModule(
        data_dir=DATA_DIR,
        records_list=record_files,
        split_file_path=SPLIT_FILE,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LENGTH
    )

    model = FMChagasClassifier(
        ecg_fm_checkpoint_path=checkpoint_path,
        freeze_encoder=True,
        optimizer_hparams = optimizer_hparams
    )
    
    print("\n--- Model Summary ---")
    print(model)
    
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="auto",
        precision=PRECISION,
        devices=1,
        logger=WandbLogger(project="ecg_transformer_chagas", name="foundation_model", entity="edwards_physionet"),
        # logger=pl.loggers.TensorBoardLogger("lightning_logs/", name="ecg_transformer_final"),
        callbacks=[pl.callbacks.ModelCheckpoint(monitor=CHECKPOINT_MONITOR_METRIC, mode='min', filename='best-challenge-{epoch:02d}-{val_challenge_score:.4f}.ckpt'), 
                   pl.callbacks.DeviceStatsMonitor()]
    )

    # --- Run Training and Testing ---
    print(f"\n--- Starting Training with {len(record_files)} records from '{DATA_DIR}' ---")
    trainer.fit(model, datamodule=data_module)
    print("\n--- Training Finished ---")
    
    print("\n--- Starting Testing ---")
    trainer.test(model, datamodule=data_module)
    print("\n--- Testing Finished ---")

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

# train_model(
#     data_folder='../training_data',
#     model_folder='./ckpts',
#     checkpoint_path='./ckpts/mimic_iv_ecg_finetuned.pt',
#     verbose=True
# )