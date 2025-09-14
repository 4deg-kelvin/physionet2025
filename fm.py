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
from sklearn.model_selection import train_test_split, StratifiedKFold

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
import shutil

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
        Extract AoL features from ECG-FM encoder
        Args:
            x: (batch_size, 12, seq_length) ECG signals
        Returns:
            features: (batch_size, embed_dim) AoL pooled features
        """
        with torch.no_grad():
            # Forward through the ECG-FM model
            out = self.ecg_fm_model(source=x)
            encoder = self.ecg_fm_model.encoder.encoder
            features = self.ecg_fm_model.encoder.feature_extractor(x)
            features = self.ecg_fm_model.encoder.post_extract_proj(features.transpose(1,2)).transpose(1,2)
            features = self.ecg_fm_model.encoder.conv_pos(features)
            features = self.ecg_fm_model.encoder.layer_norm(features)
            # features: (batch, seq_len, embed_dim)
            hidden_states = []
            out = features
            for layer in encoder.layers:
                out = layer(out)
                hidden_states.append(out)
            # Stack and average pool across layers
            stacked = torch.stack(hidden_states, dim=0)  # [12, batch, seq_len, 768]
            aol_features = stacked.mean(dim=0).mean(dim=1)  # [batch, 768]
        return aol_features



class FMChagasClassifier(pl.LightningModule):
    def __init__(self, ecg_fm_checkpoint_path, freeze_encoder, optimizer_hparams, info_features=0,
                 loss_type='bce', loss_top_percent=0.05, loss_margin=1.0, loss_momentum=0.99, loss_warmup_steps=100, pos_weight=5.0):
        super().__init__()
        self.save_hyperparameters()
        self.feature_extractor = ECGFMFeatureExtractor(
            ecg_fm_checkpoint_path, 
            freeze_encoder=freeze_encoder
        )
        
        self.feature_dim = 768  # ECG-FM embedding dimension

        # Demographic encode
        self.dem_encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1536),
            nn.ReLU()
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim + self.hparams.info_features, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        if self.hparams.loss_type == 'bce':
            # Weighted BCE loss: positive cases get pos_weight
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.hparams.pos_weight))
        else:
            self.criterion = PercentileRankingLoss(
                top_percent=self.hparams.loss_top_percent,
                margin=self.hparams.loss_margin,
                momentum=self.hparams.loss_momentum,
                warmup_steps=self.hparams.loss_warmup_steps
            )

        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.test_auroc = AUROC(task="binary")

        self.validation_step_outputs = []
        self.validation_step_labels = []
        self.test_step_outputs = []
        self.test_step_labels = []
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
            # Weighted BCE loss: positive cases get pos_weight
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.hparams.pos_weight))
        else:
            self.criterion = PercentileRankingLoss(
                top_percent=self.hparams.loss_top_percent,
                margin=self.hparams.loss_margin,
                momentum=self.hparams.loss_momentum,
                warmup_steps=self.hparams.loss_warmup_steps
            )
        
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.test_auroc = AUROC(task="binary")

        self.validation_step_outputs = []
        self.validation_step_labels = []
        self.test_step_outputs = []
        self.test_step_labels = []

    def forward(self, x, info=None):
        # Extract features using ECG-FM
        features = self.feature_extractor(x)  # [batch, 768]

        # info: [batch, 2] (age, sex)
        if info is None:
            raise ValueError("Demographic info (age, sex) must be provided for demographic encoding.")

        # Demographic encoder: outputs gamma and beta
        dem_out = self.dem_encoder(info)  # [batch, 1536]
        gamma, beta = dem_out[:, :features.shape[1]], dem_out[:, features.shape[1]:]
        # Modulate features
        mod_features = gamma * features + beta  # [batch, 768]

        # Classify
        logits = self.classifier(mod_features)
        return logits

    def _common_step(self, batch, batch_idx):
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
        
        return loss, logits, labels

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

# set default loss type


if __name__ == '__main__':
    
    # add args parser for debug mode
    import argparse
    parser = argparse.ArgumentParser(description="Train ECG-FM for Chagas classification")
    parser.add_argument('--debug', action='store_true', help="Run in debug mode with a smaller dataset")

    args = parser.parse_args()

    data_folder='../training_data'
    model_folder='./ckpts'
    checkpoint_path='./ckpts/mimic_iv_ecg_finetuned.pt'
    verbose=True

    # --- Hyperparameters ---
    DATA_DIR = "../training_data/" 
    BATCH_SIZE = 32
    # SEQ_LEN = utils.WINDOW_SIZE
    NUM_LEADS = 12
    WINDOWING_METHOD = 'entire_recording'
    NUM_EPOCHS = 16
    CHECKPOINT_MONITOR_METRIC = 'val_challenge_score'
    SEQ_LENGTH = utils.UNIFIED_FREQUENCY * 10  
    EPOCHS = 50
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
        "checkpoint_monitor_metric": CHECKPOINT_MONITOR_METRIC,
        # "precision": PRECISION,
        "epochs": EPOCHS,
        "lr": LR, 
        'loss_type': LOSS_TYPE,
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
    all_records = helper_code.find_records_abs(DATA_DIR)
    if args.debug:
        print("--- DEBUG MODE: Using a random subset of 1000 records. ---")
        np.random.shuffle(all_records)
        all_records = all_records[:10000]
    print(f"{len(all_records)} records found in the dataset.")
    records_meta = utils.prepare_stratification(all_records)
    df = pd.DataFrame(records_meta)

    # Exclude records where 'source' is CODE-15%
    df = df[df['source'] != 'CODE-15%']
    
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
    
    print(f"Training on {len(train_records)} records.")
    print(f"Final training set positive ratio: {train_df['label'].value_counts(normalize=True).get(1, 0):.2%}")
    print(f"Validating on {len(val_records)} records.")
    print(f"Validation set positive ratio: {val_df['label'].value_counts(normalize=True).get(1, 0):.2%}")
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
        loss_type=LOSS_TYPE
    )
    
    print("\n--- Model Summary ---")
    print(model)
    
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="auto",
        devices=1,
        logger=WandbLogger(project="foundation_model", name=f"foundation_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}", entity="edwards_physionet"),
        # logger=pl.loggers.TensorBoardLogger("lightning_logs/", name="ecg_transformer_final"),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor=CHECKPOINT_MONITOR_METRIC,
                mode='max',
                filename='best-challenge-{epoch:02d}-{val_challenge_score:.4f}.ckpt',
                dirpath=model_folder
            ),
            pl.callbacks.DeviceStatsMonitor()
        ]
    )

    # --- Run Training and Testing ---
    print(f"\n--- Starting Training with {len(train_records)} records from '{DATA_DIR}' ---")
    trainer.fit(model, datamodule=data_module)
    print("\n--- Training Finished ---")
    
    print("\n--- Starting Testing ---")
    trainer.test(model, datamodule=data_module)
    print("\n--- Testing Finished ---")
def train_model(data_folder, model_folder, is_submission=True):
    checkpoint_path='mimic_iv_ecg_finetuned.pt'

    # --- Hyperparameters ---
    DATA_DIR = data_folder
    BATCH_SIZE = 16
    # SEQ_LEN = utils.WINDOW_SIZE
    NUM_LEADS = 12
    WINDOWING_METHOD = 'entire_recording'
    NUM_EPOCHS = 10
    SEQ_LENGTH = utils.UNIFIED_FREQUENCY * 10  
    CHECKPOINT_MONITOR_METRIC = 'val_challenge_score'
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
        "checkpoint_monitor_metric": CHECKPOINT_MONITOR_METRIC,
        "epochs": NUM_EPOCHS,
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

    # If not submission mode, do a stratified train/val/test split on labels
    if not is_submission:
        from sklearn.model_selection import train_test_split
        train_df, temp_df = train_test_split(
            df, test_size=0.2, stratify=df['label'], random_state=42
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42
        )
        train_records = train_df['record'].tolist()
        val_records   = val_df['record'].tolist()
        test_records  = test_df['record'].tolist()
        print(f"Training on {len(train_records)} records.")
        print(f"Validating on {len(val_records)} records.")
        print(f"Testing on {len(test_records)} records.")
    else:
        # submission mode: train on all records
        train_records = df['record'].tolist()
        print(f"Training on {len(train_records)} records.")
        val_records = []
        test_records = []

    # 3. Create DataModule
    data_module = ECGDataModule(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LENGTH,
        windowing_method='entire_recording',

    )
    data_module.train_dataset = ECGDataset(
        train_records, DATA_DIR, is_training=True,
        seq_len=SEQ_LENGTH, windowing_method='entire_recording',
        include_wide_feats=True
    )
    if not is_submission:
        data_module.val_dataset   = ECGDataset(val_records,   DATA_DIR, is_training=False,
                                              seq_len=SEQ_LENGTH, windowing_method='entire_recording',
                                              include_wide_feats=True)
        data_module.test_dataset  = ECGDataset(test_records,  DATA_DIR, is_training=False,
                                              seq_len=SEQ_LENGTH, windowing_method='entire_recording',
                                              include_wide_feats=True)
    else:
        data_module.val_dataset = None
        data_module.test_dataset = None

    # Define optimizer hyperparameters

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
    
    if is_submission:
        trainer = pl.Trainer(
            max_epochs=NUM_EPOCHS,
            accelerator="auto",
            devices=1,
            default_root_dir=model_folder,
            logger=False,
            callbacks=[],
            limit_val_batches=0
        )
    else:
        checkpoint_cb = pl.callbacks.ModelCheckpoint(
            monitor=CHECKPOINT_MONITOR_METRIC,
            mode='max',
            save_top_k=1,
            filename='foundation_model_finetuned',
            dirpath=model_folder
         )
        trainer = pl.Trainer(
            max_epochs=NUM_EPOCHS,
            accelerator="auto",
            devices=1,
            logger=WandbLogger(project="foundation_model", name=f"foundation_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}", entity="edwards_physionet"),
            callbacks=[pl.callbacks.ModelCheckpoint(
                monitor=CHECKPOINT_MONITOR_METRIC,
                mode='max',  # was 'min' – challenge score is a higher-is-better metric
                filename='foundation_model_finetuned',
                dirpath=model_folder
            )]
        )

    # --- Run Training and Testing ---
    print(f"\n--- Starting Training with {len(train_records)} records from '{DATA_DIR}' ---")
    trainer.fit(model, datamodule=data_module)
    print("\n--- Training Finished ---")

    if is_submission:
        # Save the model to model_folder
        final_checkpoint_path = os.path.join(model_folder, "foundation_model_finetuned.ckpt")
        trainer.save_checkpoint(final_checkpoint_path, weights_only=True)
        print(f"Model weights saved to {final_checkpoint_path}")
def load_finetuned_model(
    ckpt_path, 
    map_location=None, 
    strict=True, 
    override_hparams=None,
    ecg_fm_checkpoint_path="mimic_iv_ecg_finetuned.pt"
):
    """
    Load a fine-tuned FMChagasClassifier from a Lightning .ckpt file.

    Args:
        ckpt_path (str): Path to the .ckpt file (e.g., 'foundation_model_finetuned.ckpt').
        map_location (str or torch.device, optional): Device mapping for loading.
            Defaults to 'cuda' if available, else 'cpu'.
        strict (bool): Strict state_dict loading. Default True.
        override_hparams (dict, optional): Optional kwargs to override saved hparams,
            e.g., {'loss_type': 'bce'}.
        ecg_fm_checkpoint_path (str): Path to the base ECG-FM checkpoint required for feature extraction.

    Returns:
        FMChagasClassifier: Model loaded with weights, set to eval() mode.
    """
    if map_location is None:
        map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    override_hparams = override_hparams or {}
    model = FMChagasClassifier.load_from_checkpoint(
        ckpt_path,
        ecg_fm_checkpoint_path=ecg_fm_checkpoint_path,
        map_location=map_location,
        strict=strict,
        **override_hparams
    )
    model.eval()
    return model

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