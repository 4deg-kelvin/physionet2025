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
    def __init__(self, ecg_fm_checkpoint_path, freeze_encoder, optimizer_hparams, info_features=0):
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

    def forward(self, x, info=None):
        # Extract features using ECG-FM
        features = self.feature_extractor(x)
        
        if self.hparams.info_features > 0 and info is not None:
            features = torch.cat((features, info), dim=1)

        # Classify
        logits = self.classifier(features)
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
    WINDOWING_METHOD = 'random'
    NUM_EPOCHS = 16
    CHECKPOINT_MONITOR_METRIC = 'val_challenge_score'
    PRECISION = "16-mixed"
    SEQ_LENGTH = utils.UNIFIED_FREQUENCY * 10  
    EPOCHS = 50
    LR = 1e-4

    # Add these values into the config dictionary
    config = {
        "data_dir": DATA_DIR,
        "batch_size": BATCH_SIZE,
        "seq_length": SEQ_LENGTH,
        "num_leads": NUM_LEADS,
        "windowing_method": WINDOWING_METHOD,
        "num_epochs": NUM_EPOCHS,
        "checkpoint_monitor_metric": CHECKPOINT_MONITOR_METRIC,
        "precision": PRECISION,
        "epochs": EPOCHS,
        "lr": LR
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
    
    # --- Exclude records from code15_label_issues.csv ---
    try:
        issues_df = pd.read_csv('code15_label_issues.csv')
        # Extract stem (filename without extension) from the full path for exclusion list
        stems_to_exclude = set(issues_df['record_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0]))
        
        # Create a new 'base_record' column with the stem for matching
        df['base_record'] = df['record'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
        
        initial_count = len(df)
        # Filter out the records using the new column
        df = df[~df['base_record'].isin(stems_to_exclude)]
        final_count = len(df)
        
        print(f"Excluded {initial_count - final_count} records based on 'code15_label_issues.csv'.")
        # Drop the temporary 'base_record' column
        df = df.drop(columns=['base_record'])
        
    except FileNotFoundError:
        print("Warning: 'code15_label_issues.csv' not found. No records will be excluded.")
    
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
        optimizer_hparams = optimizer_hparams,
        info_features=2 # Age and Sex
    )
    
    print("\n--- Model Summary ---")
    print(model)
    
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="auto",
        precision=PRECISION,
        devices=1,
        logger=WandbLogger(project="foundation_model", name=f"foundation_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}", entity="edwards_physionet"),
        # logger=pl.loggers.TensorBoardLogger("lightning_logs/", name="ecg_transformer_final"),
        callbacks=[pl.callbacks.ModelCheckpoint(monitor=CHECKPOINT_MONITOR_METRIC, mode='min', filename='best-challenge-{epoch:02d}-{val_challenge_score:.4f}.ckpt'), 
                   pl.callbacks.DeviceStatsMonitor()]
    )

    # --- Run Training and Testing ---
    print(f"\n--- Starting Training with {len(train_records)} records from '{DATA_DIR}' ---")
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