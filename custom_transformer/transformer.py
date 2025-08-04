# TODO: incorporate the train_val_test_sets.csv to standarize what is train/val/test
# increase the epoch count and look at the model again
# compute the challenge_score during validation
# add more hparams for better tracking/reproduction
# experiment with different windowing centering techniques (qrs centering, starting a qrs complex)
# splitting this into multiple files, adding args to be passed into this

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
import sys
import neurokit2 as nk
import pandas as pd
from tqdm import tqdm

# --- Robust Path Handling ---
# Handles running in different environments (e.g., script vs. notebook)
# This allows the script to find your helper_code and utils modules
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

# --- Standard Imports ---
# The script will now directly use these modules.
# Make sure they are available in your environment.
import helper_code
import utils

# --- 1. Core Model Components ---

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) module."""
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Dimension must be even for RotaryEmbedding.")
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, :, None, :], persistent=False)

    def forward(self, x):
        seq_len = x.shape[1]
        cos = self.cos_cached[:, :seq_len, ...]
        sin = self.sin_cached[:, :seq_len, ...]
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]
        rotated_x = torch.cat((-x2, x1), dim=-1)
        return x * cos + rotated_x * sin

class RoPETransformerEncoderLayer(nn.Module):
    """
    A custom Transformer Encoder layer that correctly incorporates RoPE.
    This  manually handles Q, K, V projections to avoid shape errors.
    """
    def __init__(self, d_model, nhead, d_ff, dropout, max_seq_len=2048):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"
        
        # Manual projection layers for Query, Key, Value
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Rotary Embedding
        self.rope = RotaryEmbedding(dim=self.head_dim, max_seq_len=max_seq_len)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # Normalization and dropout layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu
    def forward(self, src):
        """
        Forward pass for the custom encoder layer.
        Args:
            src (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = src.shape

        # 1. Project src to Q, K, V
        q = self.q_proj(src)
        k = self.k_proj(src)
        v = self.v_proj(src)

        # 2. Reshape for multi-head processing
        q = q.view(batch_size, seq_len, self.nhead, self.head_dim)
        k = k.view(batch_size, seq_len, self.nhead, self.head_dim)
        v = v.view(batch_size, seq_len, self.nhead, self.head_dim)
        
        # 3. Apply RoPE to query and key
        q = self.rope(q)
        k = self.rope(k)

        # 4. Transpose for attention calculation: [batch, n_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 5. Scaled dot-product attention
        # This function correctly handles the 4D input
        attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout1.p if self.training else 0.0)
        
        # 6. Reshape back to [batch_size, seq_len, d_model] and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_output = self.out_proj(attn_output)

        # 7. Add & Norm (first residual connection)
        # Note: self.dropout1 is applied inside scaled_dot_product_attention now
        src = self.norm1(src + attn_output)

        # 8. Feed-forward network
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        
        # 9. Add & Norm (second residual connection)
        src = self.norm2(src + self.dropout2(ff_output))
        
        return src

class Transformer(nn.Module):
    """A Transformer encoder that uses RoPE."""
    def __init__(self, d_model, h, d_ff, num_layers, dropout, max_seq_len=2048):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([
            RoPETransformerEncoderLayer(d_model, h, d_ff, dropout, max_seq_len)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        out = x.permute(0, 2, 1)
        for layer in self.layers:
            out = layer(out)
        out = self.norm(out)
        out = out.mean(dim=1)
        return out

class ChagasTransformer(nn.Module):
    """ The main model combining CNN encoder and Transformer."""
    def __init__(self, d_model, nhead, d_ff, num_layers, dropout_rate, deepfeat_sz, **kwargs):
        super(ChagasTransformer, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(12, 128, kernel_size=14, stride=3, padding=2, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=14, stride=3, padding=0, bias=False),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Conv1d(256, d_model, kernel_size=10, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(d_model), nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=10, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(d_model), nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=10, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(d_model), nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=10, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(d_model), nn.ReLU(inplace=True),
        )
        self.transformer = Transformer(d_model, nhead, d_ff, num_layers, dropout=dropout_rate)
        self.fc1 = nn.Linear(d_model, deepfeat_sz)
        self.fc2 = nn.Linear(deepfeat_sz, 1)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        z = self.encoder(x)
        out = self.transformer(z)
        out = self.dropout(F.relu(self.fc1(out)))
        out = self.fc2(out)
        return out

# --- 2. PyTorch Lightning DataModule ---

class ECGDataset(Dataset):
    def __init__(self, records_list, data_dir, is_training=True, seq_len=5000, windowing_method='qrs'):
        self.records_list = records_list
        self.data_dir = data_dir
        self.is_training = is_training
        self.seq_len = seq_len
        self.windowing_method = windowing_method

    def __len__(self):
        return len(self.records_list)

    def __getitem__(self, idx):
        record_path = os.path.join(self.data_dir, self.records_list[idx])
        try:
            # --- FIX: Load header and signal separately for robustness ---
            try:
                # Load header to get metadata and for label extraction
                signal, metadata = helper_code.load_signals(record_path)
                header_text = helper_code.load_header(record_path)
                signal = signal.T # make signal (num_leads, num_samples) shape

            except Exception as e:
                # If loading fails for any reason, skip this record
                tqdm.write(f"Error loading record {self.records_list[idx]}: {e}. Skipping.")
                return None


            # --- Gracefully handle short signals ---
            # Get frequency and check if the signal is long enough to be useful
            orig_freq = metadata['fs']
            if signal.shape[1] < orig_freq * utils.MIN_SIGNAL_DURATION:
                tqdm.write(f"Skipping record {self.records_list[idx]} due to insufficient length: {signal.shape[1]} samples.")
                return None # Return None to be filtered out by the custom collate function
            
            # --- ADDED: Get and process demographic data as wide features ---
            age, sex, label = helper_code.get_patient_info(header_text, allow_missing_label=False)
            assert label is not None and label == 0 or label == 1, f"Invalid label {label} for record {self.records_list[idx]}"
    
            # Process and normalize age. Use a neutral value for missing data.
            if age is None or np.isnan(age):
                normalized_age = 0.0 
            else:
                normalized_age = (age - utils.MEAN_AGE_TRAIN) / utils.STD_AGE_TRAIN

            # Process and normalize sex. Use a neutral value for unknown/missing.
            if sex is None or not isinstance(sex, str) or sex.lower() not in ['male', 'female']:
                numerical_sex = 0.5 
            elif sex.lower() == 'male':
                numerical_sex = 0.0
            else: # 'female'
                numerical_sex = 1.0

            # wide_feats is a 2D tensor with age, sex, and computed_features
            # Standardize sampling frequency
            if orig_freq != utils.UNIFIED_FREQUENCY:
                num_samples = int(signal.shape[1] * (utils.UNIFIED_FREQUENCY / orig_freq))
                signal = resample(signal, num_samples, axis=1)

            # Clean signal and correct polarity
            cleaned_leads = [nk.ecg_clean(lead, sampling_rate=utils.UNIFIED_FREQUENCY) for lead in signal]
            signal = np.stack(cleaned_leads)
            try:
                signal, _ = utils.correct_12_lead_polarity_lead_II_ref(signal, utils.UNIFIED_FREQUENCY)
            except Exception as e:
                tqdm.write(f"Error correcting polarity for record {self.records_list[idx]}: {e}")
            created_features = False
            created_wide = False
            try:
                # Continue with the uncorrected signal if polarity check fails
                # computed_features = utils.calculate_ecg_features(signal, utils.UNIFIED_FREQUENCY, utils.REFERENCE_POLARITY_LEAD_IDX)
                # # If cant compute features on this lead, try on other leads
                # if computed_features is None:
                #     for i in range(signal.shape[0]):
                #         if i != utils.REFERENCE_POLARITY_LEAD_IDX:
                #             computed_features = utils.calculate_ecg_features(signal, utils.UNIFIED_FREQUENCY, i)
                #             if computed_features is not None:
                #                 break
                # if computed_features is None:
                #     tqdm.write(f"Skipping record {self.records_list[idx]} because no features could be computed.")
                #     return None

                # # Fill nan values in computed features with 0
                # computed_features = np.nan_to_num(computed_features, nan=0.0, posinf=0.0, neginf=0.0)
                # created_features = True

                wide_feats = np.array([normalized_age, numerical_sex])
                # Append the wide features to the computed features
                # wide_feats = np.concatenate((wide_feats, computed_features), axis=0)
                # created_wide = True

            except Exception as e:
                tqdm.write(f"Error extracting features for record {self.records_list[idx]}: {e}")
                return None # Return None if feature extraction fails

            # Extract windows from the signal
            if utils.USE_ONE_WINDOW:
                windows = utils.get_windows(signal, method=self.windowing_method, window_size=self.seq_len)
                if len(windows) == 0:
                    tqdm.write(f"Skipping record {self.records_list[idx]} because no windows could be extracted.")
                    return None # Return None if windowing fails
                signal = windows[0]
            else:
                raise NotImplementedError("Multiple windows not implemented yet. Set USE_ONE_WINDOW to True for now.")
            
            # Normalize each lead between -1 and 1
            signal = utils.normalize(signal, smooth=1e-8)
    
            # MODIFIED: Return a 3-tuple including the wide_feats tensor
            return torch.FloatTensor(signal.copy()), torch.FloatTensor(wide_feats), torch.FloatTensor([label])
        except Exception as e:
            tqdm.write(f"Error processing record {self.records_list[idx]}: {e}. Skipping.")
            return None

def collate_fn_skip_none(batch):
    """
    Custom collate function that filters out `None` values.
    This is used to handle records that were skipped in the Dataset (e.g., too short).
    """
    # Filter out None entries
    batch = [item for item in batch if item is not None]

    # If the entire batch was filtered out (e.g., all records were short),
    # return empty tensors to prevent a crash in the training loop.
    if not batch:
        return torch.tensor([]), torch.tensor([])

    # Use the default collate function on the filtered, valid batch
    return torch.utils.data.default_collate(batch)

class ECGDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, records_list, split_file_path, batch_size=32, seq_len=utils.WINDOW_SIZE, windowing_method='qrs'):
        super().__init__()
        self.data_dir = data_dir
        self.records_list = records_list
        self.split_file_path = split_file_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.windowing_method = windowing_method

    def setup(self, stage=None):
        split_df = pd.read_csv(self.split_file_path)
        basename_to_path = {os.path.splitext(os.path.basename(p))[0]: p for p in self.records_list}
        train_ids = split_df[split_df['split'] == 'train']['exam_id'].tolist()
        val_ids = split_df[split_df['split'] == 'val']['exam_id'].tolist()
        test_ids = split_df[split_df['split'] == 'test']['exam_id'].tolist()
        train_files = [basename_to_path[id] for id in train_ids if id in basename_to_path]
        val_files = [basename_to_path[id] for id in val_ids if id in basename_to_path]
        test_files = [basename_to_path[id] for id in test_ids if id in basename_to_path]
        
        self.train_dataset = ECGDataset(train_files, self.data_dir, is_training=True, seq_len=self.seq_len, windowing_method=self.windowing_method)
        self.val_dataset = ECGDataset(val_files, self.data_dir, is_training=False, seq_len=self.seq_len, windowing_method=self.windowing_method)
        self.test_dataset = ECGDataset(test_files, self.data_dir, is_training=False, seq_len=self.seq_len, windowing_method=self.windowing_method)
        
        print(f"Data setup complete. Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
        total_records = len(train_files) + len(val_files) + len(test_files)
        print(f"Total records processed: {total_records}, %age in a split: {total_records / len(self.records_list) * 100:.2f}%")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=min(os.cpu_count(), 10), persistent_workers=True, shuffle=True, pin_memory=True, collate_fn=collate_fn_skip_none)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=min(os.cpu_count(), 10), persistent_workers=True, pin_memory=True, collate_fn=collate_fn_skip_none)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=min(os.cpu_count(), 10), persistent_workers=True, pin_memory=True, collate_fn=collate_fn_skip_none)

# --- 3. PyTorch Lightning Module ---

class ECGClassifierLightning(pl.LightningModule):
    def __init__(self, model_hparams, optimizer_hparams):
        super().__init__()
        self.save_hyperparameters()
        self.model = ChagasTransformer(**model_hparams)
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
        return self.model(x)

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

    # --- Hyperparameters ---
    DATA_DIR = "../training_data/" 
    BATCH_SIZE = 80
    SEQ_LEN = utils.WINDOW_SIZE
    NUM_LEADS = 12
    SPLIT_FILE = "../train_val_test_sets.csv"
    WINDOWING_METHOD = 'entire_recording'
    NUM_EPOCHS = 16
    CHECKPOINT_MONITOR_METRIC = 'val_challenge_score'
    # PRECISION = "16-mixed"

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

    model_hparams = {
        'd_model': 256,
        'nhead': 8,
        'num_layers': 8,
        'd_ff': 2048,
        'dropout_rate': 0.1,
        'deepfeat_sz': 256, 
        'batch_size': BATCH_SIZE,
        'data_path': DATA_DIR,
        'seq_len': SEQ_LEN,
        'num_leads': NUM_LEADS,
        'use_single_window': utils.USE_ONE_WINDOW,
        'num_records': len(record_files),
        'split_file': SPLIT_FILE,
        'pos_weight': utils.POS_WEIGHT,
        'windowing_method': WINDOWING_METHOD,
        'epochs': NUM_EPOCHS,
        'checkpoint_monitor_metric': CHECKPOINT_MONITOR_METRIC,
    }
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
        seq_len=SEQ_LEN
    )

    model = ECGClassifierLightning(model_hparams, optimizer_hparams)
    
    print("\n--- Model Summary ---")
    print(model)
    
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="auto",
        # precision=PRECISION,
        devices=1,
        logger=pl.loggers.TensorBoardLogger("lightning_logs/", name="ecg_transformer_final"),
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