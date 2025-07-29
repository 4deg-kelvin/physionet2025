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
from datetime import datetime
import wandb
from pytorch_lightning.loggers import WandbLogger
import time
from sklearn.model_selection import train_test_split
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
        # combine transformer output with wide features
        self.fc1 = nn.Linear(d_model + NUM_WIDE_FEATURES, deepfeat_sz)
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
    
    def forward(self, x, wide_feats):
        # x: (batch, leads, seq), wide_feats: (batch,2)
        z = self.encoder(x)
        features = self.transformer(z)
        combined = torch.cat((features, wide_feats), dim=1)
        out = self.dropout(F.relu(self.fc1(combined)))
        return self.fc2(out)

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
            # Only exclude records if we're not training 
            orig_freq = metadata['fs']
            if signal.shape[1] < orig_freq * utils.MIN_SIGNAL_DURATION and self.is_training:
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
    def __init__(self, data_dir, records_list=None, split_file_path=None, batch_size=32, seq_len=utils.WINDOW_SIZE, windowing_method='qrs'):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.windowing_method = windowing_method

        # ensure these attrs always exist
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=min(os.cpu_count(), 10),
                          persistent_workers=True, shuffle=True, pin_memory=True, collate_fn=collate_fn_skip_none)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=min(os.cpu_count(), 10),
                          persistent_workers=True, pin_memory=True, collate_fn=collate_fn_skip_none)

    def test_dataloader(self):
        # fallback to val_dataset if no test_dataset was set
        dataset = self.test_dataset if self.test_dataset is not None else self.val_dataset
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=min(os.cpu_count(), 10),
                          persistent_workers=True, pin_memory=True, collate_fn=collate_fn_skip_none)

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

    def forward(self, signals, wide_feats):
        return self.model(signals, wide_feats)

    def _common_step(self, batch, batch_idx):
        signals, wide_feats, labels = batch

        # Handle cases where the batch is empty after filtering
        if signals.numel() == 0:
            return None, None, None

        logits = self(signals, wide_feats)
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
def ensemble_predict(test_records, kfold_ckpts, code15_ckpt, seq_len, windowing_method, batch_size, data_dir, w_k=0.5, w_c=0.5):
        """
        test_records: list of record IDs for universal_test
        kfold_ckpts: list of two checkpoint paths (best PTB folds)
        code15_ckpt: path to CODE-15% checkpoint
        w_k, w_c: weights for k-fold avg and code15
        """
        # Create dataset in inference mode (don't skip short signals)
        ds = ECGDataset(test_records, data_dir, is_training=False,
                        seq_len=seq_len, windowing_method=windowing_method)
        loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn_skip_none)
        # select device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load and predict k-fold
        k_preds = []
        # load and predict k-fold models
        for ckpt in kfold_ckpts:
            m = ECGClassifierLightning.load_from_checkpoint(
                ckpt, model_hparams=model_hparams, optimizer_hparams=optimizer_hparams
            )
            m.to(device)
            m.eval()
            preds = []
            for batch in tqdm(loader, total=len(loader), desc=f"Predicting with {ckpt}"):
                x, w, _ = batch
                # Skip empty batches (all records filtered out)
                if isinstance(x, torch.Tensor) and x.numel() == 0:
                    continue
                x, w = x.to(device), w.to(device)
                with torch.no_grad():
                    out = m(x, w)
                    preds.append(torch.sigmoid(out).cpu().numpy())
            k_preds.append(np.concatenate(preds))
        # If no predictions were collected, return empty array
        if not k_preds:
            return np.array([])
        avg_k = np.mean(k_preds, axis=0)

        # load and predict code15
        # load and predict CODE-15% model
        m15 = ECGClassifierLightning.load_from_checkpoint(
            code15_ckpt, model_hparams=model_hparams, optimizer_hparams=optimizer_hparams
        )
        m15.to(device)
        m15.eval()
        c_preds = []
        for batch in tqdm(loader, total=len(loader), desc="Predicting with CODE-15%"):
            x, w, _ = batch
            # Skip empty batches
            if isinstance(x, torch.Tensor) and x.numel() == 0:
                continue
            x, w = x.to(device), w.to(device)
            with torch.no_grad():
                out15 = m15(x, w)
                c_preds.append(torch.sigmoid(out15).cpu().numpy())
        # If no code15 predictions, return average from k-fold only
        if not c_preds:
            return avg_k
        c_all = np.concatenate(c_preds)

        # weighted ensemble
        return w_k * avg_k + w_c * c_all
# --- 4. Training Script ---
# --- Hyperparameters ---
DATA_DIR = "../training_data/" 
BATCH_SIZE = 80
SEQ_LEN = utils.WINDOW_SIZE
NUM_LEADS = 12
SPLIT_FILE = "../train_val_test_sets.csv"
WINDOWING_METHOD = 'entire_recording'
NUM_EPOCHS = 7
CHECKPOINT_MONITOR_METRIC = 'val_challenge_score'
NUM_WIDE_FEATURES = 2
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
    # 'num_records': len(record_files),
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
if __name__ == '__main__':
    pl.seed_everything(42)
    # --- Create run directory ---
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"trained_model_{run_ts}"
    RUN_DIR = run_name
    os.makedirs(RUN_DIR, exist_ok=True)
    print(f"Run artifacts will be saved to: {RUN_DIR}")


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

    # restore hyperparameters for model and optimizer


    # --- Load metadata and stratify ---
    records_meta = utils.prepare_stratification(helper_code.find_records_abs(DATA_DIR))

    # 1. build universal 10% test set (stratified per source)
    universal_test = []
    remaining = []
    for src in ['PTB-XL','SaMi-Trop','CODE-15%']:
        src_meta = [m for m in records_meta if m['source']==src]
        labels   = [m['label'] for m in src_meta]
        # 10% for test
        train_val, test_src = train_test_split(src_meta,
                                               test_size=0.1,
                                               stratify=labels,
                                               random_state=42)
        universal_test += test_src
        remaining    += train_val

    print(f"Created test set with {len(universal_test)} records. Remaining records: {len(remaining)}")
    # 2. K-fold on PTB+SaMi using remaining entries
    ptb_sami = [m for m in remaining if m['source'] in ['PTB-XL','SaMi-Trop']]
    folds = utils.generate_k_folds(5, ptb_sami)

    # track (fold_idx, path, score)
    fold_results = []

    for fold_idx, val_fold in enumerate(folds, start=1):
        # build train/val file lists
        train_fold = [r for f in folds if f is not val_fold for r in f]
        train_files = [r['record'] for r in train_fold]
        val_files   = [r['record'] for r in val_fold]
        
        print(f"\n-- Fold {fold_idx}/5: Train={len(train_files)}, Val={len(val_files)}")
        
        # initialize DataModule for this fold (override datasets manually)
        data_module = ECGDataModule(
            data_dir=DATA_DIR,
            records_list=[],          # dummy, we'll override below
            split_file_path=None,
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            windowing_method=WINDOWING_METHOD
        )
        data_module.train_dataset = ECGDataset(train_files, DATA_DIR, is_training=True,  seq_len=SEQ_LEN, windowing_method=WINDOWING_METHOD)
        data_module.val_dataset   = ECGDataset(val_files,   DATA_DIR, is_training=False, seq_len=SEQ_LEN, windowing_method=WINDOWING_METHOD)
        # also assign test_dataset (e.g. reuse the validation fold or set your own test split)
        data_module.test_dataset  = ECGDataset(val_files,   DATA_DIR, is_training=False, seq_len=SEQ_LEN, windowing_method=WINDOWING_METHOD)
        
        # setup loggers
        tb_logger    = pl.loggers.TensorBoardLogger(os.path.join(RUN_DIR, "logs"), name=f"fold{fold_idx}")
        wandb_logger = WandbLogger(project="ecg_transformer_chagas", name=f"fold{fold_idx}", save_dir=RUN_DIR)

        # rebuild model & trainer per fold
        model   = ECGClassifierLightning(model_hparams, optimizer_hparams)
        checkpoint_cb = pl.callbacks.ModelCheckpoint(
            dirpath=RUN_DIR,
            filename=f'fold{fold_idx}-best-{{epoch:02d}}-{{val_challenge_score:.4f}}',
            monitor=CHECKPOINT_MONITOR_METRIC, mode='max'
        )
        trainer = pl.Trainer(
            max_epochs=NUM_EPOCHS,
            accelerator="auto",
            devices=1,
            logger=[tb_logger, wandb_logger],
            callbacks=[checkpoint_cb]
        )
        trainer.fit(model, datamodule=data_module)
        trainer.test(model, datamodule=data_module)

        # record best path and score (convert tensor to float)
        fold_results.append((fold_idx, checkpoint_cb.best_model_path, float(checkpoint_cb.best_model_score)))

    # Save all fold results (fold index, checkpoint path, score) to CSV
    fold_df = pd.DataFrame(fold_results, columns=["fold_idx", "ckpt_path", "score"])
    fold_df.to_csv(os.path.join(RUN_DIR, "fold_results.csv"), index=False)

    # 3. CODE-15% model
    code15_rem = [m for m in remaining if m['source']=='CODE-15%']
    labels15   = [m['label'] for m in code15_rem]
    train15, val15 = train_test_split(code15_rem,
                                      test_size=0.2,
                                      stratify=labels15,
                                      random_state=42)
    train_files15 = [r['record'] for r in train15]
    val_files15   = [r['record'] for r in val15]

    data_module15 = ECGDataModule(DATA_DIR, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, windowing_method=WINDOWING_METHOD)
    data_module15.train_dataset = ECGDataset(train_files15, DATA_DIR, seq_len=SEQ_LEN, windowing_method=WINDOWING_METHOD)
    data_module15.val_dataset   = ECGDataset(val_files15,   DATA_DIR, seq_len=SEQ_LEN, windowing_method=WINDOWING_METHOD)

    # setup loggers
    tb_logger    = pl.loggers.TensorBoardLogger("lightning_logs/", name="code15_final")
    wandb_logger = WandbLogger(project="ecg_transformer_chagas", name="code15_final")

    model   = ECGClassifierLightning(model_hparams, optimizer_hparams)
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="auto",
        devices=1,
        logger=[tb_logger, wandb_logger],
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=RUN_DIR,
                filename='code15-best-{epoch:02d}-{val_challenge_score:.4f}',
                monitor=CHECKPOINT_MONITOR_METRIC, mode='max'
            )
        ]
    )
    trainer.fit(model, datamodule=data_module15)
    trainer.test(model, datamodule=data_module15)

    # ensemble evaluation & grid search
    

    # usage:
    # final_scores = ensemble_predict(...)

    NUM_FOLDS_IN_STRONG_TRANSFORMER = 2
    # 5. Evaluate ensemble on universal test
    test_records  = [r['record'] for r in universal_test]
    test_labels   = [r['label']  for r in universal_test]
    fold_results_sorted = sorted(fold_results, key=lambda x: x[NUM_FOLDS_IN_STRONG_TRANSFORMER], reverse=True)
    top2_ckpts = [path for _, path, _ in fold_results_sorted[:NUM_FOLDS_IN_STRONG_TRANSFORMER]]
    code15_ckpt   = 'code15-best.ckpt'

    # simple grid search for best weights
    best_score = -1.0
    best_wk, best_wc = 0.5, 0.5
    for w_k in np.linspace(0, 1, 11):
        w_c = 1.0 - w_k
        preds = ensemble_predict(test_records, top2_ckpts, code15_ckpt, w_k=w_k, w_c=w_c)
        score = utils.compute_challenge_score(test_labels, preds)
        if score > best_score:
            best_score, best_wk, best_wc = score, w_k, w_c

    print(f"Best ensemble weights: w_k={best_wk:.2f}, w_c={best_wc:.2f} -> Challenge Score={best_score:.4f}")

    # save best weights to CSV in run dir
    ensemble_df = pd.DataFrame([{"w_k": best_wk, "w_c": best_wc}])
    ensemble_df.to_csv(os.path.join(RUN_DIR, "ensemble_weights.csv"), index=False)