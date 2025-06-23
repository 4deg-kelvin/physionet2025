# TODO: incorporate the train_val_test_sets.csv to standarize what is train/val/test
# increase the epoch count and look at the model again 

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
    This version manually handles Q, K, V projections to avoid shape errors.
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
    # FIX: Removed the 'helper' parameter from the constructor
    def __init__(self, records_list, data_dir, is_training=True, seq_len=5000):
        self.records_list = records_list
        self.data_dir = data_dir
        self.is_training = is_training
        self.seq_len = seq_len

    def __len__(self):
        return len(self.records_list)

    def __getitem__(self, idx):
        # Construct the full path to the record name, which helper_code functions expect
        record_path = os.path.join(self.data_dir, self.records_list[idx])
        
        # FIX: Directly call functions from the imported helper_code and utils modules
        signal, _ = helper_code.load_signals(record_path) # returns in form (num_samp, num_leads)
        header_text = helper_code.load_header(record_path)

        signal = signal.T  # Transpose to (num_leads, num_samples)

        orig_freq = helper_code.get_sampling_frequency(header_text)
        if orig_freq != utils.UNIFIED_FREQUENCY:
            num_samples = int(signal.shape[1] * (utils.UNIFIED_FREQUENCY / orig_freq))
            signal = resample(signal, num_samples, axis=1)

        cleaned_leads = [nk.ecg_clean(lead, sampling_rate=utils.UNIFIED_FREQUENCY) for lead in signal]
        signal = np.stack(cleaned_leads)
        
        # TODO: currently we only take the first window of the signal
        # however, we eventually want to get multiple windows for the val/test set
        # but that means the output dim of get_item will have an extra dimension 
        signal = utils.get_windows(signal, window_size=self.seq_len)[0] # ret (num_windows, num_leads, window_size)

        
        label = helper_code.get_label(header_text)
        if isinstance(label, str):
            label = 1 if label.lower() == "true" else 0
        elif isinstance(label, bool):
            label = int(label)
        
        # Use .copy() to prevent potential negative stride errors from numpy operations
        return torch.FloatTensor(signal.copy()), torch.FloatTensor([label])

class ECGDataModule(pl.LightningDataModule):
    # FIX: Removed the 'helper' parameter
    def __init__(self, data_dir, records_list=None, batch_size=32, seq_len=utils.WINDOW_SIZE):
        super().__init__()
        self.data_dir = data_dir
        self.records_list = records_list
        self.batch_size = batch_size
        self.seq_len = seq_len

    def setup(self, stage=None):
        # FIX: The dataset is instantiated without the helper parameter
        full_dataset = ECGDataset(self.records_list, self.data_dir, is_training=True, seq_len=self.seq_len)
        
        train_size = int(0.8 * len(full_dataset))
        val_size = int(0.1 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        # Re-assign the is_training flag for validation and test sets
        self.val_dataset.dataset.is_training = False
        self.test_dataset.dataset.is_training = False


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, persistent_workers=True)

# --- 3. PyTorch Lightning Module ---

class ECGClassifierLightning(pl.LightningModule):
    def __init__(self, model_hparams, optimizer_hparams):
        super().__init__()
        self.save_hyperparameters()
        self.model = ChagasTransformer(**model_hparams)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([utils.POS_WEIGHT]))  # Using label smoothing for better generalization
        
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.test_auroc = AUROC(task="binary")

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx):
        signals, labels = batch
        logits = self(signals)
        loss = self.criterion(logits, labels)
        preds = torch.sigmoid(logits)
        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_acc(preds, labels.int())
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        self.val_acc(preds, labels.int())
        self.val_auroc(preds, labels)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=True)
        self.log('val_auroc', self.val_auroc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self._common_step(batch, batch_idx)
        self.log('test_loss', loss)
        self.test_acc(preds, labels.int())
        self.test_auroc(preds, labels)
        self.log('test_acc', self.test_acc, on_epoch=True)
        self.log('test_auroc', self.test_auroc, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.optimizer_hparams['lr'])

        # Define the scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.estimated_stepping_batches # Total training steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", # Update the LR at every step
            },
        }

# --- 4. Training Script ---

if __name__ == '__main__':
    pl.seed_everything(42)

    # --- Hyperparameters ---
    # IMPORTANT: Change this to the actual directory containing your training data
    DATA_DIR = "../training_data" 
    BATCH_SIZE = 8
    SEQ_LEN = utils.WINDOW_SIZE # Assuming WINDOW_SIZE is defined in your utils.py
    NUM_LEADS = 12

    if not os.path.isdir(DATA_DIR):
        print(f"Error: Data directory not found at '{DATA_DIR}'")
        print("Please update the DATA_DIR variable to point to your dataset.")
        sys.exit(1)

    model_hparams = {
        'd_model': 128,
        'nhead': 8,
        'num_layers': 3,
        'd_ff': 512,
        'dropout_rate': 0.1,
        'deepfeat_sz': 256,
    }
    optimizer_hparams = {'lr': 0.0001}

    print("\n--- Starting Training Script ---")
    print(f"Using data directory: {DATA_DIR}")
    # --- Setup Data and Model ---
    record_files = helper_code.find_records_abs(DATA_DIR)
    
    print(f"Found {len(record_files)} records in '{DATA_DIR}'")
    # FIX: Instantiated DataModule without the helper parameter
    data_module = ECGDataModule(
        data_dir=DATA_DIR,
        records_list=record_files, # Pass the found records
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN
    )

    model = ECGClassifierLightning(model_hparams, optimizer_hparams)
    
    print("\n--- Model Summary ---")
    print(model)
    
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices=1,
        logger=pl.loggers.TensorBoardLogger("lightning_logs/", name="ecg_transformer_final"),
        callbacks=[pl.callbacks.ModelCheckpoint(monitor='val_loss', mode='min')]
    )

    # --- Run Training and Testing ---
    print(f"\n--- Starting Training with {len(record_files)} records from '{DATA_DIR}' ---")
    trainer.fit(model, datamodule=data_module)
    print("\n--- Training Finished ---")
    
    print("\n--- Starting Testing ---")
    trainer.test(model, datamodule=data_module)
    print("\n--- Testing Finished ---")
