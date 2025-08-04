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
from models.Cardioformer import Model as Cardioformer
from types import SimpleNamespace
import wandb
from pytorch_lightning.loggers import WandbLogger
import warnings
warnings.filterwarnings(
    "ignore",
    message="Detected call of `lr_scheduler.step\\(\\) before `optimizer.step\\(\\)`.*"
)





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
import utils1 as utils

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
                # Continue with the uncorrected signal if polarity check fails

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
            
            # Get the label from the loaded header
            label = helper_code.get_label(header_text)

            assert label == 0 or label == 1, f"Invalid label {label} for record {self.records_list[idx]}"

            # Use .copy() to prevent potential negative stride errors from numpy operations
            return torch.FloatTensor(signal.copy()), torch.FloatTensor([label])
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

        model_config = SimpleNamespace(**model_hparams)
        self.model = Cardioformer(model_config)
        # self.model = Cardioformer(**model_hparams)
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

        # logits = self(signals)
        logits = self(signals.permute(0, 2, 1))  # [B, seq_len, enc_in] -> [B, enc_in, seq_len]
        # print(logits.shape)
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
    BATCH_SIZE = 8
    SEQ_LEN = utils.UNIFIED_FREQUENCY * 7 
    NUM_LEADS = 12
    SPLIT_FILE = "../train_val_test_sets.csv"
    WINDOWING_METHOD = 'random'
    WINDOWING_METHOD = 'random'
    NUM_EPOCHS = 10
    CHECKPOINT_MONITOR_METRIC = 'val_challenge_score'
    PRECISION = "16-mixed"

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
        # -e_layers 6 --batch_size 128 --d_model 128 --d_ff 256 --patch_len_list 2,4,8,8,16,16,16,16,32,32,32,32,32,32,32,32 --augmentations jitter0.2,scale0.2,drop0.5 --swa --no_inter_attn --des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --pat
        'task_name': 'classification',
        'pred_len': 1,  # Not used in classification but required by Cardioformer
        'output_attention': False,
        'enc_in': NUM_LEADS,
        'single_channel': False,
        'patch_len_list': '2,4,8,8,16,16,16,32,32,32,32,32',#'2,4,8,8,16,16,16,16,32,32,32,32,32,32,32,32',
        'augmentations': 'jitter0.2,scale0.2,drop0.5',
        'd_model': 128,
        'n_heads': 8,
        'dropout': 0.2,
        'no_inter_attn': False,
        'activation': 'gelu',
        'e_layers': 6,  # from num_layers
        'd_ff': 256,
        'seq_len': SEQ_LEN,
        'num_class': 1,  # Binary classification
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
        precision=PRECISION,
        devices=1,
        logger=WandbLogger(
            project="ecg_transformer_chagas",
            name=f"cardioformer_{WINDOWING_METHOD}"
        ),
        callbacks=[
            # save best model by val_challenge_score
            pl.callbacks.ModelCheckpoint(
                monitor=CHECKPOINT_MONITOR_METRIC,
                mode='max',
                filename='best-challenge-{epoch:02d}-{val_challenge_score:.4f}'
            ),
            # always keep the most recent checkpoint
            pl.callbacks.ModelCheckpoint(
                save_last=True,
                filename='last'
            )
        ]
    )

    # Initialize wandb, resume existing run if WANDB_RUN_ID is set
    wandb.init(
        entity="edwards_physionet",
        project="ecg_transformer_chagas",
        name=f"custom_transformer_{WINDOWING_METHOD}_5s",
        config={
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "num_leads": NUM_LEADS,
            "windowing_method": WINDOWING_METHOD,
            "num_epochs": NUM_EPOCHS,
            "data_dir": DATA_DIR,
            "split_file": SPLIT_FILE
        }
    )

    # --- Run Training and Testing ---
    print(f"\n--- Starting Training with {len(record_files)} records from '{DATA_DIR}' ---")
    trainer.fit(model, datamodule=data_module)
    print("\n--- Training Finished ---")
    
    print("\n--- Starting Testing ---")
    trainer.test(model, datamodule=data_module)
    print("\n--- Testing Finished ---")
    wandb.finish()
