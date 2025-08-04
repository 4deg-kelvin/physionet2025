import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
import sys
import os
from datetime import datetime
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.classification import Accuracy, AUROC
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dataloader import ECGDataModule, ECGDataset

from functools import cache

# --- Robust Path Handling ---
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)
    # also add mae directory to resolve utils
    mae_dir = os.path.join(project_root, 'se_blocks')
    if mae_dir not in sys.path:
        sys.path.insert(0, mae_dir)
except NameError:
    pass
import helper_code
import utils
import model

def main():
    pl.seed_everything(42)
    
    DATA_DIR = '../training_data'
    BATCH_SIZE = 32
    SEQ_LENGTH = utils.UNIFIED_FREQUENCY * 10  # Length of each ECG sequence
    EPOCHS = 15
    
    # --- Load metadata and prepare data splits ---
    records_meta = utils.prepare_stratification(helper_code.find_records_abs(DATA_DIR))
    df = pd.DataFrame(records_meta)

    # Isolate data from different sources
    ptb_df = df[df['source'] == 'PTB-XL'].copy()
    sami_df = df[df['source'] == 'SaMi-Trop'].copy()
    other_sources_df = df[~df['source'].isin(['PTB-XL', 'SaMi-Trop'])].copy()

    # --- Create Validation Set (5% positive) ---
    # Use 10% of the positive samples for validation
    val_pos_count = int(len(sami_df) * 0.10)
    # Calculate the number of negative samples needed for a 5% positive ratio
    val_neg_count = int(val_pos_count * (0.95 / 0.05))

    val_pos = sami_df.sample(n=val_pos_count, random_state=42)
    val_neg = ptb_df.sample(n=val_neg_count, random_state=42)
    val_set = pd.concat([val_pos, val_neg]).sample(frac=1, random_state=42)

    # --- Create Test Set (5% positive) from remaining data ---
    sami_remaining = sami_df.drop(val_pos.index)
    ptb_remaining = ptb_df.drop(val_neg.index)
    
    # Use 10% of the original positive samples for testing as well
    test_pos_count = int(len(sami_df) * 0.10)
    test_neg_count = int(test_pos_count * (0.95 / 0.05))

    test_pos = sami_remaining.sample(n=test_pos_count, random_state=42)
    test_neg = ptb_remaining.sample(n=test_neg_count, random_state=42)
    test_set = pd.concat([test_pos, test_neg]).sample(frac=1, random_state=42)

    # --- Create Training Set from the rest of the data ---
    train_pos = sami_remaining.drop(test_pos.index)
    train_neg = ptb_remaining.drop(test_neg.index)
    train_df = pd.concat([train_pos, train_neg, other_sources_df]).sample(frac=1, random_state=42)

    # Turn the final sets into lists of record paths
    train_records = train_df['record'].tolist()
    val_records = val_set['record'].tolist()
    test_records = test_set['record'].tolist()

    print(f"Training on {len(train_records)} records.")
    print(f"Validating on {len(val_records)} records. Positive ratio: {len(val_pos) / len(val_set):.2%}")
    print(f"Testing on {len(test_records)} records. Positive ratio: {len(test_pos) / len(test_set):.2%}")
    print("-" * 40)

    # 2. Instantiate the Lightning module
    # Pass model-specific arguments directly to the constructor
    lightning_model = model.LitSE_ECGNet(
        in_channels=12,  # 12-lead ECG
        info_features=2,
        learning_rate=1e-4,
        weight_decay=1e-5,
    )

    # 3. Instantiate the PyTorch Lightning Trainer
    # This will handle the training loop, device placement (CPU/GPU), etc.
    # Using fast_dev_run to check if a single batch runs without errors.
    
    # Setup Wandb logger
    wandb_logger = WandbLogger(
        entity="edwards_physionet",
        project="ecg_se_blocks",
        name=f"se_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    # Setup model checkpointing
    checkpoint_cb = ModelCheckpoint(
        filename='se-best-{epoch:02d}-{val_challenge_score:.4f}',
        monitor='val_challenge_score',
        mode='max'
    )
    # Wire up ECGDataModule
    data_module = ECGDataModule(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LENGTH,
        windowing_method='entire_recording'
    )
    data_module.train_dataset = ECGDataset(train_records, DATA_DIR, is_training=True,
                                           seq_len=SEQ_LENGTH, windowing_method='entire_recording')
    data_module.val_dataset   = ECGDataset(val_records,   DATA_DIR, is_training=False,
                                           seq_len=SEQ_LENGTH, windowing_method='entire_recording')
    data_module.test_dataset  = ECGDataset(test_records,  DATA_DIR, is_training=False,
                                           seq_len=SEQ_LENGTH, windowing_method='entire_recording')
    trainer = pl.Trainer(
        max_epochs=EPOCHS, 
        accelerator='auto',
        gradient_clip_val=1.0,
        gradient_clip_algorithm='norm',
        logger=wandb_logger,
        callbacks=[checkpoint_cb]
    )

    # 4. Start training
    print("Starting training...")
    trainer.fit(model=lightning_model, datamodule=data_module)
    print("Training finished.")

    # Run test set
    print("Starting testing...")
    test_results = trainer.test(model=lightning_model, datamodule=data_module)
    print("Testing finished.")

if __name__ == "__main__":
    main()