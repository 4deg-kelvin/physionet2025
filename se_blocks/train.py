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
    model_kwargs = {
        'num_leads': 12,  # 12-lead ECG
        'num_channels': [64, 128, 256, 512],
        'info_features': 2,  # Set to number of additional features if using
        'dropout_rate': 0.3
    }
    # --- Load metadata and stratify ---all th
    records_meta = utils.prepare_stratification(helper_code.find_records_abs(DATA_DIR))

    # Convert to pandas DataFrame for easier manipulation
    df = pd.DataFrame(records_meta)

    # Isolate data from different sources
    ptb_df = df[df['source'] == 'PTB-XL'].copy()
    code_df = df[df['source'] == 'CODE-15%'].copy()
    other_sources_df = df[~df['source'].isin(['PTB-XL', 'CODE-15%'])].copy()

    # 1. Create a pool for validation and test sets from PTB-XL and CODE-15%
    # We'll take 20% from each source for this pool. You can adjust this fraction.
    VAL_TEST_FRAC = 0.20

    ptb_train, ptb_val_test = train_test_split(
        ptb_df,
        test_size=VAL_TEST_FRAC,
        stratify=ptb_df['label'],
        random_state=42
    )
    code_train, code_val_test = train_test_split(
        code_df,
        test_size=VAL_TEST_FRAC,
        stratify=code_df['label'],
        random_state=42
    )

    # Combine the portions from PTB-XL and CODE-15% to form the main val/test pool
    val_test_pool = pd.concat([ptb_val_test, code_val_test])

    # 2. Split the pool 50/50 into final validation and test sets
    val_set, test_set = train_test_split(
        val_test_pool,
        test_size=0.5,
        stratify=val_test_pool['label'],
        random_state=42
    )
    # Reset index to prevent KeyError in DataLoader
    val_set = val_set.reset_index(drop=True)
    test_set = test_set.reset_index(drop=True)

    print(f"✅ Created validation set with {len(val_set)} records.")
    print(f"✅ Created test set with {len(test_set)} records.")
    print("--- Validation/Test Set Source Composition ---")
    print("Validation Set:\n", val_set['source'].value_counts(normalize=True))
    print("\nTest Set:\n", test_set['source'].value_counts(normalize=True))
    print("-" * 40)


    # 3. Prepare the initial training set from the remaining data
    initial_train_set = pd.concat([ptb_train, code_train, other_sources_df])

    print(f"Initial training set has {len(initial_train_set)} records.")
    print("Initial training set label distribution:\n", initial_train_set['label'].value_counts(normalize=True))
    print("-" * 40)

    # 4. Oversample the training set to 5% positive class
    pos_df = initial_train_set[initial_train_set['label'] == 1]
    neg_df = initial_train_set[initial_train_set['label'] == 0]

    # Calculate the required number of positive samples for a 5% ratio
    target_pos_count = int(np.ceil((0.05 / 0.95) * len(neg_df)))
    num_to_add = target_pos_count - len(pos_df)

    final_train_set = initial_train_set.copy()

    if num_to_add > 0:
        # Sample with replacement from the existing positive samples
        oversampled_pos = pos_df.sample(n=num_to_add, replace=True, random_state=42)
        # Add the new samples to the training set
        final_train_set = pd.concat([initial_train_set, oversampled_pos])

    # Shuffle the final training set
    final_train_set = final_train_set.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"🚀 Oversampling complete. Final training set has {len(final_train_set)} records.")
    print("Final training set label distribution:\n", final_train_set['label'].value_counts(normalize=True))
    print("-" * 40)

    # Turn the final sets into lists of record paths
    train_records = final_train_set['record'].tolist()
    val_records = val_set['record'].tolist()
    test_records = test_set['record'].tolist()

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
        max_epochs=5, 
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

if __name__ == "__main__":
    main()