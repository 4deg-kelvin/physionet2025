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
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dataloader import ECGDataModule, ECGDataset

from functools import cache
import argparse
from wavelnet import create_daubechies_wavelet  # new import

import custom_helper_code

# --- Robust Path Handling ---

import helper_code
import utils
import se_model as model

def parse_args():
    p = argparse.ArgumentParser(description="Train SE‐block ECG net with optional multi‐task heads and WaveletCNN")
    p.add_argument("--data-dir", default="../training_data")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--seq-seconds", type=int, default=10,
                   help="Length of each ECG seq in seconds")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--do-multitask", action="store_true",
                   help="Enable RBBB and 1dAVb auxiliary tasks", default=True)
    p.add_argument("--rbbb-loss-weight", type=float, default=0.25)
    p.add_argument("--d1avb-loss-weight", type=float, default=0.25)
    p.add_argument("--debug", action="store_true", help="Enable debug mode with fast_dev_run", default=False)
    p.add_argument("--use-wavelet-cnn", action="store_true", help="Prepend WaveletConv on each lead", default=False)
    p.add_argument("--wavelet-name", type=str, default="db6", help="Daubechies wavelet name for WaveletConv")
    p.add_argument("--wavelet-kernel-size", type=int, default=129, help="Kernel size for WaveletConv (odd integer)")
    p.add_argument("--wavelet-out-channels", type=int, default=8, help="Number of filters in WaveletConv")
    p.add_argument("--no-pooling", action="store_true", default=False)
    p.add_argument("--oversample", action="store_true",default=False)
    return p.parse_args()

def main():
    args = parse_args()
    pl.seed_everything(42)

    DATA_DIR = args.data_dir
    BATCH_SIZE = args.batch_size
    SEQ_LENGTH = utils.UNIFIED_FREQUENCY * args.seq_seconds
    EPOCHS = args.epochs
    
    # --- Load metadata and prepare data splits ---
    records_raw = helper_code.find_records_abs(DATA_DIR)
    if args.debug:
        # Sample a smaller subset for debugging
        records_raw = np.random.choice(records_raw, size=10000, replace=False).tolist()   
    records_meta = utils.prepare_stratification(records_raw)
    df = pd.DataFrame(records_meta)

    if args.oversample:
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

        train_records = train_df['record'].tolist()
        val_records = val_set['record'].tolist()
        test_records = test_set['record'].tolist()
    else:
        from sklearn.model_selection import train_test_split
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
        train_records = train_df['record'].tolist()
        val_records = val_df['record'].tolist()
        test_records = test_df['record'].tolist()



    print(f"Training on {len(train_records)} records.")
    print(f"Validating on {len(val_records)} records. ")
    print(f"Testing on {len(test_records)} records.")
    print("-" * 40)

    # 2. Instantiate Lightning model with multitask & wavelet flags
    lightning_model = model.LitSE_ECGNet(
        in_channels=12,  # 12-lead ECG
        info_features=2,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        do_multitask=args.do_multitask,
        rbbb_loss_weight=args.rbbb_loss_weight,
        d1avb_loss_weight=args.d1avb_loss_weight,
        use_wavelet_cnn=args.use_wavelet_cnn,
        wavelet_name=args.wavelet_name,
        wavelet_kernel_size=args.wavelet_kernel_size,
        wavelet_out_channels=args.wavelet_out_channels,
        no_pooling=args.no_pooling
    )

    # 3. Instantiate the PyTorch Lightning Trainer
    # This will handle the training loop, device placement (CPU/GPU), etc.
    # Using fast_dev_run to check if a single batch runs without errors.
    config = {
        "in_channels": lightning_model.hparams.in_channels,
        "info_features": lightning_model.hparams.info_features,
        "learning_rate": lightning_model.hparams.learning_rate,
        "weight_decay": lightning_model.hparams.weight_decay,
        "do_multitask": lightning_model.hparams.do_multitask,
        "rbbb_loss_weight": lightning_model.hparams.rbbb_loss_weight,
        "d1avb_loss_weight": lightning_model.hparams.d1avb_loss_weight,
        "use_wavelet_cnn": lightning_model.hparams.use_wavelet_cnn,
        "wavelet_name": lightning_model.hparams.wavelet_name,
        "wavelet_kernel_size": lightning_model.hparams.wavelet_kernel_size,
        "wavelet_out_channels": lightning_model.hparams.wavelet_out_channels,
        "no_pooling": lightning_model.hparams.no_pooling,
        "batch_size": BATCH_SIZE,
        "seq_length": SEQ_LENGTH,
        "epochs": EPOCHS,
        "data_dir": DATA_DIR,
        "debug": args.debug,
        "oversample": args.oversample
    }


    # Setup Wandb logger
    wandb_logger = WandbLogger(
        entity="edwards_physionet",
        project="ecg_se_blocks",
        name=f"se_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=config
    )

    # Setup model checkpointing
    checkpoint_cb = ModelCheckpoint(
        filename='se-best-{epoch:02d}-{val_challenge_score:.4f}',
        monitor='val_challenge_score',
        mode='max'
    )
    # Wire up ECGDataModule and pass do_multitask to dataset
    data_module = ECGDataModule(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LENGTH,
        windowing_method='entire_recording'
    )
    data_module.train_dataset = ECGDataset(
        train_records, DATA_DIR, is_training=True,
        seq_len=SEQ_LENGTH, windowing_method='entire_recording',
        include_wide_feats=True, do_multitask=args.do_multitask
    )
    data_module.val_dataset = ECGDataset(
        val_records, DATA_DIR, is_training=False,
        seq_len=SEQ_LENGTH, windowing_method='entire_recording',
        include_wide_feats=True, do_multitask=args.do_multitask
    )
    data_module.test_dataset = ECGDataset(
        test_records, DATA_DIR, is_training=False,
        seq_len=SEQ_LENGTH, windowing_method='entire_recording',
        include_wide_feats=True, do_multitask=args.do_multitask
    )
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
def train_se(data_folder, model_folder):
    # --- fixed hyperparameters ---
    DATA_DIR = data_folder
    MODEL_DIR = model_folder
    BATCH_SIZE = 32
    EPOCHS = 1
    LR = 1e-4
    WEIGHT_DECAY = 1e-5
    SEQ_SECONDS = 10
    SEQ_LEN = utils.UNIFIED_FREQUENCY * SEQ_SECONDS

    DO_MULTITASK = True
    RBBB_WEIGHT = 0.25
    D1AVB_WEIGHT = 0.25
    USE_WAVELET_CNN = False


    # --- prepare splits ---
    records = custom_helper_code.find_records_abs(DATA_DIR)
    meta = utils.prepare_stratification(records)
    df = pd.DataFrame(meta)

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    train_records = train_df['record'].tolist()
    val_records   = val_df['record'].tolist()


    # --- data module ---
    dm = ECGDataModule(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        windowing_method='entire_recording'
    )
    dm.train_dataset = ECGDataset(
        train_records, DATA_DIR, is_training=True,
        seq_len=SEQ_LEN, windowing_method='entire_recording',
        include_wide_feats=True, do_multitask=DO_MULTITASK
    )
    dm.val_dataset = ECGDataset(
        val_records, DATA_DIR, is_training=False,
        seq_len=SEQ_LEN, windowing_method='entire_recording',
        include_wide_feats=True, do_multitask=DO_MULTITASK
    )
    dm.test_dataset = None

    # --- model ---
    se_model = model.LitSE_ECGNet(
        in_channels=12,
        info_features=2,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        do_multitask=DO_MULTITASK,
        rbbb_loss_weight=RBBB_WEIGHT,
        d1avb_loss_weight=D1AVB_WEIGHT,
        use_wavelet_cnn=USE_WAVELET_CNN
    )

    # Add early stopping callback
    early_stopping_cb = pl.callbacks.EarlyStopping(
        monitor='val_challenge_score',
        patience=3,
        mode='max'
    )
    # --- training setup ---
    ckpt_cb = ModelCheckpoint(
        dirpath=MODEL_DIR,
        filename='se_trained',
        monitor='val_challenge_score',
        mode='max',
        save_top_k=1,
        save_weights_only=True
    )
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator='auto',
        default_root_dir=model_folder,
        logger=False,
        callbacks=[ckpt_cb, early_stopping_cb],
        limit_test_batches=0
    )

    # --- run training ---
    trainer.fit(se_model, datamodule=dm)
if __name__ == "__main__":
    main()

def load_model(model_folder, verbose=False):
    from se_model import LitSE_ECGNet
    # load the best checkpoint
    ckpt_path = os.path.join(model_folder, 'se_trained.ckpt')
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    # rebuild and load the Lightning model
    model = LitSE_ECGNet.load_from_checkpoint(ckpt_path)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()
    return model

def run_model(record, model, verbose=False):
    import torch, os
    import utils

    # Preprocess the signal (same logic as in Dataset.__getitem__)
    signal, wide_feats = utils.preprocess_signal(
        record,
        windowing_method='entire_recording',
        is_training=False
    )

    device = next(model.parameters()).device
    sig_t = torch.as_tensor(signal, dtype=torch.float32, device=device).unsqueeze(0)
    wf_t = torch.as_tensor(wide_feats, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        # disable multi-task heads: use only the primary SE‐block model
        main_logits = model.model(sig_t, wf_t)
        prob = torch.sigmoid(main_logits).item()
        pred = int(prob > 0.5)

    return pred, prob