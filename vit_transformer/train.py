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


# --- Robust Path Handling ---
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)
    # also add this module's directory to resolve local modules
    this_dir = os.path.dirname(os.path.abspath(__file__))
    if this_dir not in sys.path:
        sys.path.insert(0, this_dir)
    # also add mae directory to resolve utils
    mae_dir = os.path.join(project_root, 'vit_transformer')
    if mae_dir not in sys.path:
        sys.path.insert(0, mae_dir)
except NameError:
    pass
import helper_code
import utils
from model import HeartBEiTMAELightning
from dataloader import ECGDataModule, ECGMixedDataset


def main():
    pl.seed_everything(42)
    
    # --- Configuration ---
    config = {
        # Data parameters
        'data_dir': '../training_data',
        'addit_data_dir': r"/juice2/scr2/kelvinkn/other_work/edwards/prna_2020_pooled_inputs",
        'sampling_rate': utils.UNIFIED_FREQUENCY,
        'duration': 10, # seconds
        
        # Model parameters
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': 768,
        'depth': 12,
        'n_heads': 12,
        'mlp_ratio': 4.0,
        'decoder_embed_dim': 512,
        'decoder_depth': 2,
        'decoder_n_heads': 8,
        'norm_pix_loss': False,
        'mask_ratio': 0.80,
        
        # Training parameters
        'batch_size': 32,
        'pretrain_epochs': 100,
        'learning_rate': 5e-4,
        'weight_decay': 0.05,
        'patience': 10,
        'precision': 16,
        
        # Other parameters
        'num_workers': 4,
        'seed': 42,
    }

    # --- Data Loading and Splitting ---
    addit_data = helper_code.find_records_abs(config['addit_data_dir'])
    records_meta = helper_code.find_records_abs(config['data_dir'])

    # Combine the two datasets
    all_meta = records_meta + addit_data

    # 10% for test
    tv_meta, test_meta = train_test_split(all_meta, test_size=0.1, random_state=42)
    # ~11.11% of train_val for validation gives 10% overall
    train_meta, val_meta = train_test_split(tv_meta, test_size=1/9, random_state=42)
    
    train_records = train_meta
    val_records   = val_meta
    test_records  = test_meta

    # --- DataModule Setup ---
    data_module = ECGDataModule(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        seq_len=config['sampling_rate'] * config['duration'],
        windowing_method='entire_recording' # This is not used by ECGAbsPathDataset but kept for consistency
    )
    data_module.train_dataset = ECGMixedDataset(train_records, is_training=True)
    data_module.val_dataset   = ECGMixedDataset(val_records, is_training=False)
    data_module.test_dataset  = ECGMixedDataset(test_records, is_training=False)

    # --- Model Instantiation ---
    mae_module = HeartBEiTMAELightning(config)

    # --- Trainer Setup ---
    wandb_logger = WandbLogger(
        entity="edwards_physionet",
        project="ecg_vit_mae_pretrain",
        name=f"heartbeit_pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=config
    )

    checkpoint_cb = ModelCheckpoint(
        filename='heartbeit-mae-best-{epoch:02d}-{val_loss:.4f}',
        monitor='val/loss',
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=config['pretrain_epochs'], 
        accelerator='auto', 
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_cb],
        precision=config['precision']
    )

    # --- Training and Testing ---
    print("Starting HeartBEiT MAE pre-training...")
    trainer.fit(model=mae_module, datamodule=data_module)
    
    print("\nTesting best model on test set...")
    trainer.test(mae_module, datamodule=data_module, ckpt_path='best')

    # --- Save Final Artifacts ---
    encoder_save_path = 'heartbeit_mae_encoder_pretrained.pth'
    torch.save(mae_module.mae.state_dict(), encoder_save_path)
    print(f"\nPre-trained MAE model saved to {encoder_save_path}")
    print("Training finished.")

if __name__ == "__main__":
    main()