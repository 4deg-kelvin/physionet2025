"""
Pretraining script for MAE-based ViT ECG model
Supports both WFDB (.hea/.mat) and .npy ECG files

Usage:
    python pretrain_mae_vit_ecg.py [--checkpoint CHECKPOINT_PATH] [--resume_from_checkpoint RESUME_PATH] 
                                  [--batch_size BATCH_SIZE] [--seq_len SEQ_LEN] [--epochs EPOCHS]
    
Options:
    --checkpoint: Path to a Lightning checkpoint to load model weights from. This initializes 
                 only the model weights and starts training from epoch 0 with a fresh optimizer.
                 
    --resume_from_checkpoint: Path to a Lightning checkpoint to fully resume training from, 
                             including optimizer state, epoch count, and learning rate scheduler.
                             This is the recommended way to continue training from a previous run.
                             
    --batch_size: Batch size for training (default: 64)
    --seq_len: Sequence length for ECG data (default: 5000)
    --epochs: Number of epochs to train (default: 20)
    
Examples:
    # Start a new training run
    python pretrain_mae_vit_ecg.py
    
    # Initialize model from checkpoint but start a new training run
    python pretrain_mae_vit_ecg.py --checkpoint /path/to/checkpoint.ckpt
    
    # Resume training completely from a previous checkpoint
    python pretrain_mae_vit_ecg.py --resume_from_checkpoint /path/to/checkpoint.ckpt
"""
from typing import Tuple, List, Optional, Union
import torch
import torch.optim as optim
import wandb
from mae_vit_ecg import MAEViTECG, MAEViTECG_Lightning
import helper_code
import os
import numpy as np
import sys
from dataloader import ECGDataset, ECGDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

import custom_helper_code

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

import utils


def compute_reconstruction_metrics(recon_patches, target_patches, mask):
    """Compute reconstruction quality metrics for pretraining"""
    metrics = {}
    
    # Compute per-sample reconstruction metrics
    batch_size = recon_patches.shape[0]
    mse_scores = []
    correlation_scores = []
    
    for b in range(batch_size):
        masked_idx = mask[b]
        if masked_idx.any():
            # Get reconstructed and target patches for masked regions
            recon_masked = recon_patches[b][masked_idx].flatten().detach().cpu().numpy()
            target_masked = target_patches[b][masked_idx].flatten().detach().cpu().numpy()
            
            # Mean Squared Error
            mse = np.mean((recon_masked - target_masked) ** 2)
            mse_scores.append(mse)
            
            # Pearson correlation coefficient
            if len(recon_masked) > 1 and np.std(recon_masked) > 0 and np.std(target_masked) > 0:
                corr = np.corrcoef(recon_masked, target_masked)[0, 1]
                if not np.isnan(corr):
                    correlation_scores.append(corr)
    
    # Aggregate metrics
    if mse_scores:
        metrics['reconstruction_mse'] = np.mean(mse_scores)
        metrics['reconstruction_rmse'] = np.sqrt(np.mean(mse_scores))
    else:
        metrics['reconstruction_mse'] = float('inf')
        metrics['reconstruction_rmse'] = float('inf')
        
    if correlation_scores:
        metrics['reconstruction_correlation'] = np.mean(correlation_scores)
        # Convert correlation to a "score" between 0-1 (higher is better)
        metrics['reconstruction_score'] = (np.mean(correlation_scores) + 1) / 2
    else:
        metrics['reconstruction_correlation'] = 0.0
        metrics['reconstruction_score'] = 0.0
    
    return metrics
def evaluate_mae_loss_dataloader(model: MAEViTECG, dataloader, device: str) -> Tuple[float, dict]:
    """Evaluate MAE loss using a DataLoader"""
    model.eval()
    total_loss = 0
    num_batches = 0
    all_metrics = {'reconstruction_mse': [], 'reconstruction_rmse': [], 
                   'reconstruction_correlation': [], 'reconstruction_score': []}
    
    with torch.no_grad():
        for batch_data in dataloader:
            ecg_data = batch_data['ecg'].to(device)
            
            recon, mask = model(ecg_data)
            target = ecg_data
            
            # Same loss calculation as training
            patch_size = model.patch_embed.patch_size
            channels = model.patch_embed.proj.in_channels
            num_patches = recon.shape[2] // patch_size
            recon_patches = recon.view(recon.shape[0], channels, num_patches, patch_size)
            recon_patches = recon_patches.permute(0, 2, 3, 1)
            target_patches = target.view(target.shape[0], channels, num_patches, patch_size)
            target_patches = target_patches.permute(0, 2, 3, 1)
            
            losses = []
            for b in range(recon_patches.shape[0]):
                masked_idx = mask[b]
                if masked_idx.any():
                    diff = recon_patches[b][masked_idx] - target_patches[b][masked_idx]
                    losses.append(diff.pow(2).mean())
            if losses:
                loss = torch.stack(losses).mean()
                total_loss += loss.item()
                num_batches += 1
                
                # Compute reconstruction quality metrics
                batch_metrics = compute_reconstruction_metrics(recon_patches, target_patches, mask)
                for key, value in batch_metrics.items():
                    if not np.isnan(value) and not np.isinf(value):
                        all_metrics[key].append(value)
    
    # Aggregate metrics
    aggregated_metrics = {}
    for key, values in all_metrics.items():
        if values:
            aggregated_metrics[key] = np.mean(values)
        else:
            aggregated_metrics[key] = 0.0 if 'correlation' in key or 'score' in key else float('inf')
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    return avg_loss, aggregated_metrics
def evaluate_mae_loss(model: MAEViTECG, file_list: List[str], file_type: str, batch_size: int, device: str) -> Tuple[float, dict]:
    """Helper function to evaluate MAE loss on a set of files"""
    model.eval()
    total_loss = 0
    num_batches = 0
    all_metrics = {'reconstruction_mse': [], 'reconstruction_rmse': [], 'reconstruction_correlation': [], 'reconstruction_score': []}

if __name__ == "__main__":
    # Data loading pattern following your specification
    np.random.seed(42)

    DATA_DIR = r"/sailhome/kelvinkn/scr2_juice/other_work/edwards/physionet2025/training_data"
    ADDIT_DIRS = [
        r"/juice2/scr2/kelvinkn/other_work/edwards/prna_2020_pooled_inputs",
        r"/juice2/scr2/kelvinkn/other_work/edwards/physionet2021_data",
    ]
    BATCH_SIZE = 64
    SEQ_LEN = 500 * 10
    EPOCHS = 20
    CHECKPOINT_PATH = None  # Path to a Lightning checkpoint to load model weights from
    
    # --- Data Loading and Splitting ---
    # 1. Load ALL records for pretraining
    records_meta = utils.prepare_stratification(custom_helper_code.find_records_abs(DATA_DIR))

    print("loading records")
    # # Exclude our finetuning ECGs
    records_meta = [record for record in records_meta if record['source'] not in ['PTB-XL', 'SaMi-Trop']]
    records_meta = custom_helper_code.find_records_abs(DATA_DIR)
    print("Found {} records in main directory".format(len(records_meta)))
    all_addit_records = []
    # Add additional directories if specified
    for additional_dir in ADDIT_DIRS:
        if additional_dir and os.path.exists(additional_dir):
            additional_records = custom_helper_code.find_records_abs(additional_dir)
            all_addit_records.extend(additional_records)
            print(f"Added {len(additional_records)} records from {additional_dir}")

    # 2. Combine all records into a single list
    print("WARNING: EXCLUDING UNLABELED RECORDS, THIS IS FOR COMPETITION MODEL TRAINING")
    all_records = records_meta + all_addit_records
    print(f"Total records found: {len(all_records)}")
    if len(all_records) == 0:
        raise ValueError("No records found in the specified directories. Please check the paths.")

    # 3. Shuffle and split the records into train/val/test sets
    np.random.shuffle(all_records)
    
    # Define split ratios
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    
    # Calculate split indices
    n_total = len(all_records)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split the records
    train_records = all_records[:n_train]
    val_records = all_records[n_train:n_train + n_val]
    test_records = all_records[n_train + n_val:]
    
    print(f"Split summary:")
    print(f"  Train: {len(train_records)} records ({len(train_records)/n_total*100:.1f}%)")
    print(f"  Val: {len(val_records)} records ({len(val_records)/n_total*100:.1f}%)")
    print(f"  Test: {len(test_records)} records ({len(test_records)/n_total*100:.1f}%)")
    
    # Check if MPS is available for faster training on Mac
    if torch.backends.mps.is_available():
        device = 'mps'
        print("Using MPS (Mac GPU) for training")
    elif torch.cuda.is_available():
        device = 'cuda'
        print("Using CUDA (GPU) for training")
    else:
        device = 'cpu'
        print("Using CPU for training")

    # 4. Create DataModule with proper splits
    data_module = ECGDataModule(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        windowing_method='entire_recording' 
    )
    
    # Set the datasets with our splits
    data_module.train_dataset = ECGDataset(
        train_records, 
        DATA_DIR, 
        is_training=True, 
        seq_len=SEQ_LEN, 
        windowing_method='entire_recording', 
        include_wide_feats=False, 
        no_labels=True
    )
    data_module.val_dataset = ECGDataset(
        val_records, 
        DATA_DIR, 
        is_training=False, 
        seq_len=SEQ_LEN, 
        windowing_method='entire_recording', 
        include_wide_feats=False,
        no_labels=True
    )
    data_module.test_dataset = ECGDataset(
        test_records, 
        DATA_DIR, 
        is_training=False, 
        seq_len=SEQ_LEN, 
        windowing_method='entire_recording', 
        include_wide_feats=False,
        no_labels=True
    )
    if CHECKPOINT_PATH:
        print(f"Loading model weights from checkpoint: {CHECKPOINT_PATH}")
        # Initialize the model with the checkpoint path
        vit_module = MAEViTECG_Lightning.load_from_checkpoint(
            CHECKPOINT_PATH)
    else:
        # Initialize the model with the checkpoint path if provided
        vit_module = MAEViTECG_Lightning(
            in_channels=12,
            seq_len=SEQ_LEN,
            patch_size=250,
            embed_dim=768,
            mask_ratio=0.75,
            lr=1e-4,
            warmup_epochs=5,
            weight_decay=0.05
        )
    
    # Train MAE via Lightning with validation monitoring
    checkpoint_cb = ModelCheckpoint(
        filename='mae_encoder_pretrained',       
        save_last=True
    )

    # Add wandb logging
    run_name = "mae-vit-ecg-pretraining"
    if CHECKPOINT_PATH:
        run_name += "-resumed"
        
    wandb_logger = WandbLogger(
        entity="edwards_physionet",
        project="mae-vit-ecg-pretraining",
        name=run_name
    )
    
    # Configure the trainer with optional resumption from checkpoint
    trainer_kwargs = {
        'max_epochs': EPOCHS,
        'accelerator': 'auto',
        'devices': 1,
        'logger': wandb_logger,
        'callbacks': [checkpoint_cb, LearningRateMonitor(logging_interval='epoch')],
        'gradient_clip_val': 1.0
    }

    trainer = pl.Trainer(**trainer_kwargs)
    
    trainer.fit(
        vit_module, 
        datamodule=data_module
    )
    # # Train the model with proper splits
    # # Pass both file lists (for fallback) and data_module (preferred)
    # train_mae(
    #     model=model, 
    #     val_files=val_records,
    #     test_files=test_records,
    #     epochs=20, 
    #     batch_size=BATCH_SIZE, 
    #     device=device, 
    #     use_wandb=True,
    #     data_module=data_module  # Pass the data module for proper DataLoader usage
    # )
def train(data_folder, model_folder, verbose):
    # Data loading pattern following your specification
    np.random.seed(42)

    DATA_DIR = data_folder
    BATCH_SIZE = 32
    SEQ_LEN = 500 * 10
    EPOCHS = 18
    CHECKPOINT_PATH = "mae_vit_encoder_partially_pretrained.ckpt"  # Path to a Lightning checkpoint to load model weights from
    
    # --- Data Loading and Splitting ---
    # 1. Load ALL records for pretraining
    records_meta = custom_helper_code.find_records_abs(DATA_DIR)


    print("Found {} records in main directory".format(len(records_meta)))
    # 2. Combine all records into a single list
    print("WARNING: EXCLUDING UNLABELED RECORDS, THIS IS FOR COMPETITION MODEL TRAINING")
    all_records = records_meta 
    print(f"Total records found: {len(all_records)}")
    if len(all_records) == 0:
        raise ValueError("No records found in the specified directories. Please check the paths.")

    # 3. Shuffle and split the records into train/val/test sets
    np.random.shuffle(all_records)
    
    # Check if MPS is available for faster training on Mac
    if torch.backends.mps.is_available():
        device = 'mps'
        print("Using MPS (Mac GPU) for training")
    elif torch.cuda.is_available():
        device = 'cuda'
        print("Using CUDA (GPU) for training")
    else:
        device = 'cpu'
        print("Using CPU for training")

    # 4. Create DataModule with proper splits
    data_module = ECGDataModule(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        windowing_method='entire_recording' 
    )
    
    # Set the datasets with our splits
    data_module.train_dataset = ECGDataset(
        all_records, 
        DATA_DIR, 
        is_training=True, 
        seq_len=SEQ_LEN, 
        windowing_method='entire_recording', 
        include_wide_feats=False, 
        no_labels=True
    )
    # No validation or test sets for this training function
    data_module.val_dataset = None
    data_module.test_dataset = None

    # Initialize the model with the checkpoint path if provided
    vit_module = MAEViTECG_Lightning(
        in_channels=12,
        seq_len=SEQ_LEN,
        patch_size=250,
        embed_dim=768,
        mask_ratio=0.75,
        lr=1e-4,
        warmup_epochs=5,
        weight_decay=0.05
    )
    if CHECKPOINT_PATH:
        print(f"Loading model weights from checkpoint: {CHECKPOINT_PATH}")
        # Initialize the model with the checkpoint path
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
        if 'state_dict' in checkpoint:
            vit_module.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            vit_module.load_state_dict(checkpoint, strict=False)
    else:
        print("No checkpoint provided, starting from scratch")
        raise ValueError(f"Checkpoint path at {CHECKPOINT_PATH} must be provided to load model weights")
    
    # Train MAE via Lightning with validation monitoring
    # checkpoint_cb = ModelCheckpoint(
    #     dirpath=model_folder,  # Save to specified model folder
    #     filename='mae_encoder_pretrained',       
    #     save_last=True
    # )


    # Configure the trainer with optional resumption from checkpoint
    trainer_kwargs = {
        'max_epochs': EPOCHS,
        'accelerator': 'auto',
        'devices': 1,
        'gradient_clip_val': 1.0,
        'num_sanity_val_steps': 0, # Disable validation sanity check
        'enable_checkpointing': False,  # Enable checkpointing
        'limit_val_batches': 0,  # Explicitly disable the validation loop
        'limit_test_batches': 0,  # Explicitly disable the test loop
        'logger': False,  # Disable logging for pretraining
    }

    trainer = pl.Trainer(**trainer_kwargs)
    
    trainer.fit(
        vit_module, 
        datamodule=data_module
    )
    # Save the pretrained model weights
    trainer.save_checkpoint(os.path.join(model_folder, 'mae_vit_encoder_pretrained.ckpt'), weights_only=True)