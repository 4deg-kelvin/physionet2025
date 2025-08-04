from typing import Tuple, List, Optional, Union
import torch
import torch.optim as optim
import wandb
from mae_vit_ecg import MAEViTECG, MAEViTECG_Lightning

import helper_code
import custom_helper_code
import os
import numpy as np
import sys
from dataloader import ECGDataset, ECGDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from torchmetrics import AUROC, Accuracy
from typing import Optional, Union, List
from mae_vit_ecg import MAEViTECGFinetune, MAEViTECGClassifier

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
# from helper_code import find_records, load_signals, find_records_abs




def main():
    """Main training function with PTB-XL and SaMi-Trop stratified by label"""
    from sklearn.model_selection import train_test_split
    
    # Configuration
    PRETRAINED_CHECKPOINT = "models/mae_encoder_pretrained.ckpt"
    DATA_DIR = r"/juice2/scr2/kelvinkn/other_work/edwards/physionet2025/training_data"  # Main data directory
    BATCH_SIZE = 32
    SEQ_LEN = 5000
    EPOCHS = 25
    
    # Freezing strategies examples:
    # freeze_layers = None  # Train all parameters
    # freeze_layers = 6  # Freeze first 6 transformer blocks
    # freeze_layers = [0, 1, 2, 3]  # Freeze specific blocks
    # freeze_layers = 'patch_embed'  # Freeze only patch embedding
    # freeze_layers = 'all_but_head'  # Freeze everything except classifier
    # freeze_layers = 'all_but_last_3'  # Freeze all but last 3 transformer blocks
    
    freeze_layers = 'all_but_last_2'  # Good default for finetuning
    
    # Load data
    print("Loading finetuning data from PTB-XL and SaMi-Trop...")
    records_meta = utils.prepare_stratification(helper_code.find_records_abs(DATA_DIR))
    
    # Filter records and labels to only include PTB-XL and SaMi-Trop data
    filtered_records_meta = [record for record in records_meta if record['source'] in ['PTB-XL', 'SaMi-Trop']]
    records = [record['record'] for record in filtered_records_meta]
    labels = np.array([record['label'] for record in filtered_records_meta])
    print(f"Found {len(records)} labeled records from PTB-XL and SaMi-Trop")
    
    # Print label distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\nLabel distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  Label {label}: {count} samples ({count/len(labels)*100:.1f}%)")
    
    # Stratified train/test split using sklearn
    X_temp, X_test, y_temp, y_test = train_test_split(
        records, labels, 
        test_size=0.15, 
        stratify=labels,
        random_state=42
    )
    
    # Stratified train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.15/0.85,  # 0.15 out of remaining 0.85
        stratify=y_temp,
        random_state=42
    )
    
    print(f"\nSplit results:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Verify stratification
    for split_name, split_labels in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        unique, counts = np.unique(split_labels, return_counts=True)
        print(f"\n{split_name} set label distribution:")
        for label, count in zip(unique, counts):
            print(f"  Label {label}: {count} samples ({count/len(split_labels)*100:.1f}%)")
    
    train_paths = X_train
    val_paths = X_val
    test_paths = X_test
    
    # Create data module
    data_module = ECGDataModule(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        windowing_method='entire_recording'
    )
    
    # Set datasets with paths only (ECGDataset expects paths, not dicts)
    data_module.train_dataset = ECGDataset(
        train_paths, DATA_DIR, is_training=True, seq_len=SEQ_LEN,
        windowing_method='entire_recording', include_wide_feats=True,
        no_labels=False  # We need labels for finetuning
    )
    data_module.val_dataset = ECGDataset(
        val_paths, DATA_DIR, is_training=False, seq_len=SEQ_LEN,
        windowing_method='entire_recording', include_wide_feats=True,
        no_labels=False
    )
    data_module.test_dataset = ECGDataset(
        test_paths, DATA_DIR, is_training=False, seq_len=SEQ_LEN,
        windowing_method='entire_recording', include_wide_feats=True,
        no_labels=False
    )
    
    # Create model
    model = MAEViTECGFinetune(
        pretrained_checkpoint_path=PRETRAINED_CHECKPOINT,
        num_classes=1,  # Binary classification
        freeze_layers=freeze_layers,
        lr=1e-4,
        weight_decay=0.01,
        warmup_epochs=5,
        dropout=0.1,
        use_cls_token=True,
        lr_schedule='cosine',
        min_lr=1e-6,
        num_wide_feats=2  # Add this to handle age and sex features
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_auroc',
        mode='max',
        filename='mae-vit-finetuned-{epoch:02d}-{val_auroc:.3f}',
        save_top_k=3,
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_auroc',
        mode='max',
        patience=10,
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Logger with dataset info
    label_dist = {str(k): v for k, v in zip(*np.unique(labels, return_counts=True))}
    wandb_logger = WandbLogger(
        entity='edwards_physionet',
        project="mae-vit-ecg-finetuning",
        name=f"ptbxl-samitrop-freeze-{freeze_layers}",
        config={
            "datasets": "PTB-XL + SaMi-Trop",
            "freeze_strategy": freeze_layers,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "label_distribution": label_dist
        }
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator='auto',
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        gradient_clip_val=1.0,
        val_check_interval=0.5,  # Check validation twice per epoch
        accumulate_grad_batches=2,  # Effective batch size = 64
    )
    
    # Train
    trainer.fit(model, datamodule=data_module)
    
    # Test
    trainer.test(model, datamodule=data_module)
    
    print("Finetuning complete!")


if __name__ == "__main__":
    main()

def load_model(model_folder, verbose):
    """
    Load the finetuned MAE ViT ECG model from a checkpoint.

    Args:
        model_folder (str): The directory where the model checkpoint is saved.
        verbose (int): Verbosity level.

    Returns:
        MAEViTECGFinetune: The loaded model, in evaluation mode.
    """
    model_path = os.path.join(model_folder, 'mae_vit_finetuned.ckpt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    # Load the model directly from the full checkpoint.
    # This works because we will now save the hyperparameters with the checkpoint.
    model = MAEViTECGFinetune.load_from_checkpoint(checkpoint_path=model_path)
    
    # Set the model to evaluation mode
    model.eval()
    
    if verbose > 0:
        print(f"Successfully loaded finetuned model from {model_path}")
        
    return model

def train(data_folder, model_folder, verbose):
    """Main training function with PTB-XL and SaMi-Trop stratified by label"""    
    # Configuration
    PRETRAINED_CHECKPOINT = os.path.join(model_folder, "mae_vit_encoder_pretrained.ckpt")
    DATA_DIR = data_folder  # Main data directory
    BATCH_SIZE = 32
    SEQ_LEN = 5000
    EPOCHS = 16
    
    # Freezing strategies examples:
    # freeze_layers = None  # Train all parameters
    # freeze_layers = 6  # Freeze first 6 transformer blocks
    # freeze_layers = [0, 1, 2, 3]  # Freeze specific blocks
    # freeze_layers = 'patch_embed'  # Freeze only patch embedding
    # freeze_layers = 'all_but_head'  # Freeze everything except classifier
    # freeze_layers = 'all_but_last_3'  # Freeze all but last 3 transformer blocks
    
    freeze_layers = 'all_but_last_2'  # Good default for finetuning
    
    # Load data
    print("Loading finetuning data from PTB-XL and SaMi-Trop...")
    records_meta = custom_helper_code.find_records_abs(DATA_DIR)
    
    # Create data module
    data_module = ECGDataModule(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        windowing_method='entire_recording'
    )
    
    # Set datasets with paths only (ECGDataset expects paths, not dicts)
    data_module.train_dataset = ECGDataset(
        records_meta, DATA_DIR, is_training=True, seq_len=SEQ_LEN,
        windowing_method='entire_recording', include_wide_feats=True,
        no_labels=False  # We need labels for finetuning
    )
    data_module.val_dataset = None
    data_module.test_dataset = None
    
    # Create model
    model = MAEViTECGFinetune(
        pretrained_checkpoint_path=PRETRAINED_CHECKPOINT,
        num_classes=1,  # Binary classification
        freeze_layers=freeze_layers,
        lr=1e-4,
        weight_decay=0.01,
        warmup_epochs=5,
        dropout=0.1,
        use_cls_token=True,
        lr_schedule='cosine',
        min_lr=1e-6,
        num_wide_feats=2  # Add this to handle age and sex features
    )
    
    # # Callbacks
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=model_folder,
    #     filename='mae-vit-finetuned',
    #     save_last=True
    # )    

    # Trainer
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator='auto',
        devices=1,
        gradient_clip_val=1.0,
        val_check_interval=0.5,  # Check validation twice per epoch
        accumulate_grad_batches=2,  # Effective batch size = 64
        num_sanity_val_steps=0,  # Skip sanity check for faster training
        enable_checkpointing=False,  # Disable checkpointing for finetuning
        limit_val_batches=0,  # Disable validation loop
        limit_test_batches=0,  # Disable test loop
    )
    
    # Train
    trainer.fit(model, datamodule=data_module)

    # Save
    model_path = os.path.join(model_folder, 'mae_vit_finetuned.ckpt')
    trainer.save_checkpoint(model_path, weights_only=True)
    print(f"Finetuned model saved to {model_path}")

    
    print("Finetuning complete!")