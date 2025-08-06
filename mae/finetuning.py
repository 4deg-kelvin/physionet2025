import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import sys
from sklearn.model_selection import train_test_split
import utils
import os
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy, AUROC
from datetime import datetime
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
import argparse

from mae import Patching, MAELightningModule

# --- 1. Re-define necessary components from pre-training ---
# Note: These must match the definitions used during pre-training.
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)
    # also add this module's directory so local utils/dataloader resolve
    this_dir = os.path.dirname(os.path.abspath(__file__))
    if this_dir not in sys.path:
        sys.path.insert(0, this_dir)
except NameError:
    pass

# --- Standard Imports ---
# Make sure these modules are on Python path after path hack
import helper_code
from dataloader import ECGDataset, ECGDataModule, collate_fn_skip_none
class FineTuningModel(nn.Module):
    """
    A model for fine-tuning the pre-trained MAE encoder for classification.
    """
    def __init__(self, mae_encoder, num_classes=1, embed_dim=768, dropout=0.1, info_features=0):
        super().__init__()
        self.encoder = mae_encoder
        self.info_features = info_features
        
        classifier_input_dim = embed_dim + info_features
        
        self.classification_head = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, patches, info=None):
        encoded_features = self.encoder(patches)
        pooled_features = encoded_features.mean(dim=1)
        
        # Concatenate pooled ECG features with wide features if provided
        if self.info_features > 0 and info is not None:
            combined_features = torch.cat([pooled_features, info], dim=1)
        else:
            combined_features = pooled_features
            
        logits = self.classification_head(combined_features)
        return logits

class FineTuningLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for fine-tuning the MAE for classification.
    """
    def __init__(self, mae_encoder, patch_size, overlap_ratio=0.0, 
                 num_classes=1, lr=1e-4, weight_decay=0.05, num_unfrozen_layers=0, info_features=0,
                 patching_type='linear', wavelnet_out_channels=32, use_se=True):
        """
        Args:
            mae_encoder (nn.Module): The pre-trained MAEEncoder.
            patch_size (int): Size of the input patches.
            overlap_ratio (float): Overlap ratio for patching.
            num_classes (int): Number of classes for classification.
            lr (float): Learning rate.
            weight_decay (float): Weight decay for the optimizer.
            num_unfrozen_layers (int): Number of final encoder layers to unfreeze for training.
                                       If 0, the entire encoder is frozen.
            info_features (int): Number of wide demographic/clinical features.
            patching_type (str): Type of patching to use ('linear' or 'wavelnet')
            wavelnet_out_channels (int): Number of output channels for wavelnet patching
            use_se (bool): Whether to use Squeeze-and-Excitation blocks
        """
        super().__init__()
        self.save_hyperparameters(ignore=['mae_encoder'])
        
        # Create patching module with same architecture as used in pre-training
        self.patching = Patching(
            patch_size=patch_size, 
            overlap_ratio=overlap_ratio,
            use_se=use_se,
            patching_type=patching_type,
            embed_dim=mae_encoder.pos_embedding.shape[-1],
            wavelnet_out_channels=wavelnet_out_channels,
            num_leads=12 # Added to match Patching definition in mae.py
        )
        
        self.model = FineTuningModel(
            mae_encoder, 
            num_classes, 
            embed_dim=mae_encoder.pos_embedding.shape[-1],
            info_features=self.hparams.info_features
        )
        
        self.criterion = nn.BCEWithLogitsLoss()

        # Metrics
        self.train_accuracy = Accuracy(task="binary")
        self.val_accuracy = Accuracy(task="binary")
        self.test_accuracy = Accuracy(task="binary")
        self.train_auroc = AUROC(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.test_auroc = AUROC(task="binary")

        # For storing outputs for epoch-level metrics
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # --- Selective Freezing Logic ---
        # 1. Freeze the entire encoder initially
        for param in self.model.encoder.parameters():
            param.requires_grad = False

        # 2. Unfreeze the last 'num_unfrozen_layers' of the transformer encoder
        if num_unfrozen_layers > 0:
            num_encoder_layers = len(self.model.encoder.transformer_encoder.layers)
            layers_to_unfreeze = self.model.encoder.transformer_encoder.layers[num_encoder_layers - num_unfrozen_layers:]
            
            print(f"Unfreezing the last {len(layers_to_unfreeze)} layers of the encoder.")
            for layer in layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True
            
            # It's also common to unfreeze the final normalization layer of the encoder
            for param in self.model.encoder.norm.parameters():
                param.requires_grad = True


    @staticmethod
    def load_from_mae_checkpoint(checkpoint_path, **kwargs):
        """
        Loads the MAE encoder from a pre-trained checkpoint.
        Handles state dict key migration for older checkpoints.
        """
        # Manually load the checkpoint to inspect its contents first
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        hparams = checkpoint['hyper_parameters']
        state_dict = checkpoint['state_dict']

        # Instantiate the MAE module with its saved hyperparameters and load the state dict
        mae_lightning_module = MAELightningModule(**hparams)
        mae_lightning_module.load_state_dict(state_dict)
        
        encoder = mae_lightning_module.model.encoder
        
        # Extract necessary hparams for the fine-tuning module
        patch_size = hparams.get('patch_size')
        overlap_ratio = hparams.get('overlap_ratio', 0.5)
        patching_type = hparams.get('patching_type', 'linear')
        wavelnet_out_channels = hparams.get('wavelnet_out_channels', 32)
        use_se = hparams.get('use_se', True)

        return FineTuningLightningModule(
            mae_encoder=encoder,
            patch_size=patch_size,
            overlap_ratio=overlap_ratio,
            patching_type=patching_type,
            wavelnet_out_channels=wavelnet_out_channels,
            use_se=use_se,
            **kwargs
        )

    def forward(self, x, info=None):
        patches = self.patching(x)
        return self.model(patches, info)

    def _common_step(self, batch, batch_idx):
        # Unpack batch, which may or may not have info features
        if self.hparams.info_features > 0:
            signals, info, labels = batch
        else:
            signals, labels = batch
            info = None
            
        logits = self(signals, info).squeeze(1)
        
        labels = labels.float().squeeze(1)
        loss = self.criterion(logits, labels)
        
        return loss, logits, labels

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        self.train_accuracy(logits, labels.int())
        self.train_auroc(logits, labels.int())
        self.log('train_acc', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_auroc', self.train_auroc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        self.validation_step_outputs.append({'logits': logits.detach(), 'labels': labels.detach()})

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
        
        all_logits = torch.cat([x['logits'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
        
        self.val_accuracy(all_logits, all_labels.int())
        self.val_auroc(all_logits, all_labels.int())
        self.log('val_acc', self.val_accuracy, prog_bar=True)
        self.log('val_auroc', self.val_auroc, prog_bar=True)

        all_probs = torch.sigmoid(all_logits)
        challenge_score = utils.compute_challenge_score(all_labels.cpu().long(), all_probs.cpu())
        self.log('val_challenge_score', challenge_score, prog_bar=True)
        
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        loss, logits, labels = self._common_step(batch, batch_idx)
        self.log('test_loss', loss)
        self.test_step_outputs.append({'logits': logits.detach(), 'labels': labels.detach()})

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return

        all_logits = torch.cat([x['logits'] for x in self.test_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.test_step_outputs])

        self.test_accuracy(all_logits, all_labels.int())
        self.test_auroc(all_logits, all_labels.int())
        self.log('test_acc', self.test_accuracy)
        self.log('test_auroc', self.test_auroc)

        all_probs = torch.sigmoid(all_logits)
        challenge_score = utils.compute_challenge_score(all_labels.cpu().long(), all_probs.cpu())
        self.log('test_challenge_score', challenge_score)

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        # Create an optimizer that only updates the parameters with requires_grad=True.
        # This will include the classification head and any unfrozen encoder layers.
        params_to_update = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = optim.AdamW(params_to_update, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


# --- 4. Main Execution ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tuning script for ECG MAE.")
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with a small subset of data (1000 records).')
    args = parser.parse_args()

    pl.seed_everything(42)
    # --- Configuration ---
    config = {
        "data_dir": '../training_data/',
        "batch_size": 32,
        "epochs": 20,
        "lr": 1e-4,
        "weight_decay": 0.05,
        "num_unfrozen_layers": 2,
        "info_features": 2, # Age and Sex
        "pretrained_checkpoint_path": '/sailhome/kelvinkn/scr2_juice/other_work/edwards/physionet2025/mae/ecg_mae/4kzv2t9n/checkpoints/mae_encoder_pretrained.ckpt',
        "oversample" : False  # Set to True if you want to oversample the positive class in training
    }

    # 1. Load the fine-tuning module from the pre-trained MAE checkpoint
    try:
        fine_tune_module = FineTuningLightningModule.load_from_mae_checkpoint(
            config["pretrained_checkpoint_path"],
            num_classes=1,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            num_unfrozen_layers=config["num_unfrozen_layers"],
            info_features=config["info_features"]
        )
        print(f"Successfully loaded MAE encoder from {config['pretrained_checkpoint_path']}")
        
        # Correctly calculate SEQ_LENGTH based on pre-trained model's parameters
        patch_size = fine_tune_module.hparams.patch_size
        overlap_ratio = fine_tune_module.hparams.overlap_ratio
        num_patches = fine_tune_module.model.encoder.pos_embedding.shape[1]
        stride = int(patch_size * (1 - overlap_ratio))
        SEQ_LENGTH = (num_patches - 1) * stride + patch_size
        config["seq_length"] = SEQ_LENGTH # Add to config for logging
        print(f"Re-calculated SEQ_LENGTH for fine-tuning: {SEQ_LENGTH}")

    except FileNotFoundError:
        print(f"Error: Pre-trained checkpoint not found at '{config['pretrained_checkpoint_path']}'.")
        print("Please run the pre-training script first and update the path.")
        exit()

    # 2. Prepare data for fine-tuning
    all_records = helper_code.find_records_abs(config["data_dir"])
    if args.debug:
        print("--- DEBUG MODE: Using a random subset of 1000 records. ---")
        np.random.shuffle(all_records)
        all_records = all_records[:10000]
    print(f"{len(all_records)} records found in the dataset.")
    records_meta = utils.prepare_stratification(all_records)
    df = pd.DataFrame(records_meta)
    
    # Isolate data from PTB-XL (negative) and SaMi-Trop (positive)
    ptb_df = df[df['source'] == 'PTB-XL'].copy()
    sami_df = df[df['source'] == 'SaMi-Trop'].copy()

    # --- Create Validation and Test Sets with 5% positive ratio ---
    # Define the number of positive samples for validation and test sets
    # Use 10% of the positive samples for validation and 10% for testing
    val_pos_count = int(len(sami_df) * 0.10)
    test_pos_count = int(len(sami_df) * 0.10)

    # Calculate the number of negative samples needed for a 5% positive ratio
    val_neg_count = int(val_pos_count * (0.95 / 0.05))
    test_neg_count = int(test_pos_count * (0.95 / 0.05))

    # Sample for validation set
    val_pos = sami_df.sample(n=val_pos_count, random_state=42)
    val_neg = ptb_df.sample(n=val_neg_count, random_state=42)
    val_df = pd.concat([val_pos, val_neg]).sample(frac=1, random_state=42)

    # Get remaining data for test and train sets
    sami_remaining = sami_df.drop(val_pos.index)
    ptb_remaining = ptb_df.drop(val_neg.index)

    # Sample for test set from remaining data
    test_pos = sami_remaining.sample(n=test_pos_count, random_state=42)
    test_neg = ptb_remaining.sample(n=test_neg_count, random_state=42)
    test_df = pd.concat([test_pos, test_neg]).sample(frac=1, random_state=42)

    # --- Create and Oversample Training Set ---
    # Initial training set is the rest of the data
    train_pos_initial = sami_remaining.drop(test_pos.index)
    train_neg_initial = ptb_remaining.drop(test_neg.index)
    initial_train_df = pd.concat([train_pos_initial, train_neg_initial])

    # Oversample the positive class in the training set to 5%
    if config["oversample"]:
        target_pos_count = int(np.ceil((0.05 / 0.95) * len(train_neg_initial)))
        num_to_add = target_pos_count - len(train_pos_initial)

        if num_to_add > 0:
            oversampled_pos = train_pos_initial.sample(n=num_to_add, replace=True, random_state=42)
            train_df = pd.concat([initial_train_df, oversampled_pos])
        else:
            train_df = initial_train_df
    else:
        train_df = initial_train_df

    # Shuffle the final training set
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Convert to lists of record paths
    train_records = train_df['record'].tolist()
    val_records = val_df['record'].tolist()
    test_records = test_df['record'].tolist()
    
    print(f"Training on {len(train_records)} records (oversampled).")
    print(f"Final training set positive ratio: {train_df['label'].value_counts(normalize=True)[1]:.2%}")
    print(f"Validating on {len(val_records)} records.")
    print(f"Validation set positive ratio: {val_df['label'].value_counts(normalize=True)[1]:.2%}")
    print(f"Testing on {len(test_records)} records.")
    print(f"Test set positive ratio: {test_df['label'].value_counts(normalize=True)[1]:.2%}")

    # 3. Create DataModule
    data_module = ECGDataModule(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        seq_len=config["seq_length"],
        windowing_method='entire_recording' 
    )
    data_module.train_dataset = ECGDataset(train_records, config["data_dir"], is_training=True, seq_len=config["seq_length"], windowing_method='entire_recording', include_wide_feats=True)
    data_module.val_dataset   = ECGDataset(val_records,   config["data_dir"], is_training=False, seq_len=config["seq_length"], windowing_method='entire_recording', include_wide_feats=True)
    data_module.test_dataset  = ECGDataset(test_records,  config["data_dir"], is_training=False, seq_len=config["seq_length"], windowing_method='entire_recording', include_wide_feats=True)

    # 4. Setup logger and checkpointing
    wandb_logger = WandbLogger(
        entity="edwards_physionet",
        project="ecg_finetune",
        name=f"finetune_unfreeze_{config['num_unfrozen_layers']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=config
    )
    checkpoint_cb = ModelCheckpoint(
        filename='finetune-best-{epoch:02d}-{val_challenge_score:.4f}',
        monitor='val_challenge_score',
        mode='max'
    )

    # 5. Create and run trainer
    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        accelerator='auto',
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_cb],
        gradient_clip_val=1.0
    )

    trainer.fit(fine_tune_module, datamodule=data_module)
    print("Fine-tuning finished.")

    # 6. Test the best model
    print("Testing the best model...")
    trainer.test(datamodule=data_module, ckpt_path='best')



