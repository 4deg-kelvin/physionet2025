import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
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
import custom_helper_code
from dataloader import ECGDataset, ECGDataModule, collate_fn_skip_none
from mae import Patching, MAELightningModule


# --- Standard Imports ---
# Make sure these modules are on Python path after path hack
import custom_helper_code
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
                 num_classes=1, lr=1e-4, weight_decay=0.05, num_unfrozen_layers=0, info_features=0):
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
        """
        super().__init__()
        self.save_hyperparameters(ignore=['mae_encoder'])
        
        self.patching = Patching(patch_size, overlap_ratio)
        self.model = FineTuningModel(
            mae_encoder, 
            num_classes, 
            embed_dim=mae_encoder.pos_embedding.shape[-1],
            info_features=self.hparams.info_features
        )
        
        self.criterion = nn.BCEWithLogitsLoss()

        # Metrics
        self.train_accuracy = Accuracy(task="binary")
        self.train_auroc = AUROC(task="binary")

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
        """
        mae_lightning_module = MAELightningModule.load_from_checkpoint(checkpoint_path)
        encoder = mae_lightning_module.model.encoder
        
        hparams = mae_lightning_module.hparams
        patch_size = hparams.get('patch_size')
        overlap_ratio = hparams.get('overlap_ratio', 0.5)  # Default to 50% overlap if not specified

        return FineTuningLightningModule(
            mae_encoder=encoder,
            patch_size=patch_size,
            overlap_ratio=overlap_ratio,
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
                "monitor": "train_loss",
            },
        }


def load_finetuned_model(model_folder, **kwargs):
    """
    Loads the final fine-tuned model from the 'finetuned_model.ckpt' file in the model_folder.
    """
    # 1. Load the pre-trained MAE encoder first
    mae_ckpt_path = os.path.join(model_folder, "mae_encoder_pretrained.ckpt")
    if not os.path.exists(mae_ckpt_path):
        raise FileNotFoundError(f"Could not find pre-trained MAE checkpoint at '{mae_ckpt_path}'.")
    
    mae_lightning_module = MAELightningModule.load_from_checkpoint(mae_ckpt_path)
    encoder = mae_lightning_module.model.encoder

    # 2. Load the fine-tuned model, providing the encoder
    finetuned_ckpt_path = os.path.join(model_folder, 'finetuned_model.ckpt')
    if not os.path.exists(finetuned_ckpt_path):
        raise FileNotFoundError(f"Could not find 'finetuned_model.ckpt' in {model_folder}. Please ensure training was completed.")
    
    # Provide the mae_encoder when loading the checkpoint
    model = FineTuningLightningModule.load_from_checkpoint(
        finetuned_ckpt_path, 
        mae_encoder=encoder, 
        **kwargs
    )
    return model


# --- 4. Main Execution ---

def train_finetune_model(data_folder, model_folder, verbose):
    pl.seed_everything(42)

    # checkpoint path is model_folder + "mae_encoder_pretrained.pth"
    ckpt_path = os.path.join(model_folder, "mae_encoder_pretrained.ckpt")

    if not os.path.exists(ckpt_path):
        raise Exception(f"Error: Pre-trained checkpoint not found at '{ckpt_path}'.")
        

    # --- Configuration ---
    config = {
        "data_dir": data_folder,
        "batch_size": 32,
        "epochs": 16,
        "lr": 1e-4,
        "weight_decay": 0.05,
        "num_unfrozen_layers": 2,
        "info_features": 2, # Age and Sex
        "pretrained_checkpoint_path": ckpt_path,
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

    # 2. Prepare training records (no validation/test)
    records_meta = custom_helper_code.find_records_abs(config["data_dir"])
    if not records_meta:
        raise ValueError(f"No records found in {config['data_dir']}.")
    train_records = records_meta

    # 3. Create DataModule with training only
    data_module = ECGDataModule(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        seq_len=config["seq_length"],
        windowing_method='entire_recording'
    )
    data_module.train_dataset = ECGDataset(
        train_records,
        config["data_dir"],
        is_training=True,
        seq_len=config["seq_length"],
        windowing_method='entire_recording',
        include_wide_feats=True
    )

    # 4. Checkpoint only last model
    checkpoint_cb = ModelCheckpoint(
        dirpath=model_folder,
        filename='finetuned_model',  # extension “.ckpt” will be added automatically
        save_last=True
    )

    # 5. Trainer without logger, no val/test loaders
    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        accelerator='auto',
        devices=1,
        callbacks=[checkpoint_cb],
        gradient_clip_val=1.0,
        num_sanity_val_steps=0  # No sanity check, disables validation/test
    )

    trainer.fit(fine_tune_module, datamodule=data_module)
    print("Fine-tuning finished.")


