import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torchmetrics import Accuracy, AUROC, F1Score, ConfusionMatrix
from pathlib import Path

# This assumes utils.py is findable via the path modification in mae.py
import sys
sys.path.append('../mae')
import utils

from model import MaskedAutoencoderViT, HeartBEiT, ECGDataset, get_2d_sincos_pos_embed

# Import the base models (from previous implementation)
# from heartbeit_model import MaskedAutoencoderViT, HeartBEiT, ECGDataset, get_2d_sincos_pos_embed




class ECGDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for ECG data."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.batch_size = config.get('batch_size', 32)
        self.num_workers = config.get('num_workers', 4)
        
    def prepare_data(self):
        """Download or generate data if needed."""
        # In practice, download/prepare your ECG dataset here
        pass
        
    def setup(self, stage: Optional[str] = None):
        """Create train, val, test datasets."""
        # Generate synthetic data for demonstration
        n_samples = self.config.get('n_samples', 10000)
        n_leads = 12
        n_timesteps = self.config.get('sampling_rate', 500) * self.config.get('duration', 10)
        
        # Generate synthetic ECG signals
        ecg_signals = self._generate_synthetic_ecg(n_samples, n_leads, n_timesteps)
        # For binary classification, labels are 0 or 1
        labels = np.random.randint(0, 2, n_samples)
        
        # Create full dataset
        full_dataset = ECGDataset(
            ecg_signals,
            labels,
            sampling_rate=self.config.get('sampling_rate', 500),
            duration=self.config.get('duration', 10),
            img_size=self.config.get('img_size', 224)
        )
        
        # Split dataset
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # For pretraining (no labels needed)
        self.pretrain_dataset = ECGDataset(
            ecg_signals,
            labels=None,
            sampling_rate=self.config.get('sampling_rate', 500),
            duration=self.config.get('duration', 10),
            img_size=self.config.get('img_size', 224)
        )
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
    def predict_dataloader(self):
        return self.test_dataloader()
    
    def pretrain_dataloader(self):
        """Dataloader for MAE pretraining (no labels)."""
        return torch.utils.data.DataLoader(
            self.pretrain_dataset,
            batch_size=self.config.get('pretrain_batch_size', 64),
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
    def _generate_synthetic_ecg(self, n_samples, n_leads, n_timesteps):
        """Generate synthetic ECG-like signals."""
        ecg_signals = []
        
        for _ in range(n_samples):
            t = np.linspace(0, 10, n_timesteps)
            ecg = np.zeros((n_leads, n_timesteps))
            
            for lead in range(n_leads):
                # Basic ECG pattern
                heart_rate = 60 + np.random.randint(-10, 20)
                beat_interval = 60 / heart_rate
                
                for beat_time in np.arange(0, 10, beat_interval):
                    # P wave
                    p_wave = 0.1 * np.exp(-((t - beat_time - 0.1)**2) / 0.01)
                    # QRS complex
                    qrs = 1.0 * np.exp(-((t - beat_time - 0.2)**2) / 0.001)
                    # T wave
                    t_wave = 0.2 * np.exp(-((t - beat_time - 0.4)**2) / 0.02)
                    
                    ecg[lead] += p_wave + qrs + t_wave
                    
                # Add noise
                ecg[lead] += 0.05 * np.random.randn(n_timesteps)
                ecg[lead] *= (0.8 + 0.4 * np.random.rand())
                
            ecg_signals.append(ecg)
            
        return np.array(ecg_signals)


# Training script
def train_heartbeit(config: Dict[str, Any]):
    """Complete HeartBEiT training pipeline with PyTorch Lightning."""
    
    # Set random seeds
    pl.seed_everything(config.get('seed', 42))
    
    # Initialize WandB logger
    wandb_logger = WandbLogger(
        project=config.get('project_name', 'heartbeit'),
        name=config.get('experiment_name', None),
        config=config,
        save_dir=config.get('save_dir', 'logs')
    )
    
    # Create data module
    data_module = ECGDataModule(config)
    
    # Phase 1: MAE Pretraining
    if config.get('do_pretrain', True):
        print("\n" + "="*50)
        print("Phase 1: Masked Autoencoder Pretraining")
        print("="*50)
        
        # Create MAE model
        mae_model = HeartBEiTMAELightning(config)
        
        # Callbacks for MAE
        mae_callbacks = [
            ModelCheckpoint(
                dirpath=Path(config.get('save_dir', 'checkpoints')) / 'mae',
                filename='mae-{epoch:02d}-{val_loss:.4f}',
                monitor='val/loss',
                mode='min',
                save_top_k=3,
                save_last=True
            ),
            LearningRateMonitor(logging_interval='epoch'),
            EarlyStopping(
                monitor='val/loss',
                patience=config.get('patience', 10),
                mode='min'
            )
        ]
        
        # Create trainer for MAE
        mae_trainer = pl.Trainer(
            max_epochs=config.get('pretrain_epochs', 100),
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            logger=wandb_logger,
            callbacks=mae_callbacks,
            gradient_clip_val=config.get('gradient_clip', 1.0),
            accumulate_grad_batches=config.get('accumulate_grad_batches', 1),
            precision=config.get('precision', 32),
            log_every_n_steps=10,
            val_check_interval=0.5,  # Validate 4 times per epoch
            enable_model_summary=True,
            enable_progress_bar=True
        )
        
        # Use pretrain dataloader for MAE
        mae_trainer.fit(
            mae_model,
            train_dataloaders=data_module.pretrain_dataloader(),
            val_dataloaders=data_module.val_dataloader()
        )
        
        # Get best checkpoint path
        best_mae_path = mae_trainer.checkpoint_callback.best_model_path
        print(f"Best MAE checkpoint: {best_mae_path}")
        
        # Load best MAE for classifier
        mae_checkpoint = torch.load(best_mae_path)
        pretrained_mae = MaskedAutoencoderViT(**config)
        pretrained_mae.load_state_dict(mae_checkpoint['state_dict'])
    else:
        pretrained_mae = None
        
    # Phase 2: Classifier Fine-tuning
    print("\n" + "="*50)
    print("Phase 2: Classification Fine-tuning")
    print("="*50)
    
    # Update config with MAE checkpoint path if needed
    if pretrained_mae is None and config.get('mae_checkpoint'):
        config['mae_checkpoint'] = config.get('mae_checkpoint')
    
    # Create classifier model
    classifier_model = HeartBEiTClassifierLightning(config, pretrained_mae)
    
    # Callbacks for classifier
    classifier_callbacks = [
        ModelCheckpoint(
            dirpath=Path(config.get('save_dir', 'checkpoints')) / 'classifier',
            filename='classifier-{epoch:02d}-{val_auc:.4f}',
            monitor='val/auc',
            mode='max',
            save_top_k=3,
            save_last=True
        ),
        LearningRateMonitor(logging_interval='epoch'),
        EarlyStopping(
            monitor='val/auc',
            patience=config.get('patience', 10),
            mode='max'
        )
    ]
    
    # Create trainer for classifier
    classifier_trainer = pl.Trainer(
        max_epochs=config.get('finetune_epochs', 50),
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=wandb_logger,
        callbacks=classifier_callbacks,
        gradient_clip_val=config.get('gradient_clip', 1.0),
        accumulate_grad_batches=config.get('accumulate_grad_batches', 1),
        precision=config.get('precision', 32),
        log_every_n_steps=10,
        val_check_interval=1.0,
        enable_model_summary=True,
        enable_progress_bar=True
    )
    
    # Train classifier
    classifier_trainer.fit(classifier_model, data_module)
    
    # Test on best checkpoint
    print("\n" + "="*50)
    print("Testing Best Model")
    print("="*50)
    
    classifier_trainer.test(
        classifier_model,
        dataloaders=data_module.test_dataloader(),
        ckpt_path='best'
    )
    
    # Finish WandB run
    wandb.finish()
    
    return classifier_trainer


# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        # Data parameters
        'n_samples': 10000,
        'sampling_rate': 500,
        'duration': 10,
        'num_classes': 1,
        
        # Model parameters
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': 768,
        'depth': 12,
        'n_heads': 12,
        'mlp_ratio': 4,
        'decoder_embed_dim': 512,
        'decoder_depth': 2,
        'decoder_n_heads': 8,
        'norm_pix_loss': False,
        'mask_ratio': 0.4,
        
        # Training parameters
        'batch_size': 32,
        'pretrain_batch_size': 64,
        'pretrain_epochs': 100,
        'finetune_epochs': 50,
        'learning_rate': 5e-4,  # For MAE
        'encoder_lr': 1e-5,     # For classifier encoder
        'head_lr': 1e-4,        # For classifier head
        'weight_decay': 0.05,
        'gradient_clip': 1.0,
        'accumulate_grad_batches': 1,
        'patience': 10,
        'precision': 16,  # Use 16 for mixed precision
        
        # Other parameters
        'num_workers': 4,
        'seed': 42,
        'do_pretrain': True,
        'save_dir': 'experiments',
        'project_name': 'heartbeit',
        'experiment_name': 'heartbeit_full_pipeline'
    }
    
    # Run training
    trainer = train_heartbeit(config)