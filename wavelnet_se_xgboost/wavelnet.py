# main_ecg_classifier.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import xgboost as xgb
from sklearn.metrics import accuracy_score
import numpy as np
import math
import pywt # Added for Daubechies wavelet generation. Ensure you have it installed: pip install PyWavelets
import sys
import os
from tqdm import tqdm
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
import torch.optim as optim

# Import parent directory to access helper_code and utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Standard Imports ---
import helper_code
import utils
from dataloader import ECGDataset, collate_fn_skip_none, ECGDataModule

# --- Wavelet Convolution Layer (User Provided) ---

def scale_wavelet(mother_wavelet, a, len_wavelet):
    """
    Scales the mother wavelet by a factor 'a' and pads it to the desired length.
    """
    wavelet_scaled = torch.zeros(a.size(dim=0), len_wavelet)
    wavelet_scaled = wavelet_scaled.to(mother_wavelet.device)
    # Reshape mother wavelet to be 3D for interpolation
    mother_wavelet_3d = mother_wavelet.view(1, 1, -1)

    for i in range(a.size(dim=0)):
        # Use interpolate for resampling
        wavelet_scaled_i = F.interpolate(mother_wavelet_3d, scale_factor=a[i].item(), mode='linear', align_corners=False, recompute_scale_factor=True)
        wavelet_scaled_i = wavelet_scaled_i.view(-1)
        len_scaled = len(wavelet_scaled_i)

        # Center the scaled wavelet
        offset = (len_wavelet - len_scaled) // 2
        if offset >= 0:
            wavelet_scaled[i, offset:offset + len_scaled] = torch.div(wavelet_scaled_i, torch.sqrt(a[i]))

    return wavelet_scaled

class WaveletConv(nn.Module):
    """
    Wavelet-based convolution layer where the scale of the wavelets is learnable.

    Parameters
    ----------
    in_channels : int
        Number of input channels. Must be 1.
    out_channels : int
        Number of output filters (wavelets).
    kernel_size : int
        Filter length. Must be an odd number.
    mother_wavelet : np.ndarray
        The mother wavelet function to be scaled.
    a_min : float, optional
        Minimum scale parameter. If None, it's determined automatically.
    """
    def __init__(self, in_channels, out_channels, kernel_size, mother_wavelet, a_min=None, stride=1, dilation=1, bias=None, groups=1):
        super(WaveletConv, self).__init__()

        if in_channels != 1:
            raise ValueError(f'WaveletConv only supports one input channel (got {in_channels})')
        if kernel_size % 2 != 1:
            raise ValueError(f'WaveletConv only supports an odd kernel size (got {kernel_size})')
        if bias:
            raise ValueError('WaveletConv does not support bias.')
        if groups > 1:
            raise ValueError('WaveletConv does not support groups.')
        if np.abs(np.sum(mother_wavelet)) > 0.01:
            raise ValueError('Mother wavelet does not satisfy the zero-mean condition.')
        if np.abs(np.sum(mother_wavelet**2) - 1) > 0.01:
            raise ValueError('Mother wavelet does not satisfy the unit-energy condition.')
        if a_min is not None and (a_min >= 1 or a_min <= 0):
            raise ValueError('Minimum scale parameter `a_min` must be between 0 and 1.')

        # If a_min is not provided, we determine it automatically.
        # This is a heuristic to ensure the smallest wavelet is reasonably sampled.
        if a_min is None:
            a_min = 11 / kernel_size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.mother_wavelet = torch.from_numpy(mother_wavelet.astype(np.float32))
        self.a_min = a_min
        self.stride = stride
        self.padding = int(np.floor(self.kernel_size / 2))
        self.dilation = dilation
        self.bias = bias
        self.groups = groups

        # Initialize scale parameters, evenly spaced between a_min and 1
        a_init = np.linspace(self.a_min, 1, self.out_channels)
        self.a = nn.Parameter(torch.Tensor(a_init).view(-1, 1))

    def forward(self, x):
        """
        Performs the wavelet-based convolution.
        """
        self.mother_wavelet = self.mother_wavelet.to(x.device)

        # Clamp scale parameters to be within the valid range [a_min, 1]
        a_clamped = torch.clamp(self.a, min=self.a_min, max=1)

        # Generate wavelet kernels by scaling the mother wavelet
        wavelet_kernels = scale_wavelet(self.mother_wavelet, a_clamped, self.kernel_size)
        wavelet_kernels = wavelet_kernels.view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(
            input=x,
            weight=wavelet_kernels,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=self.bias,
            groups=self.groups
        )


# --- SE Block ---
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for 1D convolutions.
    This block learns to re-weight channel features.
    """
    def __init__(self, num_channels, reduction_ratio=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        b, c, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)


# --- Attention Block ---
class AttentionBlock(nn.Module):
    """A self-attention mechanism for 1D sequences."""
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.query = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv1d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (B, C, T)
        query = self.query(x).permute(0, 2, 1) # (B, T, C')
        key = self.key(x) # (B, C', T)
        
        # Attention map
        attention_map = torch.bmm(query, key) # (B, T, T)
        attention_map = F.softmax(attention_map, dim=-1)
        
        value = self.value(x) # (B, C, T)
        
        # Apply attention
        out = torch.bmm(value, attention_map.permute(0, 2, 1)) # (B, C, T)
        out = self.gamma * out + x # Learnable skip connection
        
        return out, attention_map


# --- WaveNet Residual Block ---
class WaveNetResidualBlock(nn.Module):
    """
    A WaveNet-style residual block with dilated convolutions, SE block, and residual connections.
    """
    def __init__(self, in_channels, out_channels, dilation, kernel_size=3):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.padding = (self.kernel_size - 1) * self.dilation // 2

        # Dilated convolution with gated activation
        self.conv_dilated = nn.Conv1d(in_channels, 2 * out_channels, kernel_size,
                                      padding=self.padding, dilation=dilation)
        self.bn_dilated = nn.BatchNorm1d(2 * out_channels)
        
        # Attention Block for temporal feature refinement
        self.attention_block = AttentionBlock(out_channels)

        # SE Block for channel-wise attention
        self.se_block = SEBlock(out_channels)

        # 1x1 convolution for the output and skip connection
        self.conv_1x1_out = nn.Conv1d(out_channels, out_channels, 1)
        self.bn_out = nn.BatchNorm1d(out_channels)
        self.conv_1x1_skip = nn.Conv1d(out_channels, out_channels, 1)
        self.bn_skip = nn.BatchNorm1d(out_channels)

        # 1x1 convolution for the residual connection if channels don't match
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual_conv = None

    def forward(self, x):
        residual = x

        # Dilated convolution and gated activation
        x = self.conv_dilated(x)
        x = self.bn_dilated(x)
        tanh_out = torch.tanh(x[:, :x.size(1)//2, :])
        sigmoid_out = torch.sigmoid(x[:, x.size(1)//2:, :])
        x = tanh_out * sigmoid_out

        # Apply Attention block
        x, _ = self.attention_block(x) # We don't need the attention map here

        # Apply SE block
        x = self.se_block(x)

        # Generate skip connection and output for the next block
        skip_connection = self.conv_1x1_skip(x)
        skip_connection = self.bn_skip(skip_connection)
        block_output = self.conv_1x1_out(x)
        block_output = self.bn_out(block_output)

        # Add residual connection
        if self.residual_conv:
            residual = self.residual_conv(residual)
        
        return block_output + residual, skip_connection


# --- Model Architecture ---

# 1. PyTorch Lightning Module for Feature Extraction
class ECGFeatureExtractor(pl.LightningModule):
    """
    A hybrid model architecture combining learnable wavelet convolutions for feature
    extraction with an XGBoost classifier.

    The architecture consists of two main stages:

    1.  **PyTorch-based Feature Extractor (this class):**
        -   **Input:** A 12-lead ECG signal of shape (batch, 12, seq_len).
        -   **Parallel Wavelet Convolutions:** The model uses a `ModuleList` of 12
            `WaveletConv` layers. Each of the 12 ECG leads is processed
            independently by its own `WaveletConv` layer. This allows the model
            to learn optimal wavelet scales for each lead.
        -   **Feature Concatenation:** The outputs from the 12 parallel wavelet
            layers are concatenated along the channel dimension, resulting in a
            rich feature map of shape (batch, 12 * out_channels, seq_len).
        -   **Feature Projection:** The concatenated feature map is then passed
            through a projector:
            1.  `AdaptiveAvgPool1d`: Global average pooling is applied across the
                sequence length to create a fixed-size representation.
            2.  `Flatten`: The pooled features are flattened.
            3.  `Linear`: A fully connected layer projects the features into a
                128-dimensional vector.
        -   **Pre-training Classifier:** A final linear layer acts as a classifier
            head. During the PyTorch training phase, this is used with a binary
            cross-entropy loss to learn the parameters of the `WaveletConv` layers.

    2.  **XGBoost Classifier (in the main block):**
        -   **Feature Extraction:** After the PyTorch model is trained, the
            `extract_features` method is used. It passes the data through the
            network but stops before the final classifier head, yielding the
            128-dimensional feature vectors.
        -   **Classification:** These extracted features are then used to train a
            powerful `XGBClassifier`, which handles the final prediction task.
    """
    def __init__(self, num_leads=12, out_channels=32,
                 kernel_size=129, mother_wavelet=None, learning_rate=1e-3,
                 wavenet_layers=6, wavenet_channels=32, dropout_rate=0.5):
        super().__init__()
        self.save_hyperparameters('num_leads', 'out_channels', 'kernel_size', 'learning_rate', 'wavenet_layers', 'wavenet_channels', 'dropout_rate')
        
        # Create a list of WaveletConv layers, one for each lead
        self.wavelet_convs = nn.ModuleList([
            WaveletConv(1, out_channels, kernel_size, mother_wavelet) for _ in range(num_leads)
        ])
        
        # Initial 1x1 convolution to prepare for WaveNet blocks
        self.conv_1x1_pre = nn.Conv1d(num_leads * out_channels, wavenet_channels, 1)

        # WaveNet-style residual blocks
        self.wavenet_blocks = nn.ModuleList()
        for i in range(wavenet_layers):
            dilation = 2**i
            self.wavenet_blocks.append(
                WaveNetResidualBlock(wavenet_channels, wavenet_channels, dilation)
            )

        # Final processing layers
        self.final_convs = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(wavenet_channels, wavenet_channels, 1),
            nn.ReLU(),
            nn.Conv1d(wavenet_channels, wavenet_channels, 1)
        )
        
        # Spatial Dropout to be applied after the final convs
        self.spatial_dropout = nn.Dropout1d(p=self.hparams.dropout_rate)

        # Projector to create a fixed-size feature vector for XGBoost
        self.feature_projector = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(wavenet_channels, 128),
            nn.ReLU()
        )
        
        # Classifier for the pre-training task (to learn the wavelet scales) - now binary
        self.classifier = nn.Linear(128, 1)

        # Lists to store step outputs for epoch-end hooks
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        # x shape: (batch, num_leads, seq_len)
        
        # Apply WaveletConv to each lead independently
        lead_outputs = []
        for i in range(self.hparams.num_leads):
            lead_input = x[:, i:i+1, :] # Shape: (batch, 1, seq_len)
            lead_output = F.relu(self.wavelet_convs[i](lead_input))
            lead_outputs.append(lead_output)
            
        # Concatenate features along the channel dimension
        x = torch.cat(lead_outputs, dim=1) # Shape: (batch, num_leads * out_channels, seq_len)
        
        # Prepare for WaveNet blocks
        x = self.conv_1x1_pre(x)

        # Pass through WaveNet blocks and collect skip connections
        skip_connections = []
        for block in self.wavenet_blocks:
            x, skip = block(x)
            skip_connections.append(skip)
        
        # Sum skip connections
        x = torch.sum(torch.stack(skip_connections), dim=0)

        # Final processing
        x = self.final_convs(x)
        
        # Apply spatial dropout
        x = self.spatial_dropout(x)

        # Project features
        projected_features = self.feature_projector(x)
        # Classify for pre-training loss
        output = self.classifier(projected_features)
        return output

    def extract_features(self, x):
        """Extracts features to be used by the XGBoost model."""
        self.eval() # Ensure model is in evaluation mode
        with torch.no_grad():
            # Apply WaveletConv to each lead independently
            lead_outputs = []
            for i in range(self.hparams.num_leads):
                lead_input = x[:, i:i+1, :]
                lead_output = F.relu(self.wavelet_convs[i](lead_input))
                lead_outputs.append(lead_output)
            
            x = torch.cat(lead_outputs, dim=1)
            x = self.conv_1x1_pre(x)

            skip_connections = []
            for block in self.wavenet_blocks:
                x, skip = block(x)
                skip_connections.append(skip)
            
            x = torch.sum(torch.stack(skip_connections), dim=0)
            x = self.final_convs(x)
            
            # Apply spatial dropout (will be inactive in eval mode, which is correct)
            x = self.spatial_dropout(x)
            
            projected_features = self.feature_projector(x)
        return projected_features

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze() # Squeeze for binary classification
        loss = F.binary_cross_entropy_with_logits(y_hat, y.squeeze().float())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze() # Squeeze for binary classification
        loss = F.binary_cross_entropy_with_logits(y_hat, y.squeeze().float())
        self.log('val_loss', loss, prog_bar=True)
        output = {'loss': loss, 'preds': y_hat, 'targets': y.squeeze()}
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        if not outputs:
            return
        preds = torch.cat([x['preds'] for x in outputs])
        targets = torch.cat([x['targets'] for x in outputs])
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(preds).cpu().numpy()
        targets = targets.cpu().numpy()
        
        challenge_score = utils.compute_challenge_score(targets, probs)
        self.log('val_challenge_score', challenge_score, prog_bar=True)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze() # Squeeze for binary classification
        loss = F.binary_cross_entropy_with_logits(y_hat, y.squeeze().float())
        self.log('test_loss', loss)
        output = {'loss': loss, 'preds': y_hat, 'targets': y.squeeze()}
        self.test_step_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        if not outputs:
            return
        preds = torch.cat([x['preds'] for x in outputs])
        targets = torch.cat([x['targets'] for x in outputs])
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(preds).cpu().numpy()
        targets = targets.cpu().numpy()
        
        challenge_score = utils.compute_challenge_score(targets, probs)
        self.log('test_challenge_score', challenge_score, prog_bar=True)
        self.test_step_outputs.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        features = self.extract_features(x)
        return features, y

    def configure_optimizers(self):
        # Use AdamW optimizer
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.05)
        
        # Cosine annealing with warmup
        def lr_lambda(epoch):
            warmup_epochs = 5
            if self.trainer.max_epochs <= warmup_epochs:
                # If total epochs is less than warmup, just use linear warmup
                return epoch / max(1, self.trainer.max_epochs)
            
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (self.trainer.max_epochs - warmup_epochs)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

# --- Main Execution Block ---

def create_daubechies_wavelet(name='db6', kernel_size=129):
    """ 
    Creates a Daubechies wavelet, resamples it to the kernel_size, 
    and normalizes it for use as a mother wavelet.
    """
    # Generate the wavelet function from pywt
    wavelet = pywt.Wavelet(name)
    # We need to choose a level that gives enough points to resample from.
    # Level 8 is generally sufficient for good resolution.
    phi, psi, x = wavelet.wavefun(level=8)

    # Resample the wavelet to the desired kernel_size
    # We use a simple linear interpolation for this.
    x_new = np.linspace(x.min(), x.max(), kernel_size)
    psi_resampled = np.interp(x_new, x, psi)
    
    # Normalize to have zero mean and unit energy
    psi_normalized = psi_resampled - np.mean(psi_resampled)
    psi_normalized /= np.sqrt(np.sum(psi_normalized**2))
    
    return psi_normalized


if __name__ == '__main__':
    # --- Configuration ---
    SEQ_LENGTH = 5000 # Using a longer sequence length for real data
    BATCH_SIZE = 32
    EPOCHS = 20 # For demonstration; increase for real tasks
    WAVELET_KERNEL_SIZE = 129 # Must be odd
    WAVELET_OUT_CHANNELS = 32
    WAVENET_LAYERS = 6      # Number of residual blocks in the WaveNet
    WAVENET_CHANNELS = 32   # Number of channels in the WaveNet blocks
    LEARNING_RATE = 1e-3
    DROPOUT_RATE = 0.5      # Dropout rate for the feature projector
    NUM_LEADS = 12
    DATA_DIR = "../training_data" # Adjust if needed

    # --- WandB and Config Setup ---
    config = {
        "seq_length": SEQ_LENGTH,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "wavelet_kernel_size": WAVELET_KERNEL_SIZE,
        "wavelet_out_channels": WAVELET_OUT_CHANNELS,
        "wavenet_layers": WAVENET_LAYERS,
        "wavenet_channels": WAVENET_CHANNELS,
        "learning_rate": LEARNING_RATE,
        "dropout_rate": DROPOUT_RATE,
        "num_leads": NUM_LEADS,
    }
    
    wandb_logger = WandbLogger(
        entity="edwards_physionet",
        project="wavelnet_se_xgboost",
        name=f"wavelnet_xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=config
    )

    # --- 1. Generate Mother Wavelet and Load ECG Data ---
    print("Generating Daubechies (db6) mother wavelet...")
    mother_wavelet = create_daubechies_wavelet(name='db6', kernel_size=WAVELET_KERNEL_SIZE)
    
    print("Loading ECG data...")
    # Load records and labels using the setup from mae.py
    records = helper_code.find_records_abs(DATA_DIR)
    # records = np.random.choice(records, 1000, replace=False)  # For demonstration, use a subset of 1000 records
    all_records_meta = utils.prepare_stratification(records)
    all_records = [rec['record'] for rec in all_records_meta]
    all_labels_list = [rec['label'] for rec in all_records_meta]
    
    # For binary classification, we check for the presence of any diagnosis.
    # Label is 1 if any diagnosis is present (label is not 0), 0 otherwise.
    labels = np.array([1.0 if l != 0 else 0.0 for l in all_labels_list], dtype=np.float32)
    
    print(f"Created binary labels. Positive samples (any diagnosis): {int(np.sum(labels))}/{len(labels)}")

    # Split records into train, validation, and test sets
    train_size = int(0.7 * len(all_records))
    val_size = int(0.15 * len(all_records))
    test_size = len(all_records) - train_size - val_size
    
    indices = np.arange(len(all_records))
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]

    train_records = [all_records[i] for i in train_indices]
    train_labels = labels[train_indices]
    val_records = [all_records[i] for i in val_indices]
    val_labels = labels[val_indices]
    test_records = [all_records[i] for i in test_indices]
    test_labels = labels[test_indices]

    # Wire up ECGDataModule
    data_module = ECGDataModule(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LENGTH,
        windowing_method='entire_recording' # Default, will be overridden by dataset
    )

    # Create datasets for 12-lead ECGs and assign to the data module
    data_module.train_dataset = ECGDataset(train_records, DATA_DIR, is_training=True, no_labels=False,
                                           seq_len=SEQ_LENGTH, windowing_method='entire_recording')
    data_module.val_dataset = ECGDataset(val_records, DATA_DIR, is_training=False, no_labels=False,
                                         seq_len=SEQ_LENGTH, windowing_method='entire_recording')
    data_module.test_dataset = ECGDataset(test_records, DATA_DIR, is_training=False, no_labels=False, 
                                          seq_len=SEQ_LENGTH, windowing_method='entire_recording')
    
    # --- 2. Train the PyTorch Lightning Feature Extractor ---
    print("\nTraining the WaveletConv feature extractor...")
    pl.seed_everything(42)
    feature_extractor_model = ECGFeatureExtractor(
        num_leads=NUM_LEADS,
        out_channels=WAVELET_OUT_CHANNELS,
        kernel_size=WAVELET_KERNEL_SIZE,
        mother_wavelet=mother_wavelet,
        wavenet_layers=WAVENET_LAYERS,
        wavenet_channels=WAVENET_CHANNELS,
        learning_rate=LEARNING_RATE,
        dropout_rate=DROPOUT_RATE
    )
    
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=3, verbose=True, mode='min')
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        callbacks=[early_stop_callback],
        accelerator='auto',
        logger=wandb_logger
    )
    trainer.fit(feature_extractor_model, datamodule=data_module)

    # --- 3. Extract Features for XGBoost ---
    print("\nExtracting features for the XGBoost model...")
    
    # Create non-shuffled dataloaders for prediction
    predict_train_loader = DataLoader(data_module.train_dataset, batch_size=data_module.batch_size, num_workers=min(os.cpu_count(), 10), shuffle=False, collate_fn=collate_fn_skip_none)
    predict_val_loader = DataLoader(data_module.val_dataset, batch_size=data_module.batch_size, num_workers=min(os.cpu_count(), 10), shuffle=False, collate_fn=collate_fn_skip_none)
    predict_test_loader = DataLoader(data_module.test_dataset, batch_size=data_module.batch_size, num_workers=min(os.cpu_count(), 10), shuffle=False, collate_fn=collate_fn_skip_none)

    # Use trainer.predict for optimized inference
    train_outputs = trainer.predict(feature_extractor_model, dataloaders=predict_train_loader)
    val_outputs = trainer.predict(feature_extractor_model, dataloaders=predict_val_loader)
    test_outputs = trainer.predict(feature_extractor_model, dataloaders=predict_test_loader)

    # Concatenate results from batches
    X_train_features = np.concatenate([batch[0].cpu().numpy() for batch in train_outputs])
    y_train = np.concatenate([batch[1].cpu().numpy() for batch in train_outputs])

    X_val_features = np.concatenate([batch[0].cpu().numpy() for batch in val_outputs])
    y_val = np.concatenate([batch[1].cpu().numpy() for batch in val_outputs])

    X_test_features = np.concatenate([batch[0].cpu().numpy() for batch in test_outputs])
    y_test = np.concatenate([batch[1].cpu().numpy() for batch in test_outputs])
    
    print(f"Shape of extracted training features: {X_train_features.shape}")

    # --- 4. Train and Evaluate the XGBoost Classifier ---
    print("\nTraining the XGBoost classifier...")
    xgb_classifier = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    xgb_classifier.fit(X_train_features, y_train,
                       eval_set=[(X_val_features, y_val)],
                       verbose=False)

    print("\nEvaluating the XGBoost classifier on the test set...")
    # Get predictions for accuracy
    y_pred_xgb = xgb_classifier.predict(X_test_features)
    accuracy = accuracy_score(y_test, y_pred_xgb)
    
    # Get prediction probabilities for the challenge score
    y_pred_proba_xgb = xgb_classifier.predict_proba(X_test_features)[:, 1]
    challenge_score = utils.compute_challenge_score(y_test, y_pred_proba_xgb)
    
    print(f"\n--- Final Results ---")
    print(f"Accuracy of XGBoost on extracted features: {accuracy * 100:.2f}%")
    print(f"Challenge Score of XGBoost on extracted features: {challenge_score:.4f}")
