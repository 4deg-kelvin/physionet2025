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


# --- Model Architecture ---

# 1. PyTorch Lightning Module for Feature Extraction
class ECGFeatureExtractor(pl.LightningModule):
    """
    PyTorch Lightning module that uses a WaveletConv layer to extract features from ECG signals.
    The extracted features are then used to pre-train a simple classifier.
    """
    def __init__(self, in_channels=1, out_channels=32,
                 kernel_size=129, mother_wavelet=None, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters('in_channels', 'out_channels', 'kernel_size', 'learning_rate')
        
        # The main feature extractor is now just the WaveletConv layer
        self.wavelet_conv = WaveletConv(in_channels, out_channels, kernel_size, mother_wavelet)
        
        # Projector to create a fixed-size feature vector for XGBoost
        self.feature_projector = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(out_channels, 128), # Input to Linear is now out_channels from WaveletConv
            nn.ReLU()
        )
        
        # Classifier for the pre-training task (to learn the wavelet scales)
        self.classifier = nn.Linear(128, 1)

    def forward(self, x):
        # Pass through WaveletConv and ReLU activation
        wavelet_features = F.relu(self.wavelet_conv(x))
        # Project features
        projected_features = self.feature_projector(wavelet_features)
        # Classify for pre-training loss
        output = self.classifier(projected_features)
        return output

    def extract_features(self, x):
        """Extracts features to be used by the XGBoost model."""
        self.eval() # Ensure model is in evaluation mode
        with torch.no_grad():
            wavelet_features = F.relu(self.wavelet_conv(x))
            projected_features = self.feature_projector(wavelet_features)
        return projected_features

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat.squeeze(), y.float())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat.squeeze(), y.float())
        self.log('val_loss', loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat.squeeze(), y.float())
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

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
    NUM_LEADS = 12
    DATA_DIR = "../training_data" # Adjust if needed

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
        mother_wavelet=mother_wavelet
    )
    
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=3, verbose=True, mode='min')
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        callbacks=[early_stop_callback],
        accelerator='auto'
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
