import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import scipy.io
from scipy.signal import resample
import os
import math
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy, AUROC
import sys
import neurokit2 as nk
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import wandb
from pytorch_lightning.loggers import WandbLogger
import time
from sklearn.model_selection import train_test_split
# --- Robust Path Handling ---
# Handles running in different environments (e.g., script vs. notebook)
# This allows the script to find your helper_code and utils modules
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

# --- Standard Imports ---
# The script will now directly use these modules.
# Make sure they are available in your environment.
import helper_code
import utils

class ECGMixedDataset(Dataset):
    '''
    Dataset that takes a list of absolute paths to ECG records.
    Modified so that it can handle absolute paths directly, useful for training from multiple years of Physionet data. 
    Note: does NOT extract labels, only returns the signal and wide features.
    '''
    def __init__(self, records_list, is_training=True, seq_len=5000, windowing_method='qrs', img_size=224):
        self.records_list = records_list
        self.is_training = is_training
        self.seq_len = seq_len
        self.windowing_method = windowing_method
        self.img_size = img_size

    def __len__(self):
        return len(self.records_list)

    def __getitem__(self, idx):
        record_path = self.records_list[idx]
        try:
            # --- FIX: Load header and signal separately for robustness ---
            try:
                # Load header to get metadata and for label extraction
                signal, metadata = helper_code.load_signals(record_path)
                header_text = helper_code.load_header(record_path)
                signal = signal.T # make signal (num_leads, num_samples) shape

            except Exception as e:
                # If loading fails for any reason, skip this record
                tqdm.write(f"Error loading record {self.records_list[idx]}: {e}. Skipping.")
                return None


            # --- Gracefully handle short signals ---
            # Get frequency and check if the signal is long enough to be useful
            # Only exclude records if we're not training 
            orig_freq = metadata['fs']
            if signal.shape[1] < orig_freq * utils.MIN_SIGNAL_DURATION and self.is_training:
                tqdm.write(f"Skipping record {self.records_list[idx]} due to insufficient length: {signal.shape[1]} samples.")
                return None # Return None to be filtered out by the custom collate function
            
            # --- ADDED: Get and process demographic data as wide features ---
            # LABEL CAN BE NONE, AS WE EXTRACT FROM PREV YEARS PHYSIONET DATA
            age, sex = helper_code.get_patient_info(header_text, allow_missing_label=True, get_label=False)
    
            # Process and normalize age. Use a neutral value for missing data.
            if age is None or np.isnan(age):
                normalized_age = 0.0 
            else:
                normalized_age = (age - utils.MEAN_AGE_TRAIN) / utils.STD_AGE_TRAIN

            # Process and normalize sex. Use a neutral value for unknown/missing.
            if sex is None or not isinstance(sex, str) or sex.lower() not in ['male', 'female']:
                numerical_sex = 0.5 
            elif sex.lower() == 'male':
                numerical_sex = 0.0
            else: # 'female'
                numerical_sex = 1.0

            # wide_feats is a 2D tensor with age, sex, and computed_features
            # Standardize sampling frequency
            if orig_freq != utils.UNIFIED_FREQUENCY:
                num_samples = int(signal.shape[1] * (utils.UNIFIED_FREQUENCY / orig_freq))
                signal = resample(signal, num_samples, axis=1)

            # Clean signal and correct polarity
            cleaned_leads = [nk.ecg_clean(lead, sampling_rate=utils.UNIFIED_FREQUENCY) for lead in signal]
            signal = np.stack(cleaned_leads)
            # try:
            #     signal, _ = utils.correct_12_lead_polarity_lead_II_ref(signal, utils.UNIFIED_FREQUENCY)
            # except Exception as e:
            #     tqdm.write(f"Error correcting polarity for record {self.records_list[idx]}: {e}")
            created_features = False
            created_wide = False
            try:
                # Continue with the uncorrected signal if polarity check fails
                # computed_features = utils.calculate_ecg_features(signal, utils.UNIFIED_FREQUENCY, utils.REFERENCE_POLARITY_LEAD_IDX)
                # # If cant compute features on this lead, try on other leads
                # if computed_features is None:
                #     for i in range(signal.shape[0]):
                #         if i != utils.REFERENCE_POLARITY_LEAD_IDX:
                #             computed_features = utils.calculate_ecg_features(signal, utils.UNIFIED_FREQUENCY, i)
                #             if computed_features is not None:
                #                 break
                # if computed_features is None:
                #     tqdm.write(f"Skipping record {self.records_list[idx]} because no features could be computed.")
                #     return None

                # # Fill nan values in computed features with 0
                # computed_features = np.nan_to_num(computed_features, nan=0.0, posinf=0.0, neginf=0.0)
                # created_features = True

                wide_feats = np.array([normalized_age, numerical_sex])
                # Append the wide features to the computed features
                # wide_feats = np.concatenate((wide_feats, computed_features), axis=0)
                # created_wide = True

            except Exception as e:
                tqdm.write(f"Error extracting features for record {self.records_list[idx]}: {e}")
                return None # Return None if feature extraction fails

            # Extract windows from the signal
            if utils.USE_ONE_WINDOW:
                windows = utils.get_windows(signal, method=self.windowing_method, window_size=self.seq_len)
                if len(windows) == 0:
                    tqdm.write(f"Skipping record {self.records_list[idx]} because no windows could be extracted.")
                    return None # Return None if windowing fails
                signal = windows[0]
            else:
                raise NotImplementedError("Multiple windows not implemented yet. Set USE_ONE_WINDOW to True for now.")
            
            # Normalize each lead between -1 and 1
            signal = utils.normalize(signal, smooth=1e-8)
            
            # Convert ECG signal to image format for Vision Transformer
            ecg_image = self.ecg_to_image(signal)
    
            # MODIFIED: Return ECG image, wide features, and dummy label
            return ecg_image, torch.FloatTensor(wide_feats), 0
        except Exception as e:
            tqdm.write(f"Error processing record {self.records_list[idx]}: {e}. Skipping.")
            return None
    
    def ecg_to_image(self, signal):
        """Convert 12-lead ECG signal to image format for Vision Transformer.
        
        Args:
            signal: numpy array of shape (12, seq_len)
            
        Returns:
            torch.Tensor: Image tensor of shape (1, img_size, img_size)
        """
        import matplotlib.pyplot as plt
        from PIL import Image
        
        # Create figure with proper size
        fig, axes = plt.subplots(12, 1, figsize=(10, 12), dpi=self.img_size/10)
        fig.patch.set_facecolor('white')
        
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
                     'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        for i, (ax, lead_name) in enumerate(zip(axes, lead_names)):
            if i < signal.shape[0]:  # Make sure we don't exceed available leads
                ax.plot(signal[i], 'k-', linewidth=0.5)
                ax.set_ylabel(lead_name, fontsize=8)
                ax.set_xlim(0, len(signal[i]))
                ax.grid(True, alpha=0.3)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
        
        plt.tight_layout()
        
        # Convert to numpy array
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        # Convert to grayscale
        img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
        
        # Resize to target size
        img = Image.fromarray(img.astype(np.uint8))
        img = img.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        img = np.array(img).astype(np.float32) / 255.0
        
        # Add channel dimension for Vision Transformer (1, H, W)
        img = np.expand_dims(img, axis=0)
        
        return torch.from_numpy(img).float()

def collate_fn_skip_none(batch):
    """
    Custom collate function that filters out `None` values.
    This is used to handle records that were skipped in the Dataset (e.g., too short).
    """
    # Filter out None entries
    batch = [item for item in batch if item is not None]

    # If the entire batch was filtered out (e.g., all records were short),
    # return empty tensors to prevent a crash in the training loop.
    if not batch:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])

    # Separate the components of each item in the batch
    ecg_images = []
    wide_feats = []
    labels = []
    
    for item in batch:
        ecg_img, wide_feat, label = item
        ecg_images.append(ecg_img)
        wide_feats.append(wide_feat)
        labels.append(label)
    
    # Stack the tensors properly
    try:
        ecg_images = torch.stack(ecg_images)
        wide_feats = torch.stack(wide_feats)
        labels = torch.tensor(labels)
        return ecg_images, wide_feats, labels
    except Exception as e:
        print(f"Error in collate function: {e}")
        # Return empty tensors if stacking fails
        return torch.tensor([]), torch.tensor([]), torch.tensor([])

class ECGDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, records_list=None, split_file_path=None, batch_size=32, seq_len=utils.WINDOW_SIZE, windowing_method='qrs', img_size=224):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.windowing_method = windowing_method
        self.img_size = img_size

        # ensure these attrs always exist
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        # This method should be implemented to create the datasets
        # For now, it's a placeholder
        pass
    
    def create_datasets(self, train_records, val_records, test_records=None):
        """Helper method to create datasets with proper parameters."""
        self.train_dataset = ECGMixedDataset(
            train_records, 
            is_training=True, 
            seq_len=self.seq_len, 
            windowing_method=self.windowing_method,
            img_size=self.img_size
        )
        self.val_dataset = ECGMixedDataset(
            val_records, 
            is_training=False, 
            seq_len=self.seq_len, 
            windowing_method=self.windowing_method,
            img_size=self.img_size
        )
        if test_records is not None:
            self.test_dataset = ECGMixedDataset(
                test_records, 
                is_training=False, 
                seq_len=self.seq_len, 
                windowing_method=self.windowing_method,
                img_size=self.img_size
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=min(os.cpu_count(), 10),
                          persistent_workers=True, shuffle=True, pin_memory=True, collate_fn=collate_fn_skip_none)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=min(os.cpu_count(), 10),
                          persistent_workers=True, pin_memory=True, collate_fn=collate_fn_skip_none)

    def test_dataloader(self):
        # fallback to val_dataset if no test_dataset was set
         return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=min(os.cpu_count(), 10),
                          persistent_workers=True, pin_memory=True, collate_fn=collate_fn_skip_none) if self.test_dataset else self.val_dataloader()