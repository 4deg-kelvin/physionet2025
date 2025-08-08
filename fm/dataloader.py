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

class ECGDataset(Dataset):
    def __init__(self, records_list, data_dir, is_training=True, seq_len=5000, windowing_method='qrs', no_labels=False, include_wide_feats=False):
        self.records_list = records_list
        self.data_dir = data_dir
        self.is_training = is_training
        self.seq_len = seq_len
        self.windowing_method = windowing_method
        self.no_labels = no_labels
        self.include_wide_feats = include_wide_feats

    def __len__(self):
        return len(self.records_list)

    def __getitem__(self, idx):
        record_path = os.path.join(self.data_dir, self.records_list[idx])
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
            # if signal.shape[1] < orig_freq * utils.MIN_SIGNAL_DURATION and self.is_training:
            #     tqdm.write(f"Skipping record {self.records_list[idx]} due to insufficient length: {signal.shape[1]} samples.")
            #     return None # Return None to be filtered out by the custom collate function
            
            # --- ADDED: Get and process demographic data as wide features ---
            age, sex, label = None, None, None
            if self.no_labels:
                age, sex = helper_code.get_patient_info(header_text, allow_missing_label=True, get_label=False)
                label = -1 # Use -1 to indicate no label
            else:
                age, sex, label = helper_code.get_patient_info(header_text, allow_missing_label=False)
                assert label is not None and label == 0 or label == 1, f"Invalid label {label} for record {self.records_list[idx]}"
    
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
    
            # MODIFIED: Return a 3-tuple including the wide_feats tensor
            if self.include_wide_feats:
                return torch.FloatTensor(signal.copy()), torch.FloatTensor(wide_feats), torch.FloatTensor([label])
            else:
                return torch.FloatTensor(signal.copy()), torch.FloatTensor([label])
        except Exception as e:
            tqdm.write(f"Error processing record {self.records_list[idx]}: {e}. Skipping.")
            return None

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
        # Cannot determine structure, so return a structure that fine-tuning expects to avoid breaking the loop
        return torch.tensor([]), torch.tensor([]), torch.tensor([])

    # Use the default collate function on the filtered, valid batch
    return torch.utils.data.default_collate(batch)



class ECGDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, records_list=None, split_file_path=None, batch_size=32, seq_len=utils.WINDOW_SIZE, windowing_method='qrs'):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.windowing_method = windowing_method

        # ensure these attrs always exist
        full_dataset = ECGDataset(records_list, self.data_dir, is_training=True, seq_len=self.seq_len, windowing_method=self.windowing_method)
        # Split dataset
        train_size = int(0.85 * len(full_dataset))
        val_size = len(full_dataset) - train_size #int(0.15 * len(full_dataset))
        test_size = 0 #len(full_dataset) - train_size - val_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        # self.train_dataset = None
        # self.val_dataset = None
        # self.test_dataset = None

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=min(os.cpu_count(), 10),
                          persistent_workers=True, shuffle=True, pin_memory=True, collate_fn=collate_fn_skip_none)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=min(os.cpu_count(), 10),
                          persistent_workers=True, pin_memory=True, collate_fn=collate_fn_skip_none)

    def test_dataloader(self):
        # fallback to val_dataset if no test_dataset was set
        dataset = self.test_dataset if self.test_dataset is not None else self.val_dataset
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=min(os.cpu_count(), 10),
                          persistent_workers=True, pin_memory=True, collate_fn=collate_fn_skip_none)