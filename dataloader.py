import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
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

import custom_helper_code

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
    def __init__(self, records_list, data_dir, is_training=True, seq_len=5000, windowing_method='qrs', no_labels=False, include_wide_feats=False,
                 multitask_csv_path='code15_exams.csv', do_multitask=False, aug_config=None):
        self.records_list = records_list
        self.data_dir = data_dir
        self.is_training = is_training
        self.seq_len = seq_len
        self.windowing_method = windowing_method
        self.no_labels = no_labels
        self.include_wide_feats = include_wide_feats
        self.multitask_labels = {}
        self.do_multitask = do_multitask
        self.aug_config = aug_config
        
        # Initialize augmentation pipeline
        self.augmenter = None
        if self.is_training and self.aug_config:
            from augmentations import ECGAugmentations  # Import locally
            self.augmenter = ECGAugmentations(config=aug_config)

        if self.do_multitask:
            # Load multitask labels from code15_exams.csv
            try:
                if os.path.isfile(multitask_csv_path):
                    df = pd.read_csv(multitask_csv_path)
                    df['exam_id'] = df['exam_id'].astype(str)
                    # Robustly select columns that exist; primary: RBBB, 1dAVb; secondary: AF, SB; ignore ST by design
                    available_cols = [c for c in ['RBBB', '1dAVb', 'AF', 'SB'] if c in df.columns]
                    if not available_cols:
                        print(f"Warning: No expected columns found in {multitask_csv_path}.")
                        available_cols = []
                    self.multitask_labels = df.set_index('exam_id')[available_cols].to_dict('index')
                    print(f"Successfully loaded {len(self.multitask_labels)} records from {multitask_csv_path} for multi-task learning.")
            except Exception as e:
                print(f"Warning: Could not load or process {multitask_csv_path}: {e}. Multi-task labels will not be available.")

    def __len__(self):
        return len(self.records_list)

    # Convert CSV field to 0/1 robustly (handles 'True'/'False', bools, numbers, NaN)
    def _to01(self, v):
        if pd.isna(v):
            return 0
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (int, np.integer)):
            return 1 if v == 1 else 0
        if isinstance(v, (float, np.floating)):
            return 1 if int(v) == 1 else 0
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("true", "1", "yes", "y", "t"):
                return 1
            if s in ("false", "0", "no", "n", "f", ""):
                return 0
        return 0

    # --- Soft label computation per provided strategy ---
    def _compute_soft_label(self, chagas_label, has_rbbb, has_1davb, has_af, has_sb):
        """
        Returns a soft label in [0, 1] using:
          - Base score from primary indicators (RBBB, 1dAVb) and original Chagas label
          - +0.05 per secondary indicator (AF, SB)
          - Cap at 0.98 only when original label is positive
        """
        # Base score
        if chagas_label == 1:
            if has_rbbb and has_1davb:
                base = 0.95
            elif has_rbbb or has_1davb:
                base = 0.85
            else:
                base = 0.60
            adj = 0.05 * (int(has_af) + int(has_sb))
            return min(base + adj, 0.98)
        else:
            if not has_rbbb and not has_1davb:
                base = 0.05
            elif has_rbbb and has_1davb:
                base = 0.40
            else:
                base = 0.15
            adj = 0.05 * (int(has_af) + int(has_sb))
            return base + adj

    def __getitem__(self, idx):
        record_path = self.records_list[idx] # This is now an absolute path
        try:
            # --- FIX: Load header and signal separately for robustness ---
            try:
                # Load header to get metadata and for label extraction
                signal, metadata = helper_code.load_signals(record_path)
                header_text = helper_code.load_header(record_path)

                # Enforce the backbone's lead order, before the transpose. Must stay
                # identical to utils.preprocess_signal (the inference path).
                sig_names = metadata.get('sig_name') if hasattr(metadata, 'get') else None
                if sig_names:
                    signal = custom_helper_code.reorder_signal(
                        signal, list(sig_names), list(utils.ECG_FM_LEAD_ORDER)
                    )

                signal = signal.T # make signal (num_leads, num_samples) shape

                if signal.shape[0] != len(utils.ECG_FM_LEAD_ORDER):
                    tqdm.write(f"Skipping record {record_path}: got {signal.shape[0]} leads, "
                               f"expected {len(utils.ECG_FM_LEAD_ORDER)}.")
                    return None

            except Exception as e:
                # If loading fails for any reason, skip this record
                tqdm.write(f"Error loading record {record_path}: {e}. Skipping.")
                return None


            # --- Gracefully handle short signals ---
            # Get frequency and check if the signal is long enough to be useful
            # Only exclude records if we're not training 
            orig_freq = metadata['fs']
            # if signal.shape[1] < orig_freq * utils.MIN_SIGNAL_DURATION and self.is_training:
            #     tqdm.write(f"Skipping record {record_path} due to insufficient length: {signal.shape[1]} samples.")
            #     return None # Return None to be filtered out by the custom collate function
            
            # --- ADDED: Get and process demographic data as wide features ---
            age, sex, label = None, None, None
            if self.no_labels:
                age, sex = custom_helper_code.get_patient_info(header_text, get_label=False)
                label = -1 # Use -1 to indicate no label
            else:
                age, sex, label = custom_helper_code.get_patient_info(header_text, get_label=True)
                assert label is not None and label == 0 or label == 1, f"Invalid label {label} for record {record_path}"
    
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

            # Get multitask labels if enabled
            if self.do_multitask:
                exam_id = os.path.splitext(os.path.basename(record_path))[0]
                multitask_info = self.multitask_labels.get(exam_id)
                if multitask_info:
                    # CSV contains 'True'/'False'; convert to 0/1
                    rbbb_label = float(self._to01(multitask_info.get('RBBB', 'False')))
                    d1avb_label = float(self._to01(multitask_info.get('1dAVb', 'False')))
                    af_label   = float(self._to01(multitask_info.get('AF', 'False')))
                    sb_label   = float(self._to01(multitask_info.get('SB', 'False')))
                else:
                    rbbb_label = -1.0  # Sentinel for missing exam_id
                    d1avb_label = -1.0
                    af_label = 0.0
                    sb_label = 0.0

            # wide_feats is a 2D tensor with age, sex, and computed_features
            # Standardize sampling frequency
            if orig_freq != utils.UNIFIED_FREQUENCY:
                num_samples = int(signal.shape[1] * (utils.UNIFIED_FREQUENCY / orig_freq))
                signal = resample(signal, num_samples, axis=1)

            # Clean signal and correct polarity
            cleaned_leads = [nk.ecg_clean(lead, sampling_rate=utils.UNIFIED_FREQUENCY) for lead in signal]
            signal = np.stack(cleaned_leads)  # shape (num_leads, num_samples)

            # Convert to torch tensor for augmentation
            signal_tensor = torch.FloatTensor(signal.copy())
            
            # Apply augmentation after cleaning but before windowing
            if self.augmenter:
                signal_tensor = self.augmenter(signal_tensor)
                # Convert back to numpy for further processing
                signal = signal_tensor.numpy()

            # Per-lead z-score within this record, before windowing. Must stay identical
            # to utils.preprocess_signal (the inference path) -- see utils.zscore_per_record.
            signal = utils.zscore_per_record(signal)

            # try:
            #     signal, _ = utils.correct_12_lead_polarity_lead_II_ref(signal, utils.UNIFIED_FREQUENCY)
            # except Exception as e:
            #     tqdm.write(f"Error correcting polarity for record {record_path}: {e}")
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
                #     tqdm.write(f"Skipping record {record_path} because no features could be computed.")
                #     return None

                # # Fill nan values in computed features with 0
                # computed_features = np.nan_to_num(computed_features, nan=0.0, posinf=0.0, neginf=0.0)
                # created_features = True

                wide_feats = np.array([normalized_age, numerical_sex])
                # Append the wide features to the computed features
                # wide_feats = np.concatenate((wide_feats, computed_features), axis=0)
                # created_wide = True

            except Exception as e:
                tqdm.write(f"Error extracting features for record {record_path}: {e}")
                return None # Return None if feature extraction fails

            # Extract windows (skip if augmenter already handled length)
            if utils.USE_ONE_WINDOW and not self.augmenter:
                windows = utils.get_windows(signal, method=self.windowing_method, window_size=self.seq_len)
                if len(windows) == 0:
                    tqdm.write(f"Skipping record {record_path} because no windows could be extracted.")
                    return None # Return None if windowing fails
                signal = windows[0]
            elif not utils.USE_ONE_WINDOW:
                raise NotImplementedError("Multiple windows not implemented yet. Set USE_ONE_WINDOW to True for now.")
            # If augmenter was used, signal already has correct length (5000 samples)

            # (normalization already applied above, before windowing)

            # --- Soft labeling: replace hard label with soft label when possible ---
            if not self.no_labels and self.do_multitask and ('multitask_info' in locals()) and (multitask_info is not None):
                has_rbbb = bool(rbbb_label == 1.0)
                has_1davb = bool(d1avb_label == 1.0)
                has_af = bool(af_label == 1.0)
                has_sb = bool(sb_label == 1.0)
                soft_label = float(self._compute_soft_label(int(label), has_rbbb, has_1davb, has_af, has_sb))
            else:
                soft_label = float(label)

            # Return tuple based on configuration (use soft_label)
            if self.include_wide_feats:
                base_tuple = (torch.FloatTensor(signal.copy()), torch.FloatTensor(wide_feats), torch.FloatTensor([soft_label]))
            else:
                base_tuple = (torch.FloatTensor(signal.copy()), torch.FloatTensor([soft_label]))

            if self.do_multitask:
                return base_tuple + (torch.FloatTensor([rbbb_label]), torch.FloatTensor([d1avb_label]))
            else:
                return base_tuple
        except Exception as e:
            tqdm.write(f"Error processing record {record_path}: {e}. Skipping.")
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
        # Cannot determine structure, so return None and let the trainer skip it.
        return None

    # Use the default collate function on the filtered, valid batch
    return torch.utils.data.default_collate(batch)


class ECGDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, seq_len=utils.WINDOW_SIZE, windowing_method='entire_recording', aug_config=None,
                 sample_weights=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.windowing_method = windowing_method
        self.aug_config = aug_config
        # Per-sample weights aligned with train_dataset.records_list. When set,
        # train_dataloader draws with a WeightedRandomSampler instead of shuffling.
        self.sample_weights = sample_weights

        # ensure these attrs always exist
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
    def setup(self, stage=None):
        pass


    def train_dataloader(self):
        # Records arrive grouped by source dataset, so the order MUST be broken up --
        # otherwise training sees a long run of all-negative records followed by a run
        # of positives, identically every epoch. Use weighted sampling when weights were
        # supplied (the positive rate is ~2%, so unweighted batches of 16 are usually
        # all-negative), otherwise plain shuffling. sampler and shuffle are exclusive.
        sampler = None
        shuffle = True
        if self.sample_weights is not None:
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(self.sample_weights, dtype=torch.double),
                num_samples=len(self.train_dataset),
                replacement=True,
            )
            shuffle = False
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=min(os.cpu_count(), 10),
                          sampler=sampler, shuffle=shuffle,
                          persistent_workers=True, pin_memory=True, collate_fn=collate_fn_skip_none)

    def val_dataloader(self):
        # Returning None keeps Lightning from building DataLoader(None) on paths that
        # train without a validation split.
        if self.val_dataset is None:
            return None
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=min(os.cpu_count(), 10),
                          persistent_workers=True, pin_memory=True, collate_fn=collate_fn_skip_none)

    def test_dataloader(self):
        # fallback to val_dataset if no test_dataset was set
        dataset = self.test_dataset if self.test_dataset is not None else self.val_dataset
        if dataset is None:
            return None
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=min(os.cpu_count(), 10),
                          persistent_workers=True, pin_memory=True, collate_fn=collate_fn_skip_none)