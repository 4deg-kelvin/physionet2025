from torch.utils.data import Dataset, DataLoader
import os 
import sys
import neurokit2 as nk 
from tqdm import tqdm 
import numpy as np
from scipy.signal import resample
import torch
import pytorch_lightning as pl
import pandas as pd
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
    def __init__(self, records_list, data_dir, is_training=True, seq_len=5000, windowing_method='qrs'):
        self.records_list = records_list
        self.data_dir = data_dir
        self.is_training = is_training
        self.seq_len = seq_len
        self.windowing_method = windowing_method

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
            orig_freq = metadata['fs']
            if signal.shape[1] < orig_freq * utils.MIN_SIGNAL_DURATION:
                tqdm.write(f"Skipping record {self.records_list[idx]} due to insufficient length: {signal.shape[1]} samples.")
                return None # Return None to be filtered out by the custom collate function


            # Standardize sampling frequency
            if orig_freq != utils.UNIFIED_FREQUENCY:
                num_samples = int(signal.shape[1] * (utils.UNIFIED_FREQUENCY / orig_freq))
                signal = resample(signal, num_samples, axis=1)

            # Clean signal and correct polarity
            cleaned_leads = [nk.ecg_clean(lead, sampling_rate=utils.UNIFIED_FREQUENCY) for lead in signal]
            signal = np.stack(cleaned_leads)
            try:
                signal, _ = utils.correct_12_lead_polarity_lead_II_ref(signal, utils.UNIFIED_FREQUENCY)
            except Exception as e:
                tqdm.write(f"Error correcting polarity for record {self.records_list[idx]}: {e}")
                # Continue with the uncorrected signal if polarity check fails

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
            
            # Get the label from the loaded header
            label = helper_code.get_label(header_text)

            assert label == 0 or label == 1, f"Invalid label {label} for record {self.records_list[idx]}"

            # Use .copy() to prevent potential negative stride errors from numpy operations
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
        return torch.tensor([]), torch.tensor([])

    # Use the default collate function on the filtered, valid batch
    return torch.utils.data.default_collate(batch)

class ECGDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, records_list, split_file_path, batch_size=32, seq_len=utils.WINDOW_SIZE, windowing_method='qrs'):
        super().__init__()
        self.data_dir = data_dir
        self.records_list = records_list
        self.split_file_path = split_file_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.windowing_method = windowing_method

    def setup(self, stage=None):
        split_df = pd.read_csv(self.split_file_path)
        basename_to_path = {os.path.splitext(os.path.basename(p))[0]: p for p in self.records_list}
        train_ids = split_df[split_df['split'] == 'train']['exam_id'].tolist()
        val_ids = split_df[split_df['split'] == 'val']['exam_id'].tolist()
        test_ids = split_df[split_df['split'] == 'test']['exam_id'].tolist()
        train_files = [basename_to_path[id] for id in train_ids if id in basename_to_path]
        val_files = [basename_to_path[id] for id in val_ids if id in basename_to_path]
        test_files = [basename_to_path[id] for id in test_ids if id in basename_to_path]
        
        self.train_dataset = ECGDataset(train_files, self.data_dir, is_training=True, seq_len=self.seq_len, windowing_method=self.windowing_method)
        self.val_dataset = ECGDataset(val_files, self.data_dir, is_training=False, seq_len=self.seq_len, windowing_method=self.windowing_method)
        self.test_dataset = ECGDataset(test_files, self.data_dir, is_training=False, seq_len=self.seq_len, windowing_method=self.windowing_method)
        
        print(f"Data setup complete. Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
        total_records = len(train_files) + len(val_files) + len(test_files)
        print(f"Total records processed: {total_records}, %age in a split: {total_records / len(self.records_list) * 100:.2f}%")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=min(os.cpu_count(), 10), persistent_workers=True, shuffle=True, pin_memory=True, collate_fn=collate_fn_skip_none)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=min(os.cpu_count(), 10), persistent_workers=True, pin_memory=True, collate_fn=collate_fn_skip_none)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=min(os.cpu_count(), 10), persistent_workers=True, pin_memory=True, collate_fn=collate_fn_skip_none)

# --- 3. PyTorch Lightning Module ---

