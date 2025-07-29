import numpy as np
import torch
from ecg_data_utils import load_ecg_from_npy, normalize_ecg, patch_ecg, batch_ecg_patches
from helper_code import load_signals, get_header_file
import os


# To test a single record, set record_name below:
record_name = "training_data/samitrop/samitrop_unzipped/4991"
signal, fields = load_signals(record_name)
ecgs = signal.T  # (channels, seq_len)
print("Loaded ECG shape (from WFDB):", ecgs.shape)
norm_ecg = normalize_ecg(ecgs)
print("Normalized ECG mean:", np.mean(norm_ecg), "std:", np.std(norm_ecg))
patches = patch_ecg(norm_ecg, patch_size=250)
print("Patches shape:", patches.shape)
batch = batch_ecg_patches([ecgs, ecgs], patch_size=250)
print("Batch shape:", batch.shape)
print("Batch tensor type:", type(batch))

# To batch all records in a folder, use find_records from helper_code.py:
from helper_code import find_records
data_folder = "training_data/samitrop/samitrop_unzipped"
record_list = find_records(data_folder, file_extension='.hea')
ecg_list = []

# Padding utility
def pad_ecg(ecg, target_len):
    channels, seq_len = ecg.shape
    if seq_len < target_len:
        pad_width = target_len - seq_len
        ecg = np.pad(ecg, ((0, 0), (0, pad_width)), mode='constant')
    elif seq_len > target_len:
        ecg = ecg[:, :target_len]
    return ecg

for record in record_list:
    record_path = os.path.join(data_folder, record)
    try:
        signal, fields = load_signals(record_path)
        ecgs = signal.T  # (channels, seq_len)
        ecg_list.append(ecgs)
    except Exception as e:
        print(f"Error loading {record_path}: {e}")
if ecg_list:
    # Find the maximum length
    max_len = max(ecg.shape[1] for ecg in ecg_list)
    ecg_list_padded = [pad_ecg(ecg, max_len) for ecg in ecg_list]
    batch_all = batch_ecg_patches(ecg_list_padded, patch_size=250)
    print(f"Batch of all records shape: {batch_all.shape}")
    print(f"Batch tensor type: {type(batch_all)}")
else:
    print("No valid ECGs found in folder.")
