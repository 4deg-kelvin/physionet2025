"""
ECG Data Utilities for MAE-ViT ECG modeling
- Loading, normalization, patching
"""
import numpy as np
import torch

def load_ecg_from_npy(file_path):
    """Load ECG data from a numpy file. Expected shape: (channels, seq_len)"""
    arr = np.load(file_path)
    return arr

def normalize_ecg(ecg, mean=None, std=None):
    """Normalize ECG data to zero mean, unit variance."""
    if mean is None:
        mean = np.mean(ecg)
    if std is None:
        std = np.std(ecg)
    return (ecg - mean) / (std + 1e-8)

def patch_ecg(ecg, patch_size=250):
    """Split ECG into non-overlapping patches. Input shape: (channels, seq_len) Output: (num_patches, channels, patch_size)"""
    channels, seq_len = ecg.shape
    num_patches = seq_len // patch_size
    ecg = ecg[:, :num_patches * patch_size]
    patches = ecg.reshape(channels, num_patches, patch_size)
    patches = np.transpose(patches, (1, 0, 2))  # (num_patches, channels, patch_size)
    return patches

def batch_ecg_patches(ecg_list, patch_size=250):
    """Batch multiple ECGs into torch tensors for model input."""
    batch = []
    for ecg in ecg_list:
        norm = normalize_ecg(ecg)
        patches = patch_ecg(norm, patch_size)
        batch.append(patches)
    batch = np.stack(batch)  # (batch, num_patches, channels, patch_size)
    batch = torch.tensor(batch, dtype=torch.float32)
    # Rearrange to (batch, channels, seq_len) for PatchEmbed1D
    batch = batch.permute(0, 2, 1, 3).reshape(batch.shape[0], batch.shape[2], -1)
    return batch
