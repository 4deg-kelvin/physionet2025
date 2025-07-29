
"""
Pretraining script for MAE-based ViT ECG model
Supports both WFDB (.hea/.mat) and .npy ECG files
"""
from typing import Tuple, List, Optional, Union
import torch
import torch.optim as optim
import wandb
from mae_vit_ecg import MAEViTECG
from ecg_data_utils import batch_ecg_patches, normalize_ecg, load_ecg_from_npy
from helper_code import find_records, load_signals
import os

def get_ecg_file_list(data_folder: str) -> Tuple[List[str], Optional[str]]:
    # Prefer WFDB files if present
    wfdb_records = find_records(data_folder, file_extension='.hea')
    if wfdb_records:
        return wfdb_records, 'wfdb'
    # Otherwise, look for .npy files
    npy_files = [f for f in os.listdir(data_folder) if f.endswith('.npy')]
    if npy_files:
        return npy_files, 'npy'
    return [], None

def evaluate_mae_loss(model: MAEViTECG, file_list: List[str], file_type: str, data_folder: str, batch_size: int, device: str) -> float:
    """Helper function to evaluate MAE loss on a set of files"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for i in range(0, len(file_list), batch_size):
            batch_files = file_list[i:i+batch_size]
            ecg_list = []
            
            if file_type == 'wfdb':
                for record in batch_files:
                    record_path = os.path.join(data_folder, record)
                    try:
                        signal, fields = load_signals(record_path)
                        ecg = signal.T
                        ecg_normalized = normalize_ecg(ecg)
                        ecg_list.append(ecg_normalized)
                    except Exception as e:
                        continue
            elif file_type == 'npy':
                for f in batch_files:
                    try:
                        ecg = load_ecg_from_npy(os.path.join(data_folder, f))
                        ecg_normalized = normalize_ecg(ecg)
                        ecg_list.append(ecg_normalized)
                    except Exception as e:
                        continue
            
            if not ecg_list:
                continue
            
            # Same preprocessing as training
            max_len = max(ecg.shape[1] for ecg in ecg_list)
            patch_size = model.patch_embed.patch_size
            target_len = ((max_len + patch_size - 1) // patch_size) * patch_size
            ecg_list_padded = []
            for ecg in ecg_list:
                channels, seq_len = ecg.shape
                if seq_len < target_len:
                    pad_width = target_len - seq_len
                    ecg = torch.tensor(ecg, dtype=torch.float32)
                    ecg = torch.nn.functional.pad(ecg, (0, pad_width), mode='constant', value=0)
                elif seq_len > target_len:
                    ecg = torch.tensor(ecg[:, :target_len], dtype=torch.float32)
                else:
                    ecg = torch.tensor(ecg, dtype=torch.float32)
                ecg_list_padded.append(ecg)
            
            batch = torch.stack(ecg_list_padded).to(device)
            recon, mask = model(batch)
            target = batch
            
            # Same loss calculation as training
            patch_size = model.patch_embed.patch_size
            channels = model.patch_embed.proj.in_channels
            num_patches = recon.shape[2] // patch_size
            recon_patches = recon.view(recon.shape[0], channels, num_patches, patch_size)
            recon_patches = recon_patches.permute(0, 2, 3, 1)
            target_patches = target.view(target.shape[0], channels, num_patches, patch_size)
            target_patches = target_patches.permute(0, 2, 3, 1)
            
            losses = []
            for b in range(recon_patches.shape[0]):
                masked_idx = mask[b]
                if masked_idx.any():
                    diff = recon_patches[b][masked_idx] - target_patches[b][masked_idx]
                    losses.append(diff.pow(2).mean())
            if losses:
                loss = torch.stack(losses).mean()
                total_loss += loss.item()
                num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else float('inf')

def train_mae(model: MAEViTECG, data_folder: str, epochs: int = 10, batch_size: int = 8, lr: float = 1e-4, device: str = 'cpu', use_wandb: bool = True) -> None:
    file_list, file_type = get_ecg_file_list(data_folder)
    if not file_list:
        print(f"No ECG files found in {data_folder}")
        return

    # Split data into train/validation sets (80/20 split)
    import random
    random.seed(42)
    random.shuffle(file_list)
    split_idx = int(0.8 * len(file_list))
    train_files = file_list[:split_idx]
    val_files = file_list[split_idx:]
    
    print(f"Found {len(file_list)} {file_type} files for training")
    print(f"Train files: {len(train_files)}, Validation files: {len(val_files)}")
    
    if use_wandb:
        try:
            wandb.init(
                project="mae-vit-ecg-pretraining",
                config={
                    "learning_rate": lr,
                    "architecture": "ViT-Base",
                    "dataset": f"{file_type}-ECG",
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "mask_ratio": model.mask_ratio,
                    "num_records": len(file_list),
                    "train_records": len(train_files),
                    "val_records": len(val_files)
                }
            )
        except Exception as e:
            print(f"Warning: wandb initialization failed: {e}")
            use_wandb = False

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Track global step for step-wise logging
    global_step = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        print(f"\nStarting epoch {epoch+1}/{epochs}")

        for i in range(0, len(train_files), batch_size):
            batch_files = train_files[i:i+batch_size]
            ecg_list = []
            
            if i % (batch_size * 20) == 0:  # Progress update every 20 batches
                print(f"Processing batch {i//batch_size + 1}/{(len(train_files) + batch_size - 1)//batch_size}")

            if file_type == 'wfdb':
                # Load ECG data from WFDB files
                for record in batch_files:
                    record_path = os.path.join(data_folder, record)
                    try:
                        signal, fields = load_signals(record_path)
                        ecg = signal.T  # (channels, seq_len)
                        ecg_normalized = normalize_ecg(ecg)
                        ecg_list.append(ecg_normalized)
                    except Exception as e:
                        print(f"Error loading {record_path}: {e}")
                        continue
            elif file_type == 'npy':
                # Load ECG data from .npy files
                for f in batch_files:
                    try:
                        ecg = load_ecg_from_npy(os.path.join(data_folder, f))
                        ecg_normalized = normalize_ecg(ecg)
                        ecg_list.append(ecg_normalized)
                    except Exception as e:
                        print(f"Error loading {f}: {e}")
                        continue

            if not ecg_list:
                continue

            # Handle variable length ECGs by padding to next multiple of patch_size
            max_len = max(ecg.shape[1] for ecg in ecg_list)
            patch_size = model.patch_embed.patch_size
            target_len = ((max_len + patch_size - 1) // patch_size) * patch_size
            ecg_list_padded = []
            for ecg in ecg_list:
                channels, seq_len = ecg.shape
                if seq_len < target_len:
                    pad_width = target_len - seq_len
                    ecg = torch.tensor(ecg, dtype=torch.float32)
                    ecg = torch.nn.functional.pad(ecg, (0, pad_width), mode='constant', value=0)
                elif seq_len > target_len:
                    ecg = torch.tensor(ecg[:, :target_len], dtype=torch.float32)
                else:
                    ecg = torch.tensor(ecg, dtype=torch.float32)
                ecg_list_padded.append(ecg)

            batch = torch.stack(ecg_list_padded).to(device)

            recon, mask = model(batch)
            target = batch
            # Reshape recon and target to (batch, num_patches, patch_size, channels)
            patch_size = model.patch_embed.patch_size
            channels = model.patch_embed.proj.in_channels
            num_patches = recon.shape[2] // patch_size
            recon_patches = recon.view(recon.shape[0], channels, num_patches, patch_size)
            recon_patches = recon_patches.permute(0, 2, 3, 1)  # (batch, num_patches, patch_size, channels)
            target_patches = target.view(target.shape[0], channels, num_patches, patch_size)
            target_patches = target_patches.permute(0, 2, 3, 1)  # (batch, num_patches, patch_size, channels)
            # Mask shape: (batch, num_patches)
            # Compute loss only on masked patches using boolean indexing
            losses = []
            for b in range(recon_patches.shape[0]):
                masked_idx = mask[b]
                if masked_idx.any():
                    diff = recon_patches[b][masked_idx] - target_patches[b][masked_idx]
                    losses.append(diff.pow(2).mean())
            if losses:
                loss = torch.stack(losses).mean()
            else:
                loss = torch.tensor(0.0, device=device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            # Log train_loss_step and learning_rate for each step
            if use_wandb:
                try:
                    wandb.log({
                        "train_loss_step": loss.item(),
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        "global_step": global_step
                    })
                except Exception as e:
                    print(f"Warning: wandb step logging failed: {e}")

        if num_batches > 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}")
            
            # Evaluate validation loss
            val_loss = evaluate_mae_loss(model, val_files, file_type, data_folder, batch_size, device)
            print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}")
            
            if use_wandb:
                try:
                    wandb.log({
                        "epoch": epoch + 1, 
                        "train_loss_epoch": avg_loss,
                        "val_loss": val_loss
                    })
                except Exception as e:
                    print(f"Warning: wandb epoch logging failed: {e}")
        else:
            print(f"Epoch {epoch+1}/{epochs}, No valid batches processed")

    # Save the pretrained model weights
    print("Saving pretrained model weights...")
    torch.save(model.state_dict(), 'mae_vit_ecg_pretrained.pth')
    print("Pretrained model saved as 'mae_vit_ecg_pretrained.pth'")
    
    # Add test loss evaluation
    if use_wandb and num_batches > 0:
        print("Evaluating final test loss on validation set...")
        test_loss = evaluate_mae_loss(model, val_files, file_type, data_folder, batch_size, device)
        print(f"Final Test Loss: {test_loss:.4f}")
        try:
            wandb.log({"test_loss": test_loss})
        except Exception as e:
            print(f"Warning: wandb test loss logging failed: {e}")
    
    if use_wandb:
        try:
            wandb.finish()
        except Exception as e:
            print(f"Warning: wandb finish failed: {e}")

if __name__ == "__main__":
    # Example usage: point to a folder containing WFDB files (.hea/.mat) or .npy files
    data_folder = "training_data/samitrop/samitrop_unzipped"
    
    # Check if MPS is available for faster training on Mac
    if torch.backends.mps.is_available():
        device = 'mps'
        print("Using MPS (Mac GPU) for training")
    else:
        device = 'cpu'
        print("Using CPU for training")
    
    model = MAEViTECG()
    
    # Enable wandb and increase epochs for better chart visualization
    train_mae(model, data_folder, epochs=5, batch_size=4, device=device, use_wandb=True)