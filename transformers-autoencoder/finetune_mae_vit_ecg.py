"""
Fine-tuning script for MAE-based ViT ECG model (classification)
"""
import torch
import torch.optim as optim
import wandb
import torch.nn as nn
from mae_vit_ecg import MAEViTECG
from ecg_data_utils import batch_ecg_patches, load_ecg_from_npy, normalize_ecg
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from helper_code import find_records, load_signals, load_label, compute_challenge_score

def finetune_mae(model, train_files, train_labels, train_folders, val_files, val_labels, val_folders, test_files, test_labels, test_folders, file_type, epochs=10, batch_size=8, lr=1e-4, weight_decay: float = 1e-2, warmup_epochs: int = 1, min_lr: float = 1e-6, device='cuda', patience=3):
    # Add classifier head
    # 1. Start a new run
    wandb.init(
        project="mae-vit-ecg-finetuning",
        config={
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "min_lr": min_lr,
            "architecture": "ViT-Base with Classifier",
            "dataset": "SaMi-Trop + PTB-XL",
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
        }
    )

    embed_dim = model.patch_embed.embed_dim
    model.classifier = nn.Sequential(
        nn.Linear(embed_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    
    # Initialize classifier weights properly to prevent extreme loss values
    def init_classifier_weights(m):
        if isinstance(m, nn.Linear):
            # Xavier/Glorot initialization for better convergence
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    model.classifier.apply(init_classifier_weights)
    print("✅ Classifier head initialized with Xavier weights")
    # Automatically select device (prioritize MPS for Mac, then CUDA, then CPU)
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Using device: {device}")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    
    # Initialize learning rate scheduler with cosine annealing after warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs - warmup_epochs, 
        eta_min=min_lr
    )
    
    # Early stopping variables
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"Training with early stopping (patience={patience})")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        print(f"\nStarting epoch {epoch+1}/{epochs}")
        
        # Implement linear warmup for the first warmup_epochs
        if epoch < warmup_epochs:
            warmup_lr = lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            print(f"Warmup epoch {epoch+1}/{warmup_epochs}, LR: {warmup_lr:.6f}")
        elif epoch == warmup_epochs:
            # Reset to base learning rate for cosine annealing
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print(f"Starting cosine annealing from epoch {epoch+1}, LR: {lr:.6f}")
        
        # --- Training loop ---
        for i in range(0, len(train_files), batch_size):
            batch_files = train_files[i:i+batch_size]
            batch_folders = train_folders[i:i+batch_size]
            batch_labels = torch.tensor(train_labels[i:i+batch_size], dtype=torch.float32).to(device)
            ecg_list = []
            if (i // batch_size) % 10 == 0:
                print(f"Training batch {i//batch_size + 1}/{(len(train_files) + batch_size - 1)//batch_size}")
            if file_type == 'wfdb':
                for record, folder in zip(batch_files, batch_folders):
                    record_path = os.path.join(folder, record)
                    try:
                        signal, fields = load_signals(record_path)
                        ecg = signal.T
                        ecg_normalized = normalize_ecg(ecg)
                        ecg_list.append(ecg_normalized)
                    except Exception as e:
                        print(f"Error loading {record_path}: {e}")
                        continue
            elif file_type == 'npy':
                for f in batch_files:
                    ecg = load_ecg_from_npy(f)
                    ecg_normalized = normalize_ecg(ecg)
                    ecg_list.append(ecg_normalized)
            if not ecg_list:
                continue
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
            patches = model.patch_embed(batch)
            mask = model.random_masking(patches)
            masked_patches = patches.clone()
            masked_patches[mask] = 0
            x = masked_patches
            for block in model.transformer_encoder_blocks:
                x = block(x)
            encoded = x
            recon = model.decoder(encoded)
            batch_size_, num_patches, _ = encoded.shape
            recon = recon.view(batch_size_, num_patches, model.patch_embed.patch_size, model.patch_embed.proj.in_channels)
            recon = recon.permute(0, 3, 1, 2).contiguous()
            recon = recon.view(batch_size_, model.patch_embed.proj.in_channels, num_patches * model.patch_embed.patch_size)
            cls_features = encoded.mean(dim=1)
            cls_features = torch.nn.functional.normalize(cls_features, p=2, dim=1)
            preds = model.classifier(cls_features)
            if num_batches == 0 and epoch == 0:
                print(f"Debug - cls_features shape: {cls_features.shape}")
                print(f"Debug - cls_features mean: {cls_features.mean().item():.6f}")
                print(f"Debug - cls_features std: {cls_features.std().item():.6f}")
                print(f"Debug - predictions mean: {preds.mean().item():.6f}")
                print(f"Debug - target labels: {batch_labels.tolist()}")
            loss = criterion(preds.squeeze(), batch_labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "train_loss": avg_loss})

        # --- Validation loop ---
        model.eval()
        val_total_loss = 0
        val_batches = 0
        val_preds = []
        val_true = []
        with torch.no_grad():
            for i in range(0, len(val_files), batch_size):
                batch_files = val_files[i:i+batch_size]
                batch_folders = val_folders[i:i+batch_size]
                batch_labels = torch.tensor(val_labels[i:i+batch_size], dtype=torch.float32).to(device)
                ecg_list = []
                if (i // batch_size) % 10 == 0:
                    print(f"Validation batch {i//batch_size + 1}/{(len(val_files) + batch_size - 1)//batch_size}")
                if file_type == 'wfdb':
                    for record, folder in zip(batch_files, batch_folders):
                        record_path = os.path.join(folder, record)
                        try:
                            signal, fields = load_signals(record_path)
                            ecg = signal.T
                            ecg_normalized = normalize_ecg(ecg)
                            ecg_list.append(ecg_normalized)
                        except Exception as e:
                            print(f"Error loading {record_path}: {e}")
                            continue
                elif file_type == 'npy':
                    for f in batch_files:
                        ecg = load_ecg_from_npy(f)
                        ecg_normalized = normalize_ecg(ecg)
                        ecg_list.append(ecg_normalized)
                if not ecg_list:
                    continue
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
                patches = model.patch_embed(batch)
                mask = model.random_masking(patches)
                masked_patches = patches.clone()
                masked_patches[mask] = 0
                x = masked_patches
                for block in model.transformer_encoder_blocks:
                    x = block(x)
                encoded = x
                cls_features = encoded.mean(dim=1)
                cls_features = torch.nn.functional.normalize(cls_features, p=2, dim=1)
                preds = model.classifier(cls_features)
                loss = criterion(preds.view(-1), batch_labels.view(-1))
                val_total_loss += loss.item()
                val_batches += 1
                pred_out = preds.squeeze().cpu().numpy()
                if pred_out.ndim == 0:
                    val_preds.extend([float(pred_out)])
                else:
                    val_preds.extend(pred_out.tolist())
                val_true.extend(batch_labels.cpu().numpy().tolist())
        avg_val_loss = val_total_loss / val_batches if val_batches > 0 else float('inf')
        val_pred_bin = [1 if p >= 0.5 else 0 for p in val_preds]
        val_accuracy = accuracy_score(val_true, val_pred_bin) if len(set(val_true)) > 1 else None
        val_auroc = roc_auc_score(val_true, val_preds) if len(set(val_true)) > 1 else None
        
        # Calculate challenge score
        val_challenge_score = compute_challenge_score(val_true, val_preds)
        
        print(f"Epoch {epoch+1}/{epochs}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val AUROC: {val_auroc:.4f}, Val Challenge Score: {val_challenge_score:.4f}")
        wandb.log({
            "epoch": epoch + 1,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
            "val_auroc": val_auroc,
            "val_challenge_score": val_challenge_score
        })

        # Early stopping logic (now based on validation loss)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"✅ New best val loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            print(f"⏳ No improvement for {patience_counter}/{patience} epochs")
        if patience_counter >= patience:
            print(f"🛑 Early stopping triggered! Best val loss: {best_loss:.4f}")
            print(f"Restoring best model from epoch {epoch + 1 - patience}")
            model.load_state_dict(best_model_state)
            break
        
        # Step the scheduler after warmup period
        if epoch >= warmup_epochs:
            scheduler.step()
            print(f"Scheduler step: New LR = {optimizer.param_groups[0]['lr']:.6f}")
            
    # Save the best fine-tuned model
    if best_model_state is not None:
        torch.save(best_model_state, 'mae_vit_ecg_finetuned.pth')
        print("💾 Best fine-tuned model saved as 'mae_vit_ecg_finetuned.pth'")
    else:
        torch.save(model.state_dict(), 'mae_vit_ecg_finetuned.pth')
        print("💾 Final fine-tuned model saved as 'mae_vit_ecg_finetuned.pth'")

    # --- Final Test Set Evaluation ---
    print("\nEvaluating on the held-out test set...")
    model.eval()
    test_total_loss = 0
    test_batches = 0
    test_preds = []
    test_true = []
    with torch.no_grad():
        for i in range(0, len(test_files), batch_size):
            batch_files = test_files[i:i+batch_size]
            batch_folders = test_folders[i:i+batch_size]
            batch_labels = torch.tensor(test_labels[i:i+batch_size], dtype=torch.float32).to(device)

            # This section is identical to the validation loop's data loading
            ecg_list = []
            if file_type == 'wfdb':
                for record, folder in zip(batch_files, batch_folders):
                    record_path = os.path.join(folder, record)
                    try:
                        signal, fields = load_signals(record_path)
                        ecg = signal.T
                        ecg_normalized = normalize_ecg(ecg)
                        ecg_list.append(ecg_normalized)
                    except Exception as e:
                        print(f"Error loading {record_path}: {e}")
                        continue
            elif file_type == 'npy':
                for f in batch_files:
                    ecg = load_ecg_from_npy(f)
                    ecg_normalized = normalize_ecg(ecg)
                    ecg_list.append(ecg_normalized)

            if not ecg_list:
                continue

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
            patches = model.patch_embed(batch)
            mask = torch.zeros_like(patches, dtype=torch.bool) # No masking during testing
            x = patches
            for block in model.transformer_encoder_blocks:
                x = block(x)
            encoded = x
            cls_features = encoded.mean(dim=1)
            cls_features = torch.nn.functional.normalize(cls_features, p=2, dim=1)
            preds = model.classifier(cls_features)

            loss = criterion(preds.view(-1), batch_labels.view(-1))
            test_total_loss += loss.item()
            test_batches += 1
            pred_out = preds.squeeze().cpu().numpy()
            if pred_out.ndim == 0:
                test_preds.extend([float(pred_out)])
            else:
                test_preds.extend(pred_out.tolist())
            test_true.extend(batch_labels.cpu().numpy().tolist())

    # Calculate and log final test metrics
    avg_test_loss = test_total_loss / test_batches if test_batches > 0 else float('inf')
    test_pred_bin = [1 if p >= 0.5 else 0 for p in test_preds]
    test_accuracy = accuracy_score(test_true, test_pred_bin) if len(set(test_true)) > 1 else None
    test_auroc = roc_auc_score(test_true, test_preds) if len(set(test_true)) > 1 else None
    test_challenge_score = compute_challenge_score(test_true, test_preds)

    print(f"--- Test Set Results ---")
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test AUROC: {test_auroc:.4f}")
    print(f"Test Challenge Score: {test_challenge_score:.4f}")

    # Log final test metrics to wandb
    wandb.log({
        "test_loss": avg_test_loss,
        "test_accuracy": test_accuracy,
        "test_auroc": test_auroc,
        "test_challenge_score": test_challenge_score
    })

    wandb.finish()

if __name__ == "__main__":
    # Example usage: Use both datasets for balanced positive/negative labels
    
    # Load positive cases from SaMi-Trop
    samitrop_folder = "training_data/samitrop/samitrop_unzipped"
    samitrop_records = find_records(samitrop_folder, file_extension='.hea')
    
    # Load negative cases from PTB-XL  
    ptbxl_folder = "training_data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    ptbxl_records = find_records(ptbxl_folder, file_extension='.hea')
    
    # Combine datasets
    data_files = []
    labels = []
    data_folders = []  # Track which folder each record comes from
    
    # Add SaMi-Trop records (positive cases)
    for record in samitrop_records:
        record_path = os.path.join(samitrop_folder, record)
        try:
            chagas_label = load_label(record_path)
            data_files.append(record)
            data_folders.append(samitrop_folder)
            labels.append(chagas_label)
        except Exception as e:
            print(f"Error loading label for {record_path}: {e}")
    
    # Add PTB-XL records (negative cases) 
    for record in ptbxl_records:
        record_path = os.path.join(ptbxl_folder, record)
        try:
            chagas_label = load_label(record_path)
            data_files.append(record)
            data_folders.append(ptbxl_folder)
            labels.append(chagas_label)
        except Exception as e:
            print(f"Error loading label for {record_path}: {e}")
    
    file_type = 'wfdb'
    print(f"Found {len(data_files)} WFDB records total")
    print(f"Chagas positive: {sum(labels)}, Chagas negative: {len(labels) - sum(labels)}")
    
    model = MAEViTECG()
    
    # Load pretrained weights from MAE pretraining
    try:
        model.load_state_dict(torch.load('mae_vit_ecg_pretrained.pth', map_location='cpu'))
        print("✅ Successfully loaded pretrained MAE weights!")
    except FileNotFoundError:
        print("⚠️  Pretrained weights not found. Using randomly initialized model.")
        print("   Run pretrain_mae_vit_ecg.py first to generate pretrained weights.")
    

    # Split data into training + validation (90%) and test (10%)
    train_val_files, test_files, train_val_labels, test_labels, train_val_folders, test_folders = train_test_split(
        data_files, labels, data_folders, test_size=0.1, random_state=42, stratify=labels
    )

    # Split the training + validation set into final training (80%) and validation (10%)
    train_files, val_files, train_labels, val_labels, train_folders, val_folders = train_test_split(
        train_val_files, train_val_labels, train_val_folders, test_size=1/9, random_state=42, stratify=train_val_labels # 1/9 of 90% is 10% of the total
    )

    print(f"Train size: {len(train_files)}, Validation size: {len(val_files)}, Test size: {len(test_files)}")

    # Pass train/val/test splits to finetune_mae
    finetune_mae(
        model,
        train_files, train_labels, train_folders,
        val_files, val_labels, val_folders,
        test_files, test_labels, test_folders,
        file_type,
        epochs=10, batch_size=4, lr=1e-5, patience=3
    )
