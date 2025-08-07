import torch
import numpy as np
import pandas as pd
import os
import sys
import argparse
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import cleanlab
from cleanlab.count import estimate_cv_predicted_probabilities


# Add project root to path to import from mae and dataloader
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)
    this_dir = os.path.dirname(os.path.abspath(__file__))
    if this_dir not in sys.path:
        sys.path.insert(0, this_dir)
except NameError:
    pass

from finetuning import FineTuningLightningModule
from dataloader import ECGDataset, ECGDataModule
import helper_code
import utils

def get_predictions_for_weak_labels(config, predict_records):
    """
    Loads a fine-tuned model and returns predictions for a given set of records.
    """
    print(f"--- Loading fine-tuned model to predict on {len(predict_records)} records ---")
    
    # 1. Load the fine-tuning module from the provided checkpoint.
    # This checkpoint is assumed to be ALREADY fine-tuned on the strong labels.
    finetuning_params = {
        "num_classes": 1,
        "lr": config["lr"],
        "weight_decay": config["weight_decay"],
        "num_unfrozen_layers": config["num_unfrozen_layers"],
        "info_features": config["info_features"]
    }
    
    fine_tune_module = FineTuningLightningModule.load_from_mae_checkpoint(
        config["pretrained_checkpoint_path"],
        **finetuning_params
    )

    # 2. Create DataModule for the records to be predicted.
    patch_size = fine_tune_module.hparams.patch_size
    overlap_ratio = fine_tune_module.hparams.overlap_ratio
    stride = int(patch_size * (1 - overlap_ratio))
    num_patches = fine_tune_module.model.encoder.pos_embedding.shape[1]
    SEQ_LENGTH = (num_patches - 1) * stride + patch_size

    data_module = ECGDataModule(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        seq_len=SEQ_LENGTH,
        windowing_method='entire_recording'
    )
    data_module.predict_dataset = ECGDataset(predict_records, config["data_dir"], is_training=False, seq_len=SEQ_LENGTH, include_wide_feats=True)

    # 3. Create a trainer instance to run prediction.
    trainer = pl.Trainer(
        accelerator='auto',
        devices=1,
        logger=False,
        enable_progress_bar=True
    )
    
    # 4. Get predictions on the weak-label validation set.
    predictions = trainer.predict(fine_tune_module, datamodule=data_module)
    
    all_logits = torch.cat([p[0] for p in predictions]).squeeze()
    all_probs = torch.sigmoid(all_logits).cpu().numpy()
    
    return all_probs

def main(args):
    pl.seed_everything(42)
    
    # Configuration for the cleaning process.
    config = {
        "data_dir": args.data_dir,
        "pretrained_checkpoint_path": args.checkpoint_path,
        "batch_size": 32,
        "lr": 1e-4,
        "weight_decay": 0.05,
        "num_unfrozen_layers": 2,
        "info_features": 2,
    }

    # --- Step 1: Identify weak label set ---
    print("--- Step 1: Identifying weak label set for analysis ---")
    all_records = helper_code.find_records_abs(config["data_dir"])
    records_meta = utils.prepare_stratification(all_records)
    df = pd.DataFrame(records_meta)
    
    # Weak labels to be validated are from CODE-15%
    weak_df = df[df['source'] == 'CODE-15%'].copy()

    if len(weak_df) == 0:
        raise ValueError("No records found from 'CODE-15%'. These are required as the weak labels to be cleaned.")

    weak_records = weak_df['record'].values
    weak_labels = weak_df['label'].values

    print(f"Found {len(weak_records)} records with weak labels to be validated.")

    # --- Step 2: Get predictions for the weak label set ---
    pred_probs = get_predictions_for_weak_labels(config, weak_records)
    
    # --- Step 3: Find label issues in the weak label set ---
    print("\n--- Step 3: Finding label issues with cleanlab ---")
    pred_probs_cl = np.stack([1 - pred_probs, pred_probs], axis=1)

    label_issues_indices = cleanlab.filter.find_label_issues(
        labels=weak_labels,
        pred_probs=pred_probs_cl,
        return_indices_ranked_by='self_confidence'
    )
    
    num_issues = len(label_issues_indices)
    print(f"Found {num_issues} potential label issues in the 'CODE-15%' dataset.")
    
    # --- Step 4: Create and save cleaned dataset ---
    print("\n--- Step 4: Reporting and saving results ---")
    
    if num_issues > 0:
        # Get the full record paths for the identified issues
        issue_record_paths = weak_records[label_issues_indices]
        
        # Create a DataFrame of the issues
        issues_df = pd.DataFrame({
            'record_path': issue_record_paths,
            'given_label': weak_labels[label_issues_indices],
            'predicted_prob_of_label_1': pred_probs[label_issues_indices]
        })
        
        output_dir = os.path.dirname(args.checkpoint_path)
        if not output_dir:
            output_dir = '.'
        
        issues_csv_path = os.path.join(output_dir, 'code15_label_issues.csv')
        issues_df.to_csv(issues_csv_path, index=False)

        print(f"\nSaved {num_issues} potential label issues to: {issues_csv_path}")
        print("\nTop 10 most likely label errors found in 'CODE-15%':")
        print(issues_df.head(10))
    else:
        print("No label issues found in 'CODE-15%'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Use cleanlab to find label errors in the dataset.")
    parser.add_argument('--data_dir', type=str, default='../training_data/', help='Path to the training data directory.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the pre-trained MAE checkpoint to use as a base for fine-tuning.')
    
    args = parser.parse_args()
    main(args)
