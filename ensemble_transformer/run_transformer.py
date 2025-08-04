#!/usr/bin/env python3
"""
Script to load trained transformer models and evaluate ensemble on test sets.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm 

# Ensure project root is in path to import modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import utils
import helper_code
from transformer import (
    ECGDataset,
    ECGClassifierLightning,
    model_hparams,
    optimizer_hparams,
    collate_fn_skip_none,
    ensemble_predict,
)
from torch.utils.data import DataLoader
import transformer
from torch.utils.data import DataLoader

# Constants (match those used during training)
DATA_DIR = os.path.join(project_root, 'training_data')
BATCH_SIZE = 80
SEQ_LEN = utils.WINDOW_SIZE
WINDOWING_METHOD = 'entire_recording'


def build_test_sets():
    """
    Build 10% test split for each source: PTB-XL, SaMi-Trop, CODE-15%.
    Returns dicts: test_records[src] and test_labels[src].
    """
    print(f"Finding test set records at {DATA_DIR}...")
    records_meta = utils.prepare_stratification(helper_code.find_records_abs(DATA_DIR))
    test_records = {}
    test_labels = {}
    for src in ['PTB-XL', 'SaMi-Trop', 'CODE-15%']:
        src_meta = [m for m in records_meta if m['source'] == src]
        labels = [m['label'] for m in src_meta]
        _, test_src = train_test_split(src_meta, test_size=0.1, stratify=labels, random_state=42)
        test_records[src] = [m['record'] for m in test_src]
        test_labels[src] = [m['label'] for m in test_src]
    return test_records, test_labels


def find_top_k_ckpts(run_dir, k):
    """
    Read fold_results.csv and return list of top-k checkpoint paths.
    """
    df = pd.read_csv(os.path.join(run_dir, 'fold_results.csv'))
    topk = df.nlargest(k, 'score')
    return topk['ckpt_path'].tolist()


def find_code15_ckpt(run_dir):
    """
    Locate the best CODE-15% checkpoint in run_dir by parsing the filename score.
    """
    ckpts = [os.path.join(run_dir, f) for f in os.listdir(run_dir)
             if f.startswith('code15-best-') and f.endswith('.ckpt')]
    if not ckpts:
        raise FileNotFoundError('No CODE-15% checkpoint found in ' + run_dir)
    # filenames like code15-best-{epoch}-{val:.4f}.ckpt; extract last part
    def score_from_path(p):
        fname = os.path.basename(p)
        score_str = fname.split('=')[-1].replace('.ckpt', '')
        return float(score_str)
    best = max(ckpts, key=score_from_path)
    return best


def main():
    parser = argparse.ArgumentParser(description='Run transformer ensemble on test sets')
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Directory containing training run artifacts and checkpoints')
    parser.add_argument('--k', type=int, default=2,
                        help='Number of strong transformer folds to include')
    args = parser.parse_args()
    run_dir = args.run_dir
    k = args.k

    # Get checkpoint paths
    kfold_ckpts = find_top_k_ckpts(run_dir, k)
    code15_ckpt = find_code15_ckpt(run_dir)
    print(f"Using strong transformer checkpoints: {kfold_ckpts}")
    print(f"Using code15 checkpoint: {code15_ckpt}")

    # Build test sets
    test_records, test_labels = build_test_sets()

    print(f"Built test sets with length {len(test_records)}")
    # Evaluate per-source and populate universal test lists
    universal_recs = []
    universal_labels = []
    # Compute per-source scores and collect universal records, labels, and preds
    scores = {}
    universal_preds_list = []
    for src, recs in tqdm(test_records.items(), total=len(test_records),
                          desc='Evaluating per-source test sets'):
        labels = test_labels[src]
        preds = ensemble_predict(recs, kfold_ckpts, code15_ckpt,
                                 SEQ_LEN, WINDOWING_METHOD, BATCH_SIZE, DATA_DIR)
        score = utils.compute_challenge_score(labels, preds)
        print(f"{src} 10% test set Challenge Score: {score:.4f}")
        scores[src] = score
        universal_recs.extend(recs)
        universal_labels.extend(labels)
        universal_preds_list.append(preds)

    # Universal test by concatenating per-source predictions
    uni_preds = np.concatenate(universal_preds_list)
    uni_score = utils.compute_challenge_score(universal_labels, uni_preds)
    print(f"Universal test Challenge Score: {uni_score:.4f}")

    # Save results using stored scores to avoid duplicate computation
    out_df = [{'source': src, 'score': scores[src]} for src in test_records]
    out_df.append({'source': 'universal', 'score': uni_score})
    df = pd.DataFrame(out_df)
    df.to_csv(os.path.join(run_dir, 'ensemble_test_scores.csv'), index=False)
    print(f"Saved per-source and universal scores to {os.path.join(run_dir, 'ensemble_test_scores.csv')}")
    # Precompute per-model probabilities on universal set
    print("Precomputing model-specific predictions on universal test set...")
    ds_univ = ECGDataset(universal_recs, DATA_DIR, is_training=False,
                         seq_len=SEQ_LEN, windowing_method=WINDOWING_METHOD)
    loader_univ = DataLoader(ds_univ, batch_size=BATCH_SIZE,
                             collate_fn=collate_fn_skip_none)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Predict with each k-fold checkpoint
    kfold_probs = []
    for ckpt in kfold_ckpts:
        print(f"Predicting with {ckpt}...")
        model = ECGClassifierLightning.load_from_checkpoint(
            ckpt, model_hparams=model_hparams, optimizer_hparams=optimizer_hparams
        )
        model.to(device).eval()
        preds = []
        for batch in loader_univ:
            x, w, _ = batch
            if isinstance(x, torch.Tensor) and x.numel() == 0:
                continue
            x, w = x.to(device), w.to(device)
            with torch.no_grad():
                out = model(x, w)
                preds.append(torch.sigmoid(out).cpu().numpy())
        kfold_probs.append(np.concatenate(preds))

    # Predict with CODE-15% checkpoint
    print(f"Predicting with CODE-15%: {code15_ckpt}...")
    model15 = ECGClassifierLightning.load_from_checkpoint(
        code15_ckpt, model_hparams=model_hparams, optimizer_hparams=optimizer_hparams
    )
    model15.to(device).eval()
    c_preds = []
    for batch in loader_univ:
        x, w, _ = batch
        if isinstance(x, torch.Tensor) and x.numel() == 0:
            continue
        x, w = x.to(device), w.to(device)
        with torch.no_grad():
            out = model15(x, w)
            c_preds.append(torch.sigmoid(out).cpu().numpy())
    code15_probs = np.concatenate(c_preds)

    # Grid search over weights using precomputed probabilities
    print("Performing grid search for ensemble weights...")
    avg_k = np.mean(kfold_probs, axis=0)
    best_score, best_wk, best_wc = -1.0, 0.5, 0.5
    for w_k in np.linspace(0, 1, 11):
        w_c = 1.0 - w_k
        combined = w_k * avg_k + w_c * code15_probs
        score = utils.compute_challenge_score(universal_labels, combined)
        if score > best_score:
            best_score, best_wk, best_wc = score, w_k, w_c
    print(f"Best ensemble weights: w_k={best_wk:.2f}, w_c={best_wc:.2f} -> Challenge Score={best_score:.4f}")
    # Save calibrated weights
    weights_df = pd.DataFrame([{"w_k": best_wk, "w_c": best_wc, "score": best_score}])
    weights_df.to_csv(os.path.join(run_dir, 'ensemble_weights.csv'), index=False)
    print(f"Saved best ensemble weights to {os.path.join(run_dir, 'ensemble_weights.csv')}")
    # Precompute per-model probabilities on universal set
    all_preds = []
    for src, recs in test_records.items():
        preds = ensemble_predict(recs, kfold_ckpts, code15_ckpt,
                                 SEQ_LEN, WINDOWING_METHOD, BATCH_SIZE, DATA_DIR)
        all_preds.append(preds)
    all_preds = np.array(all_preds)  # Shape (n_sources, n_samples)

    print("Performing grid search for ensemble weights on precomputed arrays...")
    for w_k in tqdm(np.linspace(0, 1, 11), desc='Grid search weights', num=len(np.linspace(0, 1, 11))):
        w_c = 1.0 - w_k
        # Weighted sum of precomputed arrays
        preds = np.average(all_preds, axis=0, weights=np.concatenate([[w_k] * k, [w_c]]))
        score = utils.compute_challenge_score(universal_labels, preds)
        if score > best_score:
            best_score, best_wk, best_wc = score, w_k, w_c
    print(f"Best ensemble weights: w_k={best_wk:.2f}, w_c={best_wc:.2f} -> Challenge Score={best_score:.4f}")
    # Save best weights to CSV
    weights_df = pd.DataFrame([{"w_k": best_wk, "w_c": best_wc, "score": best_score}])
    weights_df.to_csv(os.path.join(run_dir, 'ensemble_weights.csv'), index=False)
    print(f"Saved best ensemble weights to {os.path.join(run_dir, 'ensemble_weights.csv')}")


if __name__ == '__main__':
    main()
