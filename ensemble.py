import os
import torch
import utils
import numpy as np
import pandas as pd
import custom_helper_code
from tqdm import tqdm

import fm 
from train_se import train_se, load_model as load_se_model, run_model as run_se_model
from sklearn.model_selection import train_test_split

def train_ensemble(data_folder, model_folder):
    # 1) Fine‐tune the FM foundation model
    fm.train_model(data_folder, model_folder, is_submission=True)
    # 2) Train the SE‐block model
    train_se(data_folder, model_folder)

def load_ensemble_models(model_folder):
    # Load FM classifier
    fm_ckpt = os.path.join(model_folder, 'foundation_model_finetuned.ckpt')
    fm_model = fm.load_finetuned_model(fm_ckpt)
    # Load SE classifier
    se_model = load_se_model(model_folder)
    return fm_model, se_model

def run_fm_model(record, fm_model):
    # Preprocess (reuse SE‐style preprocessing for 1-window inference)
    signal, wide_feats = utils.preprocess_signal(
        record,
        windowing_method='entire_recording',
        is_training=False
    )
    device = next(fm_model.parameters()).device
    sig_t = torch.as_tensor(signal, dtype=torch.float32, device=device).unsqueeze(0)
    wf_t  = torch.as_tensor(wide_feats, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = fm_model(sig_t, wf_t)
        prob = torch.sigmoid(logits).item()
        pred = int(prob > 0.5)
    return pred, prob

def run_ensemble(record, fm_model, se_model, fm_weight=0.2, se_weight=0.8):
    # enforce hard‐coded weights
    fm_weight, se_weight = 0.2, 0.8
    # Get FM prediction
    pred_fm, prob_fm = run_fm_model(record, fm_model)
    # Get SE prediction
    pred_se, prob_se = run_se_model(record, se_model)
    # Weighted average of probabilities
    prob = fm_weight * prob_fm + se_weight * prob_se
    pred = int(prob > 0.5)
    return pred, prob

def search_best_weights(data_folder, fm_model, se_model, step=0.05):
    """
    Grid search for fm_weight in [0,1] (se_weight = 1 - fm_weight)
    to maximize utils.compute_challenge_score on the validation split.
    """
    # Prepare records & labels
    meta = utils.prepare_stratification(custom_helper_code.find_records_abs(data_folder))
    df = pd.DataFrame(meta)
    
    # Use ONLY the val/test split for searching weights
    _, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    records = val_df['record'].tolist()
    labels = val_df['label'].tolist()

    # Build dataset and DataLoader for batched inference
    from torch.utils.data import DataLoader
    from dataloader import ECGDataset, collate_fn_skip_none

    seq_len = utils.UNIFIED_FREQUENCY * 10  # match default 10s windows
    ds = ECGDataset(
        records, data_folder, is_training=False,
        seq_len=seq_len, windowing_method='entire_recording',
        include_wide_feats=True, do_multitask=False
    )
    loader = DataLoader(ds, batch_size=32, collate_fn=collate_fn_skip_none, num_workers=9)

    # Run both models in eval mode
    fm_model.eval(); se_model.eval()
    fm_probs_batches, se_probs_batches = [], []
    for batch in tqdm(loader, desc="Finding weights", total=len(loader)):
        signals, wide_feats, _ = batch
        device = next(fm_model.parameters()).device
        signals, wide_feats = signals.to(device), wide_feats.to(device)
        with torch.no_grad():
            fm_logits = fm_model(signals, wide_feats)
            se_logits = se_model.model(signals, wide_feats)
            fm_probs_batches.append(torch.sigmoid(fm_logits.squeeze(-1)).cpu().numpy())
            se_probs_batches.append(torch.sigmoid(se_logits.squeeze(-1)).cpu().numpy())
    fm_probs = np.concatenate(fm_probs_batches)
    se_probs = np.concatenate(se_probs_batches)

    print("Found probabilities for both models, starting search...")
    best_score = -1.0
    best_fm, best_se = 0.0, 1.0
    # add progress bar over candidate weights
    for w in tqdm(np.arange(0, 1+step, step), desc="Searching ensemble weights"):
        probs = w*fm_probs + (1-w)*se_probs
        score = utils.compute_challenge_score(labels, probs, use_gpu=True)
        if score > best_score:
            best_score, best_fm, best_se = score, w, 1-w
    return best_fm, best_se, best_score

def run_ensemble_batch(records, fm_model, se_model, fm_weight=0.2, se_weight=0.8, batch_size=32):
    """
    Preprocess a list of records in batch, run both models together,
    and return lists of (pred, prob).
    """
    device = next(fm_model.parameters()).device
    # preprocess all signals and wide_feats
    data = [utils.preprocess_signal(r, windowing_method='entire_recording', is_training=False)
            for r in records]
    sigs = torch.stack([torch.as_tensor(d[0], dtype=torch.float32) for d in data]).to(device)
    wfs  = torch.stack([torch.as_tensor(d[1], dtype=torch.float32) for d in data]).to(device)
    # enforce hard‐coded weights
    fm_weight, se_weight = 0.2, 0.8
    with torch.no_grad():
        fm_logits = fm_model(sigs, wfs)
        se_logits = se_model.model(sigs, wfs)
        fm_probs = torch.sigmoid(fm_logits.squeeze(-1)).cpu().numpy()
        se_probs = torch.sigmoid(se_logits.squeeze(-1)).cpu().numpy()
    ens_probs = fm_weight * fm_probs + se_weight * se_probs
    ens_preds = (ens_probs > 0.5).astype(int)
    return ens_preds.tolist(), ens_probs.tolist()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", "-d")
    parser.add_argument("--fm_model_folder", "-fm")
    parser.add_argument("--se_model_folder", "-se")
    parser.add_argument("--fm-weight", type=float, default=None)
    parser.add_argument("--se-weight", type=float, default=None)
    parser.add_argument("--no_train", action='store_true',default=False)
    args = parser.parse_args()

    # 1) Train both models
    if args.no_train is False:
        train_ensemble(args.data_folder, args.fm_model_folder, args.se_model_folder)
    # 2) Load them
        
        fm_model = fm.load_finetuned_model(os.path.join(args.fm_model_folder,'foundation_model_finetuned.ckpt'))
        se_model = load_se_model(args.se_model_folder, verbose=False)
    else:
        # these are 5w03csgw on se model, and 72v4t91v on fm
        fm_model = fm.load_finetuned_model(os.path.join(args.fm_model_folder,'foundation_model_finetuned.ckpt'))
        se_model = load_se_model(args.se_model_folder)
    # 3) Search for best weights if not provided
    if args.fm_weight is None or args.se_weight is None:
        fm_w, se_w, val_score = search_best_weights(args.data_folder, fm_model, se_model)
        print("--- Best ensemble weights found ---")
        print(f"Best ensemble weights found: fm={fm_w:.2f}, se={se_w:.2f}, validation score={val_score:.4f}")
        print("--------------------")
    else:
        fm_w, se_w = args.fm_weight, args.se_weight

    # Output the best weights as a csv
    output_path = os.path.join(args.fm_model_folder, 'ensemble_weights.csv')
    with open(output_path, 'w') as f:
        f.write(f"fm_weight,se_weight\n{fm_w},{se_w}\n")
    print(f"Ensemble weights saved to {output_path}")

    # # 4) Batched inference example on first few records
    # files = sorted(os.listdir(args.data_folder))[:32]  # choose batch size
    # paths = [os.path.join(args.data_folder, f) for f in files]
    # preds, probs = run_ensemble_batch(paths, fm_model, se_model,
    #                                   fm_weight=fm_w, se_weight=se_w,
    #                                   batch_size=len(paths))
    # for f, p, pr in zip(files, preds, probs):
    #     print(f"{f} -> pred={p}, prob={pr:.4f}")
