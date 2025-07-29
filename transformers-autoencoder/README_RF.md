
# Quickstart

## 1. Environment Setup
Install dependencies (recommended: use a virtual environment):
```bash
pip install -r training_data/requirements_rf.txt
```

## 2. Data Preparation
Prepare datasets (PTB-XL, Samitrop, etc.):
```bash
# PTB-XL
python training_data/prepare_ptbxl_data.py
# Samitrop
python training_data/prepare_samitrop_data.py
```

## 3. Pretraining (MAE-ViT)
Run MAE pretraining:
```bash
python pretrain_mae_vit_ecg.py
```

## 4. Finetuning
Run finetuning on downstream tasks:
```bash
python finetune_mae_vit_ecg.py
```

## 5. Monitoring & Evaluation
Metrics and logs are tracked with wandb. Challenge score, AUROC, accuracy, and loss are logged for train/val/test splits.

---

# Key Design Decisions

- **Unified Frequency:** All signals resampled to 500Hz for consistency.
- **Windowing:** Default is a single 10s window per record; future support for multiple windows/data augmentation.
- **Preprocessing:** Uses NeuroKit2 for cleaning; polarity correction (Lead II reference) available.
- **Wide Features:** Age and sex normalized and included as features.
- **Train/Val/Test Split:** Stratified 80/10/10 split for robust evaluation.
- **Metrics:** Challenge score, AUROC, accuracy, and loss tracked for all splits.
- **Early Stopping:** Implemented in finetuning for robust training.
- **Wandb Logging:** All key metrics and losses logged for experiment tracking.
- **Error Handling:** Robust to missing/short signals and shape/type mismatches.

---
