import os
import torch
import torchaudio
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch.nn.functional as F
from scipy.signal import resample
import warnings
import torch.nn as nn
from torchmetrics import Accuracy, AUROC
from tqdm import tqdm
import wandb
from pytorch_lightning.loggers import WandbLogger


import sys
import neurokit2 as nk
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
# --- Suppress specific warnings ---



# --- Spectrogram & Model Parameters ---
MODEL_NAME = 'google/vit-base-patch16-224-in21k'
NUM_WIDE_FEATURES = 2 # Age and Sex
N_FFT = 256
HOP_LENGTH = 64

# --- 1. PyTorch Dataset ---
class ECGDataset(Dataset):
    """
    INTEGRATED: This version combines your data loading/cleaning with the
    spectrogram generation and ViT processing.
    """
    def __init__(self, records_list, data_dir, processor, seq_len=5000, windowing_method='entire_recording'):
        self.records_list = records_list
        self.data_dir = data_dir
        self.processor = processor # ViT Image Processor
        self.seq_len = seq_len
        self.windowing_method = windowing_method
        self.spectrogram_transform = torchaudio.transforms.Spectrogram(
            n_fft=N_FFT, hop_length=HOP_LENGTH
        )

    def __len__(self):
        return len(self.records_list)

    def __getitem__(self, idx):
        record_name = self.records_list[idx]
        record_path = os.path.join(self.data_dir, record_name)
        
        try:
            signal, metadata = helper_code.load_signals(record_path)
            header_text = helper_code.load_header(record_path)
            signal = signal.T # Shape: (12, num_samples)

            if signal.shape[1] < metadata['fs'] * utils.MIN_SIGNAL_DURATION:
                tqdm.write(f"Skipping {record_name} due to insufficient length.")
                return None

            age, sex, label = helper_code.get_patient_info(header_text)
            
            # --- Process Wide Features ---
            normalized_age = (age - utils.MEAN_AGE_TRAIN) / utils.STD_AGE_TRAIN if age is not None else 0.0
            numerical_sex = 0.0 if sex and sex.lower() == 'male' else 1.0 if sex and sex.lower() == 'female' else 0.5
            wide_feats = np.array([normalized_age, numerical_sex], dtype=np.float32)

            # --- Signal Processing ---
            if metadata['fs'] != utils.UNIFIED_FREQUENCY:
                num_samples = int(signal.shape[1] * (utils.UNIFIED_FREQUENCY / metadata['fs']))
                signal = resample(signal, num_samples, axis=1)

                        # Clean signal and correct polarity
            cleaned_leads = [nk.ecg_clean(lead, sampling_rate=utils.UNIFIED_FREQUENCY) for lead in signal]
            signal = np.stack(cleaned_leads)
            try:
                signal, _ = utils.correct_12_lead_polarity_lead_II_ref(signal, utils.UNIFIED_FREQUENCY)
            except Exception as e:
                tqdm.write(f"Error correcting polarity for record {self.records_list[idx]}: {e}")
            
            if utils.USE_ONE_WINDOW:
                windows = utils.get_windows(signal, method=self.windowing_method, window_size=self.seq_len)
                if len(windows) == 0:
                    tqdm.write(f"Skipping {record_name} because no windows could be extracted.")
                    return None
                signal = windows[0]
            
            # signal = utils.normalize(signal)

            # --- Spectrogram Generation ---
            signal_tensor = torch.from_numpy(signal.copy()).float()
            spectrograms = [self.spectrogram_transform(lead) for lead in signal_tensor]
            log_spectrograms = [torch.log(spec + 1e-6) for spec in spectrograms]
            
            normalized_spectrograms = [(s - s.min()) / (s.max() - s.min() + 1e-6) for s in log_spectrograms]
            rows = [torch.cat(normalized_spectrograms[i:i+3], dim=1) for i in range(0, 12, 3)]
            # To create the grid, stack the rows vertically along the frequency axis (dim=0)
            grid = torch.cat(rows, dim=0)

            # --- ViT Processor ---
            grid_numpy = grid.numpy()
            scaled_grid = (grid_numpy * 255).astype(np.uint8)
            rgb_grid = np.stack([scaled_grid] * 3, axis=-1)
            pil_image = Image.fromarray(rgb_grid)
            
            inputs = self.processor(images=pil_image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)

            return {
                "pixel_values": pixel_values, 
                "wide_feats": torch.tensor(wide_feats, dtype=torch.float),
                "labels": torch.tensor([label], dtype=torch.float) # For BCEWithLogitsLoss
            }

        except Exception as e:
            tqdm.write(f"Error processing record {record_name}: {e}. Skipping.")
            return None

# --- 2. PyTorch Lightning DataModule ---
def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        # Return an empty dict if the whole batch is invalid
        return {}
    return torch.utils.data.dataloader.default_collate(batch)

class ECGDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, records_list, split_file_path, batch_size=32, seq_len=utils.WINDOW_SIZE):
        super().__init__()
        self.data_dir = data_dir
        self.records_list = records_list
        self.split_file_path = split_file_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.processor = ViTImageProcessor.from_pretrained(MODEL_NAME) # Instantiate processor here

    def setup(self, stage=None):
        split_df = pd.read_csv(self.split_file_path)
        basename_to_path = {os.path.splitext(os.path.basename(p))[0]: p for p in self.records_list}
        
        train_ids = [basename_to_path[str(id)] for id in split_df[split_df['split'] == 'train']['exam_id'] if str(id) in basename_to_path]
        val_ids = [basename_to_path[str(id)] for id in split_df[split_df['split'] == 'val']['exam_id'] if str(id) in basename_to_path]
        test_ids = [basename_to_path[str(id)] for id in split_df[split_df['split'] == 'test']['exam_id'] if str(id) in basename_to_path]
        
        self.train_dataset = ECGDataset(train_ids, self.data_dir, processor=self.processor, seq_len=self.seq_len)
        self.val_dataset = ECGDataset(val_ids, self.data_dir, processor=self.processor, seq_len=self.seq_len)
        self.test_dataset = ECGDataset(test_ids, self.data_dir, processor=self.processor, seq_len=self.seq_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True, pin_memory=True, collate_fn=custom_collate_fn)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)


# --- 3. PyTorch Lightning Module ---
class MultimodalChagasClassifier(pl.LightningModule):
    def __init__(self, model_hparams, optimizer_hparams):
        super().__init__()
        self.save_hyperparameters()
        
        # --- Model Architecture ---
        self.vit = ViTForImageClassification.from_pretrained(MODEL_NAME, ignore_mismatched_sizes=True)
        self.vit.classifier = nn.Identity() # Use ViT as a feature extractor
        vit_output_size = self.vit.config.hidden_size

        self.final_classifier = nn.Sequential(
            nn.Linear(vit_output_size + model_hparams['num_wide_features'], 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1) # Output a single logit for binary classification
        )
        
        # --- Loss & Metrics ---
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([utils.POS_WEIGHT]))
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.test_auroc = AUROC(task="binary")
        self.validation_step_outputs, self.test_step_outputs = [], []

    def forward(self, pixel_values, wide_feats):
        image_features = self.vit(pixel_values=pixel_values).logits
        combined_features = torch.cat((image_features, wide_feats), dim=1)
        return self.final_classifier(combined_features)

    def _common_step(self, batch):
        if not batch: return None, None, None # Handle empty batch
        logits = self(batch["pixel_values"], batch["wide_feats"])
        loss = self.criterion(logits, batch["labels"])
        preds = torch.sigmoid(logits)
        return loss, preds, batch["labels"]

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._common_step(batch)
        if loss is None: return None
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_acc(preds, labels.int())
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._common_step(batch)
        if loss is None: return
        self.log('val_loss', loss, prog_bar=True)
        self.val_acc(preds, labels.int())
        self.val_auroc(preds, labels)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=True)
        self.log('val_auroc', self.val_aur  oc, on_epoch=True, prog_bar=True)
        self.validation_step_outputs.append({'preds': preds, 'labels': labels})
    
    def on_validation_epoch_end(self):
        if not self.validation_step_outputs: return
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs]).flatten().cpu().numpy()
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs]).flatten().cpu().numpy()
        challenge_score = utils.compute_challenge_score(all_labels, all_preds)
        self.log('val_challenge_score', challenge_score, prog_bar=True)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self._common_step(batch)
        if loss is None: return
        self.test_acc(preds, labels.int())
        self.test_auroc(preds, labels)
        self.log('test_acc', self.test_acc, on_epoch=True)
        self.log('test_auroc', self.test_auroc, on_epoch=True)
        self.test_step_outputs.append({'preds': preds, 'labels': labels})

    def on_test_epoch_end(self):
        if not self.test_step_outputs: return
        all_preds = torch.cat([x['preds'] for x in self.test_step_outputs]).flatten().cpu().numpy()
        all_labels = torch.cat([x['labels'] for x in self.test_step_outputs]).flatten().cpu().numpy()
        challenge_score = utils.compute_challenge_score(all_labels, all_preds)
        self.log('test_challenge_score', challenge_score)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.optimizer_hparams['lr'])
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self.hparams.optimizer_hparams.get('warmup_steps', 0)
        
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, total_iters=warmup_steps)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=self.hparams.optimizer_hparams.get('lr_end', 1e-7))
        
        sequential_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
        
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": sequential_scheduler, "interval": "step"}}

# --- 4. Main Execution Block ---
if __name__ == '__main__':
    pl.seed_everything(42)

    # --- Dummy Data and Split File Generation ---
    DATA_DIR = "../training_data/"
    SPLIT_FILE = "../train_val_test_sets.csv"
    record_paths = helper_code.find_records_abs(DATA_DIR)
    print(f"Found {len(record_paths)} records in {DATA_DIR}")
    NUM_WIDE_FEATURES = 2
    NUM_EPOCHS = 15
    WINDOW_DURATION = 10  # seconds
    SEQ_LEN = utils.UNIFIED_FREQUENCY * WINDOW_DURATION
    BATCH_SIZE = 16

    
    WINDOWING_METHOD = 'entire_recording'  # or 'fixed_size', 'sliding_window'
    CHECKPOINT_MONITOR_METRIC = 'val_challenge_score'  # Metric to monitor
    
    # Initialize wandb, resume existing run if WANDB_RUN_ID is set
    wandb.init(
        entity="edwards_physionet",
        project="ecg_transformer_chagas",
        name=f"vit_transformer_{WINDOWING_METHOD}_10s",
        config={
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "windowing_method": WINDOWING_METHOD,
            "num_epochs": NUM_EPOCHS,
            "data_dir": DATA_DIR,
            "split_file": SPLIT_FILE
        }
    )

    # --- Hyperparameters ---
    model_hparams = {'num_wide_features': NUM_WIDE_FEATURES}
    optimizer_hparams = {'lr': 1e-4, 'warmup_steps': 50, 'lr_end': 1e-7}

    # --- Initialization ---
    data_module = ECGDataModule(
        data_dir=DATA_DIR,
        records_list=record_paths,
        split_file_path=SPLIT_FILE,
        batch_size=8,
        seq_len=SEQ_LEN
    )
    model = MultimodalChagasClassifier(model_hparams, optimizer_hparams)
    
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="auto",
        # precision=PRECISION,
        devices=1,
        logger=WandbLogger(
            project="ecg_transformer_chagas",
            name=f"vit_transformer_{WINDOWING_METHOD}"
        ),
        callbacks=[
            # save best model by val_challenge_score
            pl.callbacks.ModelCheckpoint(
                monitor=CHECKPOINT_MONITOR_METRIC,
                mode='max',
                filename='vit-transformer-{epoch:02d}-{val_challenge_score:.4f}'
            ),
            # always keep the most recent checkpoint
            pl.callbacks.ModelCheckpoint(
                save_last=True,
                filename='last-{epoch:02d}-{val_challenge_score:.4f}'
            )
        ]
    )
    
    print("\n--- Starting Multimodal Model Training ---")
    trainer.fit(model, data_module)
    print("\n--- Training Complete ---")

    print("\n--- Starting Model Testing ---")
    trainer.test(model, datamodule=data_module)
    print("\n--- Testing Complete ---")
