import pytorch_lightning as pl
import torch
import os
import sys
from lightning_classifier import CardioformerLightning
from dataloader import ECGDataModule

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

if __name__ == '__main__':

    # --- Hyperparameters ---
    # NOTE: Update these paths to match your dataset structure
    # Not all of these hyperparamters will be used in your model
    DATA_DIR = "../training_data/" 
    BATCH_SIZE = 80
    SEQ_LEN = utils.WINDOW_SIZE
    NUM_LEADS = 12
    SPLIT_FILE = "../train_val_test_sets.csv"
    WINDOWING_METHOD = 'entire_recording'
    NUM_EPOCHS = 16
    CHECKPOINT_MONITOR_METRIC = 'val_challenge_score' # Metric to monitor for saving checkpoints
    # PATCH_LENGTHS = [2, 4, 8, 8, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32]
    PATCH_LENGTHS = [8, 16]
    NUM_ENCODER_LAYERS = 2
    D_FF = 128
    DROPOUT = 0.1
    N_HEADS = 4
    D_MODEL = 128
    # This controls the precision of the model training, can be "32-true", "16-mixed", or "bf16-mixed"
    PRECISION = "16-mixed" 

    try:
        if torch.cuda.is_available():
            if torch.cuda.get_device_capability()[0] >= 8:
                print("Setting TF32 matmul precision for Ampere GPUs")
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.set_float32_matmul_precision('high')
            else:
                print("Warning: Not an Ampere GPU. Some precision settings may not be optimal.")
    except AttributeError:
        print("Warning: TF32 matmul precision setting not available. Skipping this step.")

    if not os.path.isdir(DATA_DIR):
        print(f"Error: Data directory not found at '{DATA_DIR}'")
        print("Please update the DATA_DIR variable to point to your dataset.")
        sys.exit(1)

    print("\n--- Starting Training Script ---")
    print(f"Using data directory: {DATA_DIR}")
    # --- Setup Data and Model ---
    record_files = helper_code.find_records_abs(DATA_DIR)
    
    print(f"Found {len(record_files)} records in '{DATA_DIR}'")
    print(f"Using split file: {SPLIT_FILE}")

    # #   def __init__(self, seq_len: int, in_channels: int,
    #              patch_lengths: List[int], d_model: int = 128, n_heads: int = 8,
    #              num_layers: int = 6, d_ff: int = 256, dropout: float = 0.1):

    # These params should match the model's __init__ signature
    # If you change the model's signature, update these accordingly
    model_hparams = {
        'seq_len': SEQ_LEN,
        'in_channels': NUM_LEADS,
        'patch_lengths': PATCH_LENGTHS,
        'd_model': D_MODEL,
        'n_heads': N_HEADS,
        'num_layers': NUM_ENCODER_LAYERS,
        'd_ff': D_FF,
        'dropout': DROPOUT,
        # 'data_path': DATA_DIR,
        # 'seq_len': SEQ_LEN,
        # 'use_single_window': utils.USE_ONE_WINDOW,
        # 'num_records': len(record_files),
        # 'split_file': SPLIT_FILE,
        # 'pos_weight': utils.POS_WEIGHT,
        # 'windowing_method': WINDOWING_METHOD,
        # 'epochs': NUM_EPOCHS,
        # 'checkpoint_monitor_metric': CHECKPOINT_MONITOR_METRIC,
    }
    # These params should match the optimizer's parameters, see `lightning_classifier.py` for the optimizers
    optimizer_hparams = {
        'lr': 2e-5,
        'warmup_steps': 700,
        'lr_end': 1e-7,
        'weight_decay': 1e-4,
    }

    # This is where the data will come from 
    data_module = ECGDataModule(
        data_dir=DATA_DIR,
        records_list=record_files,
        split_file_path=SPLIT_FILE,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN
    )

    model = CardioformerLightning(model_hparams, optimizer_hparams)
    
    print("\n--- Model Summary ---")
    print(model)
    
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="auto",
        precision=PRECISION,
        devices=1,
        logger=pl.loggers.TensorBoardLogger("lightning_logs/", name="cardioformer"),
        callbacks=[pl.callbacks.ModelCheckpoint(monitor=CHECKPOINT_MONITOR_METRIC, mode='max', filename='best-challenge-{epoch:02d}-{val_challenge_score:.4f}.ckpt'), 
                   pl.callbacks.DeviceStatsMonitor()]
    )

    # --- Run Training and Testing ---
    print(f"\n--- Starting Training with {len(record_files)} records from '{DATA_DIR}' ---")
    trainer.fit(model, datamodule=data_module)
    print("\n--- Training Finished ---")
    
    print("\n--- Starting Testing ---")
    trainer.test(model, datamodule=data_module)
    print("\n--- Testing Finished ---")
