
from run_12ECG_classifier import load_12ECG_model, run_12ECG_classifier
from xg_dataloader import ChagasDataset, ChagasIterator
from helper_code_2025 import find_records

import torch
import xgboost as xgb

import numpy as np
import argparse
import os

# Set a seed for reproducibility
seed = 42
np.random.seed(seed)

def train(model_folder, data_folder, output_dir = None, batch_size=1):
    if output_dir is None:
        output_dir = model_folder

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda':
        print("Using GPU")
    elif device == 'cpu':
        print("Using CPU!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")

    # TODO: load prna model
    prna_model = load_12ECG_model(specific_model_path=model_folder, device=device)
    assert prna_model is not None, 'Model not loaded'

    # print(prna_model)
    records = find_records(data_folder, full_path=True)
    print("Records found: ", len(records))

    data = ChagasDataset(records, limit_ptb=True)

    total_samples = len(data)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size


    assert data is not None, 'Data not loaded'

    # Split the dataset
    train_data, val_data = torch.utils.data.random_split(data, [train_size, val_size])

    print(f"Len training data, {len(train_data)}, val data {len(val_data)}")

    train_iterator = ChagasIterator(train_data, loaded_model=prna_model, exclude_ptb=True)
    val_iterator = ChagasIterator(val_data, loaded_model=prna_model, exclude_ptb=False)

    
    dtrain = xgb.DMatrix(train_iterator)
    dval = xgb.DMatrix(val_iterator)

    # define xgboost model
    xgb_params = {
        'objective': 'binary:logistic',
        'n_estimators': 100,
        'base_score': 0.5,
        'max_depth': 2,
        'learning_rate': .5, 
        'device': device, 
        'tree_method': 'hist', 
        'eval_metric': 'logloss', 
        'verbosity': 1
    }

    class ErrorHandling(xgb.callback.TrainingCallback):
        def __init__(self):
            self.epoch = 0

        def after_iteration(self, model, epoch, evals_log):
            try:
                self.epoch += 1
                print(f"Epoch {self.epoch}: {evals_log}")
            except Exception as e:
                print(f"Error at epoch {self.epoch}: {e}")
            return False  # Return True to stop training

    # Add custom logger callback
    custom_logger = ErrorHandling()

    evals = [(dval, 'eval')]
    num_round = 5


    model = xgb.train(xgb_params, dtrain, num_round, evals=evals, verbose_eval=True, callbacks=[custom_logger]) 

    # save model, seeing if the output_dir exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save_model(os.path.join(output_dir, 'xgboost_model.json'))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    train(args.model_dir, args.data_dir, args.batch_size)

