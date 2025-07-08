#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os


import train_xg_boost
from dataloader import load_challenge_data_dat
import transformer

import torch
import xgboost as xgb

import numpy as np
import argparse
import pathlib
import helper_code
import utils
from scipy.signal import resample
import neurokit2 as nk

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose):
    transformer.train(data_folder, model_folder)

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    checkpoint_path = os.path.join(model_folder, 'last.ckpt')
    model = transformer.load_model(checkpoint_path)
    # move model to GPU if available
    if torch.cuda.is_available():
        model = model.to('cuda:0' if torch.cuda.is_available() else 'cpu')
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    # --- copy of ECGDataset.__getitem__ preprocessing ---
    record_path = record
    try:
        signal, wide_feats = utils.preprocess_signal(record_path, windowing_method='entire_recording')
        sig_t = torch.FloatTensor(signal).unsqueeze(0).to("cuda:0" if torch.cuda.is_available() else "cpu")
        wf_t  = torch.FloatTensor(wide_feats).unsqueeze(0).to("cuda:0" if torch.cuda.is_available() else "cpu")
        logits = model(sig_t, wf_t)
        prob   = torch.sigmoid(logits).item()
        pred   = 1 if prob > 0.5 else 0
        return pred, prob

    except Exception as e:
        if verbose:
            print(f"run_model error: {e}")
        return None, None


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################



# Save your trained model.
def save_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)