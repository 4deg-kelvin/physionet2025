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


import transformer

import torch

import numpy as np
import argparse
import pathlib
import helper_code
import custom_helper_code
import utils
from scipy.signal import resample
import neurokit2 as nk

import pretrain_mae_vit_ecg
import finetuning_mae_vit_ecg

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose):
    pretrain_mae_vit_ecg.train(data_folder, model_folder, verbose)
    finetuning_mae_vit_ecg.train(data_folder, model_folder, verbose)

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    model = finetuning_mae_vit_ecg.load_model(model_folder, verbose)
    # move model to GPU if available
    if torch.cuda.is_available():
        model = model.to('cuda:0' if torch.cuda.is_available() else 'cpu')
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    record_path = record
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        signal, wide_feats = utils.preprocess_signal(
            record_path, windowing_method='entire_recording', is_training=False
        )
        sig_t = torch.FloatTensor(signal).unsqueeze(0).to(device)
        wf_t  = torch.FloatTensor(wide_feats).unsqueeze(0).to(device)

        # Attempt to call forward(x, info); fallback to forward(x)
        try:
            logits = model(sig_t, wf_t)
        except TypeError:
            logits = model(sig_t)

        prob = torch.sigmoid(logits).item()
        pred = 1 if prob > 0.5 else 0
        return pred, prob

    # IMPORTANT: if you get a NotImplementedError specifically, something bad happened (ie, you 
    # chose the wrong preprocessing steps, etc, so you should NOT proceed with ANY prediction, therefore we 
    # end the program here for debugging purposes.)
    except NotImplementedError as e:
        raise NotImplementedError(f"run_model error: {e}")
    except Exception as e:
        if verbose:
            print(f"run_model error: {e}")
        return None, None


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################



# # Save your trained model.
# def save_model(model_folder, model):
#     d = {'model': model}
#     filename = os.path.join(model_folder, 'model.sav')
#     joblib.dump(d, filename, protocol=0)
