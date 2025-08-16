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
from train_se import train_se, load_model as se_load_model, run_model as se_run_model
import ensemble  # new import

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose):
    # prepare separate subfolders
    # train both FM and SE models
    ensemble.train_ensemble(data_folder, model_folder)

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    fm_folder = os.path.join(model_folder, "fm")
    se_folder = os.path.join(model_folder, "se")
    # returns (fm_model, se_model)
    return ensemble.load_ensemble_models(model_folder)

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model_tuple, verbose):
    fm_model, se_model = model_tuple
    # delegate to ensemble inference
    return ensemble.run_ensemble(record, fm_model, se_model)


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