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
from run_12ECG_classifier import load_12ECG_model, run_12ECG_classifier
from dataloader import load_challenge_data_dat
import helper_code_2025

import torch
import xgboost as xgb

import numpy as np
import argparse
import pathlib

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose):
    FEATURE_MODEL_PATH = pathlib.Path("feature_model/")
    train_xg_boost.train(FEATURE_MODEL_PATH, data_folder, model_folder, batch_size=32)
    # # Find the data files.
    # if verbose:
    #     print('Finding the Challenge data...')

    # records = find_records(data_folder)
    # num_records = len(records)

    # if num_records == 0:
    #     raise FileNotFoundError('No data were provided.')

    # # Extract the features and labels from the data.
    # if verbose:
    #     print('Extracting features and labels from the data...')

    # features = np.zeros((num_records, 6), dtype=np.float64)
    # labels = np.zeros(num_records, dtype=bool)

    # # Iterate over the records.
    # for i in range(num_records):
    #     if verbose:
    #         width = len(str(num_records))
    #         print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

    #     record = os.path.join(data_folder, records[i])
    #     features[i] = extract_features(record)
    #     labels[i] = load_label(record)

    # # Train the models.
    # if verbose:
    #     print('Training the model on the data...')

    # # This very simple model trains a random forest model with very simple features.

    # # Define the parameters for the random forest classifier and regressor.
    # n_estimators = 12  # Number of trees in the forest.
    # max_leaf_nodes = 34  # Maximum number of leaf nodes in each tree.
    # random_state = 56  # Random state; set for reproducibility.

    # # Fit the model.
    # model = RandomForestClassifier(
    #     n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, labels)

    # # Create a folder for the model if it does not already exist.
    # os.makedirs(model_folder, exist_ok=True)

    # # Save the model.
    # save_model(model_folder, model)

    # if verbose:
    #     print('Done.')
    #     print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    # Load the xgboost model that we trained
    model_filename = os.path.join(model_folder, 'xgboost_model.json')
    if verbose:
        print(f'Loading xgboost model from {model_filename}')
    xgb_model = xgb.Booster()
    xgb_model.load_model(model_filename)

    feature_model_path = pathlib.Path("feature_model/")
    if verbose:
        print('Loading feature model... at ', feature_model_path)
    # Load our pretrained feature model 
    feature_model = load_12ECG_model(specific_model_path=feature_model_path, 
                                     device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    models = {'xgboost': xgb_model, 'feature_model': feature_model}
    return models

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    # Load the model.
    xgb_model = model['xgboost']
    feature_model = model['feature_model']

    feats = extract_features(record, feature_model)
    feats = feats.reshape(1, -1)
    dtest = xgb.DMatrix(feats)

    try:
        xgb_prob_output = xgb_model.predict(dtest)
        xgb_binary_output = 1 if xgb_prob_output > 0.5 else 0

        return xgb_binary_output, xgb_prob_output
    except Exception as e:
        print(e)
        return None, None


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract your features.
def extract_features(record, feature_model):
    header_path = helper_code_2025.get_header_file(record)
    assert ".hea" in header_path, "Header file not found"
    recording, _, = load_challenge_data_dat(header_path)
    # IMPORTANT: DO NOT USE HELPER_CODE_2025.LOAD_HEADER, AS THIS RETURNS A DICTIONARY
    # we need to keep it in string form so that get_age, get_sex can work properly
    # which is used in the run_12ECG_classifier function
    header = helper_code_2025.load_header(record)
    assert recording is not None, "Recording not found"
    assert header is not None, "Header not found"
    assert type(header) != dict, "Header is dict"
    preds, probs, classes, feats_t = run_12ECG_classifier(recording,
                                                            header,
                                                            feature_model, 
                                                            is_dat=True, 
                                                            return_normalized_features=True)
    feats_t = feats_t.view(-1).cpu().detach().numpy()

    # Stack these features together for the xgboost model?
    feats = np.hstack((preds, feats_t))

    return feats    


# Save your trained model.
def save_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)