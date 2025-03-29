import numpy as np
import pandas as pd
import os 

import xgboost

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import wfdb
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from wfdb import processing

from dataloader import load_challenge_data_dat
import helper_code_2025
from run_12ECG_classifier import get_normalized_features, run_12ECG_classifier

class ChagasDataset(Dataset):
    """Dataset to load Chagas data. 
    At each element, it loads a recoridng and header. It also runs the feature extraction model, 
    and returns the features and the label. The features will be passed into an xgboost model, 
    and the label is used for training.

    Args:
        Records: List of records to load (from helper_code.find_records())
        feat_means: List of means for each feature (from loaded_model['feat_means'])
        feat_stds: List of standard deviations for each feature (from loaded_model['feat_stds'])

    Returns:
        _type_: _description_
    """    ''''''
    def __init__ (self, records, loaded_model, return_classes=False):
        self.records = records
        self.loaded_model = loaded_model
        self.return_classes = return_classes

    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        header_path = helper_code_2025.get_header_file(self.records[idx])
        assert ".hea" in header_path, "Header file not found"
        recording, _, = load_challenge_data_dat(header_path)
        # IMPORTANT: DO NOT USE HELPER_CODE_2025.LOAD_HEADER, AS THIS RETURNS A DICTIONARY
        # we need to keep it in string form so that get_age, get_sex can work properly
        # which is used in the run_12ECG_classifier function
        header = helper_code_2025.load_header(self.records[idx])
        assert recording is not None, "Recording not found"
        assert header is not None, "Header not found"
        assert type(header) != dict, "Header is dict"
        preds, probs, classes, feats_t = run_12ECG_classifier(recording,
                                                                header,
                                                                self.loaded_model, 
                                                                is_dat=True, 
                                                                return_normalized_features=True)
        feats_t = feats_t.view(-1).cpu().detach().numpy()
        # print(feats_t.shape)
        # Stack these features together for the xgboost model?
        # print(preds.shape)
        feats = np.hstack((preds, feats_t))
        label = helper_code_2025.load_label(self.records[idx])

        assert feats is not None, "Features not found"
        assert label is not None, "Label not found"

        if self.return_classes:
            return feats, label, classes
        return feats, label
class ChagasIterator(xgboost.DataIter):
    def __init__(self, chagas_dataset):
        self.dataset = chagas_dataset
        self._idx = 0
        super().__init__(cache_prefix=os.path.join(os.getcwd(), 'cache'))

    def next(self, input_data) -> bool:
        if self._idx >= len(self.dataset):
            return False
        
        successful_iteration = False

        while not successful_iteration:
            try:
                X, y = self.dataset[self._idx]
                # TODO: accomadate batch sizes greater than 1
                X = X.reshape(1, -1)
                y = np.array([y]).reshape(1, -1)

                input_data(data=X, label=y)
                self._idx += 1
                successful_iteration = True
            except Exception as e:
                print(f"Error with training at {self._idx}, moving to next data point: {e} ")
                self._idx += 1
                
                if self._idx >= len(self.dataset):
                    return False

        return True
    def reset(self):
        self._idx = 0
    