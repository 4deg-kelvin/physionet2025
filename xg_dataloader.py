import numpy as np
import pandas as pd
import os 
import random

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
from run_12ECG_classifier import get_normalized_features, run_12ECG_classifier, prepare_data_for_feat_model

class ChagasDataset():
    """Dataset to load Chagas data. 
    At each element, it loads a recording and header. It also runs the feature extraction model, 
    and returns the features and the label. The features will be passed into an xgboost model, 
    and the label is used for training.

    Args:
        Records: List of records to load (from helper_code.find_records())
        limit_ptb: If True, only 10% of the PTB-XL dataset will be loaded

    Returns:
        _type_: _description_
    """    ''''''
    def __init__ (self, records, return_classes=False, return_source=False, limit_ptb=False):
        self.records = records
        self.return_classes = return_classes
        self.return_source = return_source

        self.limit_ptb = limit_ptb 


    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        header_path = helper_code_2025.get_header_file(self.records[idx])
        assert ".hea" in header_path, "Header file not found"
        recording, _, = load_challenge_data_dat(header_path)
        # IMPORTANT: DO NOT USE the header from load_challenge_data_dat, AS THIS RETURNS A DICTIONARY
        # we need to keep it in string form so that get_age, get_sex can work properly
        # which is used in the run_12ECG_classifier function
        # load_header returns a string
        header = helper_code_2025.load_header(self.records[idx])
        # assert recording is not None, "Recording not found"
        # assert header is not None, "Header not found"
        # assert type(header) != dict, "Header is dict"
        source = helper_code_2025.get_variable(header, "# Source:")

        label = helper_code_2025.load_label(self.records[idx])

        # Limiting PTB to 10% of the dataset, so pick a number between 1 and 10
        # and if we don't get this one, return Null so that ChagasIterator can skip this datapoint
        if self.limit_ptb:
            if "PTB-XL" in source and random.randint(1, 10) != 1:
                print("Skipping PTB-XL")
                # Set the source to EXCLUDE so that we can skip this data point
                return header, recording, "EXCLUDE", label

        return header, recording, source, label

        # preds, probs, classes, feats_t = run_12ECG_classifier(recording,
        #                                                         header,
        #                                                         self.loaded_model, 
        #                                                         is_dat=True, 
        #                                                         return_normalized_features=True)
        # feats_t = feats_t.view(-1).cpu().detach().numpy()
        # # print(feats_t.shape)
        # # Stack these features together for the xgboost model?
        # # print(preds.shape)
        # feats = np.hstack((preds, feats_t))
        # label = helper_code_2025.load_label(self.records[idx])

        # # assert feats is not None, "Features not found"
        # # assert label is not None, "Label not found"

        # if self.return_classes:
        #     return feats, label, classes
        # if self.return_source:
        #     return feats, label, source
        # return feats, label
class ChagasNoFeatsDataset(Dataset):
    def __init__ (self, records, loaded_model):
        self.records = records
        self.loaded_model = loaded_model

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
        # assert recording is not None, "Recording not found"
        # assert header is not None, "Header not found"
        # assert type(header) != dict, "Header is dict"
        
        windows_t, wide_feats = prepare_data_for_feat_model(recording, header, self.loaded_model, is_dat=True)

        label = helper_code_2025.load_label(self.records[idx])


        return windows_t, wide_feats, label
class ChagasIterator(xgboost.DataIter):
    def __init__(self, chagas_dataset, loaded_model, exclude_ptb=False):
        self.dataset = chagas_dataset
        self.loaded_model = loaded_model
        # If you want to exclude PTB-XL data, set this to True
        self.exclude_ptb = exclude_ptb
        self._idx = 0
        super().__init__(cache_prefix=os.path.join(os.getcwd(), 'cache'))
    
    def forward(self, recording, header, source):
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

        return feats

    def next(self, input_data) -> bool:
        if self._idx >= len(self.dataset):
            return False
        
        successful_iteration = False

        while not successful_iteration:
            try:
                heading, recording, source, y = self.dataset[self._idx]
                # If the source is PTB-XL, meaning we have EXCLUDE as the source, we skip this data point
                if source == "EXCLUDE" and self.exclude_ptb:
                    self._idx += 1
                
                    if self._idx >= len(self.dataset):
                        return False
                    continue
                X = self.forward(recording, heading, source)

                # These value are none if we are skipping ptb-xl data
                if X is None or y is None:
                    self._idx += 1
                
                    if self._idx >= len(self.dataset):
                        return False
                    continue

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
class ChagasBatchedIterator(xgboost.DataIter):
    def __init__(self, chagas_dataset, batch_size=16):
        self.dataset = chagas_dataset
        self._idx = 0
        self.batch_size = batch_size
        super().__init__(cache_prefix=os.path.join(os.getcwd(), 'cache'))

    def next(self, input_data) -> bool:
            if self._idx >= len(self.dataset):
                return False
            
            successful_iteration = False

            while not successful_iteration:
                try:
                    # Get next batch indices, starting from self._idx, making sure not to go over the dataset length
                    batch_indices = list(range(self._idx, min(self._idx + self.batch_size, len(self.dataset))))

                    # Get the features and labels for the batch using the indices
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