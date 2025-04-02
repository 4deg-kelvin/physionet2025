#!/usr/bin/env python

import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import logging
import pandas as pd
import os, sys, re
from pathlib import Path
import joblib
import wfdb
from wfdb import processing
from scipy.signal import decimate, resample
from biosppy.signals import ecg
from biosppy.signals.tools import filter_signal

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset

from model import CTN
from feats.features import *
import helper_code_2025

nb_windows=30
window_size = 15*500
deepfeat_sz = 64
dropout_rate = 0.2
fs = 500
filter_bandwidth = [3, 45]
polarity_check = []

# Transformer parameters
d_model = 256   # embedding size
nhead = 8       # number of heads
d_ff = 2048     # feed forward layer size
num_layers = 8  # number of encoding layers

ch_idx = 1
nb_demo = 2
nb_feats = 20

model_name = 'ctn'
folds = [3, 4, 6]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)

classes = sorted(['270492004', '164889003', '164890007', '426627000', '713427006', 
                  '713426002', '445118002', '39732003', '164909002', '251146004', 
                  '698252002', '10370003', '284470004', '427172004', '164947007', 
                  '111975006', '164917005', '47665007', '59118001', '427393009', 
                  '426177001', '426783006', '427084000', '63593006', '164934002', 
                  '59931005', '17338001'])

num_classes = len(classes)            

def run_12ECG_classifier(data, 
                         header_data, 
                         loaded_model, 
                         is_dat=False, 
                         standarize_sampling_rate=True, 
                         return_normalized_features=False):
    # Standardize recording
    recording = None
    if standardize_sampling_rate:
        if is_dat:
            recording = standarize_sampling_rate_dat(data, header_data)
        else:
            recording = standardize_sampling_rate(data, header_data)
    else: 
        recording = data

    # Get wide features
    feat_means = loaded_model['feat_means']
    feat_stds = loaded_model['feat_stds']
    feats_t = get_normalized_features(recording[ch_idx], feat_means, feat_stds)

    # Apply filtering and normalization
    recording = preprocess_signal(recording, filter_bandwidth, ch_idx=1)

    # Split into random windows
    inp_t = get_windows_padded(recording, window_size, nb_windows)

    # Make predictions
    models = [loaded_model['models'][fold] for fold in folds]
    thrs = [loaded_model['thrs'][fold] for fold in folds]
    probs, preds, wide_feats = None, None, None
    if is_dat:
        probs, preds, wide_feats = predict_dat(models, thrs, inp_t, feats_t, header_data)
    else:
        probs, preds = predict(models, thrs, inp_t, feats_t, header_data)

    # Return normalized features if needed
    if return_normalized_features:
        if wide_feats is None:
            print("WARNING: wide_feats is None")
        return preds, probs, classes, wide_feats
    else:
        return preds, probs, classes
def prepare_data_for_feat_model(data, header_data, loaded_model, is_dat=False, standarize_sampling_rate=True):
    recording = None
    if standardize_sampling_rate:
        if is_dat:
            recording = standarize_sampling_rate_dat(data, header_data)
        else:
            recording = standardize_sampling_rate(data, header_data)
    else: 
        recording = data

    # Get wide features
    feat_means = loaded_model['feat_means']
    feat_stds = loaded_model['feat_stds']
    feats_t = get_normalized_features(recording[ch_idx], feat_means, feat_stds)

    # Apply filtering and normalization
    recording = preprocess_signal(recording, filter_bandwidth, ch_idx=1)

    # Split into random windows
    inp_t = get_windows_padded(recording, window_size, nb_windows)

    mean_age = pd.read_csv('mean_age.csv', index_col=0).age.values[0]
    std_age = pd.read_csv('std_age.csv', index_col=0).age.values[0]
    
    # Get (normalized) demographic data
    # print("hdr", hdr)
    age_t = torch.FloatTensor((np.array([helper_code_2025.get_age(header_data)])[None].T - mean_age) / std_age)
    sex_t = torch.FloatTensor([1. if helper_code_2025.get_sex(header_data).lower() == 'female' else 0])[None].T  
    # Feats are in the order age, sex, then the rest of the features      
    wide_feats = torch.cat([age_t, sex_t, feats_t.float()], dim=1).to(device)

    return inp_t, wide_feats
def get_normalized_features(recording, feat_means, feat_stds):
    ''' Get normalized wide features '''
    
    ecg_features = Features(
        data=recording,
        fs=fs,
        feature_groups=['full_waveform_statistics', 'heart_rate_variability_statistics', 'template_statistics']
    )

    # Calculate ECG features
    ecg_features.calculate_features(
        filter_bandwidth=[3, 45], show=False,
        channel=ch_idx, normalize=True, polarity_check=True,
        template_before=0.25, template_after=0.4
    )

    feats = ecg_features.get_features()
    # print("feats", feats)
    # print("feats type", type(feats), "feats index", feats.index)
    # print('feats columns', feats.columns)
    # print("feats transposed", feats.T)  

    # Get feature names in order of importance (remove duration and demo)
    feature_names = list(np.load('top_feats.npy'))
    feature_names.remove('full_waveform_duration')
    feature_names.remove('Age')
    feature_names.remove('Gender_Male')

    # print("feature names", feature_names, len(feature_names))
    # see the differences in indices that are in feats dataframe vs feature_names
    
    #FIXME: ensure this works (we don't originally transpose)
    feats = feats.T

    # Get top (normalized) features (excludes signal duration and demo feats)
    top_feats = feats[feature_names[:nb_feats]].values
    # print("top feats", top_feats, top_feats.shape, top_feats.dtype)
    # print("top feats element type", type(top_feats[0][0]))
    # NOTE: this line is also changed (we don't originally flatten)
    top_feats = np.array(top_feats, dtype=float)
    # print("top feats", top_feats, top_feats.shape, top_feats.dtype)
    # top_feats = top_feats[0]
    # First, convert any infs to nans
    top_feats[np.isinf(top_feats)] = np.nan
    # Replace NaNs with feature means
    top_feats[np.isnan(top_feats)] = feat_means[None][np.isnan(top_feats)]
    # Normalize wide features
    feats_normalized = (top_feats - feat_means) / feat_stds
    # Use zeros (normalized mean) if cannot find patient features
    if not len(feats_normalized):
        feats_normalized = np.zeros(nb_feats)[None]

    return torch.from_numpy(feats_normalized)

def predict(models, thrs, inp_windows_t, feats_t, hdr, device=device):
    '''
    Predict using the loaded models
    models: list of pytorch model
    thrs: list of individual model thresholds
    inp_windows_t: torch tensor holding random recording windows
    hdr: header file contents
    device: cpu or gpu
    returns, probs: numpy arr of probs per class [1 x nb_classes]
    '''
    probs, preds = [], []

    # Get normalized data (as batch of 1)
    inp_windows_t = inp_windows_t.float()[None].to(device)

    mean_age = pd.read_csv('mean_age.csv', index_col=0).age.values[0]
    std_age = pd.read_csv('std_age.csv', index_col=0).age.values[0]
    
    # Get (normalized) demographic data
    age_t = torch.FloatTensor((get_age(hdr[13])[None].T - mean_age) / std_age)
    sex_t = torch.FloatTensor([1. if hdr[14].find('Female') >= 0. else 0])[None].T        
    wide_feats = torch.cat([age_t, sex_t, feats_t.float()], dim=1).to(device)

    # Predict
    outs = []
    with torch.no_grad():
        for model, thr in zip(models, thrs):
            model.eval()

            # Loop over nb_windows
            for inp_t in inp_windows_t.transpose(1, 0):
                out = model(inp_t, wide_feats)
                outs.append(out)
            out = torch.stack(outs).mean(dim=0)   # take the average of the sequence windows

            # Collect probs and preds
            prob = out.sigmoid().data.cpu().numpy()
            probs.append(prob)
            pred = prob > thr
            preds.append(pred)

    # Consolidate probs and preds
    probs = np.concatenate(probs)
    preds = np.concatenate(preds)

    probs = probs.mean(axis=0)
    preds = np.any(preds, axis=0)
    
    return probs, preds
def predict_dat(models, thrs, inp_windows_t, feats_t, hdr, device=device):
    '''
    Predict using the loaded models, MADE FOR DAT FILES 
    models: list of pytorch model
    thrs: list of individual model thresholds
    inp_windows_t: torch tensor holding random recording windows
    hdr: header file contents
    device: cpu or gpu
    returns, probs: numpy arr of probs per class [1 x nb_classes], wide_feats: torch tensor of wide features
    '''
    probs, preds = [], []

    # Get normalized data (as batch of 1)
    inp_windows_t = inp_windows_t.float()[None].to(device)

    mean_age = pd.read_csv('mean_age.csv', index_col=0).age.values[0]
    std_age = pd.read_csv('std_age.csv', index_col=0).age.values[0]
    
    # Get (normalized) demographic data
    # print("hdr", hdr)
    age_t = torch.FloatTensor((np.array([helper_code_2025.get_age(hdr)])[None].T - mean_age) / std_age)
    sex_t = torch.FloatTensor([1. if helper_code_2025.get_sex(hdr).lower() == 'female' else 0])[None].T  
    # Feats are in the order age, sex, then the rest of the features      
    wide_feats = torch.cat([age_t, sex_t, feats_t.float()], dim=1).to(device)

    # Predict
    outs = []
    with torch.no_grad():
        for model, thr in zip(models, thrs):
            model.eval()

            # Loop over nb_windows
            for inp_t in inp_windows_t.transpose(1, 0):
                out = model(inp_t, wide_feats)
                outs.append(out)
            out = torch.stack(outs).mean(dim=0)   # take the average of the sequence windows

            # Collect probs and preds
            prob = out.sigmoid().data.cpu().numpy()
            probs.append(prob)
            pred = prob > thr
            preds.append(pred)

    # Consolidate probs and preds
    probs = np.concatenate(probs)
    preds = np.concatenate(preds)

    probs = probs.mean(axis=0)
    preds = np.any(preds, axis=0)
    
    return probs, preds, wide_feats
def load_12ECG_model(top_level_model_dir=None, specific_model_path = None, device="cuda"):
    """Loads the 1ECG model

    Args:
        top_level_model_dir (str, optional): This is (from the original code) the path to the model's entire directory. It will prepend an extra path to it. I don't use this. Defaults to None.
        specific_model_path (str, optional): Specific model directory, will NOT prepend any path. This is what I use (and implemented). Defaults to None.
        device (str, optional): Device to use. Defaults to "cuda".
    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """    
    # load the model from disk
    model = CTN(d_model, nhead, d_ff, num_layers, dropout_rate, deepfeat_sz, nb_feats, nb_demo, classes).to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    
    if top_level_model_dir is None and specific_model_path is None:
        raise Exception('Please provide either model_dir or model_path')
    # Optionally load prefix_path, which is just the path to model files
    prefix_path = Path(f'{top_level_model_dir}/saved_models/{model_name}') if top_level_model_dir is not None else Path(specific_model_path)

    feat_means = np.loadtxt(prefix_path / f"fold_{folds[0]}" / "feat_means.txt")
    feat_stds = np.loadtxt(prefix_path / f"fold_{folds[0]}" / "feat_stds.txt")

    # Get ensemble model and their thrs
    models, thrs = {}, {}
    for fold in folds:
        models[fold] = load_best_model(model, f'{prefix_path}/fold_{fold}/{model_name}.tar')
        thrs[fold] = np.loadtxt(f'{prefix_path}/fold_{fold}/thrs.txt')

    loaded_model = {'models' : models, 'thrs' : thrs, 'feat_means' : feat_means, 'feat_stds' : feat_stds}
    # print(loaded_model)
    return loaded_model

def load_best_model(model, model_loc):
    checkpoint = torch.load(model_loc, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model    

def standardize_sampling_rate(recording, hdr, fs=fs):
    ''' Standardize sampling rate '''
    sampling_rate = int(hdr[0].split()[2])    
    if sampling_rate > fs:
        recording = decimate(recording, int(sampling_rate / fs))
    elif sampling_rate < fs:
        recording = resample(recording, int(recording.shape[-1] * (fs / sampling_rate)), axis=1)
    return recording
def standarize_sampling_rate_dat(recording, hdr, fs=fs):
    ''' Standardize sampling rate for dat files '''
    sampling_rate = None
    # Hdr is already loaded using helper_code_2025, and is a dictionary
    if type(hdr) == dict:
        sampling_rate = hdr['fs']
    else:
        sampling_rate = helper_code_2025.get_sampling_frequency(hdr)  

    if sampling_rate > fs:
        recording = decimate(recording, int(sampling_rate / fs))
    elif sampling_rate < fs:
        recording = resample(recording, int(recording.shape[-1] * (fs / sampling_rate)), axis=1)
    return recording
def normalize(seq, smooth=1e-8):
    ''' Normalize each sequence between -1 and 1 '''
    return 2 * (seq - np.min(seq, axis=1)[None].T) / (np.max(seq, axis=1) - np.min(seq, axis=1) + smooth)[None].T - 1    

def preprocess_signal(data, filter_bandwidth, ch_idx=1, fs=fs):
    '''
    Applies preprocessing steps to recording, including band pass filter, polarity check and normalization
    data: ecg recording
    ch_idx: channel index to use for polarity check
    returns, data: preprocessed signal
    '''
    
    # Apply band pass filter
    if filter_bandwidth is not None:
        data = apply_filter(data, filter_bandwidth)

    data = normalize(data)    
    return data

def get_windows_padded(data, window_size=window_size, nb_windows=nb_windows):
    ''' 
    Split recording into random collection of windows. Applies padding to end of signal, if not long enough
    window_size: size of window to use
    nb_windows: number of windows
    returns, ecg_segs: torch tensor of random ecg windows [nb_windows x 12 channels x window_size]
    '''
    seq_len = data.shape[-1] # get the length of the ecg sequence
    
    # Add just enough padding to allow window
    pad = np.abs(np.min(seq_len - window_size, 0))
    if pad > 0:
        data = np.pad(data, ((0,0),(0,pad+1)))
        seq_len = data.shape[-1] # get the new length of the ecg sequence
    
    starts = np.random.randint(seq_len - window_size + 1, size=nb_windows) # get start indices of ecg segment        
    ecg_segs = torch.from_numpy(np.array([data[:,start:start+window_size] for start in starts])) 
    return ecg_segs

def extract_templates(signal, rpeaks, before=0.2, after=0.4, fs=500):
    # convert delimiters to samples
    before = int(before * fs)
    after = int(after * fs)

    # Sort R-Peaks in ascending order
    rpeaks = np.sort(rpeaks)

    # Get number of sample points in waveform
    length = len(signal)

    # Create empty list for templates
    templates = []

    # Create empty list for new rpeaks that match templates dimension
    rpeaks_new = np.empty(0, dtype=int)

    # Loop through R-Peaks
    for rpeak in rpeaks:

        # Before R-Peak
        a = rpeak - before
        if a < 0:
            continue

        # After R-Peak
        b = rpeak + after
        if b > length:
            break

        # Append template list
        templates.append(signal[a:b])

        # Append new rpeaks list
        rpeaks_new = np.append(rpeaks_new, rpeak)

    # Convert list to numpy array
    templates = np.array(templates).T

    return templates, rpeaks_new

def apply_filter(signal, filter_bandwidth, fs=500):
        # Calculate filter order
        order = int(0.3 * fs)
        # Filter signal
        signal, _, _ = filter_signal(signal=signal, ftype='FIR', band='bandpass',
                                     order=order, frequency=filter_bandwidth, 
                                     sampling_rate=fs)
        return signal         

def get_age(hdr):
    ''' Get age as integer from hdr '''
    res = re.search(r': (\d+)[\n]*', hdr)
    if res is None:
        return np.array([0])
    else:
        return np.array([float(res.group(1))])
