'''
If you set the environment variable DEBUG=1 before running your pipeline, you will get:

Full error tracebacks and raised exceptions for debugging.
Logging at the DEBUG level for all errors and warnings.
In normal mode (no DEBUG=1), errors and warnings are logged at the WARNING level and the pipeline continues as before.

To enable debug mode, run your script like this:
DEBUG=1 python preprocessing.py --output EDA_output
'''


import pandas as pd
import numpy as np
import os
import logging

# --- Canonical Helper Functions ---

# Debug mode: set via environment variable DEBUG=1
DEBUG = os.environ.get('DEBUG', '0') == '1'
if DEBUG:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.WARNING)
def check_interval(waves_signals: dict) -> pd.DataFrame:
    """
    Given a dictionary of ECG wave delineation points, validate and extract intervals where all required waves are present and in physiological order.
    Returns a DataFrame of valid intervals for feature extraction.
    """
    """
    Validates the physiological order of ECG wave components between R-peaks.
    Enhanced version with better error handling and validation.
    """
    keys = ['ECG_P_Onsets', 'ECG_P_Peaks', 'ECG_P_Offsets', 'ECG_Q_Peaks', 
            'ECG_R_Peaks', 'ECG_S_Peaks', 'ECG_T_Onsets', 'ECG_T_Peaks', 'ECG_T_Offsets']
    # Create dataframe with available keys only
    available_keys = [key for key in keys if key in waves_signals and waves_signals[key] is not None]
    if len(available_keys) < 5:  # Need at least P, Q, R, S, T peaks
        return pd.DataFrame()
    df_waves = pd.DataFrame({key: waves_signals[key] for key in available_keys})
    rows = df_waves.shape[0]
    if rows < 2:
        return pd.DataFrame()
    mask = np.zeros(rows, dtype=bool)
    for i in range(1, rows):
        try:
            start_interval = df_waves.loc[i-1, 'ECG_R_Peaks']
            end_interval = df_waves.loc[i, 'ECG_R_Peaks']
            # Slice the dataframe for the interval between two R-peaks
            sliced_df = df_waves[(df_waves['ECG_R_Peaks'] >= start_interval) & 
                               (df_waves['ECG_R_Peaks'] <= end_interval)]
            if sliced_df.shape[0] < 2:
                continue
            # Check for any missing values within the interval
            interval_points = sliced_df.iloc[0:2]
            if interval_points.isnull().sum().sum() == 0:
                # Check if the points are in ascending order (physiological order)
                points_values = interval_points.iloc[0].values
                if len(points_values) > 1 and np.all(np.diff(points_values) >= 0):
                    mask[i-1] = True
        except (IndexError, KeyError, ValueError):
            continue
    return df_waves[mask]

def ecg_signal_features(row: pd.Series, frequency: int, milliseconds: bool = True) -> dict:
    """
    Given a row of wave delineation points, compute canonical ECG interval and duration features.
    Converts sample indices to milliseconds (default) or seconds.
    Returns a dict of features, or NaN if not computable.
    """
    """
    Calculates durations and intervals from ECG wave points.
    Enhanced with better error handling and validation.
    """
    try:
        wave_keys = ['ECG_P_Onsets', 'ECG_P_Peaks', 'ECG_P_Offsets', 'ECG_Q_Peaks', 
                    'ECG_R_Peaks', 'ECG_S_Peaks', 'ECG_T_Onsets', 'ECG_T_Peaks', 'ECG_T_Offsets']
        waves = dict(zip(wave_keys, row)) if not isinstance(row, dict) else row
        features = {}
        # P wave duration
        if not (pd.isna(waves['ECG_P_Offsets']) or pd.isna(waves['ECG_P_Onsets'])):
            features['P_wave_duration'] = waves['ECG_P_Offsets'] - waves['ECG_P_Onsets']
        else:
            features['P_wave_duration'] = np.nan
        # PR interval
        if not (pd.isna(waves['ECG_Q_Peaks']) or pd.isna(waves['ECG_P_Onsets'])):
            features['PR_interval'] = waves['ECG_Q_Peaks'] - waves['ECG_P_Onsets']
        else:
            features['PR_interval'] = np.nan
        # PR segment
        if not (pd.isna(waves['ECG_Q_Peaks']) or pd.isna(waves['ECG_P_Offsets'])):
            features['PR_segment'] = waves['ECG_Q_Peaks'] - waves['ECG_P_Offsets']
        else:
            features['PR_segment'] = np.nan
        # QRS duration
        if not (pd.isna(waves['ECG_S_Peaks']) or pd.isna(waves['ECG_Q_Peaks'])):
            features['QRS_duration'] = waves['ECG_S_Peaks'] - waves['ECG_Q_Peaks']
        else:
            features['QRS_duration'] = np.nan
        # QT interval
        if not (pd.isna(waves['ECG_T_Offsets']) or pd.isna(waves['ECG_Q_Peaks'])):
            features['QT_interval'] = waves['ECG_T_Offsets'] - waves['ECG_Q_Peaks']
        else:
            features['QT_interval'] = np.nan
        # ST segment
        if not (pd.isna(waves['ECG_T_Onsets']) or pd.isna(waves['ECG_S_Peaks'])):
            features['ST_segment'] = waves['ECG_T_Onsets'] - waves['ECG_S_Peaks']
        else:
            features['ST_segment'] = np.nan
        # Convert to time units
        scale = 1000 / frequency if milliseconds else 1 / frequency
        for key in features:
            if not pd.isna(features[key]):
                features[key] = features[key] * scale
        return features
    except Exception:
        return {k: np.nan for k in ['P_wave_duration','PR_interval','PR_segment','QRS_duration','QT_interval','ST_segment']}

def st_slope(signal_df: pd.DataFrame, s_peak: int, t_onset: int) -> float:
    """
    Compute the slope of the ST segment between S peak and T onset.
    Returns NaN if indices are invalid or identical.
    """
    """Calculates the slope of the ST segment, handling division by zero."""
    if t_onset == s_peak:
        return np.nan
    try:
        return (signal_df.iloc[t_onset]['ECG_Clean'] - signal_df.iloc[s_peak]['ECG_Clean']) / (t_onset - s_peak)
    except Exception:
        return np.nan

def assess_ecg_quality(info, signals, frequency):
    """
    Assess ECG signal quality for HRV analysis.
    Checks for sufficient R peaks, valid RR intervals, and physiological variability.
    Returns True if quality is sufficient for HRV calculation, else False.
    """
    """
    Assess ECG signal quality for HRV analysis.
    Returns True if quality is sufficient for HRV calculation.
    """
    try:
        if 'ECG_R_Peaks' not in info or info['ECG_R_Peaks'] is None:
            return False
        r_peaks = info['ECG_R_Peaks']
        if len(r_peaks) < 5:
            return False
        rr_intervals = np.diff(r_peaks) / frequency * 1000
        valid_rr = rr_intervals[(rr_intervals >= 300) & (rr_intervals <= 2000)]
        if len(valid_rr) < 4:
            return False
        if np.std(valid_rr) < 1.0:
            return False
        return True
    except Exception:
        return False
def extract_lead_features(lead_signal: np.ndarray, frequency: int, channel=None) -> dict:
    """
    Extracts all canonical features from a single ECG lead:
    - Morphological (intervals/durations)
    - HRV (short-ECG supported only)
    - Wavelet features
    Returns a dict of features, or NaN-filled dict if extraction fails.
    """
    import neurokit2 as nk, pywt, numpy as np, pandas as pd
    features = {}
    try:
        msg = f"[DEBUG] extract_lead_features: channel={channel}, signal shape: {lead_signal.shape}, min: {np.min(lead_signal)}, max: {np.max(lead_signal)}, mean: {np.mean(lead_signal)}"
        logging.debug(msg)
        # Process with NeuroKit2
        try:
            ecg_signals, info = nk.ecg_process(lead_signal, sampling_rate=frequency)
            # Convert binary arrays in info to lists of sample indices
            binary_keys = [
                'ECG_P_Peaks', 'ECG_P_Onsets', 'ECG_P_Offsets',
                'ECG_Q_Peaks', 'ECG_R_Peaks', 'ECG_S_Peaks',
                'ECG_T_Peaks', 'ECG_T_Onsets', 'ECG_T_Offsets',
                'ECG_R_Onsets', 'ECG_R_Offsets'
            ]
            for key in binary_keys:
                if key in info and isinstance(info[key], (np.ndarray, list, pd.Series)):
                    arr = np.asarray(info[key])
                    # If it's a binary array (same length as signal), convert to indices
                    if arr.dtype in [np.int32, np.int64, np.float32, np.float64, bool] and arr.shape[0] == lead_signal.shape[0]:
                        info[key] = np.where(arr == 1)[0]
        except ZeroDivisionError:
            msg = "[ERROR] ZeroDivisionError in nk.ecg_process; filling with NaNs."
            logging.error(msg, exc_info=True)
            import traceback
            traceback.print_exc()
            info = {}
            ecg_signals = pd.DataFrame()
        except Exception as e:
            msg = f"[ERROR] Exception in nk.ecg_process: {e}; filling with NaNs."
            logging.error(msg, exc_info=True)
            import traceback
            traceback.print_exc()
            info = {}
            ecg_signals = pd.DataFrame()

        # Morphological features
        try:
            correct_waves = check_interval(info) if info else pd.DataFrame()
            morph_feature_keys = ['P_wave_duration', 'PR_interval', 'PR_segment', 'QRS_duration', 'QT_interval', 'ST_segment', 'ST_slope']
            stats = ['mean', 'std', 'min', 'max']
            if correct_waves.shape[0] == 0:
                for key in morph_feature_keys:
                    for stat in stats:
                        features[f'{key}_{stat}'] = np.nan
            else:
                morph_features = correct_waves.apply(lambda x: ecg_signal_features(x, frequency), axis=1, result_type='expand')
                morph_features['ST_slope'] = correct_waves.apply(lambda x: st_slope(ecg_signals, int(x['ECG_S_Peaks']), int(x['ECG_T_Onsets'])), axis=1)
                agg = morph_features.aggregate(['mean', 'std', 'min', 'max']).unstack().to_frame().T
                agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
                features.update(agg.iloc[0].to_dict())
        except Exception as e:
            msg = f"[ERROR] Exception in morphological feature extraction: {e}"
            logging.error(msg, exc_info=True)
            import traceback
            traceback.print_exc()
            morph_feature_keys = ['P_wave_duration', 'PR_interval', 'PR_segment', 'QRS_duration', 'QT_interval', 'ST_segment', 'ST_slope']
            stats = ['mean', 'std', 'min', 'max']
            for key in morph_feature_keys:
                for stat in stats:
                    features[f'{key}_{stat}'] = np.nan

        # HRV features
        hrv_feature_names = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_SDSD', 'HRV_CVNN', 'HRV_CVSD',
                            'HRV_MedianNN', 'HRV_MadNN', 'HRV_MCVNN', 'HRV_IQRNN', 'HRV_SDRMSSD', 'HRV_Prc20NN',
                            'HRV_Prc80NN', 'HRV_pNN50', 'HRV_pNN20', 'HRV_MinNN', 'HRV_MaxNN', 'HRV_HTI', 'HRV_TINN']
        is_good_quality = False
        try:
            if info and not ecg_signals.empty:
                is_good_quality = assess_ecg_quality(info, lead_signal.reshape(-1, 1), frequency)
            if is_good_quality:
                hrv_features_full = nk.hrv_time(ecg_signals, sampling_rate=frequency)
                for col in hrv_feature_names:
                    if col in hrv_features_full.columns:
                        features[col] = hrv_features_full.iloc[0][col]
                    else:
                        features[col] = np.nan
            else:
                for col in hrv_feature_names:
                    features[col] = np.nan
        except Exception as e:
            msg = f"[ERROR] Exception in HRV feature extraction: {e}"
            logging.error(msg, exc_info=True)
            import traceback
            traceback.print_exc()
            for col in hrv_feature_names:
                features[col] = np.nan

        # Wavelet features
        try:
            coeffs = pywt.wavedec(lead_signal, 'db4', level=4)
            cA4, cD4, cD3, cD2, cD1 = coeffs
            features.update({
                'wavelet_energy_d1': np.sum(np.square(cD1)),
                'wavelet_energy_d2': np.sum(np.square(cD2)),
                'wavelet_energy_d3': np.sum(np.square(cD3)),
                'wavelet_energy_d4': np.sum(np.square(cD4)),
                'wavelet_energy_a4': np.sum(np.square(cA4)),
                'wavelet_std_d1': np.std(cD1),
                'wavelet_std_d2': np.std(cD2),
                'wavelet_std_d3': np.std(cD3),
                'wavelet_std_d4': np.std(cD4),
                'wavelet_std_a4': np.std(cA4),
            })
        except Exception as e:
            msg = f"[ERROR] Exception in wavelet feature extraction: {e}"
            logging.error(msg, exc_info=True)
            import traceback
            traceback.print_exc()
            for key in ['wavelet_energy_d1', 'wavelet_energy_d2', 'wavelet_energy_d3', 'wavelet_energy_d4', 'wavelet_energy_a4',
                        'wavelet_std_d1', 'wavelet_std_d2', 'wavelet_std_d3', 'wavelet_std_d4', 'wavelet_std_a4']:
                features[key] = np.nan
    except Exception as e:
        msg = f"[ERROR] Exception in extract_lead_features outer block: {e}"
        logging.error(msg, exc_info=True)
        import traceback
        traceback.print_exc()
        # Fill all features with NaN in case of catastrophic failure
        morph_feature_keys = ['P_wave_duration', 'PR_interval', 'PR_segment', 'QRS_duration', 'QT_interval', 'ST_segment', 'ST_slope']
        stats = ['mean', 'std', 'min', 'max']
        for key in morph_feature_keys:
            for stat in stats:
                features[f'{key}_{stat}'] = np.nan
        hrv_feature_names = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_SDSD', 'HRV_CVNN', 'HRV_CVSD',
                            'HRV_MedianNN', 'HRV_MadNN', 'HRV_MCVNN', 'HRV_IQRNN', 'HRV_SDRMSSD', 'HRV_Prc20NN',
                            'HRV_Prc80NN', 'HRV_pNN50', 'HRV_pNN20', 'HRV_MinNN', 'HRV_MaxNN', 'HRV_HTI', 'HRV_TINN']
        for col in hrv_feature_names:
            features[col] = np.nan
        for key in ['wavelet_energy_d1', 'wavelet_energy_d2', 'wavelet_energy_d3', 'wavelet_energy_d4', 'wavelet_energy_a4',
                    'wavelet_std_d1', 'wavelet_std_d2', 'wavelet_std_d3', 'wavelet_std_d4', 'wavelet_std_a4']:
            features[key] = np.nan
    return features
    try:
        coeffs = pywt.wavedec(lead_signal, 'db4', level=4)
        cA4, cD4, cD3, cD2, cD1 = coeffs
        features.update({
            'wavelet_energy_d1': np.sum(np.square(cD1)),
            'wavelet_energy_d2': np.sum(np.square(cD2)),
            'wavelet_energy_d3': np.sum(np.square(cD3)),
            'wavelet_energy_d4': np.sum(np.square(cD4)),
            'wavelet_energy_a4': np.sum(np.square(cA4)),
            'wavelet_std_d1': np.std(cD1),
            'wavelet_std_d2': np.std(cD2),
            'wavelet_std_d3': np.std(cD3),
            'wavelet_std_d4': np.std(cD4),
            'wavelet_std_a4': np.std(cA4),
        })
    except Exception:
        for key in ['wavelet_energy_d1', 'wavelet_energy_d2', 'wavelet_energy_d3', 'wavelet_energy_d4', 'wavelet_energy_a4',
                    'wavelet_std_d1', 'wavelet_std_d2', 'wavelet_std_d3', 'wavelet_std_d4', 'wavelet_std_a4']:
            features[key] = np.nan
    return features
import argparse
import math
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from wfdb import processing

import torch
import torch.nn as nn
import torch.nn.functional as F
from helper_code import load_header, get_age, get_sex, get_label, load_signals, get_sampling_frequency

# Parameters
DEBUG = True
patience = 10
batch_size = 1 * torch.cuda.device_count() if torch.cuda.is_available() else 1
assert batch_size != 0, 'Batch size is 0 (most likely due to non-detected cuda'
window = 15*500
num_epochs = 30
dropout_rate = 0.2
deepfeat_sz = 64
padding = 'zero' # 'zero', 'qrs', or 'none'
fs = 500
filter_bandwidth = [3, 45]
polarity_check = []
model_name = 'ctn'

# Transformer parameters
d_model = 256   # embedding size
nhead = 8       # number of heads
d_ff = 2048     # feed forward layer size
num_layers = 8  # number of encoding layers

do_train = True
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if not torch.cuda.is_available:
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

ch_idx = 1
nb_demo = 2
nb_feats = 20
thrs_per_class = False
class_weights = None

classes = sorted(['270492004', '164889003', '164890007', '426627000', '713427006', 
                  '713426002', '445118002', '39732003', '164909002', '251146004', 
                  '698252002', '10370003', '284470004', '427172004', '164947007', 
                  '111975006', '164917005', '47665007', '59118001', '427393009', 
                  '426177001', '426783006', '427084000', '63593006', '164934002', 
                  '59931005', '17338001'])

char2dir = {
        'Q' : 'Training_2',
        'A' : 'Training_WFDB',
        'E' : 'WFDB',
        'S' : 'WFDB',
        'H' : 'WFDB',
        'I' : 'WFDB'
    }

# Load all features dataframe
# data_df = pd.read_csv('records_stratified_10_folds_v2.csv', index_col=0)
feats_dir = Path('feats/')
feats_files = list(feats_dir.glob(f'*/*all_feats_ch_{ch_idx}.zip'))
if feats_files:
    all_feats = pd.concat([pd.read_csv(f, index_col=0) for f in feats_files])
else:
    all_feats = pd.DataFrame()

leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
lead2idx = dict(zip(leads, range(len(leads))))

# dx_mapping_scored = pd.read_csv('eval/dx_mapping_scored.csv')
# snomed2dx = dict(zip(dx_mapping_scored['SNOMED CT Code'].values, dx_mapping_scored['Dx']))

beta = 2
num_classes = len(classes)

weights_file = 'eval/weights.csv'
normal_class = '426783006'
normal_index = classes.index(normal_class)
normal_lbl = [0. for i in range(num_classes)]
normal_lbl[normal_index] = 1.
equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

top_feats_path = Path('top_feats.npy')
if top_feats_path.exists():
    feature_names = list(np.load(top_feats_path))
    feature_names.remove('full_waveform_duration')
    feature_names.remove('Age')
    feature_names.remove('Gender_Male')
    # Compute top feature means and stds
    feats = all_feats[feature_names[:nb_feats]].values
    # First, convert any infs to nans
    feats[np.isinf(feats)] = np.nan
    # Store feature means and stds
    feat_means = np.nanmean(feats, axis=0)
    feat_stds = np.nanstd(feats, axis=0)
else:
    feature_names = []
    feats = np.array([])
    feat_means = np.array([])
    feat_stds = np.array([])



def extract_all_ecg_features_from_signal(signals, frequency, age=np.nan, is_male=np.nan, chagas=np.nan, record=None, channel_count=12):
    """
    Extracts ECG features from a raw ECG signal array (shape: [n_samples, n_leads]) and frequency.
    Optionally takes demographic info (age, is_male, chagas, record).
    Returns a flat dict with single-lead and aggregated features, matching extract_all_ecg_features output.
    """
    feature_dicts = []
    for channel in range(channel_count):
        try:
            lead_features = extract_lead_features(signals[:, channel], frequency, channel=channel)
            feature_dicts.append(lead_features)
        except Exception as e:
            msg = f"[ERROR] Feature extraction failed for channel {channel}: {e}"
            if DEBUG:
                logging.error(msg, exc_info=True)
                import traceback
                traceback.print_exc()
            else:
                logging.warning(msg)
            continue

    if not feature_dicts:
        msg = f"[ERROR] Feature extraction failed for all channels in record {record}. Raising exception for debug."
        if DEBUG:
            logging.error(msg)
            import traceback
            traceback.print_stack()
        raise RuntimeError(msg)

    single_lead_features = feature_dicts[0].copy()
    single_lead_features_renamed = {f"{k}_single_lead": v for k, v in single_lead_features.items()}

    features_df = pd.DataFrame(feature_dicts)
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    means = features_df[numeric_cols].mean(axis=0)
    stds = features_df[numeric_cols].std(axis=0)
    aggregated_features = {}
    for col in numeric_cols:
        aggregated_features[f"{col}_mean_all_leads"] = means[col]
        aggregated_features[f"{col}_std_all_leads"] = stds[col]

    combined = {}
    combined.update(single_lead_features_renamed)
    combined.update(aggregated_features)
    combined['age'] = age
    combined['is_male'] = is_male
    combined['chagas'] = chagas
    combined['record'] = record
    return combined

def extract_all_ecg_features_from_path(record_path, header_path=None, channel_count=12):
    """
    Loads metadata and signals for a record, then iterates through all leads, calling extract_lead_features.
    Returns the first successful feature dict, or a NaN-filled dict if all channels fail.
    Handles all error cases robustly and logs issues.
    """
    try:
        if header_path is None:
            header_path = record_path + '.hea'
        header = load_header(header_path)
        age = get_age(header)
        sex = get_sex(header)
        is_male = 1 if str(sex).strip().lower() == 'male' else 0 if str(sex).strip().lower() == 'female' else np.nan
        chagas = get_label(header)
    except Exception as e:
        msg = f"[ERROR] Demographic extraction failed for {record_path}: {e}"
        if DEBUG:
            logging.error(msg, exc_info=True)
        else:
            logging.warning(msg)
        age, is_male, chagas = np.nan, np.nan, np.nan
    try:
        signals, _ = load_signals(record_path)
        if DEBUG:
            logging.debug(f"[DEBUG] Loaded signals for {record_path}: type={type(signals)}, shape={getattr(signals, 'shape', None)}, dtype={getattr(signals, 'dtype', None)}")
            if hasattr(signals, 'shape') and signals.shape[0] > 0 and signals.shape[1] > 0:
                logging.debug(f"[DEBUG] First 5 samples, first channel: {signals[:5,0]}")
                logging.debug(f"[DEBUG] First 5 samples, all channels: {signals[:5,:]}")
            else:
                logging.debug(f"[DEBUG] signals appears empty or malformed: {signals}")
    except Exception as e:
        msg = f"[ERROR] Signal loading failed for {record_path}: {e}"
        if DEBUG:
            logging.error(msg, exc_info=True)
        else:
            logging.warning(msg)
        if DEBUG:
            pass
    # No debug logging for lead_signal here; handled in extract_lead_features
    # Actually run feature extraction and raise if all channels fail
    try:
        frequency = get_sampling_frequency(header) if 'header' in locals() else 500
        return extract_all_ecg_features_from_signal(signals, frequency, age=age, is_male=is_male, chagas=chagas, record=record_path, channel_count=signals.shape[1])
    except Exception as e:
        msg = f"[ERROR] Feature extraction failed for all channels in record {record_path}: {e}"
        if DEBUG:
            logging.error(msg, exc_info=True)
            import traceback
            traceback.print_exc()
        raise

def extract_all_ecg_features(record_path, header_path=None, channel_count=12):
    """
    Backward-compatible alias for extract_all_ecg_features_from_path.
    """
    return extract_all_ecg_features_from_path(record_path, header_path=header_path, channel_count=channel_count)

def assess_ecg_quality(info, signals, frequency):
    """
    Assess ECG signal quality for HRV analysis.
    Returns True if quality is sufficient for HRV calculation.
    
     checks:

    Presence of R peaks in the ECG signal
    Minimum of 5 R peaks for meaningful HRV analysis
    R-R intervals within reasonable physiological range (300-2000 ms)
    Sufficient R-R interval variability (>1ms standard deviation)
    """
    try:
        # Check if R peaks were detected
        if 'ECG_R_Peaks' not in info or info['ECG_R_Peaks'] is None:
            return False
        
        r_peaks = info['ECG_R_Peaks']
        
        # Need at least 5 R peaks for meaningful HRV analysis
        if len(r_peaks) < 5:
            return False
        
        # Calculate R-R intervals in milliseconds
        rr_intervals = np.diff(r_peaks) / frequency * 1000
        
        # Check for reasonable R-R interval range (300-2000 ms)
        # This filters out obvious detection errors
        valid_rr = rr_intervals[(rr_intervals >= 300) & (rr_intervals <= 2000)]
        
        # Need at least 4 valid R-R intervals (from 5 R peaks)
        if len(valid_rr) < 4:
            return False
        
        # Check if R-R intervals have reasonable variability
        # If all intervals are identical, it's likely a detection artifact
        if np.std(valid_rr) < 1.0:  # Less than 1ms std deviation
            return False
        
        return True
    
    except Exception:
        return False


def check_interval(waves_signals: dict) -> pd.DataFrame:
    """Creates a DataFrame from ECG wave signals and filters for valid R-R intervals."""
    keys = ['ECG_P_Onsets', 'ECG_P_Peaks', 'ECG_P_Offsets', 'ECG_Q_Peaks', 'ECG_R_Peaks', 'ECG_S_Peaks', 'ECG_T_Onsets', 'ECG_T_Peaks', 'ECG_T_Offsets']
    df_waves = pd.DataFrame({key: waves_signals[key] for key in keys})
    rows = df_waves.shape[0]
    mask = np.zeros(rows, dtype=bool)
    for i in range(1, rows):
        start_interval = df_waves.loc[i-1, 'ECG_R_Peaks']
        end_interval = df_waves.loc[i, 'ECG_R_Peaks']
        # Slice the dataframe for the current R-R interval
        sliced_df = df_waves.iloc[i-1:i+1]
        
        # Condition 1: Check if there are any missing wave delineations within the interval.
        first_condition = sliced_df.isnull().sum().sum() == 0
        
        # Condition 2: Check if the wave points are in the correct chronological order.
        if first_condition:
            sorted_points = sliced_df.iloc[0].sort_values(ascending=True)
            second_condition = keys == sorted_points.index.tolist()
            if second_condition:
                mask[i-1] = True # Mark the start of the valid interval as True
    
    return df_waves[mask]

def ecg_signal_features(row: pd.Series, frequency: int, milliseconds: bool = True) -> dict:
    """Computes various ECG interval and duration features from a row of wave points."""
    (ECG_P_Peaks, ECG_P_Onsets, ECG_P_Offsets, ECG_Q_Peaks, ECG_S_Peaks, 
     ECG_T_Peaks, ECG_T_Onsets, ECG_T_Offsets, ECG_R_Peaks) = row
     
    P_wave_duration = (ECG_P_Offsets - ECG_P_Onsets)
    PR_interval = (ECG_Q_Peaks - ECG_P_Onsets)
    PR_segment = (ECG_Q_Peaks - ECG_P_Offsets)
    QRS_duration = (ECG_S_Peaks - ECG_Q_Peaks)
    QT_interval = (ECG_T_Offsets - ECG_Q_Peaks)
    ST_segment = (ECG_T_Onsets - ECG_S_Peaks)
    
    # Convert from samples to time (milliseconds or seconds)
    conversion_factor = 1000 / frequency if milliseconds else 1 / frequency
    
    return {
        'P_wave_duration': P_wave_duration * conversion_factor,
        'PR_interval': PR_interval * conversion_factor,
        'PR_segment': PR_segment * conversion_factor,
        'QRS_duration': QRS_duration * conversion_factor,
        'QT_interval': QT_interval * conversion_factor,
        'ST_segment': ST_segment * conversion_factor
    }

def st_slope(signal_df: pd.DataFrame, s_peak: int, t_onset: int) -> float:
    """Calculates the slope of the ST segment, handling division by zero."""
    # Robustly compute ST slope, handling divide by zero, NaN, and index errors
    try:
        # Check for NaN or non-integer indices
        if pd.isna(s_peak) or pd.isna(t_onset):
            return np.nan
        if not isinstance(s_peak, (int, np.integer)) or not isinstance(t_onset, (int, np.integer)):
            return np.nan
        # Prevent division by zero
        if t_onset == s_peak:
            return np.nan
        # Check index bounds
        if s_peak < 0 or t_onset < 0 or s_peak >= len(signal_df) or t_onset >= len(signal_df):
            return np.nan
        # Compute slope
        return (signal_df.iloc[t_onset]['ECG_Clean'] - signal_df.iloc[s_peak]['ECG_Clean']) / (t_onset - s_peak)
    except Exception:
        return np.nan
