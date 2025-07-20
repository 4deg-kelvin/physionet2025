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
debug = False
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



def extract_all_ecg_features(record_path, header_path=None, channel_count=12):
    """
    Extracts demographic, morphological, HRV, and wavelet features from a WFDB record.
    Args:
        record_path (str): Path to the record (without extension)
        header_path (str, optional): Path to the header file. If None, uses record_path + '.hea'
        channel_count (int): Number of ECG channels to try (default: 12)
    Returns:
        dict: All extracted features, or None if extraction fails
    """
    import wfdb, neurokit2 as nk, pywt, numpy as np, pandas as pd, scipy.stats as stats
    # Demographic features
    try:
        if header_path is None:
            header_path = record_path + '.hea'
        header = load_header(header_path)
        age = get_age(header)
        is_male = 1 if get_sex(header) == 'Male' else 0
        chagas = get_label(header)
    except Exception as e:
        print(f"[ERROR] Demographic extraction failed for {record_path}: {e}")
        age, is_male, chagas = np.nan, np.nan, np.nan
    # Load signals
    try:
        signals, _ = load_signals(record_path)
    except Exception as e:
        print(f"[ERROR] Signal loading failed for {record_path}: {e}")
        return None
    # Get sampling frequency
    try:
        frequency = get_sampling_frequency(header)
    except Exception as e:
        print(f"[WARN] Sampling frequency extraction failed for {record_path}: {e}. Using default 500Hz.")
        frequency = 500
    # Try all channels for feature extraction
    for channel in range(channel_count):
        try:
            print(f"[DEBUG] Processing {record_path} channel {channel}")
            try:
                ecg_signals, info = nk.ecg_process(signals[:, channel], sampling_rate=frequency)
                print(f"[DEBUG] nk.ecg_process succeeded for {record_path} channel {channel}")
                print(f"[DEBUG] ecg_signals shape: {ecg_signals.shape}")
                print(f"[DEBUG] info keys: {list(info.keys())}")
            except Exception as e:
                print(f"[ERROR] nk.ecg_process failed for {record_path} channel {channel}: {e}")
                continue
            # Morphological features
            correct_waves = check_interval(info)
            print(f"[DEBUG] correct_waves shape: {correct_waves.shape}")
            if correct_waves.shape[0] == 0:
                print(f"[WARN] No valid wave intervals for {record_path} channel {channel}")
                # Create a DataFrame with NaNs for all expected morph features
                morph_feature_keys = ['P_wave_duration', 'PR_interval', 'PR_segment', 'QRS_duration', 'QT_interval', 'ST_segment', 'ST_slope']
                stats = ['mean', 'std', 'min', 'max']
                nan_dict = {f'{key}_{stat}': [np.nan] for key in morph_feature_keys for stat in stats}
                agg_morph_features = pd.DataFrame(nan_dict)
            else:
                print(f"[DEBUG] correct_waves head: {correct_waves.head()}")
                morph_features = correct_waves.apply(lambda x: ecg_signal_features(x, frequency), axis=1, result_type='expand')
                print(f"[DEBUG] morph_features shape: {morph_features.shape}")
                morph_features['ST_slope'] = correct_waves.apply(lambda x: st_slope(ecg_signals, int(x['ECG_S_Peaks']), int(x['ECG_T_Onsets'])), axis=1)
                print(f"[DEBUG] morph_features with ST_slope head: {morph_features.head()}")
                # Only aggregate if morph_features is not empty
                if morph_features.shape[0] == 0:
                    morph_feature_keys = ['P_wave_duration', 'PR_interval', 'PR_segment', 'QRS_duration', 'QT_interval', 'ST_segment', 'ST_slope']
                    stats = ['mean', 'std', 'min', 'max']
                    nan_dict = {f'{key}_{stat}': [np.nan] for key in morph_feature_keys for stat in stats}
                    agg_morph_features = pd.DataFrame(nan_dict)
                else:
                    agg_morph_features = morph_features.aggregate(['mean', 'std', 'min', 'max']).unstack().to_frame().T
                    agg_morph_features.columns = ['_'.join(col).strip() for col in agg_morph_features.columns.values]
            # HRV features
            try:
                # Assess signal quality before attempting HRV calculation
                is_good_quality = assess_ecg_quality(info, signals, frequency)
                
                if is_good_quality:
                    hrv_features = nk.hrv_time(ecg_signals, sampling_rate=frequency)
                    print(f"[DEBUG] hrv_features shape: {hrv_features.shape}")
                else:
                    # Create NaN HRV features for poor quality signals
                    hrv_feature_names = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_SDANN1', 'HRV_SDNNI1', 'HRV_SDANN2', 'HRV_SDNNI2', 
                                       'HRV_SDANN5', 'HRV_SDNNI5', 'HRV_RMSSD', 'HRV_SDSD', 'HRV_CVNN', 'HRV_CVSD', 
                                       'HRV_MedianNN', 'HRV_MadNN', 'HRV_MCVNN', 'HRV_IQRNN', 'HRV_SDRMSSD', 'HRV_Prc20NN', 
                                       'HRV_Prc80NN', 'HRV_pNN50', 'HRV_pNN20', 'HRV_MinNN', 'HRV_MaxNN', 'HRV_HTI', 'HRV_TINN']
                    hrv_features = pd.DataFrame({col: [np.nan] for col in hrv_feature_names})
                    print(f"[DEBUG] Poor quality signal - created NaN HRV features")
            except Exception as e:
                print(f"[ERROR] HRV extraction failed for {record_path} channel {channel}: {e}")
                hrv_feature_names = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_SDANN1', 'HRV_SDNNI1', 'HRV_SDANN2', 'HRV_SDNNI2', 
                                   'HRV_SDANN5', 'HRV_SDNNI5', 'HRV_RMSSD', 'HRV_SDSD', 'HRV_CVNN', 'HRV_CVSD', 
                                   'HRV_MedianNN', 'HRV_MadNN', 'HRV_MCVNN', 'HRV_IQRNN', 'HRV_SDRMSSD', 'HRV_Prc20NN', 
                                   'HRV_Prc80NN', 'HRV_pNN50', 'HRV_pNN20', 'HRV_MinNN', 'HRV_MaxNN', 'HRV_HTI', 'HRV_TINN']
                hrv_features = pd.DataFrame({col: [np.nan] for col in hrv_feature_names})
            # Wavelet features
            try:
                coeffs = pywt.wavedec(signals[:, channel], 'db4', level=4)
                print(f"[DEBUG] wavelet coeffs lengths: {[len(c) for c in coeffs]}")
                cA4, cD4, cD3, cD2, cD1 = coeffs
                wavelet_features = {
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
                }
                wavelet_df = pd.DataFrame([wavelet_features])
            except Exception as e:
                print(f"[ERROR] Wavelet extraction failed for {record_path} channel {channel}: {e}")
                wavelet_df = pd.DataFrame()
            # Combine all features, fill missing with NaN
            print(f"[DEBUG] agg_morph_features shape: {agg_morph_features.shape if 'agg_morph_features' in locals() else 'N/A'}")
            print(f"[DEBUG] hrv_features shape: {hrv_features.shape}")
            print(f"[DEBUG] wavelet_df shape: {wavelet_df.shape if 'wavelet_df' in locals() else 'N/A'}")
            
            # Ensure all DataFrames have exactly 1 row for concatenation
            dataframes_to_concat = []
            
            # Reset index and take first row only to ensure single-row dataframes
            if not agg_morph_features.empty:
                agg_morph_features = agg_morph_features.iloc[[0]].reset_index(drop=True)
                dataframes_to_concat.append(agg_morph_features)
                
            if not hrv_features.empty:
                hrv_features = hrv_features.iloc[[0]].reset_index(drop=True) 
                dataframes_to_concat.append(hrv_features)
                
            if not wavelet_df.empty:
                wavelet_df = wavelet_df.iloc[[0]].reset_index(drop=True)
                dataframes_to_concat.append(wavelet_df)
            
            # Concatenate only non-empty dataframes
            if dataframes_to_concat:
                if debug:
                    for i, df in enumerate(dataframes_to_concat):
                        print(f"[DEBUG] DataFrame {i} shape: {df.shape}")
                        print(f"[DEBUG] DataFrame {i} head:\n{df.head()}")
                try:
                    combined_features = pd.concat(dataframes_to_concat, axis=1)
                except Exception as e:
                    if debug:
                        print(f"[DEBUG] Concatenation error: {e}")
                        print(f"[DEBUG] Number of dataframes: {len(dataframes_to_concat)}")
                        for i, df in enumerate(dataframes_to_concat):
                            print(f"[DEBUG] DataFrame {i}: shape={df.shape}, columns={list(df.columns)}")
                    raise e
            else:
                combined_features = pd.DataFrame(index=[0])  # Empty dataframe with single row
                
            combined_features['age'] = age
            combined_features['is_male'] = is_male
            combined_features['chagas'] = chagas
            combined_features['record'] = os.path.basename(record_path)
            
            # Add missing expected columns - but only if they don't already exist
            morph_feature_keys = ['P_wave_duration', 'PR_interval', 'PR_segment', 'QRS_duration', 'QT_interval', 'ST_segment', 'ST_slope']
            morph_stats = ['mean', 'std', 'min', 'max']
            # Use the actual HRV feature names returned by neurokit2
            hrv_keys = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_SDANN1', 'HRV_SDNNI1', 'HRV_SDANN2', 'HRV_SDNNI2', 
                       'HRV_SDANN5', 'HRV_SDNNI5', 'HRV_RMSSD', 'HRV_SDSD', 'HRV_CVNN', 'HRV_CVSD', 
                       'HRV_MedianNN', 'HRV_MadNN', 'HRV_MCVNN', 'HRV_IQRNN', 'HRV_SDRMSSD', 'HRV_Prc20NN', 
                       'HRV_Prc80NN', 'HRV_pNN50', 'HRV_pNN20', 'HRV_MinNN', 'HRV_MaxNN', 'HRV_HTI', 'HRV_TINN']
            wavelet_keys = ['wavelet_energy_d1', 'wavelet_energy_d2', 'wavelet_energy_d3', 'wavelet_energy_d4', 'wavelet_energy_a4',
                            'wavelet_std_d1', 'wavelet_std_d2', 'wavelet_std_d3', 'wavelet_std_d4', 'wavelet_std_a4']
            
            # Build expected columns list - ensure no duplicates
            expected_cols = ['age', 'is_male', 'chagas', 'record']
            expected_cols.extend([f'{key}_{stat}' for key in morph_feature_keys for stat in morph_stats])
            expected_cols.extend(hrv_keys)
            expected_cols.extend(wavelet_keys)
            
            # Only add missing columns, avoiding duplicates
            for col in expected_cols:
                if col not in combined_features.columns:
                    combined_features[col] = np.nan
                    
            return combined_features.iloc[0].to_dict()
        except Exception as e:
            print(f"[ERROR] Feature extraction failed for {record_path} channel {channel}: {e}")
            import traceback
            traceback.print_exc()
            continue
    # If all channels fail, return NaNs for all features
    print(f"[FAIL] All channels failed for {record_path}")
    nan_features = {col: np.nan for col in ['age', 'is_male', 'chagas', 'record']}
    # Add expected morph, hrv, wavelet columns as NaN
    morph_feature_keys = ['P_wave_duration', 'PR_interval', 'PR_segment', 'QRS_duration', 'QT_interval', 'ST_segment', 'ST_slope']
    for key in morph_feature_keys:
        for stat in ['mean', 'std', 'min', 'max']:
            nan_features[f'{key}_{stat}'] = np.nan
    # Add placeholder HRV and wavelet columns
    hrv_keys = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_SDANN1', 'HRV_SDNNI1', 'HRV_SDANN2', 'HRV_SDNNI2', 
               'HRV_SDANN5', 'HRV_SDNNI5', 'HRV_RMSSD', 'HRV_SDSD', 'HRV_CVNN', 'HRV_CVSD', 
               'HRV_MedianNN', 'HRV_MadNN', 'HRV_MCVNN', 'HRV_IQRNN', 'HRV_SDRMSSD', 'HRV_Prc20NN', 
               'HRV_Prc80NN', 'HRV_pNN50', 'HRV_pNN20', 'HRV_MinNN', 'HRV_MaxNN', 'HRV_HTI', 'HRV_TINN']
    for key in hrv_keys:
        nan_features[key] = np.nan
    wavelet_keys = ['wavelet_energy_d1', 'wavelet_energy_d2', 'wavelet_energy_d3', 'wavelet_energy_d4', 'wavelet_energy_a4',
                    'wavelet_std_d1', 'wavelet_std_d2', 'wavelet_std_d3', 'wavelet_std_d4', 'wavelet_std_a4']
    for key in wavelet_keys:
        nan_features[key] = np.nan
    return nan_features

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
    # Prevent division by zero if s_peak and t_onset are the same point
    if t_onset == s_peak:
        return np.nan
    return (signal_df.iloc[t_onset]['ECG_Clean'] - signal_df.iloc[s_peak]['ECG_Clean']) / (t_onset - s_peak)
