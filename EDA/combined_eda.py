
# This script performs EDA on ECG data, extracting various features.
#
# Extracted Features:
#
# Basic Features:
# - age: Age of the patient.
# - is_male: Gender of the patient (1 for Male, 0 for Female).
# - chagas: Presence of Chagas disease (label).
#
# Advanced ECG Features (from NeuroKit2, aggregated across all leads):
# - Heart Rate:
#   - heart_rate_mean: Mean heart rate.
#   - heart_rate_min: Minimum heart rate.
#   - heart_rate_max: Maximum heart rate.
# - HRV Time-Domain Features:
#   - SDNN, RMSSD, pNN50, etc. (various standard deviation and interval-based metrics).
# - HRV Frequency-Domain Features:
#   - Various frequency band powers (e.g., VLF, LF, HF).
# - HRV Non-Linear Features:
#   - Metrics like Sample Entropy (SampEn), Approximate Entropy (ApEn), etc.
# - Morphological Features:
#   - p_wave_amplitude: Mean amplitude of the P-wave.
#   - qrs_amplitude: Mean amplitude of the QRS complex.
#   - t_wave_amplitude: Mean amplitude of the T-wave.
# - ECG Wave Durations and Intervals:
#   - pr_interval: Mean PR interval.
#   - qrs_duration: Mean QRS duration.
#   - qt_interval: Mean QT interval.
#   - qtc_interval: Mean corrected QT interval.
# - Statistical Features:
#   - signal_skewness: Skewness of the ECG signal.
#   - signal_kurtosis: Kurtosis of the ECG signal.
#
# Wavelet-Based Features:
# - wavelet_energy_d{1-4}: Energy of the detail coefficients at levels 1-4.
# - wavelet_energy_a4: Energy of the approximation coefficients at level 4.
# - wavelet_std_d{1-4}: Standard deviation of the detail coefficients at levels 1-4.
# - wavelet_std_a4: Standard deviation of the approximation coefficients at level 4.

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import scipy.stats as stats
import wfdb
import neurokit2 as nk
from joblib import Parallel, delayed
from tqdm import tqdm
import orjson
import pywt

from helper_code import *

# --- Utility Functions ---

def get_json(path):
    """Loads a JSON file."""
    with open(path, 'r') as f:
        return orjson.loads(f.read())

def load_ecg_data(exam_id, search_directory):
    """
    Searches for and loads ECG data (.mat and .hea files).
    """
    mat_filename = f"{exam_id}.mat"
    hea_filename = f"{exam_id}.hea"
    
    for dirpath, _, filenames in os.walk(search_directory):
        if mat_filename in filenames and hea_filename in filenames:
            mat_filepath = os.path.join(dirpath, mat_filename)
            hea_filepath = os.path.join(dirpath, hea_filename)
            try:
                data, header = load_challenge_data(hea_filepath)
                return data, header
            except Exception as e:
                print(f"Error loading {mat_filepath}: {e}")
                return None, None
    return None, None

# --- Feature Extraction ---

def extract_basic_features(record_path):
    """
    Extracts basic demographic and clinical features from the header file.
    """
    try:
        header = load_header(record_path + '.hea')
        age = get_age(header)
        is_male = 1 if get_sex(header) == 'Male' else 0
        chagas = get_label(header)
        return {'age': age, 'is_male': is_male, 'chagas': chagas}
    except Exception:
        return None

def extract_advanced_ecg_features(signal, sampling_rate):
    """
    Extracts morphological and HRV features from a single ECG signal.
    Focuses on features robust for short recordings.
    """
    features = {}
    # Use Lead II for analysis, which is typically the second column in 12-lead ECGs
    if signal.ndim > 1 and signal.shape[1] > 1:
        lead_signal = signal[:, 1]
    else:
        lead_signal = signal.flatten()

    try:
        # Process the ECG signal with NeuroKit2
        signals, info = nk.ecg_process(lead_signal, sampling_rate=sampling_rate)
        peaks = info["ECG_R_Peaks"]

        # If not enough peaks are found, return empty features
        if len(peaks) < 4: # Need at least a few beats for meaningful stats
            return {}, []

        # --- Morphological Features (mean and std over all detected beats) ---
        morph_features = [
            'ECG_P_Duration', 'ECG_P_Amplitude', 'ECG_QRS_Duration', 
            'ECG_QRS_Amplitude', 'ECG_T_Duration', 'ECG_T_Amplitude', 'ECG_QT_Interval'
        ]
        for feature in morph_features:
            if feature in signals:
                feature_values = signals[feature].dropna()
                if not feature_values.empty:
                    features[f'{feature}_mean'] = feature_values.mean()
                    features[f'{feature}_std'] = feature_values.std()
                else: # Handle case where no valid values exist for a feature
                    features[f'{feature}_mean'] = np.nan
                    features[f'{feature}_std'] = np.nan

        # --- HRV Features (computed only if enough peaks are detected) ---
        # Based on user feedback, these calculations can fail on very short signals.
        # We will only compute them if more than 20 RR intervals (21 peaks) are found.
        if len(peaks) > 20:
            try:
                # Add a check for variability in peaks to prevent errors
                if np.std(np.diff(peaks)) > 0:
                    # Time-Domain
                    hrv_time = nk.hrv_time(peaks, sampling_rate=sampling_rate, show=False)
                    features['HRV_RMSSD'] = hrv_time['HRV_RMSSD'].iloc[0]
                    features['HRV_MeanNN'] = hrv_time['HRV_MeanNN'].iloc[0]
                    features['HRV_pNN50'] = hrv_time['HRV_pNN50'].iloc[0]

                    # Non-Linear (Poincaré)
                    hrv_poincare_df = nk.hrv_poincare(peaks, sampling_rate=sampling_rate, show=False)
                    features['HRV_SD1'] = hrv_poincare_df['HRV_SD1'].iloc[0]
                else:
                    raise ValueError("No variability in RR-intervals.")
            except Exception:
                features['HRV_RMSSD'] = np.nan
                features['HRV_MeanNN'] = np.nan
                features['HRV_pNN50'] = np.nan
                features['HRV_SD1'] = np.nan
        else:
            # If not enough peaks, fill with NaN
            features['HRV_RMSSD'] = np.nan
            features['HRV_MeanNN'] = np.nan
            features['HRV_pNN50'] = np.nan
            features['HRV_SD1'] = np.nan

        # --- Overall Heart Rate ---
        if 'ECG_Rate' in signals and not signals['ECG_Rate'].dropna().empty:
            features['Heart_Rate_Mean'] = signals['ECG_Rate'].mean()
        else:
            features['Heart_Rate_Mean'] = np.nan
        
        rr_interval_lengths = [len(peaks)]

    except Exception as e:
        # Silently fail for signals that are too noisy or short to be processed
        return {}, []

    return features, rr_interval_lengths

def extract_wavelet_features(record_path):
    """
    Extracts wavelet-based features from the ECG signal.
    """
    try:
        signals, _ = wfdb.rdsamp(record_path)
        ecg_signal = signals[:, 0] # Use lead I for wavelet features

        # Wavelet decomposition
        coeffs = pywt.wavedec(ecg_signal, 'db4', level=4)
        cA4, cD4, cD3, cD2, cD1 = coeffs

        features = {
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
        return features
    except Exception:
        return None

def process_record(record_path):
    """
    Main processing function for a single record.
    """
    try:
        # Basic features from header
        basic_features = extract_basic_features(record_path)
        if not basic_features:
            return None, []

        # Load signal for advanced feature extraction
        signal, fields = wfdb.rdsamp(record_path)
        sampling_rate = fields['fs']
        
        if signal is None or signal.size == 0:
            return None, []

        # Advanced and wavelet features
        advanced_features, rr_interval_lengths = extract_advanced_ecg_features(signal, sampling_rate)
        wavelet_features = extract_wavelet_features(record_path)

        # Combine all features
        all_features = basic_features
        if advanced_features:
            all_features.update(advanced_features)
        if wavelet_features:
            all_features.update(wavelet_features)
        
        all_features['record'] = os.path.basename(record_path)
        return all_features, rr_interval_lengths
    except Exception as e:
        # print(f"Error in process_record for {os.path.basename(record_path)}: {e}") # Uncomment for debugging
        return None, []

# --- Main Execution ---

def main():
    """
    Main function to run the EDA and feature extraction pipeline.
    """
    # --- Data Loading (from xgboost_training.ipynb) ---
    data_dir = 'training_data/samitrop/'
    print(f"Searching for .hea files in: {data_dir}")
    
    hea_files = glob.glob(os.path.join(data_dir, '**', '*.hea'), recursive=True)
    record_stems = [f[:-4] for f in hea_files]
    print(f"Found {len(record_stems)} records to process.")

    # --- Feature Extraction (with new wavelet features) ---
    all_rr_interval_lengths_collected = []
    processed_results = []

    with Parallel(n_jobs=-1) as parallel:
        for res, rr_lengths in tqdm(parallel(delayed(process_record)(record) for record in record_stems), total=len(record_stems)):
            if res is not None:
                processed_results.append(res)
                all_rr_interval_lengths_collected.extend(rr_lengths)

    df = pd.DataFrame(processed_results)
    
    # Data Cleaning
    df.dropna(subset=['chagas'], inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    print("Feature extraction complete.")
    print(f"Dataset shape: {df.shape}")
    print("Columns:", df.columns.tolist())

    # --- EDA (from notebook_template.ipynb) ---
    
    # 1. Feature Distribution Analysis
    plt.figure(figsize=(12, 5))
    sns.histplot(data=df, x='age', hue='is_male', multiple='stack', bins=30)
    plt.title('Age Distribution by Gender')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.legend(title='Gender', labels=['Female', 'Male'])
    plt.savefig('EDA/age_distribution.png')
    plt.show()

    # 2. Correlation Heatmap
    plt.figure(figsize=(20, 15))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=False, cmap='viridis')
    plt.title('Feature Correlation Heatmap')
    plt.savefig('EDA/correlation_heatmap.png')
    plt.show()

    # 3. Boxplots for key features
    key_features = ['age', 'wavelet_energy_d1', 'wavelet_energy_a4']
    
    plt.figure(figsize=(18, 4))
    for i, feature in enumerate(key_features):
        if feature in df.columns:
            plt.subplot(1, 3, i + 1)
            sns.boxplot(x='chagas', y=feature, data=df)
            plt.title(f'{feature} by Chagas Status')
    plt.tight_layout()
    plt.savefig('EDA/feature_boxplots.png')
    plt.show()

    # Save the processed data
    df.to_csv('EDA/combined_features.csv', index=False)
    print("EDA plots and processed data saved in the EDA folder.")

    # --- RR Interval Analysis ---
    if all_rr_interval_lengths_collected:
        avg_rr_length = np.mean(all_rr_interval_lengths_collected)
        zero_rr_count = all_rr_interval_lengths_collected.count(0)
        
        print(f"\nRR Interval Analysis:")
        print(f"Average length of rr_intervals: {avg_rr_length:.2f}")
        print(f"Number of rr_intervals with length 0: {zero_rr_count}")

        plt.figure(figsize=(10, 6))
        sns.histplot(all_rr_interval_lengths_collected, bins=50, kde=True)
        plt.title('Distribution of RR Interval Lengths')
        plt.xlabel('RR Interval Length')
        plt.ylabel('Frequency')
        plt.savefig('EDA/rr_interval_length_distribution.png')
        plt.show()
    else:
        print("\nNo RR interval data collected for analysis.")

if __name__ == '__main__':
    main()
