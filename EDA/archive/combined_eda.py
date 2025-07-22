
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

# Suppress common warnings for cleaner EDA output
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

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

def extract_advanced_ecg_features(signal, sampling_rate, failure_counters):
    """
    Extracts a comprehensive set of ECG features with robust, modular error handling.
    """
    features = {}
    rr_interval_lengths = []

    # --- Single-Lead Feature Extraction (using the first available lead) ---
    try:
        lead_signal = signal[:, 0] # Default to first lead
        signals, info = nk.ecg_process(lead_signal, sampling_rate=sampling_rate)
        peaks = info["ECG_R_Peaks"]
        rr_interval_lengths.append(len(peaks))

        if len(peaks) >= 4:
            # Morphological Features
            morph_features = [
                'ECG_P_Duration', 'ECG_P_Amplitude', 'ECG_QRS_Duration', 
                'ECG_QRS_Amplitude', 'ECG_T_Duration', 'ECG_T_Amplitude', 
                'ECG_QT_Interval', 'ECG_QTc_Interval'
            ]
            for feature in morph_features:
                if feature in signals:
                    feature_values = signals[feature].dropna()
                    if not feature_values.empty:
                        features[f'{feature}_mean'] = feature_values.mean()
                        features[f'{feature}_std'] = feature_values.std()

            # HRV, Stats, and TKEO
            rate_values = signals['ECG_Rate'].dropna()
            if not rate_values.empty:
                features['Heart_Rate_Mean'] = rate_values.mean()
            else:
                features['Heart_Rate_Mean'] = np.nan
            features['signal_skewness'] = stats.skew(lead_signal)
            features['signal_kurtosis'] = stats.kurtosis(lead_signal)
            tkeo_signal = nk.ecg_tkeo(lead_signal)
            features['tkeo_mean'] = np.mean(tkeo_signal)
            features['tkeo_std'] = np.std(tkeo_signal)

            # Conditional HRV Features
            if len(peaks) > 20 and np.std(np.diff(peaks)) > 0:
                try:
                    hrv_time = nk.hrv_time(peaks, sampling_rate=sampling_rate, show=False)
                    features.update(hrv_time.iloc[0].to_dict())
                    hrv_poincare = nk.hrv_poincare(peaks, sampling_rate=sampling_rate, show=False)
                    features.update(hrv_poincare.iloc[0].to_dict())
                    hrv_fragmentation = nk.hrv_fragmentation(peaks, sampling_rate=sampling_rate, show=False)
                    features.update(hrv_fragmentation.iloc[0].to_dict())
                except Exception:
                    failure_counters['hrv_failures'] += 1
                    pass # Fail silently if HRV calculations fail

    except Exception:
        failure_counters['single_lead_processing_failures'] += 1
        pass # Fail silently on single-lead processing

    # --- Multi-Lead (Axis) Feature Extraction ---
    if signal.ndim > 1 and signal.shape[1] > 1:
        try:
            lead1_signal = signal[:, 0]
            lead2_signal = signal[:, 1]
            signals_I, _ = nk.ecg_process(lead1_signal, sampling_rate=sampling_rate)
            signals_II, _ = nk.ecg_process(lead2_signal, sampling_rate=sampling_rate)
            
            features['qrs_axis'] = nk.ecg_axis(signals_I, signals_II)
            features['t_axis'] = nk.ecg_axis(signals_I, signals_II, wave='T')
            features['p_axis'] = nk.ecg_axis(signals_I, signals_II, wave='P')
            if features.get('qrs_axis') is not None and features.get('t_axis') is not None:
                features['qrs_t_angle'] = abs(features['qrs_axis'] - features['t_axis'])
        except Exception:
            failure_counters['axis_failures'] += 1
            pass # Fail silently on axis calculation

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

def process_record(record_path, failure_counters):
    """
    Main processing function for a single record.
    """
    try:
        # Basic features from header
        basic_features = extract_basic_features(record_path)
        if not basic_features:
            failure_counters['basic_feature_failures'] += 1
            return None, [], failure_counters

        # Load signal for advanced feature extraction
        signal, fields = wfdb.rdsamp(record_path)
        sampling_rate = fields['fs']
        
        if signal is None or signal.size == 0:
            failure_counters['signal_loading_failures'] += 1
            return None, [], failure_counters

        # Advanced and wavelet features
        advanced_features, rr_interval_lengths = extract_advanced_ecg_features(signal, sampling_rate, failure_counters)
        wavelet_features = extract_wavelet_features(record_path)

        # Combine all features
        all_features = basic_features
        if advanced_features:
            all_features.update(advanced_features)
        if wavelet_features:
            all_features.update(wavelet_features)
        
        all_features['record'] = os.path.basename(record_path)
        return all_features, rr_interval_lengths, failure_counters
    except Exception as e:
        failure_counters['process_record_exceptions'] += 1
        return None, [], failure_counters

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
    failure_counters = {
        'basic_feature_failures': 0,
        'signal_loading_failures': 0,
        'single_lead_processing_failures': 0,
        'hrv_failures': 0,
        'axis_failures': 0,
        'process_record_exceptions': 0
    }

    with Parallel(n_jobs=-1) as parallel:
        results = parallel(delayed(process_record)(record, failure_counters.copy()) for record in tqdm(record_stems))

    for res, rr_lengths, counters in results:
        if res is not None:
            processed_results.append(res)
            all_rr_interval_lengths_collected.extend(rr_lengths)
        for key in failure_counters:
            failure_counters[key] += counters[key]

    df = pd.DataFrame(processed_results)

    # --- Failure Analysis ---
    print("\n--- Feature Extraction Failure Report ---")
    total_records = len(record_stems)
    print(f"Total records attempted: {total_records}")
    for key, value in failure_counters.items():
        if value > 0:
            print(f"- {key}: {value} ({value/total_records:.2%})")
    print("-----------------------------------------")

    # --- Data Cleaning and Validation ---
    if df.empty:
        print("\nNo records were processed successfully. The resulting DataFrame is empty.")
        print("Cannot proceed with EDA. Please check the data source and failure report above.")
        return # Exit gracefully

    df.dropna(subset=['chagas'], inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    print("\nFeature extraction and cleaning complete.")
    print(f"Final dataset shape: {df.shape}")
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
