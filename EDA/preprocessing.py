import os
"""
Preprocessing script for ECG Chagas detection.

This file loads ECG records from SaMi-Trop and PTB-XL datasets, parses metadata, balances the dataset, and extracts ECG features for each record. Feature extraction includes:
- ECG wave delineation features (P wave, PR interval, QRS duration, QT interval, ST segment, ST slope) using custom logic and NeuroKit2 (`neurokit2`)
- Heart rate variability (HRV) features (SDNN, RMSSD, SDRMSSD, etc.) using NeuroKit2
- Signal quality assessment for HRV suitability
- Metadata parsing (age, sex, Chagas label, source) from WFDB header files

Parallel processing is performed with `joblib`. Results are saved to CSV files for downstream analysis.
"""
import os
import argparse
from datetime import datetime
from pathlib import Path
from typing import Union
from tqdm import tqdm
import numpy as np
import pandas as pd
from helper_code import find_records, get_sampling_frequency, load_header, load_signals
from utils import assess_ecg_quality, check_interval, st_slope, ecg_signal_features
from joblib import Parallel, delayed
import neurokit2 as nk
import traceback

def extract_ecg_features(hea_path: str, channel: int) -> Union[pd.DataFrame, dict]:
    """
    Extracts ECG features from a single record given its full .hea file path.
    """
    record_stem = hea_path.replace('.hea', '')
    # Use the filename without extensions or suffixes as the unique exam_id
    exam_id = Path(hea_path).stem.replace('_lr', '').replace('_hr', '')
    
    try:
        header = load_header(record_stem)
        signals, _ = load_signals(record_stem)
        frequency = get_sampling_frequency(header)

        if signals is None or signals.size == 0:
            return {'error': 'NoSignals', 'exam_id': exam_id, 'path': hea_path}

        ecg_signals, info = nk.ecg_process(signals[:, channel], sampling_rate=frequency)
        
        # Assess signal quality before attempting HRV calculation
        is_good_quality = assess_ecg_quality(info, signals, frequency)
        
        # Only compute HRV features for signals that pass quality checks
        # Drop windowed HRV features (SDANN1, SDANN2, SDANN5, SDNNI1, SDNNI2, SDNNI5) as they require long recordings
        # See documentation below for details
        hrv_feature_names = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_SDSD', 'HRV_CVNN', 'HRV_CVSD',
                            'HRV_MedianNN', 'HRV_MadNN', 'HRV_MCVNN', 'HRV_IQRNN', 'HRV_SDRMSSD', 'HRV_Prc20NN',
                            'HRV_Prc80NN', 'HRV_pNN50', 'HRV_pNN20', 'HRV_MinNN', 'HRV_MaxNN', 'HRV_HTI', 'HRV_TINN']
        if is_good_quality:
            hrv_features_full = nk.hrv_time(ecg_signals, sampling_rate=frequency)
            # Only keep the supported features
            hrv_features = hrv_features_full[[col for col in hrv_feature_names if col in hrv_features_full.columns]].copy()
            # Add any missing columns as NaN
            for col in hrv_feature_names:
                if col not in hrv_features.columns:
                    hrv_features[col] = np.nan
            hrv_features = hrv_features[hrv_feature_names]
        else:
            hrv_features = pd.DataFrame({col: [np.nan] for col in hrv_feature_names})
        
        correct_waves = check_interval(info)

        if correct_waves.empty:
            # Create a DataFrame with NaNs if no valid waves are found
            nan_dict = {k: np.nan for k in ["P_wave_duration","PR_interval","PR_segment","QRS_duration","QT_interval","ST_segment","ST_slope"]}
            nan_df = pd.DataFrame([nan_dict])
            for col in hrv_features.columns:
                nan_df[col] = np.nan
            nan_df['exam_id'] = exam_id
            return nan_df

        # Calculate features for valid waves
        expanded = correct_waves.apply(lambda x: ecg_signal_features(x, frequency), axis=1, result_type='expand')
        correct_waves[expanded.columns] = expanded
        
        # Calculate ST slope
        st_slope_values = [
            st_slope(ecg_signals, int(s_peak), int(t_onset)) if not pd.isna(s_peak) and not pd.isna(t_onset) else np.nan
            for s_peak, t_onset in zip(correct_waves['ECG_S_Peaks'], correct_waves['ECG_T_Onsets'])
        ]
        correct_waves['ST_slope'] = st_slope_values
        
        # Clean up and aggregate features - drop ECG wave position columns
        ecg_wave_columns = ['ECG_P_Onsets', 'ECG_P_Peaks', 'ECG_P_Offsets', 'ECG_Q_Peaks', 'ECG_R_Peaks', 'ECG_S_Peaks', 'ECG_T_Onsets', 'ECG_T_Peaks', 'ECG_T_Offsets']
        columns_to_drop = [col for col in ecg_wave_columns if col in correct_waves.columns]
        if columns_to_drop:
            correct_waves.drop(columns=columns_to_drop, inplace=True)
        correct_waves['exam_id'] = exam_id
        
        agg = correct_waves.groupby('exam_id').aggregate(['mean', 'std', 'min', 'max']).reset_index()
        agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
        agg.rename(columns={'exam_id_': 'exam_id'}, inplace=True)
        
        # Combine aggregated wave features with HRV features
        result = pd.concat([agg, hrv_features], axis=1)
        return result

    except Exception as e:
        return {'error': 'ProcessingError', 'exam_id': exam_id, 'path': hea_path, 'exception': str(e), 'trace': traceback.format_exc()}

"""
NOTE: Windowed HRV features (SDANN1, SDANN2, SDANN5, SDNNI1, SDNNI2, SDNNI5) are not included in the output.
These features require long-term ECG recordings (≥1-5 minutes) and are always NaN for standard 10-second clinical ECGs.
This is a limitation of short ECGs and not a bug in the extraction pipeline.
"""
def parse_metadata_from_hea(header_path):
    """Parses age, sex, Chagas label, and source from a .hea file."""
    age, sex, chagas, source = None, None, None, None
    with open(header_path, 'r') as f:
        for line in f:
            if line.startswith('# Age:'):
                try: age = int(line.split(':')[1].strip()) 
                except (ValueError, IndexError): age = None
            elif line.startswith('# Sex:'):
                sex_str = line.split(':')[1].strip().lower()
                sex = sex_str in ('male', 'm')
            elif line.startswith('# Chagas label:'):
                chagas = line.split(':')[1].strip().lower() == 'true'
            elif line.startswith('# Source:'):
                source = line.split(':')[1].strip().lower()
    return age, sex, chagas, source

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing for ECG Chagas Detection')
    parser.add_argument('--train_samitrop', type=str, default='./training_data/samitrop', help='SaMi-Trop base folder path.')
    # UPDATED: The default path now points to the parent directory of all records.
    parser.add_argument('--train_ptbxl', type=str, default='./training_data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3', help='PTB-XL base folder path containing records subdirectories.')
    parser.add_argument('--output', type=str, required=True, help='Output folder path to store the features and logs.')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility.')
    parser.add_argument('--jobs', type=int, default=-1, help='Number of jobs to run in parallel.')
    args = parser.parse_args()

    samitrop_dir = Path(args.train_samitrop)
    ptbxl_dir = Path(args.train_ptbxl)

    # Create a unique timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_folder = Path(args.output) / f'run_{timestamp}'
    output_folder.mkdir(parents=True, exist_ok=True)

    # Use find_records for both datasets
    samitrop_rel = find_records(str(samitrop_dir))
    ptbxl_rel = find_records(str(ptbxl_dir))
    print(f"Found {len(samitrop_rel)} records in SaMi-Trop and {len(ptbxl_rel)} records in PTB-XL.")

    # Parse metadata from all found .hea files
    metadata = []
    for rel_path in tqdm(samitrop_rel + ptbxl_rel, desc="Parsing metadata"):
        if rel_path in samitrop_rel:
            hea_path = os.path.join(str(samitrop_dir), rel_path + '.hea')
        else:
            hea_path = os.path.join(str(ptbxl_dir), rel_path + '.hea')
        exam_id = Path(hea_path).stem.replace('_lr', '').replace('_hr', '')
        age, is_male, chagas, source = parse_metadata_from_hea(hea_path)
        # Default the source if not specified in the header
        if source is None:
            source = 'ptbxl' if 'ptb-xl' in hea_path else 'samitrop'
        metadata.append({
            'exam_id': exam_id,
            'age': age,
            'is_male': is_male,
            'chagas': chagas,
            'source': source,
            'hea_path': hea_path
        })
    df = pd.DataFrame(metadata).dropna(subset=['chagas']) # Only keep records with a valid Chagas label

    
    # Balance the dataset between positive and negative Chagas cases
    n_positive = df[df['chagas'] == True].shape[0]
    n_negative = df[df['chagas'] == False].shape[0]
    if n_negative > 0 and n_positive > 0:
        n_samples = min(n_positive, n_negative)
        balanced_df = pd.concat([
            df[df['chagas'] == True].sample(n=n_samples, random_state=args.seed),
            df[df['chagas'] == False].sample(n=n_samples, random_state=args.seed)
        ])
    else:
        print("[WARNING] Only one class found. Proceeding with all available labeled records.")
        balanced_df = df
    
    records_to_process = balanced_df['hea_path'].tolist()

    # --- Feature Extraction ---
    # SIMPLIFIED: Pass the full 'hea_path' directly to the processing function.
    # The 'channel' is hardcoded to 1 for the initial attempt.
    all_results = Parallel(n_jobs=args.jobs)(
        delayed(extract_ecg_features)(hea_path=path, channel=1)
        for path in tqdm(records_to_process, desc="Initial feature extraction (channel 1)")
    )

    # Separate successful results from errors
    valid_features = [res for res in all_results if isinstance(res, pd.DataFrame)]
    error_records = [res for res in all_results if isinstance(res, dict) and 'error' in res]
    
    print(f"Initial run: {len(valid_features)} successful, {len(error_records)} errors.")

    # Retry failed records on other channels (0, 2-11)
    retried_features = []
    if error_records:
        retry_tasks = []
        for error in error_records:
            # Create a task for each of the other 11 channels
            for ch in [0] + list(range(2, 12)):
                retry_tasks.append(delayed(extract_ecg_features)(hea_path=error['path'], channel=ch))
        
        print(f"Retrying {len(error_records)} failed records on other channels...")
        retry_results = Parallel(n_jobs=args.jobs)(tqdm(retry_tasks, desc="Retrying on other channels"))
        
        # Keep only the first successful result for each retried record
        processed_exam_ids = set()
        for res in retry_results:
            if isinstance(res, pd.DataFrame):
                exam_id = res['exam_id'].iloc[0]
                if exam_id not in processed_exam_ids:
                    retried_features.append(res)
                    processed_exam_ids.add(exam_id)

    print(f"Successfully recovered {len(retried_features)} records on retry.")
    
    # Combine all successful results and deduplicate by exam_id
    all_successful_results = valid_features + retried_features
    if not all_successful_results:
        print("[ERROR] No features could be extracted. Exiting.")
        # Always write error_records.csv for debugging
        pd.DataFrame(error_records).to_csv(output_folder / 'error_records.csv', index=False)
        print(f"Logged {len(error_records)} initial errors to 'error_records.csv'.")
        exit()

    # Deduplicate: keep first occurrence of each exam_id
    final_features_list = []
    seen_exam_ids = set()
    for res in all_successful_results:
        exam_id = res['exam_id'].iloc[0]
        if exam_id not in seen_exam_ids:
            final_features_list.append(res)
            seen_exam_ids.add(exam_id)

    print(f"[DEBUG] Total successful extractions: {len(all_successful_results)}")
    print(f"[DEBUG] After deduplication: {len(final_features_list)} unique records")

    all_features_df = pd.concat(final_features_list, ignore_index=True)
    
    # Debug: Show what exam_ids are in each DataFrame
    print(f"[DEBUG] Total feature extraction results: {len(all_features_df)} records")
    print(f"[DEBUG] Unique exam_ids in features: {all_features_df['exam_id'].nunique()}")
    print(f"[DEBUG] Unique exam_ids in balanced metadata: {balanced_df['exam_id'].nunique()}")
    print(f"[DEBUG] Balanced metadata total records: {len(balanced_df)}")
    
    # Check for duplicates in features
    duplicate_features = all_features_df[all_features_df.duplicated(subset=['exam_id'], keep=False)]
    if len(duplicate_features) > 0:
        print(f"[DEBUG] Found {len(duplicate_features)} duplicate exam_ids in features")
        print(f"[DEBUG] Duplicate exam_ids: {sorted(duplicate_features['exam_id'].unique().tolist()[:10])}...")
    
    # Merge features with metadata
    final_df = all_features_df.merge(balanced_df[['exam_id','age','is_male','chagas']], on='exam_id', how='inner')
    
    # Show merge results
    features_before_merge = len(all_features_df)
    features_after_merge = len(final_df)
    
    if features_before_merge > features_after_merge:
        dropped_count = features_before_merge - features_after_merge
        print(f"[INFO] {dropped_count} feature records dropped during merge (failed feature extraction)")
        print(f"[INFO] Successfully processed: {features_after_merge} records") 
    else:
        print(f"[INFO] All {features_after_merge} extracted features successfully merged with metadata")
    
    # Save results
    final_df.to_csv(output_folder / 'signals_features.csv', index=False)
    pd.DataFrame(error_records).to_csv(output_folder / 'error_records.csv', index=False)
    
    print(f"Processing complete. Saved {len(final_df)} records to 'signals_features.csv'.")
    print(f"Logged {len(error_records)} initial errors to 'error_records.csv'.")