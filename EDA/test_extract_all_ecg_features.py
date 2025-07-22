"""
This script tests ECG feature extraction for WFDB records in the specified data directory.
It searches for all .hea files (WFDB headers), extracts features using the extract_all_ecg_features function,
counts NaN values per record, and saves summary statistics and results to CSV files.
Supports a debug mode to process only the first 50 records for faster testing.
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import logging
import sys


from tqdm import tqdm
from utils import extract_all_ecg_features, extract_all_ecg_features_from_signal
from helper_code import load_signals, load_header, get_age, get_sex, get_label, get_sampling_frequency
from helper_code import find_records

# Set your data directory here
DATA_DIR = './training_data/'

if __name__ == '__main__':
    # Configure logging to output to both console and file at DEBUG level
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler('eda_errors.log', mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Set DEBUG=True globally for utils.py
    import utils
    utils.DEBUG = True
    parser = argparse.ArgumentParser(description='Test ECG Feature Extraction')
    # Extra debug: check for NaNs, signal length, and R-peak detection for first 3 records
    import numpy as np
    import helper_code
    print("\n[EXTRA DEBUG] Checking first 3 records for NaNs, signal length, and R-peak detection:")
    for i, record_path in enumerate(sorted(helper_code.find_records_abs("./training_data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records100/"))[:3]):
        signals, _ = helper_code.load_signals(record_path)
        print(f"Record: {record_path}")
        print(f"  Signal shape: {signals.shape}, dtype: {signals.dtype}")
        print(f"  Any NaNs in signal? {np.isnan(signals).any()}")
        print(f"  Signal length (samples): {signals.shape[0]}")
        # Try R-peak detection on lead I (channel 0)
        try:
            import neurokit2 as nk
            rpeaks = nk.ecg_peaks(signals[:,0], sampling_rate=500)[1]["ECG_R_Peaks"]
            print(f"  Detected R-peaks (lead I): {rpeaks}")
            print(f"  Number of R-peaks: {len(rpeaks) if rpeaks is not None else 0}")
        except Exception as e:
            print(f"  R-peak detection failed: {e}")
    print("[END EXTRA DEBUG]\n")
    parser.add_argument('--debug', action='store_true', help='Run in debug mode (only first 50 files)')
    args = parser.parse_args()

    rel_records = find_records(DATA_DIR)
    if not rel_records:
        print(f"[WARN] No .hea files found in {DATA_DIR}")
        sys.exit(1)
    records = [os.path.join(DATA_DIR, r) for r in rel_records]
    print(f"Found {len(records)} records to test.")
    if args.debug:
        records = records[:50]
        print("Debug mode enabled: Only running on first 50 records.")
    results = []
    nan_counts = []
    failed = 0
    for i, record_path in tqdm(enumerate(records)):
        try:
            # --- Original function (file path) ---
            features = extract_all_ecg_features(record_path)
            if features is None:
                print(f"[WARN] Extraction failed for: {record_path}")
                failed += 1
                continue
            nan_count = sum(pd.isna(val) for val in features.values())
            nan_counts.append(nan_count)
            results.append(features)
            print(f"[{i+1}/{len(records)}] {os.path.basename(record_path)}: {nan_count} NaNs (file)")

            # --- New function (raw signal) ---
            header = load_header(record_path + '.hea')
            age = get_age(header)
            is_male = 1 if get_sex(header) == 'Male' else 0
            chagas = get_label(header)
            signals, _ = load_signals(record_path)
            frequency = get_sampling_frequency(header)
            features_signal = extract_all_ecg_features_from_signal(
                signals, frequency, age=age, is_male=is_male, chagas=chagas, record=os.path.basename(record_path), channel_count=signals.shape[1])
            nan_count_signal = sum(pd.isna(val) for val in features_signal.values())
            print(f"    [signal] {os.path.basename(record_path)}: {nan_count_signal} NaNs (raw signal)")
            # Optionally, compare keys/values
            if set(features.keys()) != set(features_signal.keys()):
                print(f"    [DIFF] Feature keys mismatch for {os.path.basename(record_path)}")
            # Optionally, compare values (not always identical due to randomness, but should be close)
        except Exception as e:
            import traceback
            print(f"[ERROR] Exception for {record_path}: {e}")
            traceback.print_exc()
            failed += 1

    # Summary
    print(f"\nTotal records processed: {len(records)}")
    print(f"Total failed: {failed}")
    print(f"Average NaNs per record: {np.mean(nan_counts) if nan_counts else 'N/A'}")

    # Save NaN counts per record
    if nan_counts:
        nan_df = pd.DataFrame({
            'record': [os.path.basename(r) for r in records[:len(nan_counts)]],
            'num_nans': nan_counts
        })
        nan_df.to_csv('test_feature_extraction_nan_counts.csv', index=False)
        print("NaN counts saved to test_feature_extraction_nan_counts.csv")

    # Optionally, save results and detailed NaN report
    if results:
        df = pd.DataFrame(results)
        df.to_csv('test_feature_extraction_results.csv', index=False)
        print("Results saved to test_feature_extraction_results.csv")

        # Detailed NaN summary
        total = len(df)
        feature_nan_counts = df.isna().sum()
        feature_nan_percent = (feature_nan_counts / total * 100).round(2)
        nan_summary = pd.DataFrame({
            'feature': feature_nan_counts.index,
            'num_nan': feature_nan_counts.values,
            'percent_nan': feature_nan_percent.values
        })
        nan_summary = nan_summary.sort_values('num_nan', ascending=False)
        nan_summary.to_csv('nan_feature_report.csv', index=False)
        print("Detailed NaN feature report saved to nan_feature_report.csv")

        # Print summary
        num_with_any_nan = (df.isna().sum(axis=1) > 0).sum()
        print(f"Records with at least one NaN: {num_with_any_nan} / {total} ({num_with_any_nan/total*100:.2f}%)")
        print("Top features with NaNs:")
        print(nan_summary.head(10).to_string(index=False))
