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


from tqdm import tqdm
from utils import extract_all_ecg_features
from helper_code import find_records

# Set your data directory here
DATA_DIR = './training_data/'






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test ECG Feature Extraction')
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
            features = extract_all_ecg_features(record_path)
            if features is None:
                print(f"[WARN] Extraction failed for: {record_path}")
                failed += 1
                continue
            # Count NaNs
            nan_count = sum(pd.isna(val) for val in features.values())
            nan_counts.append(nan_count)
            results.append(features)
            print(f"[{i+1}/{len(records)}] {os.path.basename(record_path)}: {nan_count} NaNs")
        except Exception as e:
            print(f"[ERROR] Exception for {record_path}: {e}")
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
