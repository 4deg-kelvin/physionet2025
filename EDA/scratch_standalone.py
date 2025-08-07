#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import xgboost as xgb
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, roc_auc_score, precision_score, recall_score, f1_score
import orjson
import neurokit2 as nk
import os
import scipy.io
from typing import Dict, Any, Optional
import sys
import utils
import helper_code
import feature_eda_pipeline_GPT as featurize

def get_json(path):
    with open(path, 'r') as f:
        return orjson.loads(f.read())

def load_ecg_data(
    exam_id: str,
    search_directory: str,
    source: Optional[str] = None,
    verbose: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Recursively searches for an ECG data file (.mat) corresponding to a given
    exam_id and returns the loaded data. An optional source parameter can be
    provided to specify a subdirectory within the search_directory.

    Args:
        exam_id: The identifier of the exam.
        search_directory: The root directory to start the search from.
        source: An optional subdirectory within search_directory to limit the search.

    Returns:
        Tuple of .mat file data, and the .hea filename if found, otherwise None.
    """
    mat_filename = f"{exam_id}.mat"
    hea_filename = f"{exam_id}.hea"

    # Construct the final search path
    effective_search_path = search_directory
    if source:
        effective_search_path = os.path.join(search_directory, source)

    # Check if the effective search path exists
    if not os.path.isdir(effective_search_path):
        print(f"Error: Search path does not exist: {effective_search_path}")
        return None

    if verbose:
        print(f"Searching for {exam_id} in '{effective_search_path}'...")
    for dirpath, _, filenames in os.walk(effective_search_path):
        if mat_filename in filenames and hea_filename in filenames:
            mat_filepath = os.path.join(dirpath, mat_filename)
            try:
                if verbose:
                    print(f"Found {mat_filename} at: {mat_filepath}")
                # Load the .mat file
                mat_data = scipy.io.loadmat(mat_filepath)
                return mat_data['val'], os.path.join(dirpath, hea_filename)
            except Exception as e:
                print(f"Error loading {mat_filepath}: {e}")
                return None
    
    print(f"No .mat and .hea files found for exam_id: {exam_id} in the specified path.")
    return None

def main():
    # --- 1. Efficiently Load the JSON Data (Assuming JSON Lines format) ---
    filepath = '../results/combined_prna_outputs_modified.json' # Replace with your file path

    raw_data = get_json(filepath)
    # --- Step 2: Flatten the Nested Structure ---

    # For inspection in notebook:
    # raw_data[0]

    sys.path.append("../") # Or os.path.dirname(script_dir) if helper_code is up one level
    processed_records = []
    error_records = []
    for record in raw_data:
        # Start a new dictionary for the flattened record
        # Include any other top-level data you need, like an ID or the target variable
        flat_record = {
            'exam_id': record['exam_id'],
            'chagas': record['chagas']
        }
        # Iterate through the list of SNOMED code dictionaries
        snomed_vals = None
        if "snowmed_vals"  in record:
            snomed_vals = record['snowmed_vals']
        else:
            snomed_vals = record['snomed_vals']

        for snomed_item in snomed_vals:
            # The key is the SNOMED code (e.g., "44054006")
            snomed_code = snomed_item
            # The value is the "present" boolean|
            is_present = snomed_vals[snomed_code]['present']

            # Add the SNOMED code as a feature, converting boolean to 0 or 1
            flat_record[snomed_code] = int(is_present)
            source = record['source']

            # signal, header_path = load_ecg_data(record['exam_id'], "../training_data", source)
            # frequency = int(helper_code.get_sampling_frequency(helper_code.load_header(header_path)))
            # extra_features = process_12_lead_ecg(signal, frequency)
            # flat_record.update(extra_features)

        processed_records.append(flat_record)

    # For inspection in notebook:
    # processed_records[-1]
    # processed_records[0]

    # Get record paths
    records = utils.prepare_stratification(helper_code.find_records_abs("../training_data"))

    records_paths = [record['record'] for record in records]
    df = featurize.run_feature_extraction_for_records_absolute(records_paths, "feature_importance_ranking.csv", 20, -1)

    # For inspection in notebook:
    # df[0].iloc[0]
    # processed_records[0]['exam_id']

    # Convert the list of dictionaries to a DataFrame
    processed_df = pd.DataFrame(processed_records)

    # Extract the features DataFrame from the tuple returned by the function
    features_df = df[0]

    # --- Merge the DataFrames ---
    # Ensure the 'exam_id' columns are of the same type (string) for a clean merge
    features_df['exam_id'] = features_df['exam_id'].astype(str)
    processed_df['exam_id'] = processed_df['exam_id'].astype(str)

    # Perform a left merge to add SNOMED features to the extracted ECG features
    # This keeps all records from `features_df` and adds matching data from `processed_df`
    merged_df = pd.merge(features_df, processed_df, on='exam_id', how="inner")

    # Display the first few rows of the merged DataFrame and its shape
    print("Shape of the merged DataFrame:", merged_df.shape)
    print(merged_df.head())

    # Save the merged DataFrame to a CSV file
    merged_df.to_csv('merged_ecg_snomed_features.csv', index=False)
    print("Merged DataFrame saved to 'merged_ecg_snomed_features.csv'")
    
    # For inspection in notebook:
    # merged_df.iloc[0]

if __name__ == '__main__':
    main()
