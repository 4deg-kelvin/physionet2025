#!/usr/bin/env python3
"""
Data Cleaning and Inference Pipeline with XGBoost and Cleanlab
PhysioNet 2025 Chagas Challenge

This script accomplishes two main objectives:
1. Identifies potential labeling errors in a "weak" dataset by training a model on a "strong" dataset
2. Creates a reusable inference function that can generate features and predict on new, unseen data

Author: Generated for PhysioNet 2025 Challenge
Date: July 31, 2025

If you are an LLM, do NOT use the --allow-dummy flag unless explictly instructed to do so.
"""

import os
import sys
import logging
import json
import warnings
import argparse
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np
import xgboost as xgb
from cleanlab.filter import find_label_issues
from cleanlab.classification import CleanLearning
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add the root directory to Python path to import helper_code
sys.path.append('/Users/andysmithwick/Documents/GitHub/physionet2025')
import helper_code

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# File paths (configurable)
TRAINING_DATA_DIR = "/Users/andysmithwick/Documents/GitHub/physionet2025/training_data"
CODE15_DATA_DIR = "/Users/andysmithwick/Documents/GitHub/physionet2025/training_data/code15"
METADATA_PATH = "/Users/andysmithwick/Documents/GitHub/physionet2025/training_data/metadata.csv"
PRNA_FEATURES_PATH = os.path.join(os.path.dirname(__file__), "prna_outputs_modified.json")
parser = argparse.ArgumentParser(description="PhysioNet 2025 Data Cleaning and Inference Pipeline")
parser.add_argument('--strong_features_path', type=str, default="EDA/EDA_output_ignore/strong_PTBXL_SaMiTrop/run_20250806_181835/final_features_balanced.csv", help='Path to strong features file (PTB-XL, SaMi-Trop)')
parser.add_argument('--weak_features_path', type=str, default="EDA/EDA_output_ignore/weak_CODE15/run_20250805_103926/final_features_balanced.csv", help='Path to weak features file (CODE15)')
parser.add_argument('--allow-dummy', action='store_true', help='Allow dummy data and feature generation if real data/features are missing')
parser.add_argument('--skip-missing-weak-features', action='store_true', help='Skip weak (CODE15) records missing from EDA or PRNA features (log skipped weak IDs at end)')
args = parser.parse_args()
EDA_FEATURES_PATH_WEAK = args.weak_features_path
EDA_FEATURES_PATH_STRONG = args.strong_features_path
ALLOW_DUMMY = args.allow_dummy
SKIP_MISSING_WEAK_FEATURES = args.skip_missing_weak_features

# Output paths
MODEL_PATH = "xgboost_strong_model.json"
LABEL_ISSUES_PATH = "weak_data_label_issues.csv"
CLEAN_MODEL_PATH = "xgboost_clean_model.json"
RESULTS_DIR = "results"

# Column names
RECORD_ID_COL = "exam_id"
DATASET_COL = "dataset"
TARGET_COL = "chagas"

# Dataset categories
STRONG_DATASETS = ["PTB-XL", "SaMi-Trop"]  # Updated to match helper_code output
WEAK_DATASETS = ["CODE15"]

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pipeline.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# =============================================================================
# METADATA EXTRACTION FUNCTIONS
# =============================================================================

def extract_metadata_from_records(data_dir: str, logger) -> pd.DataFrame:
    """
    Extract metadata from ECG records using helper_code functions.
    
    Args:
        data_dir: Directory containing ECG records
        logger: Logger instance
        
    Returns:
        DataFrame with record metadata including record_id, dataset, chagas, age, sex
    """
    logger.info(f"Extracting metadata from records in {data_dir}...")
    
    # Find all .hea files in the directory (recursively)
    records = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.hea'):
                records.append(os.path.join(root, file))
    logger.info(f"Found {len(records)} .hea files in {data_dir}")

    if len(records) == 0:
        logger.warning(f"No .hea files found in {data_dir}")
        return pd.DataFrame()

    metadata_list = []
    for i, record in enumerate(records):
        if i % 100 == 0:
            logger.info(f"Processing record {i+1}/{len(records)}: {record}")
        try:
            # Load header
            header = helper_code.load_header(record)
            # Extract age, sex, chagas label
            age, sex, label = helper_code.get_patient_info(header, allow_missing_label=True)
            # Infer dataset from folder name
            folder_name = os.path.basename(os.path.dirname(record))
            dataset_label = folder_name
            # Normalize dataset label if needed
            if dataset_label == 'CODE-15%':
                dataset_label = 'CODE15'
            # If helper_code.get_source(header) gives a better label, use it
            source = helper_code.get_source(header)
            if source:
                if source == 'CODE-15%':
                    source = 'CODE15'
                dataset_label = source
            # Extract record name (without .hea extension)
            record_name = os.path.splitext(os.path.basename(record))[0]
            # For CODE15, ensure record_name is pure numeric string (remove suffixes if present)
            if dataset_label == 'CODE15':
                import re
                record_name = re.sub(r'(_hr|_lr)$', '', str(record_name))
            metadata_list.append({
                RECORD_ID_COL: record_name,
                DATASET_COL: dataset_label,
                TARGET_COL: label if label is not None else np.nan,
                'age': age if age is not None else np.nan,
                'sex': sex if sex else 'Unknown'
            })
        except Exception as e:
            logger.warning(f"Error processing record {record}: {str(e)}")
            record_name = os.path.splitext(os.path.basename(record))[0]
            folder_name = os.path.basename(os.path.dirname(record))
            dataset_label = folder_name
            if dataset_label == 'CODE-15%':
                dataset_label = 'CODE15'
            # For CODE15, ensure record_name is pure numeric string (remove suffixes if present)
            if dataset_label == 'CODE15':
                import re
                record_name = re.sub(r'(_hr|_lr)$', '', str(record_name))
            metadata_list.append({
                RECORD_ID_COL: record_name,
                DATASET_COL: dataset_label,
                TARGET_COL: np.nan,
                'age': np.nan,
                'sex': 'Unknown'
            })
        
    metadata_df = pd.DataFrame(metadata_list)
    logger.info(f"Extracted metadata for {len(metadata_df)} records")
    # Log summary statistics
    if not metadata_df.empty:
        logger.info("Dataset distribution:")
        if DATASET_COL in metadata_df.columns:
            dataset_counts = metadata_df[DATASET_COL].value_counts()
            for dataset, count in dataset_counts.items():
                logger.info(f"  {dataset}: {count} records")
        if TARGET_COL in metadata_df.columns:
            label_counts = metadata_df[TARGET_COL].value_counts(dropna=False)
            logger.info("Label distribution:")
            for label, count in label_counts.items():
                logger.info(f"  {label}: {count} records")
    # Log CODE15 merge keys
    code15_meta = metadata_df[metadata_df[DATASET_COL] == 'CODE15']
    logger.info(f"CODE15 metadata merge keys (first 10): {code15_meta[RECORD_ID_COL].astype(str).tolist()[:10]}")
    return metadata_df

# =============================================================================
# FEATURE GENERATION FUNCTIONS (PLACEHOLDERS)
# =============================================================================

def get_prna_features(record_ids: List[str], allow_dummy: bool = False) -> pd.DataFrame:
    """
    Placeholder function to generate PRNA features for given record IDs.
    
    In a real implementation, this would call the actual PRNA inference code.
    For now, this returns dummy features for demonstration.
    
    Args:
        record_ids: List of record IDs to generate features for
        allow_dummy: If False, raise error if real features are missing
    Returns:
        DataFrame with record_id and PRNA features
    """
    logger = logging.getLogger(__name__)
    if not allow_dummy:
        raise RuntimeError("PRNA features file not found and dummy feature generation is not allowed. Please provide real PRNA features or run with --allow-dummy.")
    logger.warning("Using placeholder PRNA feature generation. Replace with actual implementation.")
    n_features = 50  # Assumed number of PRNA features
    features_data = {RECORD_ID_COL: record_ids}
    for i in range(n_features):
        features_data[f'prna_feature_{i}'] = np.random.randn(len(record_ids))
    return pd.DataFrame(features_data)

def get_eda_features(record_ids: List[str], allow_dummy: bool = False) -> pd.DataFrame:
    """
    Placeholder function to generate EDA features for given record IDs.
    
    In a real implementation, this would call the actual EDA feature extraction code.
    For now, this returns dummy features for demonstration.
    
    Args:
        record_ids: List of record IDs to generate features for
        allow_dummy: If False, raise error if real features are missing
    Returns:
        DataFrame with record_id and EDA features
    """
    logger = logging.getLogger(__name__)
    if not allow_dummy:
        raise RuntimeError("EDA features file not found and dummy feature generation is not allowed. Please provide real EDA features or run with --allow-dummy.")
    logger.warning("Using placeholder EDA feature generation. Replace with actual implementation.")
    n_features = 30  # Assumed number of EDA features
    features_data = {RECORD_ID_COL: record_ids}
    for i in range(n_features):
        features_data[f'eda_feature_{i}'] = np.random.randn(len(record_ids))
    return pd.DataFrame(features_data)

# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_and_prepare_data(logger, allow_dummy: bool = False, skip_missing_weak_features: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list]:

    logger.info("Loading and preparing data...")
    # Extract metadata from all available data directories (including CODE15)
    all_metadata = []
    for data_dir in [TRAINING_DATA_DIR, CODE15_DATA_DIR]:
        if os.path.exists(data_dir):
            logger.info(f"Processing directory: {data_dir}")
            dir_metadata = extract_metadata_from_records(data_dir, logger)
            if not dir_metadata.empty:
                all_metadata.append(dir_metadata)
        else:
            logger.warning(f"Data directory not found: {data_dir}")
    if all_metadata:
        metadata = pd.concat(all_metadata, ignore_index=True)
        logger.info(f"Combined metadata from {len(all_metadata)} directories")
    else:
        if not allow_dummy:
            raise RuntimeError("No data directories found and dummy data generation is not allowed. Please provide real data or run with --allow-dummy.")
        logger.warning("No data directories found. Creating synthetic metadata for demonstration...")
        np.random.seed(42)
        n_strong = 1000  # PTB-XL and Samitrop samples
        n_weak = 500     # CODE15 samples
        strong_ids = [f"strong_{i:04d}" for i in range(n_strong)]
        weak_ids = [f"weak_{i:04d}" for i in range(n_weak)]
        strong_datasets = np.random.choice(STRONG_DATASETS, n_strong)
        weak_datasets = np.random.choice(WEAK_DATASETS, n_weak)
        strong_labels = np.random.choice([0, 1], n_strong, p=[0.7, 0.3])
        weak_labels = np.random.choice([0, 1], n_weak, p=[0.6, 0.4])
        metadata = pd.DataFrame({
            RECORD_ID_COL: strong_ids + weak_ids,
            DATASET_COL: np.concatenate([strong_datasets, weak_datasets]),
            TARGET_COL: np.concatenate([strong_labels, weak_labels])
        })
    if metadata.empty:
        raise ValueError("No metadata available. Cannot proceed with pipeline.")
    metadata.columns = [c.strip() for c in metadata.columns]
    logger.info(f"Metadata columns after loading: {metadata.columns.tolist()}")
    records_with_labels = metadata.dropna(subset=[TARGET_COL])
    if len(records_with_labels) < len(metadata):
        logger.warning(f"Removed {len(metadata) - len(records_with_labels)} records with missing labels")
    metadata = records_with_labels
    logger.info(f"Using {len(metadata)} records with valid labels")

    # Load pre-computed features
    # Using named tuples to track feature source instead of positional order
    features_dfs = []  # Will contain tuples of (source_name, dataframe)
    
    # Load PRNA features if available (support CSV or JSON)
    def load_prna_features(path):
        _, ext = os.path.splitext(path)
        if ext.lower() == '.csv':
            return pd.read_csv(path)
        elif ext.lower() == '.json':
            logger.info(f"Loading PRNA features from JSON: {path}")
            import json
            with open(path, 'r') as f:
                data = json.load(f)
            # If the JSON is a list of dicts, convert directly and extract features
            if isinstance(data, list):
                records = []
                for record in data:
                    if not record or 'exam_id' not in record:
                        continue
                    row = {RECORD_ID_COL: record['exam_id']}
                    
                    # Extract basic numeric features (convert string numbers to float)
                    numeric_fields = ['nn_predicted_age', 'timey', 'age']
                    for field in numeric_fields:
                        if field in record and record[field] and record[field] != "":
                            try:
                                row[f'prna_{field}'] = float(record[field])
                            except (ValueError, TypeError):
                                pass
                    
                    # Extract boolean features as 0/1
                    bool_fields = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF', 'normal_ecg', 'death']
                    for field in bool_fields:
                        if field in record and record[field] is not None and record[field] != "":
                            if isinstance(record[field], bool):
                                row[f'prna_{field}'] = int(record[field])
                            elif isinstance(record[field], str):
                                row[f'prna_{field}'] = 1 if record[field].lower() == 'true' else 0
                    
                    # Extract gender as numeric
                    if 'is_male' in record and record['is_male'] is not None and record['is_male'] != "":
                        if isinstance(record['is_male'], bool):
                            row['prna_is_male'] = int(record['is_male'])
                        elif isinstance(record['is_male'], str):
                            row['prna_is_male'] = 1 if record['is_male'].lower() == 'true' else 0
                    
                    # Extract SNOMED probability features (top features only to avoid too many columns)
                    if 'snomed_vals' in record and isinstance(record['snomed_vals'], dict):
                        for snomed_code, snomed_data in record['snomed_vals'].items():
                            if isinstance(snomed_data, dict) and 'probability' in snomed_data:
                                try:
                                    prob_value = float(snomed_data['probability'])
                                    # Only include features with significant probability to reduce dimensionality
                                    if prob_value > 0.01:  # 1% threshold
                                        row[f'prna_snomed_{snomed_code}_prob'] = prob_value
                                    if 'present' in snomed_data:
                                        present_val = snomed_data['present']
                                        if isinstance(present_val, bool):
                                            row[f'prna_snomed_{snomed_code}_present'] = int(present_val)
                                except (ValueError, TypeError):
                                    pass
                    
                    if len(row) > 1:  # Only add if we extracted some features
                        records.append(row)
                
                return pd.DataFrame(records)
            # If the JSON is a dict of exam_id: features, convert to DataFrame
            elif isinstance(data, dict):
                records = []
                for eid, feats in data.items():
                    row = {RECORD_ID_COL: eid}
                    row.update(feats)
                    records.append(row)
                return pd.DataFrame(records)
            else:
                raise ValueError("Unrecognized JSON structure for PRNA features.")
        else:
            raise ValueError(f"Unsupported PRNA features file extension: {ext}")

    if os.path.exists(PRNA_FEATURES_PATH):
        prna_features = load_prna_features(PRNA_FEATURES_PATH)
        # Sanitize column names
        prna_features.columns = [c.strip() for c in prna_features.columns]
        # Ensure all columns are strings
        prna_features.columns = [str(c) for c in prna_features.columns]
        # Debug: log columns and first few exam_ids
        logger.info(f"[DEBUG] PRNA features columns after loading: {prna_features.columns.tolist()}")
        if RECORD_ID_COL in prna_features.columns:
            logger.info(f"[DEBUG] First 10 PRNA exam_ids: {prna_features[RECORD_ID_COL].astype(str).head(10).tolist()}")
        else:
            logger.error(f"'exam_id' column not found in PRNA features. Columns: {prna_features.columns.tolist()}")
            raise KeyError(f"'exam_id' column not found in PRNA features. Columns: {prna_features.columns.tolist()}")
        # Ensure exam_id is string type
        prna_features[RECORD_ID_COL] = prna_features[RECORD_ID_COL].astype(str)
        features_dfs.append(("prna", prna_features))
        logger.info(f"Loaded PRNA features with {len(prna_features)} records")
    else:
        prna_features = None
        logger.warning(f"PRNA features file not found: {PRNA_FEATURES_PATH}")
        if not allow_dummy:
            raise RuntimeError("PRNA features file not found and dummy feature generation is not allowed. Please provide real PRNA features or run with --allow-dummy.")
        logger.info("Will use placeholder PRNA features for demonstration")
        prna_features = get_prna_features(metadata[RECORD_ID_COL].tolist(), allow_dummy=allow_dummy)
        features_dfs.append(("prna", prna_features))
    # Load EDA features for strong and weak datasets
    eda_features_strong = None
    eda_features_weak = None
    if os.path.exists(EDA_FEATURES_PATH_STRONG):
        eda_features_strong = pd.read_csv(EDA_FEATURES_PATH_STRONG)
        eda_features_strong.columns = [c.strip() for c in eda_features_strong.columns]
        logger.info(f"EDA features (strong) columns after loading: {eda_features_strong.columns.tolist()}")
        col_counts = pd.Series(eda_features_strong.columns).value_counts()
        dups = col_counts[col_counts > 1]
        if not dups.empty:
            logger.error(f"Duplicate columns in EDA features (strong): {dups}")
            raise ValueError(f"Duplicate columns in EDA features (strong): {dups}")
        eda_features_strong.columns = [str(c) for c in eda_features_strong.columns]
        STRONG_ID_COL = 'exam_id'
        if STRONG_ID_COL not in eda_features_strong.columns:
            logger.error(f"'{STRONG_ID_COL}' column not found in EDA features (strong) after rename. Columns: {eda_features_strong.columns.tolist()}")
            raise KeyError(f"'{STRONG_ID_COL}' column not found in EDA features (strong) after rename. Columns: {eda_features_strong.columns.tolist()}")
        logger.info(f"Loaded EDA features (strong) with {len(eda_features_strong)} records")
    if os.path.exists(EDA_FEATURES_PATH_WEAK):
        eda_features_weak = pd.read_csv(EDA_FEATURES_PATH_WEAK)
        eda_features_weak.columns = [c.strip() for c in eda_features_weak.columns]
        logger.info(f"EDA features (weak) columns after loading: {eda_features_weak.columns.tolist()}")
        # No renaming needed; use 'exam_id' as the identifier
        col_counts = pd.Series(eda_features_weak.columns).value_counts()
        dups = col_counts[col_counts > 1]
        if not dups.empty:
            logger.error(f"Duplicate columns in EDA features (weak): {dups}")
            raise ValueError(f"Duplicate columns in EDA features (weak): {dups}")
        eda_features_weak.columns = [str(c) for c in eda_features_weak.columns]
        # Ensure merge key is standardized to 'exam_id' for CODE15
        if 'record_id' in eda_features_weak.columns and 'exam_id' not in eda_features_weak.columns:
            eda_features_weak = eda_features_weak.rename(columns={'record_id': 'exam_id'})
        if RECORD_ID_COL not in eda_features_weak.columns:
            logger.error(f"'{RECORD_ID_COL}' column not found in EDA features (weak) after rename. Columns: {eda_features_weak.columns.tolist()}")
            raise KeyError(f"'{RECORD_ID_COL}' column not found in EDA features (weak) after rename. Columns: {eda_features_weak.columns.tolist()}")
        logger.info(f"Loaded EDA features (weak) with {len(eda_features_weak)} records")
    # Append EDA features to features_dfs with source names
    if eda_features_strong is not None:
        features_dfs.append(("eda_strong", eda_features_strong))
    if eda_features_weak is not None:
        features_dfs.append(("eda_weak", eda_features_weak))
    if eda_features_strong is None and eda_features_weak is None:
        logger.warning("No EDA features found for either strong or weak datasets.")
        if not allow_dummy:
            raise RuntimeError("EDA features file not found and dummy feature generation is not allowed. Please provide real EDA features or run with --allow-dummy.")
        logger.info("Will use placeholder EDA features for demonstration")
        eda_features = get_eda_features(metadata[RECORD_ID_COL].tolist(), allow_dummy=allow_dummy)
        features_dfs.append(("eda_dummy", eda_features))

    # Fail loudly when required feature sets are missing
    if not allow_dummy:
        feature_sources = {source for source, _ in features_dfs}
        if "prna" not in feature_sources:
            raise RuntimeError("PRNA features are required but not loaded. Please provide real PRNA features or run with --allow-dummy.")
        if "eda_weak" not in feature_sources:
            raise RuntimeError("EDA weak features are required but not loaded. Please provide real EDA weak features or run with --allow-dummy.")
        logger.info(f"[VALIDATION] Required feature sets loaded: {feature_sources}")
    
    # === BEGIN DEBUGGING BLOCK: Compare strong class 0 IDs in metadata vs PRNA/EDA features ===
    try:
        # Extract strong class 0 exam_ids from metadata
        strong_class_0_ids = set(metadata[(metadata[DATASET_COL].isin(STRONG_DATASETS)) & (metadata[TARGET_COL] == 0)][RECORD_ID_COL].astype(str))
        # Extract all exam_ids from PRNA features
        prna_features_df = next((df for source, df in features_dfs if source == "prna"), pd.DataFrame())
        prna_ids = set(prna_features_df[RECORD_ID_COL].astype(str)) if not prna_features_df.empty and RECORD_ID_COL in prna_features_df.columns else set()
        # Extract all exam_ids from EDA features (use strong if available, otherwise weak)
        eda_features_df = next((df for source, df in features_dfs if source in ["eda_strong", "eda_weak"]), pd.DataFrame())
        eda_ids = set(eda_features_df[RECORD_ID_COL].astype(str)) if not eda_features_df.empty and RECORD_ID_COL in eda_features_df.columns else set()
        # Log total counts and first 5 examples
        logger.info(f"[DIAG] strong_class_0_ids: {len(strong_class_0_ids)} IDs. Examples: {list(strong_class_0_ids)[:5]}")
        logger.info(f"[DIAG] prna_ids: {len(prna_ids)} IDs. Examples: {list(prna_ids)[:5]}")
        logger.info(f"[DIAG] eda_ids: {len(eda_ids)} IDs. Examples: {list(eda_ids)[:5]}")
        # Compare sets
        missing_in_prna = strong_class_0_ids - prna_ids
        missing_in_eda = strong_class_0_ids - eda_ids
        logger.info(f"[DIAG] strong_class_0_ids missing from PRNA: {len(missing_in_prna)}. Examples: {list(missing_in_prna)[:5]}")
        logger.info(f"[DIAG] strong_class_0_ids missing from EDA: {len(missing_in_eda)}. Examples: {list(missing_in_eda)[:5]}")
    except Exception as e:
        logger.warning(f"[DIAG] Error in strong class 0 ID comparison debug block: {e}")
    # === END DEBUGGING BLOCK ===


    # --- Merge all features (PRNA, EDA, etc.) into a single feature tensor ---
    logger.info(f"Metadata columns before creating combined_features: {metadata.columns.tolist()}")
    if RECORD_ID_COL not in metadata.columns:
        logger.error(f"'{RECORD_ID_COL}' column not found in metadata. Columns: {metadata.columns.tolist()}")
        raise KeyError(f"'{RECORD_ID_COL}' column not found in metadata. Columns: {metadata.columns.tolist()}")
    combined_features = metadata[[RECORD_ID_COL, DATASET_COL, TARGET_COL]].copy()
    combined_features.columns = [c.strip() for c in combined_features.columns]
    # Standardize exam_id by stripping suffixes (_hr, _lr, etc.) for merge compatibility
    import re
    def strip_suffix(exam_id):
        return re.sub(r'(_hr|_lr)$', '', str(exam_id))
    combined_features[RECORD_ID_COL] = combined_features[RECORD_ID_COL].astype(str).apply(strip_suffix)
    logger.info(f"combined_features columns after creation: {combined_features.columns.tolist()}")
    strong_meta = metadata[metadata[DATASET_COL].isin(STRONG_DATASETS)]
    logger.info(f"[DEBUG] Pre-merge strong records: {len(strong_meta)}")
    logger.info(f"[DEBUG] Pre-merge strong class distribution: {strong_meta[TARGET_COL].value_counts(dropna=False).to_dict()}")

    # If skip_missing_weak_features is enabled, filter only weak metadata to those present in all features_dfs
    skipped_ids = []
    if len(features_dfs) >= 2:
        prna_source, prna_df = next(((src, df) for src, df in features_dfs if src == "prna"), (None, pd.DataFrame()))
        eda_weak_source, eda_weak_df = next(((src, df) for src, df in features_dfs if src == "eda_weak"), (None, pd.DataFrame()))
        
        if not prna_df.empty and not eda_weak_df.empty:
            prna_ids = set(prna_df[RECORD_ID_COL].astype(str))
            eda_ids = set(eda_weak_df[RECORD_ID_COL].astype(str))
            valid_ids = prna_ids & eda_ids
        # Only filter weak data (CODE15)
        weak_mask = combined_features[DATASET_COL].isin(WEAK_DATASETS)
        weak_ids = set(combined_features.loc[weak_mask, RECORD_ID_COL].astype(str))
        skipped_ids = sorted(list(weak_ids - valid_ids))
        if skip_missing_weak_features:
            if skipped_ids:
                logger.warning(f"[SKIP] Skipping {len(skipped_ids)} weak records missing from EDA or PRNA features. IDs: {skipped_ids[:10]}{' ...' if len(skipped_ids)>10 else ''}")
            # Keep all strong data, and only keep weak data present in both features
            keep_mask = (~weak_mask) | (combined_features[RECORD_ID_COL].astype(str).isin(valid_ids))
            combined_features = combined_features[keep_mask].copy()
        else:
            if skipped_ids:
                logger.error(f"Missing features for {len(skipped_ids)} CODE15 records: {skipped_ids[:10]}{' ...' if len(skipped_ids)>10 else ''}")
                raise RuntimeError(f"Missing features for {len(skipped_ids)} CODE15 records. Set --skip-missing-weak-features to skip them.")

    for i, (source_name, features_df) in enumerate(features_dfs):
        combined_features = combined_features.reset_index(drop=True)
        features_df = features_df.reset_index(drop=True)
        features_df[RECORD_ID_COL] = features_df[RECORD_ID_COL].astype(str)
        logger.info(f"Merging features from source: {source_name}")
        logger.info(f"combined_features info before merge:\n{combined_features.info()}")
        logger.info(f"features_df info before merge:\n{features_df.info()}")
        before = len(combined_features)
        strong_before = combined_features[combined_features[DATASET_COL].isin(STRONG_DATASETS)][[RECORD_ID_COL, TARGET_COL if TARGET_COL in combined_features.columns else (TARGET_COL + '_x')]].copy()

        # Use left join for strong datasets, inner for weak
        join_type = 'left'
        logger.info(f"[DEBUG] Merge {i+1} ({source_name}): Using join type '{join_type}'")
        combined_features = combined_features.merge(
            features_df,
            on=RECORD_ID_COL,
            how=join_type
        )
        # After merge, if 'chagas_x' exists, rename to 'chagas' and drop 'chagas_y' if present
        if f'{TARGET_COL}_x' in combined_features.columns:
            combined_features = combined_features.rename(columns={f'{TARGET_COL}_x': TARGET_COL})
            if f'{TARGET_COL}_y' in combined_features.columns:
                combined_features = combined_features.drop(columns=[f'{TARGET_COL}_y'])

        # Check for duplicated demographic columns and compare for consistency
        demographic_cols = ['age', 'is_male', 'sex']
        for col in demographic_cols:
            col_x = f'{col}_x'
            col_y = f'{col}_y'
            if col_x in combined_features.columns and col_y in combined_features.columns:
                # Always keep the metadata (from .hea) version
                combined_features = combined_features.drop(columns=[col_y])
                combined_features = combined_features.rename(columns={col_x: col})

        # Always preserve the metadata 'dataset' column after merge
        dataset_x = f'{DATASET_COL}_x'
        dataset_y = f'{DATASET_COL}_y'
        if dataset_x in combined_features.columns and dataset_y in combined_features.columns:
            # Keep the metadata version
            combined_features = combined_features.drop(columns=[dataset_y])
            combined_features = combined_features.rename(columns={dataset_x: DATASET_COL})
        elif dataset_x in combined_features.columns:
            combined_features = combined_features.rename(columns={dataset_x: DATASET_COL})
        elif dataset_y in combined_features.columns:
            combined_features = combined_features.rename(columns={dataset_y: DATASET_COL})

        after = len(combined_features)
        strong_after = combined_features[combined_features[DATASET_COL].isin(STRONG_DATASETS)][[RECORD_ID_COL, TARGET_COL]].copy()
        dropped = set(strong_before[RECORD_ID_COL]) - set(strong_after[RECORD_ID_COL])
        if dropped:
            dropped_df = strong_before[strong_before[RECORD_ID_COL].isin(dropped)]
            logger.warning(f"[DEBUG] Merge {i+1} ({source_name}): Dropped {len(dropped)} strong records. Class breakdown: {dropped_df[TARGET_COL].value_counts(dropna=False).to_dict()}")
            logger.warning(f"[DEBUG] Dropped strong exam_ids: {dropped_df[RECORD_ID_COL].tolist()[:10]}{' ...' if len(dropped_df)>10 else ''}")
        if after < before:
            logger.warning(f"Dropped {before - after} records missing from {source_name} features during merge. This is expected if using placeholder features.")
    logger.info(f"Combined dataset has {len(combined_features)} records with {len(combined_features.columns)-3} features")

    # Step 3: Log final class distribution in strong_data after all merges
    strong_data_tmp = combined_features[combined_features[DATASET_COL].isin(STRONG_DATASETS)]
    logger.info(f"[DEBUG] Post-merge strong records: {len(strong_data_tmp)}")
    logger.info(f"[DEBUG] Post-merge strong class distribution: {strong_data_tmp[TARGET_COL].value_counts(dropna=False).to_dict()}")

    # Reset index to ensure downstream index-based access is safe
    combined_features = combined_features.reset_index(drop=True)

    # --- Ensure feature tensor is identical for strong and weak data ---
    feature_columns = [col for col in combined_features.columns if col not in [RECORD_ID_COL, DATASET_COL, TARGET_COL]]
    strong_data = combined_features[combined_features[DATASET_COL].isin(STRONG_DATASETS)].copy().reset_index(drop=True)
    weak_data = combined_features[combined_features[DATASET_COL].isin(WEAK_DATASETS)].copy().reset_index(drop=True)
    strong_cols = set(strong_data.columns)
    weak_cols = set(weak_data.columns)
    if strong_cols != weak_cols:
        logger.error(f"Feature columns mismatch between strong and weak data! Strong-only: {strong_cols-weak_cols}, Weak-only: {weak_cols-strong_cols}")
        raise ValueError("Feature columns mismatch between strong and weak data. Ensure feature extraction is identical for both.")

    # Only keep columns that are fully numeric after conversion (int, float, bool)
    numeric_feature_columns = []
    dropped_non_numeric = []
    for col in feature_columns:
        # Skip obviously non-numeric columns
        if col in ['trace_file', 'primary_id', 'source', 'patient_id']:
            dropped_non_numeric.append(col)
            continue
            
        try:
            strong_numeric = pd.to_numeric(strong_data[col], errors='coerce')
            weak_numeric = pd.to_numeric(weak_data[col], errors='coerce')
            
            # Check if at least some values are numeric (not all NaN)
            strong_has_numeric = strong_numeric.notnull().any() if not strong_numeric.empty else False
            weak_has_numeric = weak_numeric.notnull().any() if not weak_numeric.empty else False
            
            if strong_has_numeric or weak_has_numeric:
                numeric_feature_columns.append(col)
            else:
                dropped_non_numeric.append(col)
                logger.warning(f"Dropping non-numeric feature column: {col}")
        except Exception as e:
            dropped_non_numeric.append(col)
            logger.warning(f"Dropping feature column due to error: {col} - {e}")

    X_strong = strong_data[numeric_feature_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
    X_weak = weak_data[numeric_feature_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Remove columns that are all zeros or NaN after conversion
    cols_to_keep = []
    for col in numeric_feature_columns:
        strong_col_valid = (X_strong[col] != 0).any() or X_strong[col].notnull().any()
        weak_col_valid = (X_weak[col] != 0).any() or X_weak[col].notnull().any()
        if strong_col_valid or weak_col_valid:
            cols_to_keep.append(col)
    
    dropped_all_nan = list(set(numeric_feature_columns) - set(cols_to_keep))
    if dropped_all_nan:
        logger.warning(f"Dropping columns with all NaN/zero values: {dropped_all_nan}")
    
    X_strong = X_strong[cols_to_keep]
    X_weak = X_weak[cols_to_keep]
    y_strong = strong_data[TARGET_COL]
    y_weak = weak_data[TARGET_COL]

    logger.info(f"Feature matrix shape - Strong: {X_strong.shape}, Weak: {X_weak.shape}")
    logger.info(f"Target distribution - Strong: {y_strong.value_counts().to_dict()}")
    logger.info(f"Target distribution - Weak: {y_weak.value_counts().to_dict()}")
    if dropped_non_numeric:
        logger.info(f"Dropped non-numeric feature columns: {dropped_non_numeric}")
    if dropped_all_nan:
        logger.info(f"Dropped all-NaN feature columns: {dropped_all_nan}")

    return X_strong, y_strong, X_weak, y_weak, strong_data, weak_data, skipped_ids

# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_strong_model(X_strong: pd.DataFrame, y_strong: pd.Series, logger) -> xgb.XGBClassifier:
    """
    Train XGBoost model on strong dataset.
    
    Args:
        X_strong: Features from strong dataset
        y_strong: Labels from strong dataset
        logger: Logger instance
        
    Returns:
        Trained XGBoost classifier
    """
    logger.info("Training XGBoost model on strong dataset...")
    
    # Initialize XGBoost classifier
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    # Train the model
    model.fit(X_strong, y_strong)
    
    # Evaluate on training data
    train_pred = model.predict(X_strong)
    train_accuracy = (train_pred == y_strong).mean()
    logger.info(f"Training accuracy: {train_accuracy:.4f}")
    
    # Save the model
    model.save_model(MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")
    
    return model

# =============================================================================
# LABEL ISSUE DETECTION
# =============================================================================

def find_weak_label_issues(model: xgb.XGBClassifier, X_weak: pd.DataFrame, 
                          y_weak: pd.Series, weak_data: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Find potential label issues in weak dataset using cleanlab.
    
    Args:
        model: Trained XGBoost model
        X_weak: Features from weak dataset
        y_weak: Labels from weak dataset
        weak_data: Full weak dataset with record IDs
        logger: Logger instance
        
    Returns:
        DataFrame with record IDs that have potential label issues
    """
    logger.info("Finding potential label issues in weak dataset...")
    
    # Generate predicted probabilities
    pred_probs = model.predict_proba(X_weak)
    logger.info(f"Generated predictions for {len(X_weak)} weak samples")
    
    # Find label issues using cleanlab
    label_issue_indices = find_label_issues(
        labels=y_weak.to_numpy(),
        pred_probs=pred_probs,
        return_indices_ranked_by='self_confidence'
    )
    
    logger.info(f"Found {len(label_issue_indices)} potential label issues")
    
    # Get record IDs with label issues, robust to index misalignment
    weak_data_reset = weak_data.reset_index(drop=True)
    valid_indices = [idx for idx in label_issue_indices if idx < len(weak_data_reset)]
    if len(valid_indices) < len(label_issue_indices):
        logger.warning(f"Some label_issue_indices were out of bounds after merging. Only {len(valid_indices)} of {len(label_issue_indices)} will be used.")
    issue_records = weak_data_reset.loc[valid_indices, [RECORD_ID_COL, TARGET_COL]].copy()
    issue_records['predicted_prob_positive'] = pred_probs[valid_indices, 1]
    issue_records['confidence_score'] = np.max(pred_probs[valid_indices], axis=1)

    # Save results
    issue_records.to_csv(LABEL_ISSUES_PATH, index=False)
    logger.info(f"Label issues saved to {LABEL_ISSUES_PATH}")

    # Print summary
    logger.info(f"Label issues summary:")
    logger.info(f"  Total weak samples: {len(y_weak)}")
    logger.info(f"  Potential issues: {len(valid_indices)} ({len(valid_indices)/len(y_weak)*100:.2f}%)")

    return issue_records

# =============================================================================
# INFERENCE PIPELINE
# =============================================================================

def run_inference(record_ids: List[str], allow_dummy: bool = False, model_path: str = MODEL_PATH) -> pd.DataFrame:
    """
    Reusable inference function that generates features and predictions for new record IDs.
    
    This is the critical function for running inference on validation sets where features
    are NOT pre-computed.
    
    Args:
        record_ids: List of record IDs to generate predictions for
        allow_dummy: If False, raise error if real features are missing
        model_path: Path to the trained model file
    Returns:
        DataFrame with record_id and predicted label
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running inference for record IDs: {record_ids}")
    
    # Try to load the trained model
    if os.path.exists(model_path):
        try:
            import xgboost as xgb
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
            
            # TODO: In a real implementation, you would:
            # 1. Generate PRNA features for these record_ids 
            # 2. Generate EDA features for these record_ids
            # 3. Combine the features into the same format as training data
            # 4. Run model.predict_proba() to get predictions
            
            # For now, return a placeholder message indicating the model is ready
            results = pd.DataFrame({
                RECORD_ID_COL: record_ids,
                'message': ['Model loaded successfully - implement feature generation for full inference'] * len(record_ids)
            })
            logger.info("Model loaded successfully. To complete inference, implement PRNA and EDA feature generation for new record IDs.")
            return results
            
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            if not allow_dummy:
                raise RuntimeError(f"Model loading failed and dummy predictions are not allowed: {e}")
    
    # Fall back to dummy predictions if model loading fails and dummy is allowed
    if not allow_dummy:
        raise RuntimeError("Model not found and dummy feature generation is not allowed. Please train a model first or run with --allow-dummy.")
    
    logger.warning("Using dummy inference results. Replace with actual model inference.")
    n_samples = len(record_ids)
    results = {
        RECORD_ID_COL: record_ids,
        TARGET_COL: np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'predicted_prob_positive': np.random.rand(n_samples),
        'confidence_score': np.random.rand(n_samples)
    }
    return pd.DataFrame(results)

# =============================================================================
# CLEANING PIPELINE
# =============================================================================

def train_clean_model(X_weak: pd.DataFrame, y_weak: pd.Series, logger) -> CleanLearning:
    """
    Train CleanLearning model on weak dataset with automatic label issue detection.
    
    Args:
        X_weak: Features from weak dataset
        y_weak: Labels from weak dataset
        logger: Logger instance
        
    Returns:
        Trained CleanLearning classifier
    """
    logger.info("Training CleanLearning model on noisy weak dataset...")
    
    # Initialize base classifier
    base_classifier = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        n_estimators=100
    )
    
    # Initialize CleanLearning
    clean_model = CleanLearning(clf=base_classifier)
    
    # Train with automatic label error detection and removal
    clean_model.fit(X_weak, y_weak)
    
    # Save the clean model
    # Note: CleanLearning doesn't have a direct save method, so we save the internal classifier
    clean_model.clf.save_model(CLEAN_MODEL_PATH)
    logger.info(f"Clean model saved to {CLEAN_MODEL_PATH}")
    
    # Get info about label issues found
    if hasattr(clean_model, 'label_issues_'):
        n_issues = len(clean_model.label_issues_)
        logger.info(f"CleanLearning found {n_issues} label issues and trained on cleaned data")
    
    return clean_model

# =============================================================================
# VISUALIZATION AND REPORTING
# =============================================================================

def create_results_visualizations(issue_records: pd.DataFrame, logger):
    """Create visualizations for the results."""
    logger.info("Creating result visualizations...")
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Plot 1: Distribution of confidence scores for label issues
    plt.figure(figsize=(10, 6))
    plt.hist(issue_records['confidence_score'], bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Confidence Score')
    plt.ylabel('Number of Records')
    plt.title('Distribution of Confidence Scores for Potential Label Issues')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, 'label_issues_confidence.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Predicted probabilities vs actual labels for issues
    plt.figure(figsize=(10, 6))
    colors = ['red' if label == 1 else 'blue' for label in issue_records[TARGET_COL]]
    plt.scatter(issue_records['predicted_prob_positive'], issue_records[TARGET_COL], 
                c=colors, alpha=0.6)
    plt.xlabel('Predicted Probability (Positive Class)')
    plt.ylabel('Actual Label')
    plt.title('Predicted Probabilities vs Actual Labels for Potential Label Issues')
    plt.grid(True, alpha=0.3)
    plt.legend(['Negative', 'Positive'], loc='upper right')
    plt.savefig(os.path.join(RESULTS_DIR, 'predicted_vs_actual.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualizations saved to {RESULTS_DIR}/")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""

    allow_dummy = ALLOW_DUMMY
    skip_missing_weak_features = SKIP_MISSING_WEAK_FEATURES

    logger = setup_logging()
    logger.info("Starting Data Cleaning and Inference Pipeline")
    logger.info(f"Dummy data/features allowed: {allow_dummy}")
    logger.info("=" * 60)
    try:
        # Step 1: Data Loading and Preparation
        logger.info("STEP 1: Data Loading and Preparation")
        X_strong, y_strong, X_weak, y_weak, strong_data, weak_data, skipped_ids = load_and_prepare_data(logger, allow_dummy=allow_dummy, skip_missing_weak_features=skip_missing_weak_features)
        logger.info("=" * 60)
        if skip_missing_weak_features and skipped_ids:
            logger.warning(f"[SKIP] Skipped {len(skipped_ids)} weak metadata records missing from EDA or PRNA features. Full list: {skipped_ids}")

        # Error if no weak data is present
        if len(weak_data) == 0 or len(X_weak) == 0:
            logger.error("No weak data (CODE15) is present. The pipeline requires both strong and weak datasets to run.")
            raise RuntimeError("No weak data (CODE15) is present. The pipeline requires both strong and weak datasets to run.")

        # Step 2: Train Model on Strong Data
        logger.info("STEP 2: Training Model on Strong Data")
        logger.info(f"[DEBUG] X_strong shape: {X_strong.shape}, columns: {list(X_strong.columns)}")
        logger.info(f"[DEBUG] y_strong shape: {y_strong.shape}, values: {y_strong.value_counts().to_dict()}")
        if X_strong.empty:
            logger.error("X_strong is empty after feature selection. No features available for model training.")
        if y_strong.empty:
            logger.error("y_strong is empty after merging. No strong records available for model training.")
        model = train_strong_model(X_strong, y_strong, logger)
        logger.info("=" * 60)
        # Step 3: Find Label Issues in Weak Data
        logger.info("STEP 3: Finding Label Issues in Weak Data")
        issue_records = find_weak_label_issues(model, X_weak, y_weak, weak_data, logger)
        logger.info("=" * 60)

        # Step 4: Demonstrate Inference Pipeline
        logger.info("STEP 4: Demonstrating Inference Pipeline")
        # Use unique record IDs for inference to avoid duplicate predictions
        example_record_ids = list(dict.fromkeys(weak_data[RECORD_ID_COL].head(5).tolist()))
        logger.info(f"Running inference on example records: {example_record_ids}")
        inference_results = run_inference(example_record_ids, allow_dummy=allow_dummy)
        # Ensure the number of predictions matches the number of input records
        if len(inference_results) != len(example_record_ids):
            logger.warning(f"Number of inference results ({len(inference_results)}) does not match number of input records ({len(example_record_ids)}). Results may be incomplete.")
        logger.info("Inference results:")
        logger.info(f"\n{inference_results.to_string(index=False)}")
        inference_results.to_csv("inference_example_results.csv", index=False)
        logger.info("Inference example results saved to inference_example_results.csv")
        logger.info("=" * 60)

        # Step 5: Optional - Train Clean Model
        logger.info("STEP 5: Training CleanLearning Model (Optional)")
        clean_model = train_clean_model(X_weak, y_weak, logger)
        logger.info("=" * 60)

        # Step 6: Create Visualizations
        logger.info("STEP 6: Creating Visualizations")
        if not issue_records.empty:
            create_results_visualizations(issue_records, logger)
        else:
            logger.info("No label issues found - skipping visualization creation")
        logger.info("=" * 60)
        # Final Summary
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("Generated files:")
        logger.info(f"  - {MODEL_PATH}: Trained XGBoost model")
        logger.info(f"  - {LABEL_ISSUES_PATH}: Records with potential label issues")
        logger.info(f"  - {CLEAN_MODEL_PATH}: CleanLearning model")
        logger.info(f"  - inference_example_results.csv: Example inference results")
        logger.info(f"  - {RESULTS_DIR}/: Visualization plots")
        logger.info(f"  - pipeline.log: Detailed execution log")
        logger.info("\nTo use the inference pipeline on new data:")
        logger.info("from run_pipeline import run_inference")
        logger.info("results = run_inference(['record_1', 'record_2', ...], allow_dummy=True)")
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        logger.error("Check the log file for detailed error information")
        raise

if __name__ == "__main__":
    main()
