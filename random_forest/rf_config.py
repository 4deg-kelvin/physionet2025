"""
Configuration file for Random Forest ECG Classifier
"""

import os

# Data configuration
DATA_CONFIG = {
    'sampling_rate': 500,  # ECG sampling rate in Hz
    'signal_length': 10,   # Signal length in seconds
    'leads': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
    'file_extension': '.hea'
}

# Feature extraction configuration
FEATURE_CONFIG = {
    'extract_time_domain': True,
    'extract_frequency_domain': True,
    'extract_morphological': True,
    'extract_statistical': True,
    
    # Statistical features settings
    'include_per_lead_stats': True,
    'include_cross_lead_correlations': True,
    
    # Frequency domain settings
    'fft_window_size': 1024,
    'frequency_bands': {
        'low': (0.5, 4),      # Low frequency band
        'mid': (4, 15),       # Mid frequency band  
        'high': (15, 40)      # High frequency band
    },
    
    # Time domain settings
    'hrv_features': True,
    'rr_interval_features': True,
    'peak_detection_threshold': 0.5
}

# Model configuration
MODEL_CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'validation_size': 0.15,
    
    # Random Forest default parameters
    'default_rf_params': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'bootstrap': True,
        'n_jobs': -1
    },
    
    # Hyperparameter tuning grid
    'param_grid': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None, 0.3, 0.5],
        'bootstrap': [True, False]
    },
    
    # Cross-validation settings
    'cv_folds': 5,
    'cv_scoring': 'roc_auc',
    'cv_shuffle': True
}

# Feature selection configuration
FEATURE_SELECTION_CONFIG = {
    'perform_selection': True,
    'k_best_features': 50,
    'rfe_features': 30,
    'selection_scoring': 'f_classif',
    'recursive_feature_elimination': True,
    'feature_importance_threshold': 0.001
}

# Training configuration
TRAINING_CONFIG = {
    'perform_hyperparameter_tuning': False,  # Set to True for grid search
    'perform_feature_selection': True,
    'perform_cross_validation': True,
    'stratify_splits': True,
    'balance_classes': False,  # Set to True if dealing with imbalanced data
    'class_weight': None,  # Can be 'balanced' or custom dict
    
    # Early stopping (not applicable to RF, but useful for other models)
    'early_stopping': False,
    'patience': 10,
    'min_delta': 0.001
}

# Evaluation configuration
EVALUATION_CONFIG = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    'average': 'binary',  # For binary classification
    'pos_label': 1,
    
    # Threshold optimization
    'optimize_threshold': True,
    'threshold_metric': 'f1',  # Metric to optimize when finding best threshold
    
    # Confidence intervals
    'bootstrap_ci': True,
    'bootstrap_samples': 1000,
    'confidence_level': 0.95
}

# Output configuration
OUTPUT_CONFIG = {
    'save_model': True,
    'save_predictions': True,
    'save_feature_importance': True,
    'save_metrics': True,
    'create_plots': True,
    
    # Plot settings
    'plot_feature_importance': True,
    'plot_roc_curve': True,
    'plot_confusion_matrix': True,
    'plot_precision_recall': True,
    'plot_learning_curves': False,  # Can be computationally expensive
    
    # File formats
    'model_format': 'pkl',  # 'pkl' or 'joblib'
    'plot_format': 'png',   # 'png', 'pdf', 'svg'
    'plot_dpi': 300,
    'results_format': 'csv'  # 'csv', 'json', 'xlsx'
}

# Logging configuration
LOGGING_CONFIG = {
    'log_level': 'INFO',  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    'log_to_file': True,
    'log_file': 'random_forest_training.log',
    'log_format': '%(asctime)s - %(levelname)s - %(message)s',
    'verbose': True
}

# Computational configuration
COMPUTE_CONFIG = {
    'n_jobs': -1,  # Number of CPU cores to use (-1 for all)
    'memory_limit': None,  # Memory limit in GB (None for no limit)
    'chunk_size': 1000,  # For batch processing large datasets
    'use_gpu': False,  # Not applicable for scikit-learn RF
    'parallel_backend': 'threading'  # 'threading' or 'multiprocessing'
}

# Data preprocessing configuration
PREPROCESSING_CONFIG = {
    'normalize_features': True,
    'standardize_features': True,
    'remove_outliers': False,
    'outlier_method': 'iqr',  # 'iqr', 'zscore', 'isolation_forest'
    'outlier_threshold': 3.0,
    
    # Missing value handling
    'handle_missing_values': True,
    'missing_value_strategy': 'median',  # 'mean', 'median', 'mode', 'drop'
    'missing_value_threshold': 0.1,  # Drop features with >10% missing values
    
    # Feature scaling
    'scaling_method': 'standard',  # 'standard', 'minmax', 'robust', 'quantile'
}

# Paths configuration
PATHS_CONFIG = {
    'data_folder': './data',
    'output_folder': './rf_output',
    'model_folder': './models',
    'plots_folder': './plots',
    'logs_folder': './logs',
    'temp_folder': './temp'
}

# Create output directories if they don't exist
def create_output_directories():
    """Create necessary output directories."""
    for folder in PATHS_CONFIG.values():
        os.makedirs(folder, exist_ok=True)

# Validation configuration
VALIDATION_CONFIG = {
    'validate_input_data': True,
    'check_data_quality': True,
    'min_samples_per_class': 10,
    'max_missing_per_sample': 0.2,
    'check_feature_variance': True,
    'min_feature_variance': 1e-6
}

# Export all configurations
__all__ = [
    'DATA_CONFIG',
    'FEATURE_CONFIG', 
    'MODEL_CONFIG',
    'FEATURE_SELECTION_CONFIG',
    'TRAINING_CONFIG',
    'EVALUATION_CONFIG',
    'OUTPUT_CONFIG',
    'LOGGING_CONFIG',
    'COMPUTE_CONFIG',
    'PREPROCESSING_CONFIG',
    'PATHS_CONFIG',
    'VALIDATION_CONFIG',
    'create_output_directories'
]
