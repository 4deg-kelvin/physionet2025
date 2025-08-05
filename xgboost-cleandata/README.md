# XGBoost CleanData Pipeline

A robust data cleaning and inference pipeline using XGBoost and Cleanlab for the PhysioNet 2025 Chagas Challenge.

## Overview

This pipeline accomplishes two main objectives:

1. **Data Cleaning**: Identifies potential labeling errors in a "weak" dataset (CODE15) by training a model on a "strong" dataset (PTB-XL, Samitrop)
2. **Inference Pipeline**: Creates a reusable inference function that can generate features and predict on new, unseen data

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
python run_pipeline.py
```

## Pipeline Components

### 1. Data Loading and Preparation
- Loads metadata and pre-computed features (PRNA, EDA)
- Merges datasets on `record_id` 
- Handles missing values gracefully
- Splits data into STRONG (PTB-XL, Samitrop) and WEAK (CODE15) datasets

### 2. Model Training
- Trains XGBoost classifier on the strong dataset
- Uses optimized hyperparameters for binary classification
- Saves trained model to `xgboost_strong_model.json`

### 3. Label Issue Detection
- Generates predicted probabilities for weak dataset
- Uses Cleanlab's `find_label_issues` to identify potential mislabeled records
- Saves suspicious records to `weak_data_label_issues.csv`

### 4. Inference Pipeline
- **`run_inference(record_ids)`**: Main inference function
- Generates features for new record IDs
- Combines PRNA and EDA features
- Loads trained model and generates predictions
- Returns DataFrame with probabilities and predicted classes

### 5. CleanLearning (Optional)
- Demonstrates robust training on noisy data using Cleanlab
- Automatically detects and handles label errors during training
- Saves clean model to `xgboost_clean_model.json`

### 6. Visualization and Reporting
- Creates confidence score distributions
- Plots predicted vs actual labels for potential issues
- Saves visualizations to `results/` directory

## Generated Files

After running the pipeline, the following files are created:

- `xgboost_strong_model.json`: Trained XGBoost model on strong data
- `xgboost_clean_model.json`: CleanLearning model trained on weak data
- `weak_data_label_issues.csv`: Records with potential label issues
- `inference_example_results.csv`: Example inference results
- `pipeline.log`: Detailed execution log
- `results/`: Directory containing visualization plots
  - `label_issues_confidence.png`: Distribution of confidence scores
  - `predicted_vs_actual.png`: Predicted probabilities vs actual labels

## Usage for Inference

To use the trained model for inference on new data:

```python
from run_pipeline import run_inference

# Generate predictions for new record IDs
record_ids = ['new_record_1', 'new_record_2', 'new_record_3']
results = run_inference(record_ids)

print(results)
# Output:
#     record_id  predicted_prob_negative  predicted_prob_positive  predicted_class
# 0  new_record_1                    0.75                     0.25                0
# 1  new_record_2                    0.30                     0.70                1
# 2  new_record_3                    0.85                     0.15                0
```

## Configuration

Key parameters can be modified at the top of `run_pipeline.py`:

```python
# File paths
METADATA_PATH = "/path/to/metadata.csv"
PRNA_FEATURES_PATH = "/path/to/prna_outputs_modified.json"
EDA_FEATURES_PATH = "/path/to/eda_features.csv"

# Dataset categories
STRONG_DATASETS = ["PTB-XL", "Samitrop"]
WEAK_DATASETS = ["CODE15"]

# Column names
RECORD_ID_COL = "record_id"
TARGET_COL = "chagas"
```

## Data Sources

The pipeline expects the following data files:

1. **Metadata** (`metadata.csv`):
   - `record_id`: Unique identifier for each ECG record
   - `dataset`: Source dataset ('PTB-XL', 'Samitrop', 'CODE15')
   - `chagas`: Target label (0 or 1)

2. **PRNA Features** (`combined_features.csv`):
   - Pre-computed PRNA features for each record

3. **EDA Features** (`eda_features.csv`):
   - Pre-computed EDA/morphological features for each record

## Feature Generation Functions

The pipeline includes placeholder functions for feature generation:

- `get_prna_features(record_ids)`: Generates PRNA features
- `get_eda_features(record_ids)`: Generates EDA features

**Note**: Replace these placeholder functions with actual feature generation code for production use.

## Label Issue Analysis

The pipeline identifies potential label issues by:

1. Training a robust model on high-quality strong data
2. Applying this model to predict labels for weak data
3. Using Cleanlab to find discrepancies between predicted and actual labels
4. Ranking issues by confidence scores

Records with low confidence in their current labels are flagged as potential issues.

## Performance Metrics

The pipeline provides:

- Training accuracy on strong dataset
- Number and percentage of potential label issues found
- Confidence score distributions
- Visual analysis of prediction quality

## Troubleshooting

1. **Missing data files**: The pipeline creates synthetic data for demonstration if actual files are not found
2. **Memory issues**: Reduce dataset size or adjust XGBoost parameters
3. **Feature mismatch**: Ensure inference features match training features exactly
4. **Model loading errors**: Check that model files exist and are not corrupted

## Dependencies

- pandas: Data manipulation and analysis
- numpy: Numerical computing
- xgboost: Gradient boosting classifier
- cleanlab: Label error detection and robust learning
- scikit-learn: Machine learning utilities
- matplotlib/seaborn: Visualization

## Contact

For questions or issues, refer to the pipeline logs or contact the development team.
