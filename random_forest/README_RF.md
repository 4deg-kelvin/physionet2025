# Random Forest ECG Classifier for PhysioNet 2025

This directory contains a comprehensive Random Forest implementation using scikit-learn for ECG-based Chagas disease detection in the PhysioNet 2025 Challenge.

## Overview

The Random Forest classifier is designed to analyze 12-lead ECG signals and predict the presence of Chagas disease. The implementation includes feature extraction, preprocessing, model training, evaluation, and visualization components.

## Files Structure

```
training_data/
├── random_forest_outline.py    # Main Random Forest implementation classes
├── rf_config.py                # Configuration settings and hyperparameters  
├── rf_example.py               # Example usage scripts
├── requirements_rf.txt         # Python dependencies
└── README_RF.md               # This file
```

## Key Components

### 1. ECGFeatureExtractor
Extracts relevant features from ECG signals:
- **Time domain features**: RR intervals, heart rate variability
- **Frequency domain features**: Power spectral density, frequency components
- **Morphological features**: QRS complex characteristics, wave measurements
- **Statistical features**: Mean, std, skewness, kurtosis for each lead

### 2. ECGDataProcessor
Handles data loading and preprocessing:
- Loads ECG records from various formats
- Applies feature scaling and normalization
- Handles missing values and outliers
- Prepares data for machine learning

### 3. RandomForestTrainer
Manages model training and evaluation:
- Hyperparameter tuning with GridSearchCV
- Feature selection using SelectKBest and RFE
- Cross-validation for model assessment
- Performance evaluation with multiple metrics

### 4. ModelPersistence
Handles saving and loading trained models:
- Saves model, scaler, and feature names
- Enables model deployment and reuse

### 5. ResultsVisualizer
Creates visualizations for analysis:
- Feature importance plots
- ROC curves and AUC scores
- Confusion matrices
- Precision-recall curves

## Installation

1. Install required dependencies:
```bash
pip install -r requirements_rf.txt
```

2. Ensure you have the helper_code.py and other necessary data processing scripts in the same directory.

## Usage

### Quick Start with Demo Data

Run the demo with synthetic data to test the implementation:

```bash
python rf_example.py --mode demo --output_dir ./demo_results
```

### Training with Real ECG Data

For quick training without hyperparameter tuning:

```bash
python rf_example.py --mode quick --data_folder /path/to/ecg/data --output_dir ./quick_results
```

For advanced training with hyperparameter tuning:

```bash
python rf_example.py --mode advanced --data_folder /path/to/ecg/data --output_dir ./advanced_results
```

### Using the Main Training Script

Run the full training pipeline:

```bash
python random_forest_outline.py \
    --data_folder /path/to/ecg/data \
    --output_dir ./rf_output \
    --hyperparameter_tuning \
    --feature_selection \
    --cross_validation \
    --visualize
```

## Configuration

Modify `rf_config.py` to adjust:

- **Data settings**: Sampling rate, signal length, ECG leads
- **Feature extraction**: Which feature types to extract
- **Model parameters**: Random Forest hyperparameters
- **Training settings**: Cross-validation, feature selection
- **Output options**: Which plots and files to save

## Feature Extraction Details

The implementation extracts several types of features:

### Statistical Features (Per Lead)
- Mean, standard deviation
- Skewness, kurtosis  
- Min, max, range
- Signal energy

### Time Domain Features
- Heart rate variability metrics
- RR interval statistics
- Peak detection and analysis
- Temporal morphological features

### Frequency Domain Features
- Power spectral density
- Frequency band powers (low, mid, high)
- Dominant frequency components
- Spectral entropy

### Morphological Features
- QRS complex characteristics
- P-wave and T-wave measurements
- Wave duration and amplitude
- Inter-lead correlations

## Model Training Pipeline

1. **Data Loading**: Load ECG records and labels
2. **Feature Extraction**: Extract comprehensive feature set
3. **Preprocessing**: Scale features and handle missing values
4. **Feature Selection**: Select most informative features
5. **Model Training**: Train Random Forest with cross-validation
6. **Hyperparameter Tuning**: Optimize model parameters (optional)
7. **Evaluation**: Assess performance on test set
8. **Visualization**: Create plots for analysis
9. **Model Saving**: Save trained model for deployment

## Expected Outputs

After training, the following files will be generated:

- `random_forest_model.pkl`: Trained model and preprocessors
- `feature_importance.png`: Feature importance visualization
- `roc_curve.png`: ROC curve and AUC score
- `confusion_matrix.png`: Confusion matrix
- `training_log.txt`: Detailed training log
- `metrics_summary.csv`: Performance metrics summary

## Performance Metrics

The implementation evaluates models using:
- **Accuracy**: Overall classification accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under ROC curve
- **Precision/Recall**: Class-specific performance
- **Cross-validation scores**: Model stability assessment

## Customization

### Adding New Features

To add custom features, extend the `ECGFeatureExtractor` class:

```python
def extract_custom_features(self, signal: np.ndarray) -> Dict[str, float]:
    # Implement your custom feature extraction
    custom_features = {}
    # ... your code here ...
    return custom_features
```

### Modifying Model Parameters

Update the hyperparameter grid in `rf_config.py`:

```python
MODEL_CONFIG['param_grid'] = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, None],
    # Add your parameters
}
```

### Custom Evaluation Metrics

Add custom metrics to the evaluation pipeline:

```python
from sklearn.metrics import your_custom_metric

# In the evaluation function
custom_score = your_custom_metric(y_true, y_pred)
```

## Implementation Notes

### Current Limitations

1. **Signal Loading**: The `_load_ecg_signal()` method needs to be implemented based on your specific data format
2. **Feature Implementation**: Some advanced feature extraction methods are outlined but need implementation
3. **Memory Usage**: Large datasets may require batch processing

### TODO Items

- [ ] Implement ECG signal loading for your data format
- [ ] Complete advanced feature extraction methods
- [ ] Add support for multi-class classification
- [ ] Implement ensemble methods with other classifiers
- [ ] Add real-time prediction capability

### Performance Optimization

For large datasets:
- Use `n_jobs=-1` for parallel processing
- Implement batch processing for feature extraction
- Consider feature selection to reduce dimensionality
- Use early stopping for hyperparameter tuning

## Integration with PhysioNet Challenge

To integrate with the official challenge structure:

1. Ensure compatibility with `helper_code.py`
2. Implement the required evaluation functions
3. Format outputs according to challenge specifications
4. Test with official evaluation scripts

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Issues**: Reduce batch size or use feature selection
3. **Slow Training**: Reduce hyperparameter grid size or use fewer features
4. **Poor Performance**: Check data quality and feature relevance

### Debug Mode

Enable detailed logging by setting in `rf_config.py`:
```python
LOGGING_CONFIG['log_level'] = 'DEBUG'
LOGGING_CONFIG['verbose'] = True
```

## Contributing

To contribute improvements:
1. Follow the existing code structure
2. Add comprehensive docstrings
3. Include unit tests for new features
4. Update this README with new functionality

## References

- PhysioNet 2025 Challenge: [Official Website]
- Scikit-learn Random Forest: https://scikit-learn.org/stable/modules/ensemble.html#forest
- ECG Feature Extraction: Relevant papers and methods
- Chagas Disease Detection: Clinical literature

## Contact

For questions about this implementation, please refer to the PhysioNet 2025 Challenge documentation or create an issue in the repository.
