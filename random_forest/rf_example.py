#!/usr/bin/env python
"""
Example usage of the Random Forest ECG Classifier

This script demonstrates how to use the random_forest_outline.py classes
for training a Random Forest model on ECG data for Chagas disease detection.

Usage:
    python rf_example.py --data_folder /path/to/ecg/data --output_dir ./results
"""

import argparse
import os
import numpy as np
from pathlib import Path

# Import the Random Forest components
from .random_forest_outline import (
    ECGFeatureExtractor,
    ECGDataProcessor, 
    RandomForestTrainer,
    ModelPersistence,
    ResultsVisualizer
)

# Import configuration
from .rf_config import (
    DATA_CONFIG,
    MODEL_CONFIG,
    TRAINING_CONFIG,
    OUTPUT_CONFIG,
    create_output_directories
)

def quick_example(data_folder: str, output_dir: str = './rf_example_output'):
    """
    Quick example of Random Forest training pipeline.
    
    Args:
        data_folder: Path to ECG data folder
        output_dir: Directory to save results
    """
    
    print("=== Random Forest ECG Classifier - Quick Example ===\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Initialize components
    print("1. Initializing components...")
    feature_extractor = ECGFeatureExtractor(
        sampling_rate=DATA_CONFIG['sampling_rate']
    )
    
    data_processor = ECGDataProcessor(data_folder, feature_extractor)
    
    trainer = RandomForestTrainer(
        random_state=MODEL_CONFIG['random_state']
    )
    
    # Step 2: Load and process data
    print("2. Loading and processing ECG data...")
    try:
        X, y, feature_names = data_processor.load_and_process_dataset()
        print(f"   - Loaded {X.shape[0]} samples with {X.shape[1]} features")
        print(f"   - Class distribution: {np.bincount(y)}")
    except Exception as e:
        print(f"   Error loading data: {e}")
        print("   Note: You need to implement the signal loading functions in the outline")
        return None
    
    # Step 3: Preprocess features  
    print("3. Preprocessing features...")
    X_processed = data_processor.preprocess_features(X, fit_scaler=True)
    
    # Step 4: Train model with basic settings
    print("4. Training Random Forest model...")
    results = trainer.train_model(
        X_processed, 
        y, 
        feature_names,
        test_size=MODEL_CONFIG['test_size'],
        perform_hyperparameter_tuning=TRAINING_CONFIG['perform_hyperparameter_tuning'],
        feature_selection=TRAINING_CONFIG['perform_feature_selection']
    )
    
    # Step 5: Save model
    print("5. Saving model...")
    model_path = os.path.join(output_dir, 'rf_model.pkl')
    ModelPersistence.save_model(
        results['model'],
        data_processor.scaler,
        results['selected_features'],
        model_path
    )
    
    # Step 6: Create basic visualizations
    if OUTPUT_CONFIG['create_plots']:
        print("6. Creating visualizations...")
        visualizer = ResultsVisualizer()
        
        # Feature importance
        importance_path = os.path.join(output_dir, 'feature_importance.png')
        visualizer.plot_feature_importance(
            results['feature_importance'],
            top_n=20,
            save_path=importance_path
        )
        
        # ROC curve
        y_pred_proba = results['model'].predict_proba(results['X_test'])[:, 1]
        roc_path = os.path.join(output_dir, 'roc_curve.png')
        visualizer.plot_roc_curve(
            results['y_test'],
            y_pred_proba,
            save_path=roc_path
        )
    
    print("\n=== Example Complete ===")
    print(f"Results saved to: {output_dir}")
    print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
    print(f"Test F1 Score: {results['test_metrics']['f1_score']:.4f}")
    print(f"Test ROC AUC: {results['test_metrics']['roc_auc']:.4f}")
    
    return results

def advanced_example(data_folder: str, output_dir: str = './rf_advanced_output'):
    """
    Advanced example with hyperparameter tuning and cross-validation.
    
    Args:
        data_folder: Path to ECG data folder
        output_dir: Directory to save results
    """
    
    print("=== Random Forest ECG Classifier - Advanced Example ===\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    feature_extractor = ECGFeatureExtractor(sampling_rate=DATA_CONFIG['sampling_rate'])
    data_processor = ECGDataProcessor(data_folder, feature_extractor)
    trainer = RandomForestTrainer(random_state=MODEL_CONFIG['random_state'])
    
    # Load and process data
    print("Loading and processing data...")
    try:
        X, y, feature_names = data_processor.load_and_process_dataset()
        X_processed = data_processor.preprocess_features(X, fit_scaler=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Train with hyperparameter tuning
    print("Training with hyperparameter tuning...")
    results = trainer.train_model(
        X_processed,
        y,
        feature_names,
        test_size=0.2,
        perform_hyperparameter_tuning=True,  # Enable grid search
        feature_selection=True
    )
    
    # Perform cross-validation
    print("Performing cross-validation...")
    cv_results = trainer.cross_validate_model(X_processed, y, cv_folds=5)
    
    # Save comprehensive results
    model_path = os.path.join(output_dir, 'rf_advanced_model.pkl')
    ModelPersistence.save_model(
        results['model'],
        data_processor.scaler,
        results['selected_features'],
        model_path
    )
    
    # Create all visualizations
    visualizer = ResultsVisualizer()
    
    # Feature importance
    visualizer.plot_feature_importance(
        results['feature_importance'],
        save_path=os.path.join(output_dir, 'feature_importance.png')
    )
    
    # ROC curve
    y_pred_proba = results['model'].predict_proba(results['X_test'])[:, 1]
    visualizer.plot_roc_curve(
        results['y_test'],
        y_pred_proba,
        save_path=os.path.join(output_dir, 'roc_curve.png')
    )
    
    # Confusion matrix
    y_pred = results['model'].predict(results['X_test'])
    visualizer.plot_confusion_matrix(
        results['y_test'],
        y_pred,
        save_path=os.path.join(output_dir, 'confusion_matrix.png')
    )
    
    print("\n=== Advanced Example Complete ===")
    print(f"Best parameters: {trainer.best_params}")
    print(f"Test Performance: ACC={results['test_metrics']['accuracy']:.4f}, "
          f"F1={results['test_metrics']['f1_score']:.4f}, "
          f"AUC={results['test_metrics']['roc_auc']:.4f}")
    
    return results

def demo_with_synthetic_data(output_dir: str = './rf_demo_output'):
    """
    Demonstration using synthetic data when real ECG data is not available.
    
    Args:
        output_dir: Directory to save results
    """
    
    print("=== Random Forest Demo with Synthetic Data ===\n")
    
    # Create synthetic ECG-like data
    print("Creating synthetic ECG data...")
    np.random.seed(42)
    
    n_samples = 1000
    n_features = 100  # Simulating extracted ECG features
    
    # Create features with some correlation structure
    X = np.random.randn(n_samples, n_features)
    
    # Add some signal to make classification non-trivial
    # Features 0-9 are correlated with the outcome
    signal_features = X[:, :10]
    signal_strength = np.sum(signal_features, axis=1)
    
    # Create binary labels based on signal
    y = (signal_strength > np.median(signal_strength)).astype(int)
    
    # Add some noise to make it more realistic
    noise_factor = 0.3
    flip_indices = np.random.choice(n_samples, int(noise_factor * n_samples), replace=False)
    y[flip_indices] = 1 - y[flip_indices]
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    print(f"   - Generated {n_samples} samples with {n_features} features")
    print(f"   - Class distribution: {np.bincount(y)}")
    
    # Initialize trainer
    trainer = RandomForestTrainer(random_state=42)
    
    # Create a dummy scaler for consistency
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    print("Training Random Forest on synthetic data...")
    results = trainer.train_model(
        X_scaled,
        y,
        feature_names,
        test_size=0.2,
        perform_hyperparameter_tuning=False,
        feature_selection=True
    )
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'rf_demo_model.pkl')
    ModelPersistence.save_model(
        results['model'],
        scaler,
        results['selected_features'],
        model_path
    )
    
    # Create visualizations
    visualizer = ResultsVisualizer()
    
    visualizer.plot_feature_importance(
        results['feature_importance'],
        save_path=os.path.join(output_dir, 'demo_feature_importance.png')
    )
    
    y_pred_proba = results['model'].predict_proba(results['X_test'])[:, 1]
    visualizer.plot_roc_curve(
        results['y_test'],
        y_pred_proba,
        save_path=os.path.join(output_dir, 'demo_roc_curve.png')
    )
    
    print("\n=== Demo Complete ===")
    print(f"Demo results saved to: {output_dir}")
    print("Note: This used synthetic data for demonstration purposes")
    
    return results

def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(
        description='Random Forest ECG Classifier Example'
    )
    parser.add_argument(
        '--data_folder', 
        type=str,
        help='Path to ECG data folder'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='./rf_output',
        help='Output directory for results'
    )
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['quick', 'advanced', 'demo'],
        default='demo',
        help='Example mode to run'
    )
    
    args = parser.parse_args()
    
    # Create output directories
    create_output_directories()
    
    if args.mode == 'demo':
        print("Running demo with synthetic data...")
        results = demo_with_synthetic_data(args.output_dir)
        
    elif args.mode == 'quick':
        if not args.data_folder:
            print("Error: --data_folder required for quick mode")
            return
        results = quick_example(args.data_folder, args.output_dir)
        
    elif args.mode == 'advanced':
        if not args.data_folder:
            print("Error: --data_folder required for advanced mode")
            return
        results = advanced_example(args.data_folder, args.output_dir)
    
    return results

if __name__ == "__main__":
    results = main()
