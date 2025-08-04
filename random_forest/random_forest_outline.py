#!/usr/bin/env python
"""
Random Forest ML Model Outline for PhysioNet 2025 Challenge
===========================================================

This outline provides a comprehensive structure for implementing a Random Forest
classifier using scikit-learn for ECG-based Chagas disease detection.

Author: Generated Outline
Date: June 2025
"""

import argparse
import numpy as np
import pandas as pd
import os
import pickle
from typing import Tuple, List, Dict, Any
from pathlib import Path

# Scikit-learn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score, 
    GridSearchCV,
    StratifiedKFold
)
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    accuracy_score,
    f1_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, RFE

# Data processing imports
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Local imports
# Add EDA directory to sys.path to import helper_code
eda_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'EDA'))
if eda_dir not in sys.path:
    sys.path.insert(0, eda_dir)
from helper_code import find_records, load_label, load_signal

class ECGFeatureExtractor:
    """
    Extract relevant features from ECG signals for Random Forest classification.
    
    Features to extract:
    - Time domain features (RR intervals, heart rate variability)
    - Frequency domain features (power spectral density)
    - Morphological features (QRS complex characteristics)
    - Statistical features (mean, std, skewness, kurtosis)
    """
    
    def __init__(self, sampling_rate: int = 500):
        self.sampling_rate = sampling_rate
        
    def extract_time_domain_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extract time-domain features from ECG signal.
        
        Returns:
            Dictionary containing time-domain features
        """
        # TODO: Implement feature extraction
        # - Heart rate variability metrics
        # - RR interval statistics
        # - Signal amplitude statistics
        pass
    
    def extract_frequency_domain_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extract frequency-domain features using FFT.
        
        Returns:
            Dictionary containing frequency-domain features
        """
        # TODO: Implement FFT-based feature extraction
        # - Power spectral density features
        # - Dominant frequency components
        pass
    
    def extract_morphological_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extract morphological features from ECG waveform.
        
        Returns:
            Dictionary containing morphological features
        """
        # TODO: Implement morphological feature extraction
        # - QRS complex detection and analysis
        # - P-wave and T-wave characteristics
        # - Wave duration and amplitude measurements
        pass
    
    def extract_statistical_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extract basic statistical features from each lead.
        
        Returns:
            Dictionary containing statistical features
        """
        features = {}
        for lead_idx in range(signal.shape[0]):
            lead_signal = signal[lead_idx, :]
            features.update({
                f'lead_{lead_idx}_mean': np.mean(lead_signal),
                f'lead_{lead_idx}_std': np.std(lead_signal),
                f'lead_{lead_idx}_skewness': self._calculate_skewness(lead_signal),
                f'lead_{lead_idx}_kurtosis': self._calculate_kurtosis(lead_signal),
                f'lead_{lead_idx}_min': np.min(lead_signal),
                f'lead_{lead_idx}_max': np.max(lead_signal),
                f'lead_{lead_idx}_range': np.max(lead_signal) - np.min(lead_signal),
                f'lead_{lead_idx}_energy': np.sum(lead_signal ** 2)
            })
        return features
    
    def _calculate_skewness(self, signal: np.ndarray) -> float:
        """Calculate skewness of signal."""
        # TODO: Implement skewness calculation
        pass
    
    def _calculate_kurtosis(self, signal: np.ndarray) -> float:
        """Calculate kurtosis of signal."""
        return float(kurtosis(signal))
    
    def extract_all_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extract all feature types from ECG signal.
        
        Args:
            signal: ECG signal array of shape (n_leads, n_samples)
            
        Returns:
            Dictionary containing all extracted features
        """
        features = {}
        
        # Extract different feature types
        features.update(self.extract_statistical_features(signal))
        features.update(self.extract_time_domain_features(signal))
        features.update(self.extract_frequency_domain_features(signal))
        features.update(self.extract_morphological_features(signal))
        
        return features

class ECGDataProcessor:
    """
    Process ECG data for Random Forest training.
    """
    
    def __init__(self, data_folder: str, feature_extractor: ECGFeatureExtractor):
        self.data_folder = data_folder
        self.feature_extractor = feature_extractor
        self.scaler = StandardScaler()
        
    def load_and_process_dataset(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load ECG records and extract features for training.
        
        Returns:
            Tuple of (features, labels, feature_names)
        """
        records = find_records(self.data_folder)
        print(f"Found {len(records)} records")
        
        features_list = []
        labels_list = []
        feature_names = None
        
        for record in tqdm(records, desc="Processing ECG records"):
            try:
                # Load signal and label
                signal = self._load_ecg_signal(record)
                label = load_label(record)
                
                # Extract features
                features = self.feature_extractor.extract_all_features(signal)
                
                if feature_names is None:
                    feature_names = list(features.keys())
                
                features_list.append(list(features.values()))
                labels_list.append(label)
                
            except Exception as e:
                print(f"Error processing record {record}: {e}")
                continue
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        return X, y, feature_names
    
    def _load_ecg_signal(self, record: str) -> np.ndarray:
        """Load ECG signal from record file."""
        # TODO: Implement signal loading based on your data format
        # This should return a numpy array of shape (n_leads, n_samples)
        signal = load_signal(record)  # Placeholder - implement based on your data format
        return signal
    
    def preprocess_features(self, X: np.ndarray, fit_scaler: bool = True) -> np.ndarray:
        """
        Preprocess features (scaling, normalization).
        
        Args:
            X: Feature matrix
            fit_scaler: Whether to fit the scaler (True for training data)
            
        Returns:
            Preprocessed features
        """
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled

class RandomForestTrainer:
    """
    Train and evaluate Random Forest classifier for ECG-based Chagas detection.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.feature_selector = None
        self.best_params = None
        
    def train_model(self, 
                   X: np.ndarray, 
                   y: np.ndarray, 
                   feature_names: List[str],
                   test_size: float = 0.2,
                   perform_hyperparameter_tuning: bool = True,
                   feature_selection: bool = True) -> Dict[str, Any]:
        """
        Train Random Forest model with optional hyperparameter tuning and feature selection.
        
        Args:
            X: Feature matrix
            y: Labels
            feature_names: List of feature names
            test_size: Proportion of data for testing
            perform_hyperparameter_tuning: Whether to perform grid search
            feature_selection: Whether to perform feature selection
            
        Returns:
            Dictionary containing training results and metrics
        """
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, 
            stratify=y, random_state=self.random_state
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        print(f"Class distribution in training: {np.bincount(y_train)}")
        
        # Feature selection
        if feature_selection:
            X_train, selected_features = self._perform_feature_selection(
                X_train, y_train, feature_names
            )
            X_test = self.feature_selector.transform(X_test)
        else:
            selected_features = feature_names
        
        # Hyperparameter tuning
        if perform_hyperparameter_tuning:
            self.model = self._perform_hyperparameter_tuning(X_train, y_train)
        else:
            # Use default parameters
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_metrics = self._evaluate_model(X_train, y_train, "Training")
        test_metrics = self._evaluate_model(X_test, y_test, "Test")
        
        # Feature importance analysis
        feature_importance = self._analyze_feature_importance(selected_features)
        
        results = {
            'model': self.model,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': feature_importance,
            'selected_features': selected_features,
            'X_test': X_test,
            'y_test': y_test
        }
        
        return results
    
    def _perform_feature_selection(self, 
                                 X: np.ndarray, 
                                 y: np.ndarray, 
                                 feature_names: List[str],
                                 k_best: int = 50) -> Tuple[np.ndarray, List[str]]:
        """
        Perform feature selection using SelectKBest and RFE.
        
        Returns:
            Tuple of (selected_features_matrix, selected_feature_names)
        """
        print("Performing feature selection...")
        
        # First, use SelectKBest to reduce dimensionality
        selector_k_best = SelectKBest(score_func=f_classif, k=min(k_best, X.shape[1]))
        X_selected = selector_k_best.fit_transform(X, y)
        
        # Get selected feature names
        selected_indices = selector_k_best.get_support(indices=True)
        intermediate_features = [feature_names[i] for i in selected_indices]
        
        # Then use RFE with Random Forest for final selection
        rf_for_selection = RandomForestClassifier(
            n_estimators=50, random_state=self.random_state, n_jobs=-1
        )
        
        rfe = RFE(estimator=rf_for_selection, n_features_to_select=min(30, X_selected.shape[1]))
        X_final = rfe.fit_transform(X_selected, y)
        
        # Get final selected feature names
        final_selected_indices = rfe.get_support(indices=True)
        selected_features = [intermediate_features[i] for i in final_selected_indices]
        
        self.feature_selector = lambda x: rfe.transform(selector_k_best.transform(x))
        
        print(f"Selected {len(selected_features)} features out of {len(feature_names)}")
        
        return X_final, selected_features
    
    def _perform_hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Returns:
            Best Random Forest model
        """
        print("Performing hyperparameter tuning...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        # Create base model
        rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        
        # Perform grid search with cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        self.best_params = grid_search.best_params_
        print(f"Best parameters: {self.best_params}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def _evaluate_model(self, X: np.ndarray, y: np.ndarray, dataset_name: str) -> Dict[str, float]:
        """
        Evaluate model performance on given dataset.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'f1_score': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_pred_proba)
        }
        
        print(f"\n{dataset_name} Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        
        # Print detailed classification report
        print(f"\n{dataset_name} Classification Report:")
        print(classification_report(y, y_pred))
        
        return metrics
    
    def _analyze_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """
        Analyze and return feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        importances = self.model.feature_importances_
        feature_importance = dict(zip(feature_names, importances))
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        print("\nTop 10 Most Important Features:")
        for feature, importance in sorted_features[:10]:
            print(f"{feature}: {importance:.4f}")
        
        return feature_importance
    
    def cross_validate_model(self, 
                           X: np.ndarray, 
                           y: np.ndarray, 
                           cv_folds: int = 5) -> Dict[str, np.ndarray]:
        """
        Perform cross-validation to assess model stability.
        
        Returns:
            Dictionary containing cross-validation scores
        """
        print(f"Performing {cv_folds}-fold cross-validation...")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        scoring_metrics = ['accuracy', 'f1', 'roc_auc']
        cv_results = {}
        
        for metric in scoring_metrics:
            scores = cross_val_score(self.model, X, y, cv=cv, scoring=metric, n_jobs=-1)
            cv_results[metric] = scores
            
            print(f"{metric.upper()} - Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return cv_results

class ModelPersistence:
    """
    Handle saving and loading of trained models and preprocessors.
    """
    
    @staticmethod
    def save_model(model: RandomForestClassifier, 
                   scaler: StandardScaler, 
                   feature_names: List[str],
                   save_path: str):
        """
        Save trained model and associated components.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        model_data = {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {save_path}")
    
    @staticmethod
    def load_model(load_path: str) -> Tuple[RandomForestClassifier, StandardScaler, List[str]]:
        """
        Load trained model and associated components.
        
        Returns:
            Tuple of (model, scaler, feature_names)
        """
        with open(load_path, 'rb') as f:
            model_data = pickle.load(f)
        
        return model_data['model'], model_data['scaler'], model_data['feature_names']

class ResultsVisualizer:
    """
    Create visualizations for model results and analysis.
    """
    
    @staticmethod
    def plot_feature_importance(feature_importance: Dict[str, float], 
                              top_n: int = 20, 
                              save_path: str = None):
        """
        Plot feature importance scores.
        """
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        features, importances = zip(*sorted_features[:top_n])
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, 
                      y_pred_proba: np.ndarray, 
                      save_path: str = None):
        """
        Plot ROC curve.
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, 
                            y_pred: np.ndarray, 
                            save_path: str = None):
        """
        Plot confusion matrix.
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Chagas', 'Chagas'],
                   yticklabels=['No Chagas', 'Chagas'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """
    Main function to orchestrate the Random Forest training pipeline.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Random Forest for ECG-based Chagas detection')
    parser.add_argument('--data_folder', type=str, required=True,
                       help='Path to folder containing ECG data')
    parser.add_argument('--output_dir', type=str, default='./rf_output',
                       help='Directory to save results and model')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Proportion of data for testing')
    parser.add_argument('--hyperparameter_tuning', action='store_true',
                       help='Perform hyperparameter tuning')
    parser.add_argument('--feature_selection', action='store_true',
                       help='Perform feature selection')
    parser.add_argument('--cross_validation', action='store_true',
                       help='Perform cross-validation')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization plots')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== Random Forest ECG Classifier Training ===")
    print(f"Data folder: {args.data_folder}")
    print(f"Output directory: {args.output_dir}")
    
    # Step 1: Initialize components
    feature_extractor = ECGFeatureExtractor(sampling_rate=500)
    data_processor = ECGDataProcessor(args.data_folder, feature_extractor)
    trainer = RandomForestTrainer(random_state=42)
    
    # Step 2: Load and process data
    print("\n=== Loading and Processing Data ===")
    X, y, feature_names = data_processor.load_and_process_dataset()
    
    # Step 3: Preprocess features
    X_processed = data_processor.preprocess_features(X, fit_scaler=True)
    
    print(f"Dataset shape: {X_processed.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Step 4: Train model
    print("\n=== Training Random Forest Model ===")
    results = trainer.train_model(
        X_processed, y, feature_names,
        test_size=args.test_size,
        perform_hyperparameter_tuning=args.hyperparameter_tuning,
        feature_selection=args.feature_selection
    )
    
    # Step 5: Cross-validation (optional)
    if args.cross_validation:
        print("\n=== Cross-Validation ===")
        cv_results = trainer.cross_validate_model(X_processed, y)
    
    # Step 6: Save model
    print("\n=== Saving Model ===")
    model_path = os.path.join(args.output_dir, 'random_forest_model.pkl')
    ModelPersistence.save_model(
        results['model'], 
        data_processor.scaler, 
        results['selected_features'],
        model_path
    )
    
    # Step 7: Create visualizations (optional)
    if args.visualize:
        print("\n=== Creating Visualizations ===")
        visualizer = ResultsVisualizer()
        
        # Feature importance plot
        importance_path = os.path.join(args.output_dir, 'feature_importance.png')
        visualizer.plot_feature_importance(
            results['feature_importance'], 
            save_path=importance_path
        )
        
        # ROC curve
        y_pred_proba = results['model'].predict_proba(results['X_test'])[:, 1]
        roc_path = os.path.join(args.output_dir, 'roc_curve.png')
        visualizer.plot_roc_curve(
            results['y_test'], 
            y_pred_proba, 
            save_path=roc_path
        )
        
        # Confusion matrix
        y_pred = results['model'].predict(results['X_test'])
        cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
        visualizer.plot_confusion_matrix(
            results['y_test'], 
            y_pred, 
            save_path=cm_path
        )
    
    print("\n=== Training Complete ===")
    print(f"Final Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
    print(f"Final Test F1 Score: {results['test_metrics']['f1_score']:.4f}")
    print(f"Final Test ROC AUC: {results['test_metrics']['roc_auc']:.4f}")
    
    return results

if __name__ == "__main__":
    results = main()
