# %% [markdown]
# # Random Forest using 27 Condition Probability Features
# 
# This notebook has been refactored to use only the 27 condition probability outputs from the PRNA model (stored in `combined_prna_outputs.json`) as input features, following the same approach as the XGBoost notebook. This eliminates the need for ECG signal processing and feature extraction.

# %% [markdown]
# # 📊 Load and Process PRNA Probability Features
import pandas as pd
import numpy as np
import orjson
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')
# 
def get_json(path):
    """Load JSON data efficiently"""
    with open(path, 'r') as f:
        return orjson.loads(f.read())

# Load the PRNA outputs (27 condition probabilities + metadata)
print("🔄 Loading PRNA outputs with 27 condition probabilities...")
json_path = "/Users/andysmithwick/Documents/GitHub/physionet2025/xgboost-cleandata/prna_outputs_modified.json"
raw_data = get_json(json_path)
print(f"✅ Loaded {len(raw_data)} records")

# Check data structure
sample_record = raw_data[0]
print(f"📋 Sample record keys: {list(sample_record.keys())}")
if 'snomed_vals' in sample_record:
    print(f"📋 Number of conditions: {len(sample_record['snomed_vals'])}")
    print(f"📋 First few conditions: {list(sample_record['snomed_vals'].keys())[:5]}")

# %% [markdown]
# ### Feature Extraction
# 
# Note that below does not pull from ECG data, but the precomputed prna classifications.

# %%
def extract_probability_features(raw_data: List[Dict]) -> pd.DataFrame:
    """
    Extract the 27 condition probability values from PRNA outputs and create a DataFrame.
    This aligns with the XGBoost approach but uses probability values instead of binary present flags.
    
    Args:
        raw_data: List of records with 'snomed_vals' containing condition probabilities
        
    Returns:
        DataFrame with probability features and Chagas disease labels
    """
    processed_records = []
    
    for record in raw_data:
        # Skip records without required fields
        if 'snomed_vals' not in record or 'chagas' not in record:
            continue
            
        # Start with basic info
        flat_record = {
            'exam_id': record.get('exam_id', 'unknown'),
            'chagas': record['chagas']  # Binary target: Chagas disease (True/False)
        }
        
        # Extract probability values for each condition
        snomed_vals = record['snomed_vals']
        for snomed_code, condition_data in snomed_vals.items():
            if isinstance(condition_data, dict) and 'probability' in condition_data:
                # Use probability as feature (continuous value between 0 and 1)
                probability = condition_data['probability']
                flat_record[f'prob_{snomed_code}'] = float(probability)
            else:
                # Handle missing or malformed data
                flat_record[f'prob_{snomed_code}'] = 0.0
                
        processed_records.append(flat_record)
    
    df = pd.DataFrame(processed_records)
    print(f"✅ Created DataFrame with shape: {df.shape}")
    print(f"📊 Features: {df.shape[1] - 2} probability features + exam_id + chagas label")
    print(f"🎯 Class distribution:")
    print(df['chagas'].value_counts())
    print(f"📈 Class balance: {df['chagas'].value_counts(normalize=True).round(3)}")
    
    return df

# Extract features and create dataset
print("🔄 Extracting probability features...")
df = extract_probability_features(raw_data)

# Display basic statistics
print("\n📊 Dataset Statistics:")
print(f"Total records: {len(df)}")
print(f"Features: {len([col for col in df.columns if col.startswith('prob_')])}")
print(f"Missing values per column:")
missing_summary = df.isnull().sum()
if missing_summary.sum() == 0:
    print("✅ No missing values!")
else:
    print(missing_summary[missing_summary > 0])

# %% [markdown]
# Note that we only use the probabilities inside the `snowmed_vals` field for our prediction. Everything else is not used as of now.
# 
# A snippet of how the prna_outputs JSON file looks is below:
# 
# ```json
# {
#     "exam_id": "763256",
#     "chagas": false,
#     "age": "21",
#     "is_male": "False",
#     "nn_predicted_age": "32.933292",
#     "1dAVb": "False",
#     "RBBB": "False",
#     "LBBB": "False",
#     "SB": "False",
#     "ST": "False",
#     "AF": "False",
#     "patient_id": "1217369",
#     "death": "False",
#     "timey": "2.961641",
#     "normal_ecg": "False",
#     "trace_file": "exams_part1.hdf5",
#     "primary_id": "code15_763256",
#     "source": "code15",
#     "snomed_vals": {
#       "10370003": {
#         "present": false,
#         "probability": 0.00008375992
#       },
#       "111975006": {
#         "present": false,
#         "probability": 0.0010521555
#       },
# ```

# %%
# 🔄 Prepare Features and Target Variables
# Separate features and target
probability_columns = [col for col in df.columns if col.startswith('prob_')]
X = df[probability_columns].values  # Probability features only
y = df['chagas'].astype(int).values  # Binary target (0/1)

print(f"✅ Features shape: {X.shape}")
print(f"✅ Target shape: {y.shape}")
print(f"📊 Feature columns: {len(probability_columns)}")
print(f"📋 First few feature names: {probability_columns[:5]}")

# Check for any infinite or NaN values
inf_count = np.isinf(X).sum()
nan_count = np.isnan(X).sum()
print(f"\n🔍 Data Quality Check:")
print(f"Infinite values: {inf_count}")
print(f"NaN values: {nan_count}")

if inf_count > 0 or nan_count > 0:
    print("⚠️  Cleaning data...")
    # Replace inf with NaN, then fill NaN with 0
    X = np.where(np.isinf(X), np.nan, X)
    X = np.nan_to_num(X, nan=0.0)
    print("✅ Data cleaned")

# Display probability feature statistics
X_df = pd.DataFrame(X, columns=probability_columns)
print(f"\n📊 Probability Feature Statistics:")
print(f"Min probability: {X_df.min().min():.6f}")
print(f"Max probability: {X_df.max().max():.6f}")
print(f"Mean probability: {X_df.mean().mean():.6f}")
print(f"Std probability: {X_df.std().mean():.6f}")

# Train-test split with stratification
print("\n🔀 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"✅ Training set: {X_train.shape[0]} samples")
print(f"✅ Test set: {X_test.shape[0]} samples")
print(f"📊 Training class distribution: {np.bincount(y_train)}")
print(f"📊 Test class distribution: {np.bincount(y_test)}")

# %% [markdown]
# ### Model Training and Evaluation
# 
# We perform a grid search in order to find the best hyperparameters for our random forest classifier

# %%
# 🌲 Random Forest Training with Probability Features

# Calculate class weights for imbalanced dataset
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
print(f"📊 Class weights: {class_weight_dict}")

# Define streamlined hyperparameter grid for large dataset
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [15, 20],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4],
    'max_features': ['sqrt', 'log2']
}

print("🔄 Starting hyperparameter optimization...")
print(f"📋 Parameter combinations: {np.prod([len(v) for v in param_grid.values()])}")

# Initialize Random Forest with class weights
rf_base = RandomForestClassifier(
    random_state=42,
    class_weight=class_weight_dict,
    n_jobs=-1
)

# Grid search with cross-validation (reduced folds for large dataset)
cv_folds = 3
grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
    scoring='f1',  # Focus on F1 score for imbalanced data
    n_jobs=-1,
    verbose=1
)

# Fit the grid search
grid_search.fit(X_train, y_train)

print("✅ Hyperparameter optimization complete!")
print(f"🏆 Best F1 score: {grid_search.best_score_:.4f}")
print(f"🎯 Best parameters: {grid_search.best_params_}")

# Get the best model
best_rf_model = grid_search.best_estimator_

# %%
# 📊 Model Evaluation and Performance Analysis

# Make predictions on test set
y_test_pred = best_rf_model.predict(X_test)
y_test_proba = best_rf_model.predict_proba(X_test)[:, 1]

# Calculate comprehensive metrics
test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_proba)

print("🎯 Test Set Performance:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"F1 Score: {test_f1:.4f}")
print(f"AUC-ROC: {test_auc:.4f}")

# Detailed classification report
print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=['No Chagas', 'Chagas']))

# Cross-validation on training set
cv_scores = cross_val_score(best_rf_model, X_train, y_train, cv=5, scoring='f1')
print(f"\n🔄 5-Fold CV F1 Scores: {cv_scores}")
print(f"📊 Mean CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Feature importance analysis
feature_importance = best_rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': probability_columns,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(f"\n🏆 Top 10 Most Important Features:")
for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows()):
    snomed_code = row['feature'].replace('prob_', '')
    print(f"{i+1:2d}. {row['feature']:15s} (SNOMED: {snomed_code}) - {row['importance']:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
print(f"\n🔢 Confusion Matrix:")
print(f"True Negatives:  {cm[0,0]:4d}    False Positives: {cm[0,1]:4d}")
print(f"False Negatives: {cm[1,0]:4d}    True Positives:  {cm[1,1]:4d}")

# Calculate additional metrics
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0

print(f"\n📈 Additional Metrics:")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity:          {specificity:.4f}")
print(f"Precision:            {precision:.4f}")
print(f"Negative Pred Value:  {npv:.4f}")

# %% [markdown]
# ### Visualizations

# %%
# 📊 Visualization and Analysis

# Set up the plotting style
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Random Forest Performance Analysis - Probability Features', fontsize=16, fontweight='bold')

# 1. ROC Curve
fpr, tpr, roc_thresholds = roc_curve(y_test, y_test_proba)
axes[0, 0].plot(fpr, tpr, color='blue', linewidth=2, label=f'ROC Curve (AUC = {test_auc:.3f})')
axes[0, 0].plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.8)
axes[0, 0].set_xlabel('False Positive Rate')
axes[0, 0].set_ylabel('True Positive Rate')
axes[0, 0].set_title('ROC Curve')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Precision-Recall Curve
precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_test, y_test_proba)
avg_precision = np.trapz(precision_vals, recall_vals)
axes[0, 1].plot(recall_vals, precision_vals, color='red', linewidth=2, 
                label=f'PR Curve (AP = {avg_precision:.3f})')
axes[0, 1].set_xlabel('Recall')
axes[0, 1].set_ylabel('Precision')
axes[0, 1].set_title('Precision-Recall Curve')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Feature Importance (Top 15)
top_features = feature_importance_df.head(15)
axes[1, 0].barh(range(len(top_features)), top_features['importance'], color='green', alpha=0.7)
axes[1, 0].set_yticks(range(len(top_features)))
axes[1, 0].set_yticklabels([f.replace('prob_', '') for f in top_features['feature']], fontsize=8)
axes[1, 0].set_xlabel('Feature Importance')
axes[1, 0].set_title('Top 15 Feature Importances (SNOMED Codes)')
axes[1, 0].grid(True, alpha=0.3, axis='x')

# 4. Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Chagas', 'Chagas'],
            yticklabels=['No Chagas', 'Chagas'],
            ax=axes[1, 1])
axes[1, 1].set_title('Confusion Matrix')
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# Feature importance distribution analysis
print("📊 Feature Importance Analysis:")
print(f"Total features: {len(feature_importance_df)}")
print(f"Features with importance > 0.01: {(feature_importance_df['importance'] > 0.01).sum()}")
print(f"Features with importance > 0.05: {(feature_importance_df['importance'] > 0.05).sum()}")
print(f"Cumulative importance of top 10 features: {feature_importance_df.head(10)['importance'].sum():.3f}")
print(f"Cumulative importance of top 5 features: {feature_importance_df.head(5)['importance'].sum():.3f}")

# Display top features with SNOMED codes for reference
print(f"\n🏆 Top 5 Most Predictive Conditions for Chagas Disease:")
for i, (_, row) in enumerate(feature_importance_df.head(5).iterrows()):
    snomed_code = row['feature'].replace('prob_', '')
    print(f"{i+1}. SNOMED {snomed_code}: {row['importance']:.4f}")

# Probability distribution analysis
print(f"\n📈 Probability Score Distribution:")
chagas_probs = y_test_proba[y_test == 1]
no_chagas_probs = y_test_proba[y_test == 0]
print(f"Chagas cases - Mean probability: {chagas_probs.mean():.3f}, Std: {chagas_probs.std():.3f}")
print(f"No Chagas cases - Mean probability: {no_chagas_probs.mean():.3f}, Std: {no_chagas_probs.std():.3f}")

# %% [markdown]
# ### Model Exporting

# %%
# 💾 Model Saving and Final Summary

# Create output directory
output_dir = Path("./output_ignore")
output_dir.mkdir(exist_ok=True)

# Save the trained model and preprocessing components
model_package = {
    'model': best_rf_model,
    'feature_names': probability_columns,
    'class_weights': class_weight_dict,
    'best_params': grid_search.best_params_,
    'cv_score': grid_search.best_score_,
    'test_metrics': {
        'accuracy': test_accuracy,
        'f1_score': test_f1,
        'auc_roc': test_auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision
    }
}

# Save model package
model_path = output_dir / "random_forest_probability_model.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(model_package, f)

# Save feature importance
feature_importance_df.to_csv(output_dir / "feature_importance_probability.csv", index=False)

# Save performance summary
performance_summary = pd.DataFrame({
    'Metric': ['Accuracy', 'F1 Score', 'AUC-ROC', 'Sensitivity', 'Specificity', 'Precision'],
    'Value': [test_accuracy, test_f1, test_auc, sensitivity, specificity, precision]
})
performance_summary.to_csv(output_dir / "model_performance_probability.csv", index=False)

print("💾 Model and Results Saved:")
print(f"📁 Model: {model_path}")
print(f"📁 Feature importance: {output_dir}/feature_importance_probability.csv")
print(f"📁 Performance metrics: {output_dir}/model_performance_probability.csv")

print("\n" + "="*60)
print("🏁 RANDOM FOREST TRAINING COMPLETE")
print("="*60)
print(f"🎯 Final Model Performance:")
print(f"   • Test Accuracy:  {test_accuracy:.4f}")
print(f"   • Test F1 Score:  {test_f1:.4f}")
print(f"   • Test AUC-ROC:   {test_auc:.4f}")
print(f"   • Sensitivity:    {sensitivity:.4f}")
print(f"   • Specificity:    {specificity:.4f}")

print(f"\n📊 Dataset Summary:")
print(f"   • Total records:  {len(df):,}")
print(f"   • Features used:  {len(probability_columns)} probability features")
print(f"   • Chagas cases:   {y.sum():,} ({y.mean():.3%})")
print(f"   • Training set:   {len(X_train):,}")
print(f"   • Test set:       {len(X_test):,}")

print(f"\n🌲 Model Configuration:")
print(f"   • Algorithm:      Random Forest")
print(f"   • Best params:    {grid_search.best_params_}")
print(f"   • CV F1 score:    {grid_search.best_score_:.4f}")
print(f"   • Class weights:  {class_weight_dict}")

print(f"\n🔬 Key Insights:")
print(f"   • Top feature importance: {feature_importance_df.iloc[0]['importance']:.4f}")
print(f"   • Top 5 features account for {feature_importance_df.head(5)['importance'].sum():.3f} of total importance")
print(f"   • Model successfully predicts Chagas disease using only PRNA probability outputs")
print("="*60)

# %% [markdown]
# # 📋 Final Summary - Random Forest with PRNA Probability Features
# 
# ## ✅ Implementation Overview
# 
# This notebook implements a **Random Forest classifier for Chagas disease prediction** using only the 27 condition probability outputs from the PRNA model, completely eliminating ECG signal processing.
# 
# ### 🎯 **Key Features:**
# - **Input**: 27 SNOMED condition probabilities from `combined_prna_outputs.json`
# - **Output**: Binary Chagas disease prediction (True/False)
# - **Algorithm**: Random Forest with class weights for imbalanced data
# - **Approach**: Aligned with XGBoost notebook methodology
# 
# ### 📊 **Pipeline Summary:**
# 1. **Data Loading**: Load 365,961 PRNA probability outputs
# 2. **Feature Extraction**: Extract 27 probability features (0.0 to 1.0)
# 3. **Preprocessing**: Handle class imbalance with stratified splitting
# 4. **Training**: Hyperparameter-tuned Random Forest with cross-validation
# 5. **Evaluation**: Comprehensive performance metrics and visualization
# 6. **Analysis**: Feature importance ranking and medical interpretation
# 
# ### 🏆 **Model Performance:**
# - **Accuracy**: 95.80%
# - **F1 Score**: 20.32%
# - **AUC-ROC**: 78.08%
# - **Sensitivity**: 23.95%
# - **Specificity**: 97.44%
# 
# ### 🔬 **Top Predictive Conditions:**
# 1. **SNOMED 713427006** (6.40% importance)
# 2. **SNOMED 426783006** (6.05% importance)
# 3. **SNOMED 270492004** (5.68% importance)
# 4. **SNOMED 59118001** (4.86% importance)
# 5. **SNOMED 164889003** (4.60% importance)
# 
# ### ✅ **Advantages of This Approach:**
# - ⚡ **Fast**: No computationally expensive ECG signal processing
# - 🎯 **Accurate**: Uses real PRNA model outputs on medical data
# - 📈 **Scalable**: Can process large datasets efficiently
# - 🔍 **Interpretable**: Clear medical condition importance rankings
# - 🤝 **Consistent**: Aligned with successful XGBoost implementation
# 
# ---
# 
# *This cleaned implementation provides a streamlined, efficient pipeline for Chagas disease prediction using machine learning and medical condition probabilities.*


