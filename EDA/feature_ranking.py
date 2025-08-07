#!/usr/bin/env python
"""
Feature Ranking Script for EDA Outputs

This script loads a features CSV (e.g., final_features_balanced.csv) produced by feature_eda_pipeline_GPT.py,
performs feature importance ranking using Random Forest, and outputs a ranked CSV and console summary.

Usage:
    python feature_ranking.py --input path/to/final_features_balanced.csv [--output path/to/output_folder]

Author: Copilot (2025)
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier


def main():
    parser = argparse.ArgumentParser(description='Feature Ranking for EDA Features')
    parser.add_argument('--input', type=str, required=True, help='Path to features CSV (e.g., final_features_balanced.csv)')
    parser.add_argument('--output', type=str, default=None, help='Output folder for ranked features CSV')
    parser.add_argument('--n_top', type=int, default=20, help='Number of top features to print')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return

    df = pd.read_csv(input_path)
    if 'chagas' not in df.columns:
        print("Target column 'chagas' not found in input file.")
        return

    # Exclude non-feature columns
    non_feature_cols = {'exam_id', 'relative_path', 'chagas', 'is_male'}
    feature_cols = [col for col in df.columns if col not in non_feature_cols and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
    X = df[feature_cols].values
    y = df['chagas'].astype(int).values

    # Fit Random Forest for feature importance
    print(f"Fitting Random Forest on {len(feature_cols)} features and {len(y)} samples...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(X, y)
    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Output
    output_folder = Path(args.output) if args.output else input_path.parent
    output_folder.mkdir(parents=True, exist_ok=True)
    out_csv = output_folder / 'feature_importance_ranking.csv'
    feature_importance_df.to_csv(out_csv, index=False)
    print(f"\nTop {args.n_top} Features:")
    print(feature_importance_df.head(args.n_top).to_string(index=False))
    print(f"\nFull ranking saved to: {out_csv}")

if __name__ == '__main__':
    main()
