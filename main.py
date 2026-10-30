"""
Main Orchestration Pipeline
Project: Air Pollutant (PM2.5) Prediction Using Quantile Regression
Author: Academic Machine Learning Project
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# Ensure src directory is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from preprocessing import (
    load_dataset,
    inspect_raw_data,
    clean_and_validate,
    split_and_impute,
    RELEVANT_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN
)
from model import (
    fit_quantile_regressors,
    predict_quantiles,
    extract_coefficients_table,
    save_model_outputs
)
from evaluation import (
    compute_test_metrics,
    export_metrics_table,
    print_formatted_metrics_table
)
from visualization import generate_all_visualizations


def run_pipeline(data_path: str = None, results_dir: str = None, figures_dir: str = None):
    """
    Executes the complete machine learning workflow:
    Data Loading -> Preprocessing -> Modeling -> Evaluation -> Visualization -> Export.
    """
    print("\n" + "#" * 80)
    print("  PROJECT: AIR POLLUTANT (PM2.5) PREDICTION USING QUANTILE REGRESSION")
    print("  TECH STACK: Python, Pandas, Statsmodels, Scikit-learn, Seaborn, Matplotlib")
    print("#" * 80 + "\n")

    # Define default relative directories
    if data_path is None:
        data_path = os.path.join(PROJECT_ROOT, "data", "Data sample arvr.xlsx")
    if results_dir is None:
        results_dir = os.path.join(PROJECT_ROOT, "outputs", "results")
    if figures_dir is None:
        figures_dir = os.path.join(PROJECT_ROOT, "outputs", "figures")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # ---------------------------------------------------------
    # STEP 1: LOAD AND INSPECT DATASET
    # ---------------------------------------------------------
    try:
        raw_df = load_dataset(data_path)
    except FileNotFoundError as e:
        print(f"[FATAL ERROR] {e}")
        print("Please ensure 'Data sample arvr.xlsx' is placed in the project 'data/' folder.")
        sys.exit(1)

    raw_info = inspect_raw_data(raw_df)

    # ---------------------------------------------------------
    # STEP 2: CLEANING, PREPROCESSING & OUTLIER INVESTIGATION
    # ---------------------------------------------------------
    df_clean = clean_and_validate(raw_df)

    # Export dataset statistical summary
    summary_path = os.path.join(results_dir, "dataset_statistical_summary.csv")
    df_clean.describe().to_csv(summary_path)
    print(f"[PREPROCESSING] Statistical summary exported to: {summary_path}")

    # ---------------------------------------------------------
    # STEP 3: TRAIN-TEST SPLIT AND IMPUTATION (NO LEAKAGE)
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test, X_train_sm, X_test_sm, impute_meta = split_and_impute(
        df_clean, test_size=0.20, random_state=42
    )

    # ---------------------------------------------------------
    # STEP 4: FIT QUANTILE REGRESSION MODELS (q = 0.10, 0.50, 0.90)
    # ---------------------------------------------------------
    quantiles = [0.10, 0.50, 0.90]
    fitted_models = fit_quantile_regressors(X_train_sm, y_train, quantiles=quantiles, max_iter=2500)

    # ---------------------------------------------------------
    # STEP 5: PREDICTIONS & UNCERTAINTY INTERVALS
    # ---------------------------------------------------------
    pred_df = predict_quantiles(fitted_models, X_test_sm, y_test)
    coef_df = extract_coefficients_table(fitted_models)

    # Save model summaries and predictions
    save_model_outputs(fitted_models, pred_df, coef_df, results_dir)

    # ---------------------------------------------------------
    # STEP 6: MODEL EVALUATION ON HELD-OUT TEST DATA
    # ---------------------------------------------------------
    y_test_actual = pred_df['actual_PM2.5']
    y_test_pred_median = pred_df['pred_q50']
    q10_pred = pred_df['pred_q10']
    q90_pred = pred_df['pred_q90']

    metrics_dict, metrics_df = compute_test_metrics(
        y_test_actual, y_test_pred_median, q10_pred=q10_pred, q90_pred=q90_pred
    )
    export_metrics_table(metrics_df, results_dir)
    print_formatted_metrics_table(metrics_df)

    # ---------------------------------------------------------
    # STEP 7: GENERATE ALL 10 REQUIRED VISUALIZATIONS
    # ---------------------------------------------------------
    generate_all_visualizations(
        df_clean=df_clean,
        pred_df=pred_df,
        coef_df=coef_df,
        metrics_dict=metrics_dict,
        output_dir=figures_dir
    )

    # ---------------------------------------------------------
    # STEP 8: FINAL SUMMARY BANNER
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("                    EXECUTION SUCCESSFULLY COMPLETED!")
    print("=" * 80)
    print(f"  Total Processed Rows:       {len(df_clean)}")
    print(f"  Training Observations:      {len(X_train)} (80%)")
    print(f"  Testing Observations:       {len(X_test)} (20%)")
    print(f"  Held-out Test MSE:          {metrics_dict['MSE']:.4f}")
    print(f"  Held-out Test RMSE:         {metrics_dict['RMSE']:.4f} ug/m3")
    print(f"  Held-out Test MAPE:         {metrics_dict['MAPE']:.2f}%")
    print(f"  Held-out Test R-squared:    {metrics_dict['R2']:.4f}")
    print(f"  Empirical 80% CI Coverage:  {pred_df['inside_interval'].mean() * 100:.2f}%")
    print(f"  Figures Generated:          10 publication-quality PNGs in {figures_dir}")
    print(f"  Results Exported:           CSVs & TXT summaries in {results_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Air Pollutant PM2.5 Prediction Using Quantile Regression")
    parser.add_argument("--data", type=str, default=None, help="Path to Excel dataset file")
    parser.add_argument("--results", type=str, default=None, help="Output directory for results CSVs")
    parser.add_argument("--figures", type=str, default=None, help="Output directory for figures PNGs")
    args = parser.parse_args()

    run_pipeline(data_path=args.data, results_dir=args.results, figures_dir=args.figures)
