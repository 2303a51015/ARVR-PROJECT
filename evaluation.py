"""
Model Evaluation Pipeline for PM2.5 Prediction
Project: Air Pollutant (PM2.5) Prediction Using Quantile Regression
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """
    Computes the check-loss (pinball loss) for a specified quantile q:
    L_q(y, y_hat) = mean( max(q * (y - y_hat), (q - 1) * (y - y_hat)) )
    """
    diff = y_true - y_pred
    loss = np.maximum(q * diff, (q - 1.0) * diff)
    return float(np.mean(loss))


def compute_test_metrics(y_true: pd.Series, y_pred: pd.Series,
                         q10_pred: pd.Series = None, q90_pred: pd.Series = None) -> tuple:
    """
    Computes MSE, RMSE, MAPE, and R2 on the held-out test set for the primary median (q=0.50) model.
    Also calculates quantile check-losses and prediction interval statistics.
    Safely checks for zero values before MAPE computation.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    # 1. Mean Squared Error (MSE)
    mse = float(mean_squared_error(y_true_arr, y_pred_arr))

    # 2. Root Mean Squared Error (RMSE)
    rmse = float(np.sqrt(mse))

    # 3. Mean Absolute Percentage Error (MAPE) with safe zero handling
    zero_mask = (y_true_arr == 0)
    zero_count = int(np.sum(zero_mask))
    valid_mask = ~zero_mask
    valid_count = int(np.sum(valid_mask))

    if zero_count > 0:
        print(f"[EVALUATION WARNING] Detected {zero_count} zero actual PM2.5 values in test set.")
        print(f"[EVALUATION] Calculating MAPE over {valid_count} non-zero observations to prevent division by zero.")
        mape = float(np.mean(np.abs((y_true_arr[valid_mask] - y_pred_arr[valid_mask]) / y_true_arr[valid_mask])) * 100.0)
    else:
        mape = float(np.mean(np.abs((y_true_arr - y_pred_arr) / y_true_arr)) * 100.0)

    # 4. R-squared (R2) via sklearn.metrics.r2_score
    r2 = float(r2_score(y_true_arr, y_pred_arr))

    # 5. Quantile Check-Losses (Pinball Loss)
    loss_50 = quantile_loss(y_true_arr, y_pred_arr, 0.50)
    loss_10 = quantile_loss(y_true_arr, np.asarray(q10_pred), 0.10) if q10_pred is not None else np.nan
    loss_90 = quantile_loss(y_true_arr, np.asarray(q90_pred), 0.90) if q90_pred is not None else np.nan

    # Build metric dictionary
    metrics_dict = {
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "zero_actual_count": zero_count,
        "valid_mape_count": valid_count,
        "Quantile_Loss_q10": loss_10,
        "Quantile_Loss_q50": loss_50,
        "Quantile_Loss_q90": loss_90
    }

    # Build structured evaluation table DataFrame
    table_rows = [
        {
            "Metric Name": "Mean Squared Error (MSE)",
            "Formula / Description": "mean((actual - predicted)^2)",
            "Test Value": f"{mse:.4f}",
            "Unit": "(ug/m3)^2",
            "Interpretation": "Quantifies average squared deviation; places heavier penalty on large prediction misses."
        },
        {
            "Metric Name": "Root Mean Squared Error (RMSE)",
            "Formula / Description": "sqrt(MSE)",
            "Test Value": f"{rmse:.4f}",
            "Unit": "ug/m3",
            "Interpretation": "Expressed in native pollutant concentration units; typical magnitude of median model residual error."
        },
        {
            "Metric Name": "Mean Absolute Percentage Error (MAPE)",
            "Formula / Description": "mean(|actual - predicted| / actual) * 100",
            "Test Value": f"{mape:.2f}%",
            "Unit": "%",
            "Interpretation": f"Average relative percentage prediction error across {valid_count} valid non-zero test observations."
        },
        {
            "Metric Name": "R-squared (R2)",
            "Formula / Description": "1 - sum((y - y_hat)^2) / sum((y - y_mean)^2)",
            "Test Value": f"{r2:.4f}",
            "Unit": "Dimensionless",
            "Interpretation": "Proportion of total PM2.5 variance explained by median Quantile Regression relative to mean baseline."
        },
        {
            "Metric Name": "Median Quantile Loss (q=0.50)",
            "Formula / Description": "mean(max(0.5*(y - y_hat), -0.5*(y - y_hat)))",
            "Test Value": f"{loss_50:.4f}",
            "Unit": "ug/m3",
            "Interpretation": "Theoretical pinball objective function minimized by median Quantile Regression (0.5 * MAE)."
        },
        {
            "Metric Name": "Lower Quantile Loss (q=0.10)",
            "Formula / Description": "mean(max(0.1*(y - y_hat), -0.9*(y - y_hat)))",
            "Test Value": f"{loss_10:.4f}",
            "Unit": "ug/m3",
            "Interpretation": "Asymmetric check-loss for the 10th conditional percentile lower bound."
        },
        {
            "Metric Name": "Upper Quantile Loss (q=0.90)",
            "Formula / Description": "mean(max(0.9*(y - y_hat), -0.1*(y - y_hat)))",
            "Test Value": f"{loss_90:.4f}",
            "Unit": "ug/m3",
            "Interpretation": "Asymmetric check-loss for the 90th conditional percentile upper peak bound."
        }
    ]

    metrics_df = pd.DataFrame(table_rows)
    return metrics_dict, metrics_df


def export_metrics_table(metrics_df: pd.DataFrame, output_dir: str):
    """
    Saves the metrics table as CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, "test_metrics.csv")
    metrics_df.to_csv(out_csv, index=False)
    print(f"[EVALUATION OUTPUT] Metrics table exported to: {out_csv}")


def print_formatted_metrics_table(metrics_df: pd.DataFrame):
    """
    Prints a clean, nicely formatted ASCII table of the evaluation results.
    """
    print("\n" + "=" * 105)
    print("                    FINAL HELD-OUT TEST EVALUATION METRICS (QUANTILE REGRESSION)")
    print("=" * 105)
    header = f"{'Metric Name':<38} | {'Formula / Description':<42} | {'Test Value':<12} | {'Unit':<10}"
    print(header)
    print("-" * 105)
    for _, row in metrics_df.iterrows():
        print(f"{row['Metric Name']:<38} | {row['Formula / Description']:<42} | {row['Test Value']:<12} | {row['Unit']:<10}")
    print("=" * 105 + "\n")
