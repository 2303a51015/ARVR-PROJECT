"""
Quantile Regression Modeling Pipeline
Project: Air Pollutant (PM2.5) Prediction Using Quantile Regression
"""

import os
import pandas as pd
import numpy as np
from statsmodels.regression.quantile_regression import QuantReg


def fit_quantile_regressors(X_train_sm: pd.DataFrame, y_train: pd.Series,
                            quantiles: list = [0.10, 0.50, 0.90],
                            max_iter: int = 2500) -> dict:
    """
    Fits Quantile Regression models for the specified quantiles using training data only.
    Handles numerical convergence and returns a dictionary of fitted model wrappers.
    """
    fitted_models = {}
    print("\n--- QUANTILE REGRESSION MODEL TRAINING ---")
    print(f"Algorithm: statsmodels.regression.quantile_regression.QuantReg")
    print(f"Quantiles to estimate: {quantiles}")
    print(f"Training observations: {len(y_train)}")

    for q in quantiles:
        try:
            print(f"\n[MODEL] Fitting Quantile Regression for q = {q:.2f}...")
            mod = QuantReg(y_train, X_train_sm)
            res = mod.fit(q=q, max_iter=max_iter)
            fitted_models[q] = res

            pseudo_r2 = getattr(res, 'prsquared', np.nan)
            print(f"  -> Quantile {q:.2f} converged successfully.")
            print(f"  -> Pseudo R-squared: {pseudo_r2:.4f}")
            print(f"  -> Intercept (const): {res.params['const']:.4f}")
        except Exception as e:
            print(f"  [ERROR] Fitting failed for quantile {q}: {e}")
            raise e

    return fitted_models


def predict_quantiles(fitted_models: dict, X_test_sm: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Generates predictions for all fitted quantiles on the held-out test set.
    Creates a comprehensive prediction DataFrame containing:
    - actual PM2.5
    - predicted median PM2.5 (q=0.50)
    - 10th quantile prediction (lower bound)
    - 90th quantile prediction (upper bound)
    - prediction interval width (q90 - q10)
    - residual error for median prediction (actual - predicted_50)
    """
    pred_df = pd.DataFrame(index=X_test_sm.index)
    pred_df['actual_PM2.5'] = y_test.values

    # Predict for each quantile
    for q, res in fitted_models.items():
        q_label = f"pred_q{int(q * 100):02d}"
        pred_df[q_label] = res.predict(X_test_sm).values

    # Designate median as primary point prediction
    pred_df['predicted_PM2.5'] = pred_df['pred_q50']
    pred_df['residual'] = pred_df['actual_PM2.5'] - pred_df['predicted_PM2.5']
    pred_df['interval_width'] = pred_df['pred_q90'] - pred_df['pred_q10']

    # Coverage flag: does actual fall inside [pred_q10, pred_q90]?
    pred_df['inside_interval'] = (
        (pred_df['actual_PM2.5'] >= pred_df['pred_q10']) &
        (pred_df['actual_PM2.5'] <= pred_df['pred_q90'])
    )

    empirical_coverage = pred_df['inside_interval'].mean() * 100
    print(f"\n[PREDICTIONS] Empirical 80% Prediction Interval Coverage on Test Set: {empirical_coverage:.2f}%")
    print(f"[PREDICTIONS] Nominal coverage target: 80.00% (between 10th and 90th quantiles)")

    return pred_df


def extract_coefficients_table(fitted_models: dict) -> pd.DataFrame:
    """
    Extracts fitted parameters, standard errors, t-values, p-values,
    and 95% confidence intervals across all quantiles into a clean DataFrame.
    """
    rows = []
    for q, res in fitted_models.items():
        params = res.params
        bse = res.bse
        tvalues = res.tvalues
        pvalues = res.pvalues
        conf_int = res.conf_int()

        for var_name in params.index:
            rows.append({
                "Quantile": q,
                "Variable": var_name,
                "Coefficient": params[var_name],
                "Std_Error": bse[var_name],
                "t_Statistic": tvalues[var_name],
                "p_Value": pvalues[var_name],
                "CI_Lower_95": conf_int.loc[var_name, 0],
                "CI_Upper_95": conf_int.loc[var_name, 1]
            })

    coef_df = pd.DataFrame(rows)
    return coef_df


def save_model_outputs(fitted_models: dict, pred_df: pd.DataFrame,
                       coef_df: pd.DataFrame, output_dir: str):
    """
    Saves model summary text files, coefficient CSVs, and prediction CSVs.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save predictions CSV
    pred_path = os.path.join(output_dir, "test_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"[MODEL OUTPUT] Test predictions exported to: {pred_path}")

    # 2. Save coefficients CSV
    coef_path = os.path.join(output_dir, "model_coefficients.csv")
    coef_df.to_csv(coef_path, index=False)
    print(f"[MODEL OUTPUT] Model coefficients exported to: {coef_path}")

    # 3. Save text summaries
    summary_path = os.path.join(output_dir, "quantile_regression_summaries.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("QUANTILE REGRESSION MODEL FITTING SUMMARIES\n")
        f.write("Dataset: Air Pollutant PM2.5 Prediction\n")
        f.write("=" * 78 + "\n\n")
        for q, res in fitted_models.items():
            f.write(f"--- Quantile: {q:.2f} ---\n")
            f.write(res.summary().as_text())
            f.write("\n\n" + "=" * 78 + "\n\n")

    print(f"[MODEL OUTPUT] Detailed statsmodels summaries exported to: {summary_path}")
