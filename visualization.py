"""
Visualization Pipeline for PM2.5 Air Pollution Prediction
Project: Air Pollutant (PM2.5) Prediction Using Quantile Regression
Generates 10 publication-quality figures saved as PNG in outputs/figures/.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set global aesthetic styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['axes.labelsize'] = 11
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 9
matplotlib.rcParams['figure.titlesize'] = 14


def plot_01_pm25_distribution(df_clean: pd.DataFrame, output_dir: str):
    """Figure 1: PM2.5 Distribution Histogram with KDE and Summary Statistics."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    pm = df_clean['PM2.5']
    mean_val = pm.mean()
    median_val = pm.median()

    sns.histplot(pm, kde=True, color='#2b5c8f', bins=40, ax=ax, edgecolor='black', alpha=0.65)
    ax.axvline(mean_val, color='#d9534f', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f} $\\mu g/m^3$')
    ax.axvline(median_val, color='#28a745', linestyle='-', linewidth=2, label=f'Median: {median_val:.1f} $\\mu g/m^3$')

    ax.set_title('Figure 1: Empirical Distribution of Target Pollutant (PM2.5)', pad=12, fontweight='bold')
    ax.set_xlabel('PM2.5 Concentration ($\\mu g/m^3$)')
    ax.set_ylabel('Observation Frequency')
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()

    out_path = os.path.join(output_dir, 'fig01_pm25_distribution.png')
    plt.savefig(out_path)
    plt.close()
    print(f"[FIGURE 01] Saved: {out_path}")


def plot_02_features_boxplots(df_clean: pd.DataFrame, output_dir: str):
    """Figure 2: Multi-panel Boxplots for PM2.5 and all Five Input Features."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=300)
    axes = axes.flatten()
    cols = ['PM2.5', 'RH', 'WD', 'SR', 'BP', 'AT']
    units = ['$\\mu g/m^3$', '%', 'Degrees', '$W/m^2$', 'mm Hg', '$^\\circ C$']
    colors = ['#d9534f', '#337ab7', '#5bc0de', '#f0ad4e', '#5cb85c', '#9b59b6']

    for i, col in enumerate(cols):
        sns.boxplot(y=df_clean[col], ax=axes[i], color=colors[i], width=0.4,
                    flierprops=dict(marker='o', markersize=4, markerfacecolor='black', alpha=0.5))
        axes[i].set_title(f'{col} Distribution', fontweight='bold')
        axes[i].set_ylabel(f'{col} ({units[i]})')
        axes[i].set_xlabel('')

    fig.suptitle('Figure 2: Boxplots of Target (PM2.5) and Environmental Features (IQR & Outliers)',
                 fontweight='bold', y=0.98)
    plt.tight_layout()

    out_path = os.path.join(output_dir, 'fig02_features_boxplots.png')
    plt.savefig(out_path)
    plt.close()
    print(f"[FIGURE 02] Saved: {out_path}")


def plot_03_correlation_heatmap(df_clean: pd.DataFrame, output_dir: str):
    """Figure 3: Correlation Heatmap of All Six Environmental and Pollutant Variables."""
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
    cols = ['PM2.5', 'RH', 'WD', 'SR', 'BP', 'AT']
    corr = df_clean[cols].corr()

    sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', vmin=-1, vmax=1,
                linewidths=1, linecolor='white', cbar_kws={'label': 'Pearson Correlation Coefficient'},
                square=True, ax=ax)

    ax.set_title('Figure 3: Pearson Correlation Heatmap Across PM2.5 and Meteorological Inputs',
                 pad=12, fontweight='bold')
    plt.tight_layout()

    out_path = os.path.join(output_dir, 'fig03_correlation_heatmap.png')
    plt.savefig(out_path)
    plt.close()
    print(f"[FIGURE 03] Saved: {out_path}")


def plot_04_actual_vs_predicted_scatter(pred_df: pd.DataFrame, output_dir: str):
    """Figure 4: Actual vs Predicted PM2.5 Scatter Plot with 45-degree Reference Line."""
    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
    y_true = pred_df['actual_PM2.5']
    y_pred = pred_df['predicted_PM2.5']

    ax.scatter(y_true, y_pred, color='#1f77b4', alpha=0.55, edgecolors='none', s=35, label='Test Observations')

    # 45-degree reference line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], color='#d9534f', linestyle='--',
            linewidth=2, label='Ideal 1:1 Reference Line')

    # Linear trendline
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    x_range = np.linspace(min_val, max_val, 100)
    ax.plot(x_range, p(x_range), color='#2ca02c', linestyle='-', linewidth=1.8,
            label=f'Empirical Fit Trend: $y = {z[0]:.2f}x + {z[1]:.1f}$')

    ax.set_title('Figure 4: Actual vs. Predicted PM2.5 (Median Quantile Regression)', pad=12, fontweight='bold')
    ax.set_xlabel('Actual PM2.5 ($\\mu g/m^3$)')
    ax.set_ylabel('Predicted Median PM2.5 ($\\mu g/m^3$)')
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()

    out_path = os.path.join(output_dir, 'fig04_actual_vs_predicted_scatter.png')
    plt.savefig(out_path)
    plt.close()
    print(f"[FIGURE 04] Saved: {out_path}")


def plot_05_actual_vs_predicted_comparison(pred_df: pd.DataFrame, output_dir: str, n_samples: int = 80):
    """Figure 5: Actual and Predicted PM2.5 Comparison for Held-Out Test Observations."""
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    subset = pred_df.iloc[:n_samples].copy().reset_index(drop=True)

    ax.plot(subset.index, subset['actual_PM2.5'], marker='o', markersize=4,
            color='#d9534f', linewidth=1.5, label='Actual PM2.5')
    ax.plot(subset.index, subset['predicted_PM2.5'], marker='s', markersize=4,
            color='#1f77b4', linewidth=1.5, linestyle='--', label='Predicted Median PM2.5')

    ax.set_title(f'Figure 5: Time-Ordered Comparison of Actual vs. Predicted PM2.5 (First {n_samples} Test Points)',
                 pad=12, fontweight='bold')
    ax.set_xlabel('Test Observation Index')
    ax.set_ylabel('PM2.5 Concentration ($\\mu g/m^3$)')
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
    plt.tight_layout()

    out_path = os.path.join(output_dir, 'fig05_actual_vs_predicted_comparison.png')
    plt.savefig(out_path)
    plt.close()
    print(f"[FIGURE 05] Saved: {out_path}")


def plot_06_residual_distribution(pred_df: pd.DataFrame, output_dir: str):
    """Figure 6: Residual Error Distribution Plot (Histogram + KDE + Reference Normal)."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    resids = pred_df['residual']
    mean_res = resids.mean()
    std_res = resids.std()

    sns.histplot(resids, kde=True, color='#6f42c1', bins=35, ax=ax, edgecolor='black', alpha=0.6)
    ax.axvline(0, color='black', linestyle='-', linewidth=1.5, label='Zero Error Baseline')
    ax.axvline(mean_res, color='#d9534f', linestyle='--', linewidth=2,
               label=f'Mean Residual: {mean_res:.2f} $\\mu g/m^3$')

    ax.set_title('Figure 6: Residual Error Distribution ($e_i = Actual - Predicted$)', pad=12, fontweight='bold')
    ax.set_xlabel('Residual Error ($\\mu g/m^3$)')
    ax.set_ylabel('Observation Frequency')
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()

    out_path = os.path.join(output_dir, 'fig06_residual_distribution.png')
    plt.savefig(out_path)
    plt.close()
    print(f"[FIGURE 06] Saved: {out_path}")


def plot_07_residuals_vs_predicted(pred_df: pd.DataFrame, output_dir: str):
    """Figure 7: Residuals vs Predicted Values Scatter Plot (Homoscedasticity Assessment)."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    y_pred = pred_df['predicted_PM2.5']
    resids = pred_df['residual']

    ax.scatter(y_pred, resids, color='#e67e22', alpha=0.6, edgecolors='none', s=35)
    ax.axhline(0, color='black', linestyle='--', linewidth=1.8, label='Zero Residual Line')

    # Smoothed trend
    z = np.polyfit(y_pred, resids, 1)
    p = np.poly1d(z)
    x_range = np.linspace(y_pred.min(), y_pred.max(), 100)
    ax.plot(x_range, p(x_range), color='#c0392b', linestyle='-', linewidth=2, label='Residual Trend')

    ax.set_title('Figure 7: Residuals vs. Predicted PM2.5 Values', pad=12, fontweight='bold')
    ax.set_xlabel('Predicted Median PM2.5 ($\\mu g/m^3$)')
    ax.set_ylabel('Residual Error ($\\mu g/m^3$)')
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()

    out_path = os.path.join(output_dir, 'fig07_residuals_vs_predicted.png')
    plt.savefig(out_path)
    plt.close()
    print(f"[FIGURE 07] Saved: {out_path}")


def plot_08_metrics_comparison(metrics_dict: dict, output_dir: str):
    """Figure 8: Evaluation Metrics Comparison (Multi-Panel Standardized Scales)."""
    fig, axes = plt.subplots(1, 4, figsize=(13, 4), dpi=300)

    metric_configs = [
        ("MSE", metrics_dict["MSE"], "(\\mu g/m^3)^2", "#3498db"),
        ("RMSE", metrics_dict["RMSE"], "\\mu g/m^3", "#2ecc71"),
        ("MAPE", metrics_dict["MAPE"], "%", "#e74c3c"),
        ("R-squared (R^2)", metrics_dict["R2"], "Unitless", "#9b59b6")
    ]

    for i, (name, val, unit, col) in enumerate(metric_configs):
        bars = axes[i].bar([name], [val], color=col, width=0.45, edgecolor='black', linewidth=1)
        axes[i].set_title(f'{name}', fontweight='bold', pad=10)
        axes[i].set_ylabel(f'Value ({unit})')

        # Annotate text
        fmt = f"{val:.2f}%" if "%" in unit else f"{val:.2f}"
        axes[i].text(0, val * 1.02, fmt, ha='center', va='bottom', fontweight='bold', fontsize=11)

        # Set reasonable y limits
        if name == "R-squared (R^2)":
            axes[i].set_ylim(0, 1.0)
        else:
            axes[i].set_ylim(0, val * 1.25)

    fig.suptitle('Figure 8: Performance Evaluation Metrics on Held-Out Test Set (q = 0.50)',
                 fontweight='bold', y=1.02)
    plt.tight_layout()

    out_path = os.path.join(output_dir, 'fig08_metrics_comparison.png')
    plt.savefig(out_path)
    plt.close()
    print(f"[FIGURE 08] Saved: {out_path}")


def plot_09_quantile_predictions_interval(pred_df: pd.DataFrame, output_dir: str, n_samples: int = 70):
    """Figure 9: Actual PM2.5 alongside 10th, 50th, and 90th Quantile Predictions and 80% Uncertainty Band."""
    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=300)
    subset = pred_df.iloc[:n_samples].copy().reset_index(drop=True)

    # Shaded 80% prediction interval
    ax.fill_between(subset.index, subset['pred_q10'], subset['pred_q90'],
                    color='#74add1', alpha=0.35, label='80% Prediction Interval [q10, q90]')

    # Quantile lines
    ax.plot(subset.index, subset['pred_q90'], color='#2166ac', linestyle=':',
            linewidth=1.5, label='90th Quantile (Upper Risk Bound)')
    ax.plot(subset.index, subset['pred_q50'], color='#053061', linestyle='--',
            linewidth=2.0, label='50th Quantile (Median Point Prediction)')
    ax.plot(subset.index, subset['pred_q10'], color='#4393c3', linestyle=':',
            linewidth=1.5, label='10th Quantile (Lower Bound)')

    # Actual observations
    ax.scatter(subset.index, subset['actual_PM2.5'], color='#d6604d', s=35,
               edgecolors='black', linewidth=0.6, zorder=5, label='Actual PM2.5 Observation')

    ax.set_title('Figure 9: Quantile Regression Prediction Intervals (10th, 50th, 90th) vs. Actual PM2.5',
                 pad=12, fontweight='bold')
    ax.set_xlabel('Held-Out Test Sample Index')
    ax.set_ylabel('PM2.5 Concentration ($\\mu g/m^3$)')
    ax.legend(frameon=True, facecolor='white', framealpha=0.92, loc='upper left')
    plt.tight_layout()

    out_path = os.path.join(output_dir, 'fig09_quantile_predictions_interval.png')
    plt.savefig(out_path)
    plt.close()
    print(f"[FIGURE 09] Saved: {out_path}")


def plot_10_feature_coefficients(coef_df: pd.DataFrame, output_dir: str):
    """Figure 10: Fitted Median Quantile Regression Coefficients with 95% Confidence Intervals."""
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    # Filter to median model and exclude constant for clearer feature comparison
    median_coefs = coef_df[(coef_df['Quantile'] == 0.50) & (coef_df['Variable'] != 'const')].copy()

    y_pos = np.arange(len(median_coefs))
    coefficients = median_coefs['Coefficient'].values
    errors_low = coefficients - median_coefs['CI_Lower_95'].values
    errors_high = median_coefs['CI_Upper_95'].values - coefficients
    errors = np.array([errors_low, errors_high])

    colors = ['#27ae60' if c >= 0 else '#c0392b' for c in coefficients]
    bars = ax.barh(y_pos, coefficients, xerr=errors, capsize=5, color=colors,
                   alpha=0.8, edgecolor='black', height=0.55)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(median_coefs['Variable'], fontweight='bold')
    ax.axvline(0, color='black', linestyle='--', linewidth=1.2)

    # Annotate exact coefficient values
    for i, (c, p_val) in enumerate(zip(coefficients, median_coefs['p_Value'].values)):
        signif = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        offset = 0.2 if c >= 0 else -0.2
        ha = 'left' if c >= 0 else 'right'
        ax.text(c + offset, i, f"{c:.3f} ({signif})", va='center', ha=ha, fontsize=9.5, fontweight='bold')

    ax.set_title('Figure 10: Fitted Feature Coefficients (Median Quantile Regression, q = 0.50)',
                 pad=14, fontweight='bold')
    ax.set_xlabel('Marginal Coefficient Estimate (with 95% Confidence Interval)')

    # Mandatory academic disclaimer note as required by Master Prompt
    disclaimer_text = (
        "Note: Coefficients represent estimated conditional associations under Quantile Regression, "
        "not proven causal relationships.\nSignificance codes: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant."
    )
    fig.text(0.5, 0.01, disclaimer_text, ha='center', fontsize=8.5, style='italic', color='#555555')
    plt.subplots_adjust(bottom=0.18)

    out_path = os.path.join(output_dir, 'fig10_feature_coefficients.png')
    plt.savefig(out_path)
    plt.close()
    print(f"[FIGURE 10] Saved: {out_path}")


def generate_all_visualizations(df_clean: pd.DataFrame, pred_df: pd.DataFrame,
                                coef_df: pd.DataFrame, metrics_dict: dict,
                                output_dir: str):
    """
    Executes the complete generation of all 10 required visualization figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n--- GENERATING PUBLICATION-QUALITY FIGURES ---")
    plot_01_pm25_distribution(df_clean, output_dir)
    plot_02_features_boxplots(df_clean, output_dir)
    plot_03_correlation_heatmap(df_clean, output_dir)
    plot_04_actual_vs_predicted_scatter(pred_df, output_dir)
    plot_05_actual_vs_predicted_comparison(pred_df, output_dir)
    plot_06_residual_distribution(pred_df, output_dir)
    plot_07_residuals_vs_predicted(pred_df, output_dir)
    plot_08_metrics_comparison(metrics_dict, output_dir)
    plot_09_quantile_predictions_interval(pred_df, output_dir)
    plot_10_feature_coefficients(coef_df, output_dir)
    print("--- ALL 10 FIGURES SUCCESSFULLY GENERATED AND SAVED ---")
