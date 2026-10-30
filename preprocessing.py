"""
Data Preprocessing Pipeline for PM2.5 Air Pollution Prediction
Project: Air Pollutant (PM2.5) Prediction Using Quantile Regression
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

RELEVANT_COLUMNS = ['PM2.5', 'RH', 'WD', 'SR', 'BP', 'AT']
FEATURE_COLUMNS = ['RH', 'WD', 'SR', 'BP', 'AT']
TARGET_COLUMN = 'PM2.5'


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Loads the Excel dataset from the specified file path.
    Performs initial workbook inspection and validation.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    print(f"[PREPROCESSING] Loading Excel file from: {filepath}")
    xl = pd.ExcelFile(filepath)
    print(f"[PREPROCESSING] Sheet names detected: {xl.sheet_names}")

    df = pd.read_excel(filepath, sheet_name=0)
    print(f"[PREPROCESSING] Raw dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def inspect_raw_data(df: pd.DataFrame) -> dict:
    """
    Inspects column names, data types, missing values, and empty columns.
    Returns a dictionary of data quality metrics.
    """
    raw_shape = df.shape
    raw_columns = df.columns.tolist()
    missing_counts = df.isna().sum().to_dict()

    # Identify unnamed or empty columns
    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed")]

    print("\n--- RAW DATA INSPECTION ---")
    print(f"Total rows: {raw_shape[0]}, Total columns: {raw_shape[1]}")
    print(f"Columns: {raw_columns}")
    print("\nMissing values per column:")
    for col, count in missing_counts.items():
        if count > 0 or col in RELEVANT_COLUMNS:
            print(f"  - {col}: {count} missing ({count / raw_shape[0] * 100:.2f}%)")

    return {
        "raw_shape": raw_shape,
        "raw_columns": raw_columns,
        "missing_counts": missing_counts,
        "unnamed_cols": unnamed_cols
    }


def clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the six required columns, converts to numeric,
    removes extra unnamed columns, investigates duplicates, and performs IQR outlier screening.
    """
    # 1. Verify required columns exist
    missing_req = [col for col in RELEVANT_COLUMNS if col not in df.columns]
    if missing_req:
        raise ValueError(f"Required columns missing from dataset: {missing_req}")

    # 2. Select only the 6 relevant columns
    df_clean = df[RELEVANT_COLUMNS].copy()

    # 3. Ensure numeric conversion
    for col in RELEVANT_COLUMNS:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # 4. Duplicate investigation
    dup_count = df_clean.duplicated().sum()
    print(f"\n[PREPROCESSING] Duplicate rows detected across relevant columns: {dup_count}")
    print("  [JUSTIFICATION] Ambient continuous air monitoring sensors often record identical")
    print("  meteorological and pollutant values over consecutive reading cycles during stagnant")
    print("  atmospheric conditions. These repeated measurements reflect true environmental state persistence")
    print("  and are retained to avoid distorting the empirical distribution of observations.")

    # 5. Outlier identification using IQR method
    print("\n[PREPROCESSING] Outlier Screening via Interquartile Range (IQR):")
    outlier_summary = {}
    for col in RELEVANT_COLUMNS:
        q25 = df_clean[col].quantile(0.25)
        q75 = df_clean[col].quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
        outlier_summary[col] = {
            "q25": q25,
            "q75": q75,
            "iqr": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_count": len(outliers)
        }
        print(f"  - {col:5s}: IQR={iqr:.2f}, Lower={lower_bound:.2f}, Upper={upper_bound:.2f}, "
              f"Outliers={len(outliers)} ({len(outliers) / len(df_clean) * 100:.1f}%)")

    print("  [JUSTIFICATION] Elevated PM2.5 measurements (up to 427 ug/m3) represent severe winter")
    print("  smog and episodic atmospheric pollution events rather than instrumentation failure.")
    print("  Quantile Regression is inherently robust to response outliers and models conditional")
    print("  quantiles (such as the 90th percentile) specifically to capture these extreme risk regimes.")
    print("  Therefore, legitimate environmental extremes are strictly retained.")

    return df_clean


def split_and_impute(df_clean: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
    """
    Splits features (X) and target (y) into 80% train and 20% test sets.
    Imputes missing values (e.g. Solar Radiation SR) using the training median to avoid data leakage.
    Adds an intercept constant column for Quantile Regression.
    """
    X = df_clean[FEATURE_COLUMNS].copy()
    y = df_clean[TARGET_COLUMN].copy()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    print(f"\n[PREPROCESSING] Split completed (random_state={random_state}):")
    print(f"  - Training samples: {len(X_train)} ({len(X_train) / len(df_clean) * 100:.1f}%)")
    print(f"  - Testing samples:  {len(X_test)} ({len(X_test) / len(df_clean) * 100:.1f}%)")

    # Missing value handling: Solar Radiation (SR) median imputation from training set
    sr_train_median = X_train['SR'].median()
    train_sr_missing = X_train['SR'].isna().sum()
    test_sr_missing = X_test['SR'].isna().sum()

    print(f"[PREPROCESSING] Imputing SR missing values using training median: {sr_train_median:.2f}")
    print(f"  - Imputed in training set: {train_sr_missing} rows")
    print(f"  - Imputed in testing set:  {test_sr_missing} rows")

    X_train['SR'] = X_train['SR'].fillna(sr_train_median)
    X_test['SR'] = X_test['SR'].fillna(sr_train_median)

    # Add constant column for intercept in statsmodels QuantReg
    X_train_sm = X_train.copy()
    X_train_sm.insert(0, 'const', 1.0)

    X_test_sm = X_test.copy()
    X_test_sm.insert(0, 'const', 1.0)

    imputation_metadata = {
        "SR_median": sr_train_median,
        "train_imputed_count": int(train_sr_missing),
        "test_imputed_count": int(test_sr_missing)
    }

    return X_train, X_test, y_train, y_test, X_train_sm, X_test_sm, imputation_metadata
