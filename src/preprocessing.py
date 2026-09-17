"""
preprocessing.py — Data preprocessing for C-MAPSS turbofan engine data.

Handles:
- Identifying and removing constant/near-constant sensors
- MinMax scaling (fitted on training data only to prevent leakage)
- RUL capping (optional, justified)
- Train/test split preserving engine boundaries
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from pathlib import Path

from src.data_loader import get_sensor_columns, get_op_setting_columns


# Sensors known to be constant or near-constant in FD001
# These carry no degradation information
CONSTANT_SENSORS_FD001 = ["sensor_1", "sensor_5", "sensor_6", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]


def identify_constant_sensors(df: pd.DataFrame, threshold: float = 1e-4) -> list:
    """
    Identify sensors with near-zero variance (effectively constant).

    Parameters
    ----------
    df : pd.DataFrame
        Training data.
    threshold : float
        Standard deviation threshold below which a sensor is considered constant.

    Returns
    -------
    list
        List of constant sensor column names.
    """
    sensor_cols = [c for c in get_sensor_columns() if c in df.columns]
    constant_sensors = []
    for col in sensor_cols:
        if df[col].std() < threshold:
            constant_sensors.append(col)
    return constant_sensors


def remove_constant_sensors(df: pd.DataFrame, constant_sensors: list) -> pd.DataFrame:
    """Remove constant sensors from the dataframe."""
    return df.drop(columns=[c for c in constant_sensors if c in df.columns])


def cap_rul(df: pd.DataFrame, cap_value: int = 125) -> pd.DataFrame:
    """
    Cap RUL at a maximum value.

    Justification: Early in an engine's life, degradation is minimal and
    RUL values are very high but not meaningfully different. Capping at 125
    cycles focuses the model on the degradation region where predictions
    matter for maintenance decisions. This is a standard practice in
    C-MAPSS literature (e.g., Heimes 2008, Zheng et al. 2017).

    Parameters
    ----------
    df : pd.DataFrame
        Data with 'rul' column.
    cap_value : int
        Maximum RUL value (default: 125 based on literature).

    Returns
    -------
    pd.DataFrame
        Data with capped RUL.
    """
    df = df.copy()
    df["rul"] = df["rul"].clip(upper=cap_value)
    return df


def fit_scaler(train_df: pd.DataFrame, feature_cols: list) -> MinMaxScaler:
    """
    Fit MinMaxScaler on training data ONLY.

    This prevents data leakage from test set into scaling parameters.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data.
    feature_cols : list
        Columns to scale.

    Returns
    -------
    MinMaxScaler
        Fitted scaler.
    """
    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_cols])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: MinMaxScaler, feature_cols: list) -> pd.DataFrame:
    """
    Apply a pre-fitted scaler to data.

    Parameters
    ----------
    df : pd.DataFrame
        Data to scale.
    scaler : MinMaxScaler
        Pre-fitted scaler.
    feature_cols : list
        Columns to scale.

    Returns
    -------
    pd.DataFrame
        Scaled data.
    """
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols])
    return df


def preprocess_pipeline(
    df: pd.DataFrame,
    rul_cap: int = 125,
    constant_sensors: list = None,
) -> tuple:
    """
    Data preprocessing pipeline (capping and removing constant sensors).

    Scaling is intentionally NOT done here. The scaler must be fit on
    the training split *after* train/validation splitting to prevent leakage,
    and then applied separately.

    Parameters
    ----------
    df : pd.DataFrame
        Data with RUL.
    rul_cap : int
        RUL cap value.
    constant_sensors : list, optional
        List of constant sensors to remove. If None, it computes them from `df`.

    Returns
    -------
    df_processed : pd.DataFrame
        Processed data without scaling.
    constant_sensors : list
        The list of constant sensors removed.
    feature_cols : list
        Remaining feature columns.
    """
    # Step 1: Cap RUL if column exists
    if "rul" in df.columns:
        df = cap_rul(df, rul_cap)

    # Step 2: Identify and remove constant sensors
    if constant_sensors is None:
        constant_sensors = identify_constant_sensors(df)

    df_processed = remove_constant_sensors(df, constant_sensors)

    # Step 3: Determine feature columns
    remaining_sensors = [c for c in get_sensor_columns() if c in df_processed.columns]
    op_settings = [c for c in get_op_setting_columns() if c in df_processed.columns]
    feature_cols = ["cycle"] + op_settings + remaining_sensors

    return df_processed, constant_sensors, feature_cols
