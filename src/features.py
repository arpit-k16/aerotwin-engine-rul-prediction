"""
features.py — Temporal feature engineering for engine degradation modeling.

Creates meaningful time-series features that capture degradation trends:
- Rolling statistics (mean, std) over windows
- Rolling linear slope (trend direction)
- Exponentially weighted moving average
- Cycle normalization
"""

import numpy as np
import pandas as pd


def add_rolling_features(
    df: pd.DataFrame,
    sensor_cols: list,
    windows: list = None,
) -> pd.DataFrame:
    """
    Add rolling mean and standard deviation features for each sensor.

    These capture local trends and variability in sensor readings,
    which are key indicators of progressing degradation.

    Parameters
    ----------
    df : pd.DataFrame
        Data with engine_id, cycle, and sensor columns.
    sensor_cols : list
        Sensor columns to compute rolling features for.
    windows : list
        Rolling window sizes (in cycles).

    Returns
    -------
    pd.DataFrame
        Data with additional rolling features.
    """
    if windows is None:
        windows = [5, 10, 20]

    df = df.copy()
    df = df.sort_values(["engine_id", "cycle"])

    for window in windows:
        for col in sensor_cols:
            # Rolling mean — smoothed trend
            df[f"{col}_rmean_{window}"] = (
                df.groupby("engine_id")[col]
                .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            )
            # Rolling std — increasing variability signals degradation
            df[f"{col}_rstd_{window}"] = (
                df.groupby("engine_id")[col]
                .transform(lambda x: x.rolling(window=window, min_periods=1).std())
            )

    # Fill NaN in std columns (first few cycles have insufficient data)
    std_cols = [c for c in df.columns if "_rstd_" in c]
    df[std_cols] = df[std_cols].fillna(0)

    return df


def add_rolling_slope(
    df: pd.DataFrame,
    sensor_cols: list,
    window: int = 10,
) -> pd.DataFrame:
    """
    Add rolling linear slope for each sensor.

    A positive/negative slope indicates the direction of sensor drift,
    which is a direct indicator of degradation trajectory.

    Parameters
    ----------
    df : pd.DataFrame
        Data with engine_id, cycle, and sensor columns.
    sensor_cols : list
        Sensor columns.
    window : int
        Window size for slope calculation.

    Returns
    -------
    pd.DataFrame
        Data with slope features.
    """
    df = df.copy()
    df = df.sort_values(["engine_id", "cycle"])

    for col in sensor_cols:
        def calc_slope(series):
            """Calculate slope using linear regression over rolling window."""
            result = pd.Series(index=series.index, dtype=float)
            for i in range(len(series)):
                start = max(0, i - window + 1)
                window_data = series.iloc[start:i + 1]
                if len(window_data) < 3:
                    result.iloc[i] = 0.0
                else:
                    x = np.arange(len(window_data))
                    y = window_data.values
                    # Simple OLS slope: cov(x,y) / var(x)
                    x_mean = x.mean()
                    y_mean = y.mean()
                    slope = np.sum((x - x_mean) * (y - y_mean)) / (np.sum((x - x_mean) ** 2) + 1e-10)
                    result.iloc[i] = slope
            return result

        df[f"{col}_slope_{window}"] = (
            df.groupby("engine_id")[col].transform(calc_slope)
        )

    return df


def add_ewma_features(
    df: pd.DataFrame,
    sensor_cols: list,
    span: int = 10,
) -> pd.DataFrame:
    """
    Add Exponentially Weighted Moving Average features.

    EWMA gives more weight to recent observations, making it responsive
    to recent degradation while smoothing noise.

    Parameters
    ----------
    df : pd.DataFrame
        Data with engine_id and sensor columns.
    sensor_cols : list
        Sensor columns.
    span : int
        EWMA span parameter.

    Returns
    -------
    pd.DataFrame
        Data with EWMA features.
    """
    df = df.copy()
    df = df.sort_values(["engine_id", "cycle"])

    for col in sensor_cols:
        df[f"{col}_ewma_{span}"] = (
            df.groupby("engine_id")[col]
            .transform(lambda x: x.ewm(span=span, min_periods=1).mean())
        )

    return df





def engineer_features(
    df: pd.DataFrame,
    sensor_cols: list,
    rolling_windows: list = None,
    slope_window: int = 10,
    ewma_span: int = 10,
    top_sensors: list = None,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    To keep computation manageable, rolling slope and EWMA are computed
    only for the most important sensors (determined by initial analysis).

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed data.
    sensor_cols : list
        All sensor columns available.
    rolling_windows : list
        Windows for rolling mean/std.
    slope_window : int
        Window for rolling slope.
    ewma_span : int
        Span for EWMA.
    top_sensors : list, optional
        Subset of sensors for expensive features (slope, EWMA).
        If None, uses all sensors.

    Returns
    -------
    pd.DataFrame
        Data with engineered features.
    """
    if rolling_windows is None:
        rolling_windows = [5, 10, 20]

    if top_sensors is None:
        # Default: use top sensors known to be informative in FD001
        top_sensors = [c for c in sensor_cols if c in [
            "sensor_2", "sensor_3", "sensor_4", "sensor_7",
            "sensor_8", "sensor_9", "sensor_11", "sensor_12",
            "sensor_13", "sensor_14", "sensor_15", "sensor_17",
            "sensor_20", "sensor_21",
        ]]

    print(f"Engineering features for {len(sensor_cols)} sensors...")
    print(f"  Rolling mean/std windows: {rolling_windows}")
    print(f"  Slope/EWMA sensors: {len(top_sensors)}")

    # Rolling statistics for all sensors
    df = add_rolling_features(df, sensor_cols, rolling_windows)
    print("  ✓ Rolling mean/std features added")

    # Rolling slope for top sensors only (computationally expensive)
    df = add_rolling_slope(df, top_sensors, slope_window)
    print("  ✓ Rolling slope features added")

    # EWMA for top sensors
    df = add_ewma_features(df, top_sensors, ewma_span)
    print("  ✓ EWMA features added")



    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Get all feature columns (excluding identifiers and target).

    Parameters
    ----------
    df : pd.DataFrame
        Engineered data.

    Returns
    -------
    list
        Feature column names.
    """
    exclude = ["engine_id", "rul", "max_cycle"]
    return [c for c in df.columns if c not in exclude]
