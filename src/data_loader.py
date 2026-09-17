"""
data_loader.py — Load NASA C-MAPSS turbofan engine degradation dataset.

Dataset: NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)
Source: NASA Prognostics Center of Excellence
Reference: Saxena et al., "Damage Propagation Modeling for Aircraft Engine 
           Run-to-Failure Simulation", PHM 2008.

The dataset contains 26 space-separated columns with no header:
  Col 1:    Engine unit number
  Col 2:    Time (cycles)
  Col 3-5:  Operational settings (3)
  Col 6-26: Sensor measurements (21)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path


# Standard column names for C-MAPSS dataset
COLUMN_NAMES = (
    ["engine_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Sensor descriptions based on C-MAPSS documentation
SENSOR_DESCRIPTIONS = {
    "sensor_1": "Total temperature at fan inlet (°R)",
    "sensor_2": "Total temperature at LPC outlet (°R)",
    "sensor_3": "Total temperature at HPC outlet (°R)",
    "sensor_4": "Total temperature at LPT outlet (°R)",
    "sensor_5": "Pressure at fan inlet (psia)",
    "sensor_6": "Total pressure in bypass-duct (psia)",
    "sensor_7": "Total pressure at HPC outlet (psia)",
    "sensor_8": "Physical fan speed (rpm)",
    "sensor_9": "Physical core speed (rpm)",
    "sensor_10": "Engine pressure ratio (P50/P2)",
    "sensor_11": "Static pressure at HPC outlet (psia)",
    "sensor_12": "Ratio of fuel flow to Ps30 (pps/psi)",
    "sensor_13": "Corrected fan speed (rpm)",
    "sensor_14": "Corrected core speed (rpm)",
    "sensor_15": "Bypass ratio",
    "sensor_16": "Burner fuel-air ratio",
    "sensor_17": "Bleed enthalpy",
    "sensor_18": "Demanded fan speed (rpm)",
    "sensor_19": "Demanded corrected fan speed (rpm)",
    "sensor_20": "HPT coolant bleed (lbm/s)",
    "sensor_21": "LPT coolant bleed (lbm/s)",
}

# Dataset metadata
DATASET_INFO = {
    "FD001": {"train_engines": 100, "test_engines": 100, "conditions": 1, "fault_modes": 1},
    "FD002": {"train_engines": 260, "test_engines": 259, "conditions": 6, "fault_modes": 1},
    "FD003": {"train_engines": 100, "test_engines": 100, "conditions": 1, "fault_modes": 2},
    "FD004": {"train_engines": 248, "test_engines": 249, "conditions": 6, "fault_modes": 2},
}


def load_cmapss(data_dir: str, dataset: str = "FD001") -> tuple:
    """
    Load a C-MAPSS dataset (train, test, RUL).

    Parameters
    ----------
    data_dir : str
        Path to directory containing the raw .txt files.
    dataset : str
        Dataset identifier: 'FD001', 'FD002', 'FD003', or 'FD004'.

    Returns
    -------
    train_df : pd.DataFrame
        Training data with run-to-failure trajectories.
    test_df : pd.DataFrame
        Test data with partial trajectories.
    rul_df : pd.DataFrame
        True RUL values for test engines.
    """
    data_dir = Path(data_dir)

    if dataset not in DATASET_INFO:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {list(DATASET_INFO.keys())}")

    train_path = data_dir / f"train_{dataset}.txt"
    test_path = data_dir / f"test_{dataset}.txt"
    rul_path = data_dir / f"RUL_{dataset}.txt"

    for path in [train_path, test_path, rul_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Data file not found: {path}\n"
                f"Please download the NASA C-MAPSS dataset and place files in: {data_dir}\n"
                f"Download from: https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/xaut-bemq"
            )

    # Load training data
    train_df = pd.read_csv(
        train_path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
        engine="python",
    )

    # Load test data
    test_df = pd.read_csv(
        test_path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
        engine="python",
    )

    # Load RUL values
    rul_df = pd.read_csv(
        rul_path,
        sep=r"\s+",
        header=None,
        names=["rul"],
    )
    rul_df["engine_id"] = rul_df.index + 1

    # Type enforcement
    for df in [train_df, test_df]:
        df["engine_id"] = df["engine_id"].astype(int)
        df["cycle"] = df["cycle"].astype(int)

    return train_df, test_df, rul_df


def add_rul_to_train(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Remaining Useful Life (RUL) for training data.

    For each engine, RUL at cycle t = (max_cycle - current_cycle).
    The last cycle has RUL = 0 (failure point).

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with engine_id and cycle columns.

    Returns
    -------
    pd.DataFrame
        Training data with 'rul' column added.
    """
    df = train_df.copy()
    max_cycles = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycles.columns = ["engine_id", "max_cycle"]
    df = df.merge(max_cycles, on="engine_id", how="left")
    df["rul"] = df["max_cycle"] - df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)
    return df


def add_rul_to_test(test_df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RUL for test data using ground truth RUL values.

    The ground truth RUL gives the remaining cycles at the LAST cycle
    of each test trajectory. We extrapolate backwards.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test data.
    rul_df : pd.DataFrame
        Ground truth RUL values.

    Returns
    -------
    pd.DataFrame
        Test data with 'rul' column.
    """
    df = test_df.copy()
    max_cycles = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycles.columns = ["engine_id", "max_cycle"]
    df = df.merge(max_cycles, on="engine_id", how="left")
    df = df.merge(rul_df[["engine_id", "rul"]], on="engine_id", how="left", suffixes=("", "_true"))
    # RUL at cycle t = rul_at_last_cycle + (max_cycle - current_cycle)
    df["rul"] = df["rul"] + (df["max_cycle"] - df["cycle"])
    df.drop(columns=["max_cycle"], inplace=True)
    return df


def get_sensor_columns() -> list:
    """Return list of sensor column names."""
    return [f"sensor_{i}" for i in range(1, 22)]


def get_op_setting_columns() -> list:
    """Return list of operational setting column names."""
    return [f"op_setting_{i}" for i in range(1, 4)]


if __name__ == "__main__":
    # Quick smoke test
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    train_df, test_df, rul_df = load_cmapss(data_dir, "FD001")
    train_df = add_rul_to_train(train_df)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape:  {test_df.shape}")
    print(f"RUL entries: {len(rul_df)}")
    print(f"Train engines: {train_df['engine_id'].nunique()}")
    print(f"Test engines:  {test_df['engine_id'].nunique()}")
    print(f"\nTrain RUL range: [{train_df['rul'].min()}, {train_df['rul'].max()}]")
    print(f"\nFirst 5 rows:\n{train_df.head()}")
