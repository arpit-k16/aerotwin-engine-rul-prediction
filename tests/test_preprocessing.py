"""
Tests for data preprocessing functions.
"""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import (
    identify_constant_sensors,
    remove_constant_sensors,
    cap_rul,
    fit_scaler,
    apply_scaler,
)


def make_sample_df():
    """Create a sample dataframe for testing."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "engine_id": [1] * 50 + [2] * 50,
        "cycle": list(range(1, 51)) + list(range(1, 51)),
        "op_setting_1": np.random.randn(n) * 0.01,
        "op_setting_2": np.random.randn(n) * 0.01,
        "op_setting_3": np.ones(n) * 100.0,  # constant
        "sensor_1": np.ones(n) * 518.67,  # constant
        "sensor_2": np.random.randn(n) * 10 + 642,
        "sensor_3": np.random.randn(n) * 20 + 1590,
        "sensor_4": np.random.randn(n) * 15 + 1400,
        "rul": list(range(49, -1, -1)) + list(range(49, -1, -1)),
    })
    return df


class TestIdentifyConstantSensors:
    def test_identifies_constant_sensor(self):
        df = make_sample_df()
        constants = identify_constant_sensors(df, threshold=0.01)
        assert "sensor_1" in constants

    def test_non_constant_sensors_not_flagged(self):
        df = make_sample_df()
        constants = identify_constant_sensors(df, threshold=0.01)
        assert "sensor_2" not in constants
        assert "sensor_3" not in constants


class TestRemoveConstantSensors:
    def test_removes_columns(self):
        df = make_sample_df()
        result = remove_constant_sensors(df, ["sensor_1"])
        assert "sensor_1" not in result.columns
        assert "sensor_2" in result.columns

    def test_handles_missing_columns(self):
        df = make_sample_df()
        result = remove_constant_sensors(df, ["nonexistent"])
        assert len(result.columns) == len(df.columns)


class TestCapRul:
    def test_caps_rul(self):
        df = make_sample_df()
        # Set some high RUL values
        df.loc[0, "rul"] = 200
        result = cap_rul(df, cap_value=125)
        assert result["rul"].max() <= 125

    def test_does_not_modify_below_cap(self):
        df = make_sample_df()
        result = cap_rul(df, cap_value=125)
        # Original max is 49, should be unchanged
        assert result["rul"].max() == 49

    def test_preserves_original(self):
        df = make_sample_df()
        original_max = df["rul"].max()
        _ = cap_rul(df, cap_value=125)
        assert df["rul"].max() == original_max  # original unchanged


class TestScaler:
    def test_fit_scaler(self):
        df = make_sample_df()
        feature_cols = ["sensor_2", "sensor_3"]
        scaler = fit_scaler(df, feature_cols)
        assert hasattr(scaler, "data_min_")

    def test_apply_scaler_range(self):
        df = make_sample_df()
        feature_cols = ["sensor_2", "sensor_3"]
        scaler = fit_scaler(df, feature_cols)
        result = apply_scaler(df, scaler, feature_cols)
        for col in feature_cols:
            assert result[col].min() >= -0.01  # approximately 0
            assert result[col].max() <= 1.01   # approximately 1

    def test_scaler_preserves_other_columns(self):
        df = make_sample_df()
        feature_cols = ["sensor_2"]
        scaler = fit_scaler(df, feature_cols)
        result = apply_scaler(df, scaler, feature_cols)
        assert (result["engine_id"] == df["engine_id"]).all()
        assert (result["rul"] == df["rul"]).all()
