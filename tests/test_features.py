"""
Tests for feature engineering functions.
"""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import (
    add_rolling_features,
    add_ewma_features,
    get_feature_columns,
)


def make_sample_df():
    """Create sample dataframe for feature testing."""
    np.random.seed(42)
    df = pd.DataFrame({
        "engine_id": [1] * 20 + [2] * 20,
        "cycle": list(range(1, 21)) + list(range(1, 21)),
        "sensor_2": np.random.randn(40) * 5 + 642,
        "sensor_3": np.random.randn(40) * 10 + 1590,
        "rul": list(range(19, -1, -1)) + list(range(19, -1, -1)),
    })
    return df


class TestRollingFeatures:
    def test_adds_rolling_columns(self):
        df = make_sample_df()
        result = add_rolling_features(df, ["sensor_2"], windows=[5])
        assert "sensor_2_rmean_5" in result.columns
        assert "sensor_2_rstd_5" in result.columns

    def test_multiple_windows(self):
        df = make_sample_df()
        result = add_rolling_features(df, ["sensor_2"], windows=[5, 10])
        assert "sensor_2_rmean_5" in result.columns
        assert "sensor_2_rmean_10" in result.columns

    def test_no_nan_in_std(self):
        df = make_sample_df()
        result = add_rolling_features(df, ["sensor_2"], windows=[5])
        assert not result["sensor_2_rstd_5"].isna().any()

    def test_preserves_original_columns(self):
        df = make_sample_df()
        result = add_rolling_features(df, ["sensor_2"], windows=[5])
        assert "engine_id" in result.columns
        assert "cycle" in result.columns
        assert "rul" in result.columns


class TestEwmaFeatures:
    def test_adds_ewma_column(self):
        df = make_sample_df()
        result = add_ewma_features(df, ["sensor_2"], span=5)
        assert "sensor_2_ewma_5" in result.columns

    def test_ewma_no_nan(self):
        df = make_sample_df()
        result = add_ewma_features(df, ["sensor_2"], span=5)
        assert not result["sensor_2_ewma_5"].isna().any()





class TestGetFeatureColumns:
    def test_excludes_identifiers(self):
        df = make_sample_df()
        df["sensor_2_rmean_5"] = 0
        features = get_feature_columns(df)
        assert "engine_id" not in features
        assert "rul" not in features
        assert "cycle" in features  # cycle should be included as a feature now
        assert "sensor_2_rmean_5" in features
