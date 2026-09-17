"""
train.py — Model training for RUL prediction.

Models:
- Baseline: Naive average RUL prediction
- Random Forest Regressor
- XGBoost Regressor (primary model)

All models use proper train/validation splits that respect engine boundaries.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

RANDOM_STATE = 42


def get_naive_baseline(train_df: pd.DataFrame) -> float:
    """
    Compute naive baseline: predict average RUL for all samples.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with 'rul' column.

    Returns
    -------
    float
        Mean RUL value.
    """
    return train_df["rul"].mean()


def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> LinearRegression:
    """Train a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
) -> RandomForestRegressor:
    """
    Train a Random Forest Regressor.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        RUL target.
    n_estimators : int
        Number of trees.

    Returns
    -------
    RandomForestRegressor
        Trained model.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
    params: dict = None,
) -> "xgb.XGBRegressor":
    """
    Train an XGBoost Regressor.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        RUL target.
    X_val : pd.DataFrame, optional
        Validation features for early stopping.
    y_val : pd.Series, optional
        Validation RUL.
    params : dict, optional
        Custom XGBoost parameters.

    Returns
    -------
    xgb.XGBRegressor
        Trained model.
    """
    if not HAS_XGBOOST:
        raise ImportError("XGBoost is not installed. Run: pip install xgboost")

    if params is None:
        params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }

    model = xgb.XGBRegressor(**params)

    if X_val is not None and y_val is not None:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train, verbose=False)

    return model


def save_model(model, model_dir: str, model_name: str):
    """Save a trained model to disk."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{model_name}.joblib"
    joblib.dump(model, path)
    print(f"Model saved: {path}")
    return path


def load_model(model_dir: str, model_name: str):
    """Load a trained model from disk."""
    path = Path(model_dir) / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def split_train_val(
    df: pd.DataFrame,
    val_fraction: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple:
    """
    Split training data into train/validation sets by ENGINE.

    This prevents data leakage: all cycles from one engine are
    in either train or validation, never split across both.

    Parameters
    ----------
    df : pd.DataFrame
        Full training data.
    val_fraction : float
        Fraction of engines for validation.
    random_state : int
        Random seed.

    Returns
    -------
    train_split : pd.DataFrame
    val_split : pd.DataFrame
    """
    engine_ids = df["engine_id"].unique()
    np.random.seed(random_state)
    np.random.shuffle(engine_ids)

    n_val = max(1, int(len(engine_ids) * val_fraction))
    val_engines = engine_ids[:n_val]
    train_engines = engine_ids[n_val:]

    train_split = df[df["engine_id"].isin(train_engines)].copy()
    val_split = df[df["engine_id"].isin(val_engines)].copy()

    print(f"Train/Val split: {len(train_engines)} / {len(val_engines)} engines")
    return train_split, val_split
