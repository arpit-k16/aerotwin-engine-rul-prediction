"""
explainability.py — SHAP-based model explainability for RUL predictions.

Provides:
- Global feature importance (SHAP summary)
- Individual engine-level explanations
- Top contributing features for specific predictions

IMPORTANT DISCLAIMER:
SHAP values indicate model-internal feature contributions, NOT physical causality.
A high SHAP value for a sensor means the model relies on it for prediction,
not necessarily that the sensor directly causes engine degradation.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import joblib
from pathlib import Path


def compute_shap_values(
    model,
    X: pd.DataFrame,
    max_samples: int = 1000,
    random_state: int = 42,
) -> shap.Explanation:
    """
    Compute SHAP values for the given model and data.

    Uses TreeExplainer for tree-based models (XGBoost, RF).

    Parameters
    ----------
    model : trained model
        Tree-based model (XGBoost or RandomForest).
    X : pd.DataFrame
        Feature data.
    max_samples : int
        Maximum samples for SHAP computation (for speed).
    random_state : int
        Random seed for sampling.

    Returns
    -------
    shap.Explanation
        SHAP explanation object.
    """
    # Sample if dataset is too large
    if len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=random_state)
    else:
        X_sample = X.copy()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    return shap_values


def plot_global_importance(
    shap_values: shap.Explanation,
    max_features: int = 20,
    save_path: str = None,
):
    """
    Plot global SHAP feature importance (bar chart).

    Shows which features the model relies on most across all predictions.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=max_features, show=False, ax=ax)
    ax.set_title("Global Feature Importance (SHAP)", fontsize=14)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_summary(
    shap_values: shap.Explanation,
    max_features: int = 20,
    save_path: str = None,
):
    """
    SHAP summary (beeswarm) plot.

    Shows feature importance AND the direction of feature effects.
    Red = high feature value, Blue = low feature value.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=max_features, show=False)
    plt.title("SHAP Summary Plot", fontsize=14)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def get_top_features(
    shap_values: shap.Explanation,
    n_top: int = 10,
) -> pd.DataFrame:
    """
    Get top N features by mean absolute SHAP value.

    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values.
    n_top : int
        Number of top features to return.

    Returns
    -------
    pd.DataFrame
        Top features with mean |SHAP| values.
    """
    feature_importance = pd.DataFrame({
        "feature": shap_values.feature_names,
        "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
    })
    feature_importance = feature_importance.sort_values("mean_abs_shap", ascending=False)
    return feature_importance.head(n_top).reset_index(drop=True)


def explain_single_prediction(
    shap_values: shap.Explanation,
    index: int,
    n_top: int = 5,
) -> pd.DataFrame:
    """
    Explain a single prediction.

    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values.
    index : int
        Index of the sample to explain (within SHAP values).
    n_top : int
        Number of top features to show.

    Returns
    -------
    pd.DataFrame
        Top contributing features for this prediction.
    """
    sample_shap = shap_values[index]
    contributions = pd.DataFrame({
        "feature": sample_shap.feature_names,
        "shap_value": sample_shap.values,
        "feature_value": sample_shap.data,
    })
    contributions["abs_shap"] = contributions["shap_value"].abs()
    contributions = contributions.sort_values("abs_shap", ascending=False)
    return contributions.head(n_top).drop(columns=["abs_shap"]).reset_index(drop=True)


def plot_single_explanation(
    shap_values: shap.Explanation,
    index: int,
    save_path: str = None,
):
    """
    Waterfall plot for a single prediction explanation.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[index], max_display=10, show=False)
    plt.title(f"Prediction Explanation (Sample {index})", fontsize=14)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def run_explainability(
    model,
    X: pd.DataFrame,
    save_dir: str = None,
    max_samples: int = 1000,
) -> tuple:
    """
    Run the full explainability pipeline.

    Parameters
    ----------
    model : trained model
        XGBoost or RF model.
    X : pd.DataFrame
        Feature data.
    save_dir : str, optional
        Directory to save plots and artifacts.
    max_samples : int
        Maximum samples for SHAP.

    Returns
    -------
    shap_values : shap.Explanation
    top_features : pd.DataFrame
    """
    print("\n Computing SHAP values...")
    shap_values = compute_shap_values(model, X, max_samples=max_samples)

    top_features = get_top_features(shap_values, n_top=15)
    print("\n Top features by mean |SHAP|:")
    print(top_features.to_string(index=False))

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        plot_global_importance(
            shap_values,
            save_path=str(save_dir / "shap_global_importance.png"),
        )
        plot_summary(
            shap_values,
            save_path=str(save_dir / "shap_summary.png"),
        )
        plot_single_explanation(
            shap_values,
            index=0,
            save_path=str(save_dir / "shap_single_explanation.png"),
        )

        # Save SHAP values
        joblib.dump(shap_values, save_dir / "shap_values.joblib")
        top_features.to_csv(save_dir / "top_features.csv", index=False)

    return shap_values, top_features
