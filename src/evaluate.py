"""
evaluate.py — Model evaluation for RUL prediction.

Metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Baseline comparison

Visualizations:
- Predicted vs Actual scatter
- Residual analysis
- Per-engine performance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute regression metrics for RUL prediction.

    Parameters
    ----------
    y_true : array-like
        Actual RUL values.
    y_pred : array-like
        Predicted RUL values.

    Returns
    -------
    dict
        Dictionary with MAE, RMSE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2)}


def evaluate_baseline(y_true: np.ndarray, baseline_pred: float) -> dict:
    """Evaluate naive baseline (predicting mean RUL for all)."""
    y_pred = np.full_like(y_true, fill_value=baseline_pred, dtype=float)
    return compute_metrics(y_true, y_pred)


def compare_models(results: dict) -> pd.DataFrame:
    """
    Create comparison table of model results.

    Parameters
    ----------
    results : dict
        {model_name: {"MAE": ..., "RMSE": ...}, ...}

    Returns
    -------
    pd.DataFrame
        Comparison table.
    """
    df = pd.DataFrame(results).T
    df.index.name = "Model"
    return df


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs Actual RUL",
    save_path: str = None,
):
    """
    Scatter plot of predicted vs actual RUL.

    A perfect model would have all points on the diagonal line.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="#2196F3", edgecolors="none")

    # Perfect prediction line
    lims = [0, max(max(y_true), max(y_pred))]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")

    ax.set_xlabel("Actual RUL (cycles)", fontsize=12)
    ax.set_ylabel("Predicted RUL (cycles)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Analysis",
    save_path: str = None,
):
    """
    Residual analysis plot.

    Shows: residual distribution and residuals vs predicted values.
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residual distribution
    axes[0].hist(residuals, bins=50, color="#4CAF50", edgecolor="white", alpha=0.8)
    axes[0].axvline(x=0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Residual (Actual - Predicted)", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title("Residual Distribution", fontsize=13)
    axes[0].grid(True, alpha=0.3)

    # Residuals vs predicted
    axes[1].scatter(y_pred, residuals, alpha=0.3, s=10, color="#FF9800", edgecolors="none")
    axes[1].axhline(y=0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Predicted RUL (cycles)", fontsize=11)
    axes[1].set_ylabel("Residual", fontsize=11)
    axes[1].set_title("Residuals vs Predicted", fontsize=13)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_per_engine_performance(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    save_path: str = None,
):
    """
    Plot predicted vs actual RUL at the last cycle of each test engine.

    This is the standard C-MAPSS evaluation: predict RUL at the end
    of each test trajectory.
    """
    last_cycles = test_df.copy()
    if "predicted_rul" not in last_cycles.columns:
        last_cycles["predicted_rul"] = y_pred

    fig, ax = plt.subplots(figsize=(12, 5))

    engine_ids = last_cycles["engine_id"].values
    actual = last_cycles["rul"].values
    predicted = last_cycles["predicted_rul"].values

    x = np.arange(len(engine_ids))
    width = 0.35

    ax.bar(x - width / 2, actual, width, label="Actual RUL", color="#2196F3", alpha=0.8)
    ax.bar(x + width / 2, predicted, width, label="Predicted RUL", color="#FF5722", alpha=0.8)

    ax.set_xlabel("Engine ID", fontsize=11)
    ax.set_ylabel("RUL (cycles)", fontsize=11)
    ax.set_title("Per-Engine RUL: Actual vs Predicted (Last Cycle)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Show every 5th engine ID
    tick_indices = np.arange(0, len(engine_ids), max(1, len(engine_ids) // 20))
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(engine_ids[tick_indices], rotation=45, fontsize=8)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


def full_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    baseline_pred: float,
    save_dir: str = None,
) -> dict:
    """
    Run complete evaluation suite.

    Parameters
    ----------
    y_true : array-like
        Actual RUL values.
    y_pred : array-like
        Predicted RUL values.
    model_name : str
        Name of the model.
    baseline_pred : float
        Naive baseline prediction value.
    save_dir : str, optional
        Directory to save plots.

    Returns
    -------
    dict
        All metrics and results.
    """
    print(f"\n{'='*50}")
    print(f" Evaluation: {model_name}")
    print(f"{'='*50}")

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred)
    baseline_metrics = evaluate_baseline(y_true, baseline_pred)

    print(f"\n  {model_name}:")
    print(f"    MAE:  {metrics['MAE']}")
    print(f"    RMSE: {metrics['RMSE']}")
    print(f"\n  Baseline (mean RUL):")
    print(f"    MAE:  {baseline_metrics['MAE']}")
    print(f"    RMSE: {baseline_metrics['RMSE']}")

    improvement_mae = ((baseline_metrics["MAE"] - metrics["MAE"]) / baseline_metrics["MAE"]) * 100
    print(f"\n  MAE improvement over baseline: {improvement_mae:.1f}%")

    # Generate plots
    if save_dir:
        save_dir = Path(save_dir)
        plot_predictions_vs_actual(
            y_true, y_pred,
            title=f"{model_name} — Predicted vs Actual RUL",
            save_path=str(save_dir / f"{model_name.lower().replace(' ', '_')}_pred_vs_actual.png"),
        )
        plot_residuals(
            y_true, y_pred,
            title=f"{model_name} — Residual Analysis",
            save_path=str(save_dir / f"{model_name.lower().replace(' ', '_')}_residuals.png"),
        )

    return {
        "model": model_name,
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "improvement_mae_pct": round(improvement_mae, 1),
    }
