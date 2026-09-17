"""
health_score.py — Engine Health Score computation.

Converts model predictions and degradation signals into an interpretable
health indicator (0–100 scale) with risk categories.

DISCLAIMER:
This health score is an analytical indicator derived from model outputs
and degradation signals for portfolio/research purposes. It is NOT an
industry-certified aerospace metric and should not be used for actual
maintenance decisions.
"""

import numpy as np
import pandas as pd


# Risk category thresholds
RISK_THRESHOLDS = {
    "Healthy": (70, 100),
    "Watch": (40, 70),
    "High Risk": (0, 40),
}

# Color mapping for dashboard
RISK_COLORS = {
    "Healthy": "#4CAF50",
    "Watch": "#FF9800",
    "High Risk": "#F44336",
}


def compute_health_score(
    predicted_rul: float,
    max_rul: float = 125.0,
    min_rul: float = 0.0,
) -> float:
    """
    Compute health score from predicted RUL.

    Health Score = (predicted_rul / max_rul) * 100, clamped to [0, 100].

    The score linearly maps RUL to a 0–100 scale where:
    - 100 = engine at maximum expected remaining life
    - 0 = engine at predicted failure

    Parameters
    ----------
    predicted_rul : float
        Model's predicted RUL.
    max_rul : float
        Maximum RUL value (typically the cap value).
    min_rul : float
        Minimum RUL value.

    Returns
    -------
    float
        Health score in [0, 100].
    """
    score = ((predicted_rul - min_rul) / (max_rul - min_rul)) * 100
    return float(np.clip(score, 0, 100))


def get_risk_category(health_score: float) -> str:
    """
    Classify health score into risk category.

    Parameters
    ----------
    health_score : float
        Health score (0–100).

    Returns
    -------
    str
        Risk category: 'Healthy', 'Watch', or 'High Risk'.
    """
    for category, (low, high) in RISK_THRESHOLDS.items():
        if low <= health_score <= high:
            return category
    return "High Risk"


def compute_degradation_trend(
    rul_predictions: np.ndarray,
    window: int = 5,
) -> float:
    """
    Compute recent degradation trend from RUL predictions.

    A negative slope means RUL is declining (expected behavior).
    A steeply negative slope indicates rapid degradation.

    Parameters
    ----------
    rul_predictions : array-like
        Sequence of recent RUL predictions.
    window : int
        Number of recent cycles to consider.

    Returns
    -------
    float
        Slope of recent RUL trend (cycles per cycle).
    """
    recent = np.array(rul_predictions[-window:])
    if len(recent) < 2:
        return 0.0

    x = np.arange(len(recent))
    slope = np.polyfit(x, recent, 1)[0]
    return float(slope)


def generate_maintenance_insight(
    health_score: float,
    risk_category: str,
    predicted_rul: float,
    degradation_trend: float,
) -> str:
    """
    Generate a human-readable maintenance insight statement.

    Parameters
    ----------
    health_score : float
        Current health score.
    risk_category : str
        Current risk category.
    predicted_rul : float
        Predicted remaining useful life.
    degradation_trend : float
        Recent degradation trend slope.

    Returns
    -------
    str
        Maintenance insight statement.
    """
    insights = []

    if risk_category == "Healthy":
        insights.append(
            f"Engine health score is {health_score:.0f}/100 — operating within normal parameters."
        )
        insights.append(
            f"Estimated {predicted_rul:.0f} cycles remaining before predicted maintenance need."
        )
    elif risk_category == "Watch":
        insights.append(
            f"Engine health score is {health_score:.0f}/100 — showing early signs of degradation."
        )
        insights.append(
            f"Estimated {predicted_rul:.0f} cycles remaining. Increased monitoring recommended."
        )
    else:  # High Risk
        insights.append(
            f"Engine health score is {health_score:.0f}/100 — significant degradation detected."
        )
        insights.append(
            f"Estimated {predicted_rul:.0f} cycles remaining. Maintenance inspection recommended."
        )

    if degradation_trend < -1.5:
        insights.append("Recent sensor behavior indicates accelerating degradation.")
    elif degradation_trend < -0.5:
        insights.append("Degradation is progressing at a moderate rate.")
    else:
        insights.append("Degradation rate is within expected range.")

    # Disclaimer
    insights.append(
        "\n⚠️ This is an analytical indicator from model predictions, "
        "not a certified maintenance decision."
    )

    return "\n".join(insights)


def compute_engine_health_profile(
    engine_df: pd.DataFrame,
    predictions: np.ndarray,
    max_rul: float = 125.0,
) -> pd.DataFrame:
    """
    Compute health profile for an engine's trajectory.

    Parameters
    ----------
    engine_df : pd.DataFrame
        Data for a single engine.
    predictions : np.ndarray
        RUL predictions for each cycle.
    max_rul : float
        Maximum RUL for scoring.

    Returns
    -------
    pd.DataFrame
        Engine health profile with scores and risk categories.
    """
    profile = engine_df[["engine_id", "cycle"]].copy()
    profile["predicted_rul"] = predictions
    profile["health_score"] = profile["predicted_rul"].apply(
        lambda x: compute_health_score(x, max_rul)
    )
    profile["risk_category"] = profile["health_score"].apply(get_risk_category)

    if "rul" in engine_df.columns:
        profile["actual_rul"] = engine_df["rul"].values

    return profile
