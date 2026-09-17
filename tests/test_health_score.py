"""
Tests for health score computation.
"""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.health_score import (
    compute_health_score,
    get_risk_category,
    compute_degradation_trend,
    generate_maintenance_insight,
)


class TestComputeHealthScore:
    def test_max_rul_gives_100(self):
        score = compute_health_score(125.0, max_rul=125.0)
        assert score == 100.0

    def test_zero_rul_gives_zero(self):
        score = compute_health_score(0.0, max_rul=125.0)
        assert score == 0.0

    def test_mid_rul(self):
        score = compute_health_score(62.5, max_rul=125.0)
        assert abs(score - 50.0) < 0.01

    def test_clipping_upper(self):
        score = compute_health_score(200.0, max_rul=125.0)
        assert score == 100.0

    def test_clipping_lower(self):
        score = compute_health_score(-10.0, max_rul=125.0)
        assert score == 0.0


class TestGetRiskCategory:
    def test_healthy(self):
        assert get_risk_category(85) == "Healthy"

    def test_watch(self):
        assert get_risk_category(50) == "Watch"

    def test_high_risk(self):
        assert get_risk_category(20) == "High Risk"

    def test_boundary_healthy_watch(self):
        assert get_risk_category(70) == "Healthy"

    def test_boundary_watch_high_risk(self):
        assert get_risk_category(40) == "Watch"

    def test_zero_is_high_risk(self):
        assert get_risk_category(0) == "High Risk"

    def test_100_is_healthy(self):
        assert get_risk_category(100) == "Healthy"


class TestDegradationTrend:
    def test_declining_trend(self):
        rul_preds = [100, 90, 80, 70, 60]
        trend = compute_degradation_trend(rul_preds, window=5)
        assert trend < 0

    def test_flat_trend(self):
        rul_preds = [50, 50, 50, 50, 50]
        trend = compute_degradation_trend(rul_preds, window=5)
        assert abs(trend) < 0.01

    def test_single_value(self):
        trend = compute_degradation_trend([50], window=5)
        assert trend == 0.0

    def test_empty_returns_zero(self):
        trend = compute_degradation_trend([], window=5)
        assert trend == 0.0


class TestMaintenanceInsight:
    def test_healthy_insight(self):
        insight = generate_maintenance_insight(85, "Healthy", 100, -0.3)
        assert "normal" in insight.lower() or "healthy" in insight.lower()

    def test_high_risk_insight(self):
        insight = generate_maintenance_insight(15, "High Risk", 10, -2.0)
        assert "maintenance" in insight.lower() or "degradation" in insight.lower()

    def test_includes_disclaimer(self):
        insight = generate_maintenance_insight(50, "Watch", 60, -1.0)
        assert "not a certified" in insight.lower() or "analytical" in insight.lower()
