"""
run_pipeline.py — Main AeroTwin pipeline: data → split → preprocess → features → train → validate → evaluate → explain.

This script executes the complete ML pipeline for RUL prediction using
NASA C-MAPSS FD001 turbofan engine degradation data.

Usage:
    python run_pipeline.py [--data-dir DATA_DIR]
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_cmapss, add_rul_to_train, add_rul_to_test, get_sensor_columns
from src.preprocessing import preprocess_pipeline, fit_scaler, apply_scaler
from src.features import engineer_features, get_feature_columns
from src.train import (
    get_naive_baseline,
    train_linear_regression,
    train_random_forest,
    train_xgboost,
    save_model,
    split_train_val,
)
from src.evaluate import compute_metrics, evaluate_baseline, full_evaluation, compare_models
from src.explainability import run_explainability
from src.health_score import compute_health_score, get_risk_category


def main(data_dir: str = None):
    """Execute the full AeroTwin pipeline."""

    # ── Paths ──────────────────────────────────────────
    if data_dir is None:
        data_dir = str(PROJECT_ROOT / "data")

    models_dir = str(PROJECT_ROOT / "models")
    reports_dir = str(PROJECT_ROOT / "reports" / "figures")
    processed_dir = str(PROJECT_ROOT / "data" / "processed")

    for d in [models_dir, reports_dir, processed_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  AeroTwin — Aircraft Engine RUL Prediction Pipeline")
    print("  Dataset: NASA C-MAPSS FD001")
    print("=" * 60)

    # ══════════════════════════════════════════════════
    # PHASE 1: DATA LOADING
    # ══════════════════════════════════════════════════
    print("\n📦 Phase 1: Loading NASA C-MAPSS FD001 dataset...")
    train_df_raw, test_df_raw, rul_df = load_cmapss(data_dir, "FD001")

    # Add RUL targets
    train_df = add_rul_to_train(train_df_raw)
    test_df = add_rul_to_test(test_df_raw, rul_df)

    print(f"  Train: {train_df.shape[0]:,} rows, {train_df['engine_id'].nunique()} engines")
    print(f"  Test:  {test_df.shape[0]:,} rows, {test_df['engine_id'].nunique()} engines")
    print(f"  Train RUL range: [{train_df['rul'].min()}, {train_df['rul'].max()}]")
    print(f"  Test RUL range (last cycles): [{rul_df['rul'].min()}, {rul_df['rul'].max()}]")

    # ══════════════════════════════════════════════════
    # PHASE 2: TRAIN/VALIDATION SPLIT
    # ══════════════════════════════════════════════════
    print("\n📊 Phase 2: Train/Validation Split (by engine)...")
    train_split, val_split = split_train_val(train_df, val_fraction=0.2)
    print(f"  Train: {train_split['engine_id'].nunique()} engines, Val: {val_split['engine_id'].nunique()} engines")

    # ══════════════════════════════════════════════════
    # PHASE 3: PREPROCESSING
    # ══════════════════════════════════════════════════
    print("\n🔧 Phase 3: Preprocessing (Leakage-free)...")
    
    # 3.1 Identify constants & cap RUL ONLY on training split
    train_proc, constant_sensors, feature_cols = preprocess_pipeline(
        train_split, rul_cap=125, constant_sensors=None
    )
    print(f"  Constant/near-constant sensors removed: {constant_sensors}")
    
    # Apply to val and test
    val_proc, _, _ = preprocess_pipeline(val_split, rul_cap=125, constant_sensors=constant_sensors)
    test_proc, _, _ = preprocess_pipeline(test_df, rul_cap=None, constant_sensors=constant_sensors) # DO NOT CAP TEST RUL
    
    # 3.2 Fit scaler on training split ONLY
    scaler = fit_scaler(train_proc, feature_cols)
    train_proc = apply_scaler(train_proc, scaler, feature_cols)
    val_proc = apply_scaler(val_proc, scaler, feature_cols)
    test_proc = apply_scaler(test_proc, scaler, feature_cols)
    
    # Save preprocessing artifacts
    joblib.dump(scaler, Path(models_dir) / "scaler.joblib")
    joblib.dump(constant_sensors, Path(models_dir) / "constant_sensors.joblib")
    print(f"  RUL capped at 125 cycles for Train/Val (focus on degradation region). Test RUL remains uncapped.")

    # ══════════════════════════════════════════════════
    # PHASE 4: FEATURE ENGINEERING
    # ══════════════════════════════════════════════════
    print("\n⚙️ Phase 4: Feature Engineering...")
    remaining_sensors = [c for c in feature_cols if c.startswith("sensor_")]
    
    train_feat = engineer_features(train_proc, sensor_cols=remaining_sensors)
    val_feat = engineer_features(val_proc, sensor_cols=remaining_sensors)
    test_feat = engineer_features(test_proc, sensor_cols=remaining_sensors)

    all_feature_cols = get_feature_columns(train_feat)
    print(f"  Total features: {len(all_feature_cols)}")

    # Save processed data
    train_feat.to_parquet(os.path.join(processed_dir, "train_processed.parquet"), index=False)
    test_feat.to_parquet(os.path.join(processed_dir, "test_processed.parquet"), index=False)
    joblib.dump(all_feature_cols, os.path.join(models_dir, "all_feature_cols.joblib"))

    X_train = train_feat[all_feature_cols]
    y_train = train_feat["rul"]
    X_val = val_feat[all_feature_cols]
    y_val = val_feat["rul"]
    X_test = test_feat[all_feature_cols]
    y_test = test_feat["rul"]

    # ══════════════════════════════════════════════════
    # PHASE 5: MODEL TRAINING
    # ══════════════════════════════════════════════════
    print("\n🤖 Phase 5: Model Training...")

    # Naive baseline
    baseline_pred = get_naive_baseline(train_feat)
    print(f"  Baseline (mean RUL): {baseline_pred:.1f}")

    models = {}
    
    print("\n  Training Linear Regression...")
    lr_model = train_linear_regression(X_train, y_train)
    models["Linear Regression"] = lr_model
    save_model(lr_model, models_dir, "linear_regression")

    print("  Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train, n_estimators=100)
    models["Random Forest"] = rf_model
    save_model(rf_model, models_dir, "random_forest")

    print("  Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    models["XGBoost"] = xgb_model
    save_model(xgb_model, models_dir, "xgboost")

    # ══════════════════════════════════════════════════
    # PHASE 6: MODEL SELECTION ON VALIDATION SET
    # ══════════════════════════════════════════════════
    print("\n📈 Phase 6: Model Selection (Validation Set)...")
    val_results = {}
    
    baseline_val = evaluate_baseline(y_val.values, baseline_pred)
    val_results["Baseline"] = baseline_val
    print(f"  Baseline  → MAE: {baseline_val['MAE']}, RMSE: {baseline_val['RMSE']}")

    best_mae = float("inf")
    best_model_name = None
    best_model = None

    for name, model in models.items():
        pred_val = model.predict(X_val)
        val_metric = compute_metrics(y_val.values, pred_val)
        val_results[name] = val_metric
        print(f"  {name:9} → MAE: {val_metric['MAE']}, RMSE: {val_metric['RMSE']}")
        
        if val_metric["MAE"] < best_mae:
            best_mae = val_metric["MAE"]
            best_model_name = name
            best_model = model

    print(f"\n  ⭐ Selected Best Model: {best_model_name} (Val MAE: {best_mae})")
    
    # Save the name of the best model for the dashboard to use
    with open(os.path.join(models_dir, "best_model.json"), "w") as f:
        json.dump({"best_model": best_model_name}, f)

    # ══════════════════════════════════════════════════
    # PHASE 7: FINAL TEST EVALUATION
    # ══════════════════════════════════════════════════
    print("\n🚀 Phase 7: Final Test Evaluation (Last Cycle Only)...")
    
    # Extract last cycle for each test engine (C-MAPSS Evaluation Protocol)
    last_cycle_test = test_feat.groupby("engine_id").last().reset_index()
    X_test_last = last_cycle_test[all_feature_cols]
    y_test_last = last_cycle_test["rul"] # Uncapped true RUL

    # Evaluate all models for comparison table, but we officially only rely on the best one
    test_results = {}
    
    baseline_test = evaluate_baseline(y_test_last.values, baseline_pred)
    test_results["Baseline"] = baseline_test

    for name, model in models.items():
        pred_test = model.predict(X_test_last)
        test_results[name] = compute_metrics(y_test_last.values, pred_test)

    results_table = compare_models(test_results)
    print("\n  Test Set Results (Last Cycle):")
    print(results_table.to_string())

    # Full evaluation for the BEST model
    best_pred_test = best_model.predict(X_test_last)
    best_eval = full_evaluation(
        y_test_last.values,
        best_pred_test,
        best_model_name,
        baseline_pred,
        save_dir=reports_dir,
    )

    # Save results
    results_dict = {
        "best_model": best_model_name,
        "validation": val_results,
        "test": test_results,
        "baseline_pred": float(baseline_pred),
    }
    with open(os.path.join(reports_dir, "metrics.json"), "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n  ✓ Metrics saved to {reports_dir}/metrics.json")

    # ══════════════════════════════════════════════════
    # PHASE 8: EXPLAINABILITY (SHAP)
    # ══════════════════════════════════════════════════
    print(f"\n🔍 Phase 8: SHAP Explainability ({best_model_name})...")
    
    if best_model_name in ["Random Forest", "XGBoost"]:
        shap_values, top_features = run_explainability(
            best_model, X_val, save_dir=reports_dir, max_samples=500
        )
    else:
        print("  SHAP skipped for linear models. Refer to model coefficients.")

    # ══════════════════════════════════════════════════
    # PHASE 9: HEALTH SCORE DEMO
    # ══════════════════════════════════════════════════
    print("\n💊 Phase 9: Health Score Demonstration...")
    
    # Save test predictions for dashboard (full trajectory for plotting)
    # We predict the full trajectory for the dashboard using the best model
    best_pred_all_test = best_model.predict(X_test)
    test_predictions = test_feat[["engine_id", "cycle", "rul"]].copy()
    test_predictions["predicted_rul"] = best_pred_all_test
    test_predictions["health_score"] = test_predictions["predicted_rul"].apply(
        lambda x: compute_health_score(x, max_rul=125)
    )
    test_predictions["risk_category"] = test_predictions["health_score"].apply(get_risk_category)
    test_predictions.to_parquet(
        os.path.join(processed_dir, "test_predictions.parquet"), index=False
    )
    print(f"  ✓ Test trajectory predictions saved for dashboard")

    print("\n" + "=" * 60)
    print("  ✅ AeroTwin Pipeline Complete!")
    print("=" * 60)
    print(f"\n  Dataset: NASA C-MAPSS FD001 (100 train / 100 test engines)")
    print(f"  Features: {len(all_feature_cols)} engineered features")
    print(f"  Selected Model: {best_model_name}")
    print(f"  Final Test MAE:  {test_results[best_model_name]['MAE']}")
    print(f"  Final Test RMSE: {test_results[best_model_name]['RMSE']}")
    print(f"  Baseline MAE: {baseline_test['MAE']}")
    print(f"  MAE improvement over baseline: {best_eval['improvement_mae_pct']}%")
    print(f"\n  Run dashboard: streamlit run app/streamlit_app.py\n")

    return results_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AeroTwin RUL Prediction Pipeline")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to directory containing C-MAPSS data files",
    )
    args = parser.parse_args()
    main(data_dir=args.data_dir)
