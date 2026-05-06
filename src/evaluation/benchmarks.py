"""Benchmark harness for forecasting, scheduling, and siting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.intelligence.competition_forecaster import CompetitionForecaster
from src.intelligence.forecaster import STGCNForecaster
from src.optimization.lagrangian_optimizer import LagrangianOptimizer
from src.optimization.optimizer import optimize_charging_schedule
from src.optimization.robust_optimizer import derated_capacity, stress_label
from src.optimization.siting import recommend_station_locations
from src.spatial_grid.enhanced_simulation import generate_enhanced_synthetic_data
from src.spatial_grid.simulation import CityConfig


@dataclass
class ForecastMetricBundle:
    mae_kw: float
    rmse_kw: float
    mape_pct: float
    smape_pct: float
    peak_error_pct: float
    correlation: float
    prediction_interval_coverage_pct: float | None = None
    latency_ms: float | None = None


def compute_forecast_metrics(
    df: pd.DataFrame,
    actual_col: str = "actual_demand_kw",
    pred_col: str = "predicted_demand_kw",
    lower_col: str | None = None,
    upper_col: str | None = None,
) -> ForecastMetricBundle:
    actual = pd.to_numeric(df[actual_col], errors="coerce").fillna(0.0).astype(float)
    pred = pd.to_numeric(df[pred_col], errors="coerce").fillna(0.0).astype(float)
    error = actual - pred
    mae = float(error.abs().mean())
    rmse = float(np.sqrt(np.square(error).mean()))
    mape = float((error.abs() / actual.abs().clip(lower=1.0)).mean() * 100.0)
    smape = float((2.0 * error.abs() / (actual.abs() + pred.abs()).clip(lower=1.0)).mean() * 100.0)
    peak_error = 100.0 * float(abs(actual.max() - pred.max())) / max(float(actual.max()), 1.0)
    corr = float(actual.corr(pred)) if actual.std() > 0 and pred.std() > 0 else 0.0
    coverage = None
    if lower_col and upper_col and lower_col in df.columns and upper_col in df.columns:
        lower = pd.to_numeric(df[lower_col], errors="coerce").fillna(-np.inf)
        upper = pd.to_numeric(df[upper_col], errors="coerce").fillna(np.inf)
        coverage = float(((actual >= lower) & (actual <= upper)).mean() * 100.0)
    return ForecastMetricBundle(mae, rmse, mape, smape, peak_error, corr, coverage)


def _unmanaged_schedule(df: pd.DataFrame, demand_col: str = "predicted_demand_kw") -> Tuple[pd.DataFrame, Dict[str, object]]:
    out = df.copy()
    out["baseline_ev_load_kw"] = out[demand_col].astype(float)
    out["optimized_ev_load_kw"] = out["baseline_ev_load_kw"]
    out["effective_capacity_kw"] = out.apply(
        lambda r: derated_capacity(float(r["transformer_capacity_kw"]), float(r.get("temperature_c", 30.0))),
        axis=1,
    )
    out["baseline_total_load_kw"] = out["grid_base_load_kw"] + out["baseline_ev_load_kw"]
    out["optimized_total_load_kw"] = out["grid_base_load_kw"] + out["optimized_ev_load_kw"]
    out["baseline_transformer_utilization"] = out["baseline_total_load_kw"] / out["effective_capacity_kw"].clip(lower=1.0)
    out["optimized_transformer_utilization"] = out["optimized_total_load_kw"] / out["effective_capacity_kw"].clip(lower=1.0)
    out["stress_label"] = out["optimized_transformer_utilization"].apply(stress_label)
    peak = float(out["optimized_total_load_kw"].max())
    mean = float(out["optimized_total_load_kw"].mean())
    metrics = {
        "optimizer_type": "unmanaged",
        "optimized_peak_kw": peak,
        "optimized_par": peak / max(mean, 1.0),
        "overload_events_after": int((out["optimized_transformer_utilization"] > 1.0).sum()),
        "p95_utilization_after": float(out["optimized_transformer_utilization"].quantile(0.95)),
        "deadlines_met_pct": 100.0,
        "fairness_jain_index": 1.0,
    }
    return out, metrics


def benchmark_forecasters(
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    adjacency: Dict[str, List[str]],
    horizon_steps: int = 24,
    fast: bool = True,
) -> Tuple[Dict[str, ForecastMetricBundle], Dict[str, pd.DataFrame]]:
    metrics: Dict[str, ForecastMetricBundle] = {}
    predictions: Dict[str, pd.DataFrame] = {}

    stgcn = STGCNForecaster(
        seq_len=8 if fast else 12,
        epochs=1 if fast else 8,
        hidden_size=16 if fast else 48,
        num_blocks=1 if fast else 2,
    )
    stgcn.fit(train_df, adjacency)
    stgcn_pred = stgcn.forecast(train_df, future_df, adjacency, horizon_steps=horizon_steps)
    predictions["stgcn_guarded"] = stgcn_pred
    metrics["stgcn_selected"] = compute_forecast_metrics(stgcn_pred)
    if "stgcn_predicted_demand_kw" in stgcn_pred.columns:
        metrics["stgcn_raw"] = compute_forecast_metrics(stgcn_pred, pred_col="stgcn_predicted_demand_kw")

    gtft = CompetitionForecaster(
        seq_len=12 if fast else 24,
        forecast_horizon=horizon_steps,
        epochs=1 if fast else 10,
        hidden_size=32 if fast else 64,
        batch_size=8 if fast else 16,
    )
    gtft.fit(train_df, adjacency)
    gtft_pred = gtft.forecast(train_df, future_df, adjacency, horizon_steps=horizon_steps)
    predictions["graph_tft_quantile"] = gtft_pred
    metrics["graph_tft_selected"] = compute_forecast_metrics(
        gtft_pred,
        lower_col="p10_predicted_demand_kw",
        upper_col="p90_predicted_demand_kw",
    )
    metrics["graph_tft_raw_median"] = compute_forecast_metrics(
        gtft_pred,
        pred_col="gtft_predicted_demand_kw",
        lower_col="p10_predicted_demand_kw",
        upper_col="p90_predicted_demand_kw",
    )
    return metrics, predictions


def run_competition_benchmark(
    max_cells: int = 18,
    num_days: int = 5,
    horizon_steps: int = 12,
    fast: bool = True,
) -> Dict[str, object]:
    """Run a compact end-to-end benchmark on masked synthetic Bengaluru data."""
    config = CityConfig(max_cells=max_cells, num_days=num_days, freq="1h")
    data, grid, adjacency, _sessions, _dtrs = generate_enhanced_synthetic_data(
        config,
        include_ocpp=True,
        include_gig_fleet=True,
        apply_anonymization=True,
    )
    times = sorted(data["timestamp"].unique())
    train_times = times[:-horizon_steps]
    future_times = times[-horizon_steps:]
    train_df = data[data["timestamp"].isin(train_times)].copy()
    future_df = data[data["timestamp"].isin(future_times)].copy()

    forecast_metrics, predictions = benchmark_forecasters(
        train_df,
        future_df,
        adjacency,
        horizon_steps=horizon_steps,
        fast=fast,
    )
    chosen_pred = predictions["graph_tft_quantile"].copy()

    unmanaged_df, unmanaged_metrics = _unmanaged_schedule(chosen_pred)
    robust_df, robust_metrics = optimize_charging_schedule(chosen_pred)
    lagrangian = LagrangianOptimizer(max_iterations=3 if fast else 8)
    lagrangian_df, lagrangian_metrics = lagrangian.optimize(chosen_pred)

    recommendations, siting_summary = recommend_station_locations(
        robust_df,
        adjacency=adjacency,
        top_n=min(6, max(3, max_cells // 4)),
    )

    return {
        "config": {
            "max_cells": max_cells,
            "num_days": num_days,
            "horizon_steps": horizon_steps,
            "fast": fast,
        },
        "forecast_metrics": {name: vars(bundle) for name, bundle in forecast_metrics.items()},
        "optimization_metrics": {
            "unmanaged": unmanaged_metrics,
            "lagrangian_mcdm": lagrangian_metrics,
            "robust_lp_rolling_horizon": robust_metrics,
        },
        "siting_summary": siting_summary,
        "top_sites": recommendations.to_dict("records"),
        "frames": {
            "grid": grid,
            "prediction": chosen_pred,
            "unmanaged": unmanaged_df,
            "lagrangian": lagrangian_df,
            "robust": robust_df,
            "recommendations": recommendations,
        },
    }
