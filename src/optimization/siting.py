"""Charging infrastructure siting recommendations for Vidyut Prajna.

The scorer is intentionally transparent: it combines demand, growth, charger
gap, neighboring corridor pressure, and transformer headroom into planner-facing
rankings rather than pretending to be a black-box placement oracle.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def _norm(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    spread = float(values.max() - values.min())
    if spread <= 1e-9:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - float(values.min())) / spread


def _neighbor_pressure(
    summary: pd.DataFrame,
    adjacency: Dict[str, List[str]] | None,
    value_col: str,
) -> pd.Series:
    if not adjacency:
        return pd.Series(np.zeros(len(summary)), index=summary.index)

    demand_by_cell = dict(zip(summary["h3_cell"], summary[value_col]))
    pressures = []
    for cell in summary["h3_cell"]:
        nbr_values = [
            float(demand_by_cell[nbr])
            for nbr in adjacency.get(cell, [])
            if nbr in demand_by_cell
        ]
        pressures.append(float(np.mean(nbr_values)) if nbr_values else 0.0)
    return pd.Series(pressures, index=summary.index)


def _recommendation_reason(row: pd.Series) -> str:
    reasons = []
    if row["peak_predicted_demand_kw"] >= row["peak_predicted_demand_kw_p75"]:
        reasons.append("high forecast demand")
    if row["demand_growth_index"] >= row["growth_p75"]:
        reasons.append("fast EV adoption growth")
    if row["station_count"] <= row["station_count_p25"]:
        reasons.append("low existing charger count")
    if row["capacity_headroom_kw"] >= 22:
        reasons.append("usable transformer headroom")
    else:
        reasons.append("pair with DTR augmentation")
    return ", ".join(reasons[:3]).capitalize() + "."


def recommend_station_locations(
    optimized_df: pd.DataFrame,
    adjacency: Dict[str, List[str]] | None = None,
    top_n: int = 8,
    station_kw: float = 22.0,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Rank H3 cells for new charging-station planning.

    Args:
        optimized_df: Forecast and optimizer output.
        adjacency: H3 adjacency dict for corridor pressure.
        top_n: Number of recommended sites to return.
        station_kw: Reference station increment used for feasibility labels.

    Returns:
        recommendations: Top-ranked station sites with explainable score drivers.
        summary: Uniform-placement baseline comparison and aggregate metrics.
    """
    required = {
        "h3_cell", "baseline_ev_load_kw", "optimized_total_load_kw",
        "baseline_transformer_utilization", "optimized_transformer_utilization",
        "transformer_capacity_kw", "station_count", "charger_density_index",
        "demand_growth_index",
    }
    missing = required - set(optimized_df.columns)
    if missing:
        raise ValueError(f"Missing columns for siting recommendations: {sorted(missing)}")

    group_cols = [
        "h3_cell", "zone_name", "zone_type", "lat", "lon", "corridor_name",
        "station_count", "charger_density_index", "demand_growth_index",
        "transformer_capacity_kw",
    ]
    available_group_cols = [c for c in group_cols if c in optimized_df.columns]

    summary = (
        optimized_df.groupby(available_group_cols, as_index=False)
        .agg(
            mean_predicted_demand_kw=("baseline_ev_load_kw", "mean"),
            peak_predicted_demand_kw=("baseline_ev_load_kw", "max"),
            peak_optimized_total_kw=("optimized_total_load_kw", "max"),
            max_baseline_utilization=("baseline_transformer_utilization", "max"),
            max_optimized_utilization=("optimized_transformer_utilization", "max"),
            overload_hours_before=("baseline_transformer_utilization", lambda s: int((s > 1.0).sum())),
            overload_hours_after=("optimized_transformer_utilization", lambda s: int((s > 1.0).sum())),
        )
        .reset_index(drop=True)
    )

    summary["capacity_headroom_kw"] = (
        summary["transformer_capacity_kw"] * 0.95 - summary["peak_optimized_total_kw"]
    ).clip(lower=0.0)
    summary["projected_growth_kw"] = (
        summary["mean_predicted_demand_kw"] * summary["demand_growth_index"]
    )
    summary["neighbor_pressure_kw"] = _neighbor_pressure(
        summary, adjacency, "mean_predicted_demand_kw"
    )
    summary["charger_gap_index"] = (
        0.65 * (1.0 / (summary["station_count"].astype(float) + 1.0))
        + 0.35 * (1.0 / (summary["charger_density_index"].astype(float) + 1.0))
    )

    summary["demand_score"] = _norm(summary["peak_predicted_demand_kw"])
    summary["growth_score"] = _norm(summary["projected_growth_kw"])
    summary["charger_gap_score"] = _norm(summary["charger_gap_index"])
    summary["neighbor_score"] = _norm(summary["neighbor_pressure_kw"])
    summary["stress_score"] = _norm(summary["max_baseline_utilization"].clip(upper=1.25))
    summary["headroom_score"] = _norm(summary["capacity_headroom_kw"])

    summary["siting_score"] = 100.0 * (
        0.30 * summary["demand_score"]
        + 0.20 * summary["growth_score"]
        + 0.16 * summary["charger_gap_score"]
        + 0.14 * summary["neighbor_score"]
        + 0.12 * summary["stress_score"]
        + 0.08 * summary["headroom_score"]
    )
    summary["capacity_feasibility"] = np.where(
        summary["capacity_headroom_kw"] >= station_kw,
        "Feasible on existing headroom",
        "Needs transformer augmentation",
    )
    summary["recommended_station_kw"] = np.where(
        summary["capacity_headroom_kw"] >= station_kw,
        station_kw,
        np.maximum(0.0, summary["capacity_headroom_kw"]),
    )

    summary["peak_predicted_demand_kw_p75"] = float(summary["peak_predicted_demand_kw"].quantile(0.75))
    summary["growth_p75"] = float(summary["demand_growth_index"].quantile(0.75))
    summary["station_count_p25"] = float(summary["station_count"].quantile(0.25))
    summary["reason"] = summary.apply(_recommendation_reason, axis=1)

    ranked = summary.sort_values(
        ["siting_score", "capacity_headroom_kw"],
        ascending=[False, False],
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))

    top_n = max(1, min(int(top_n), len(ranked)))
    recommendations = ranked.head(top_n).copy()

    uniform_idx = np.linspace(0, len(ranked) - 1, top_n).round().astype(int)
    uniform = ranked.sort_values("h3_cell").iloc[uniform_idx].copy()

    recommended_capture = float(recommendations["peak_predicted_demand_kw"].sum())
    uniform_capture = float(uniform["peak_predicted_demand_kw"].sum())
    recommended_feasible = float((recommendations["capacity_headroom_kw"] >= station_kw).mean() * 100.0)
    uniform_feasible = float((uniform["capacity_headroom_kw"] >= station_kw).mean() * 100.0)

    summary_metrics: Dict[str, object] = {
        "station_budget": top_n,
        "station_kw": station_kw,
        "recommended_captured_peak_kw": recommended_capture,
        "uniform_captured_peak_kw": uniform_capture,
        "capture_improvement_pct": 100.0 * (recommended_capture - uniform_capture) / max(uniform_capture, 1.0),
        "recommended_feasible_pct": recommended_feasible,
        "uniform_feasible_pct": uniform_feasible,
        "top_corridor": str(recommendations["corridor_name"].mode().iloc[0]) if "corridor_name" in recommendations else "Demo corridor",
        "uniform_baseline_cells": uniform["h3_cell"].tolist(),
    }

    drop_helper_cols = [
        "peak_predicted_demand_kw_p75", "growth_p75", "station_count_p25",
    ]
    return recommendations.drop(columns=drop_helper_cols), summary_metrics

