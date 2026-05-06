"""Utility functions for the Vidyut Prajna dashboard."""

from __future__ import annotations

from typing import Dict, List

import h3
import numpy as np
import pandas as pd


def h3_cell_to_polygon(cell: str) -> List[List[float]]:
    """Return GeoJSON polygon coordinates [lon, lat] for one H3 cell."""
    boundary = h3.cell_to_boundary(cell)
    coords = [[float(lon), float(lat)] for lat, lon in boundary]
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def build_geojson(grid_df: pd.DataFrame) -> Dict:
    """Build GeoJSON FeatureCollection from grid DataFrame."""
    features = []
    for _, row in grid_df.iterrows():
        cell = row["h3_cell"]
        features.append({
            "type": "Feature",
            "id": cell,
            "properties": {
                "h3_cell": cell,
                "zone_name": row.get("zone_name", "Unknown"),
                "zone_type": row.get("zone_type", "Unknown"),
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [h3_cell_to_polygon(cell)]
            },
        })
    return {"type": "FeatureCollection", "features": features}


def aggregate_load_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate load data by timestamp across all cells."""
    agg_dict = {
        "baseline_total_load_kw": ("baseline_total_load_kw", "sum"),
        "optimized_total_load_kw": ("optimized_total_load_kw", "sum"),
        "baseline_ev_load_kw": ("baseline_ev_load_kw", "sum"),
        "optimized_ev_load_kw": ("optimized_ev_load_kw", "sum"),
        "safe_capacity_kw": ("transformer_capacity_kw", lambda s: float(0.95 * s.sum())),
    }
    
    if "solar_generation_kw" in df.columns:
        agg_dict["solar_generation_kw"] = ("solar_generation_kw", "sum")
    
    if "prediction_std_kw" in df.columns:
        agg_dict["prediction_std_kw"] = ("prediction_std_kw", lambda s: float(np.sqrt((s ** 2).sum())))

    if "actual_demand_kw" in df.columns:
        agg_dict["actual_ev_load_kw"] = ("actual_demand_kw", "sum")

    optional_sum_cols = {
        "stgcn_predicted_demand_kw": "stgcn_predicted_demand_kw",
        "seasonal_baseline_kw": "seasonal_baseline_kw",
        "persistence_baseline_kw": "persistence_baseline_kw",
        "blended_predicted_demand_kw": "blended_predicted_demand_kw",
    }
    for out_col, source_col in optional_sum_cols.items():
        if source_col in df.columns:
            agg_dict[out_col] = (source_col, "sum")
    
    return (
        df.groupby("timestamp", as_index=False)
        .agg(**agg_dict)
        .sort_values("timestamp")
    )


def _round_record(record: Dict[str, object], ndigits: int = 2) -> Dict[str, object]:
    rounded: Dict[str, object] = {}
    for key, value in record.items():
        if isinstance(value, pd.Timestamp):
            rounded[key] = value.isoformat()
        elif isinstance(value, (float, np.floating)):
            rounded[key] = round(float(value), ndigits)
        elif isinstance(value, (int, np.integer)):
            rounded[key] = int(value)
        else:
            rounded[key] = value
    return rounded


def build_llm_context(
    metrics: Dict[str, object],
    optimized_df: pd.DataFrame,
    time_index: int,
    recommendations: pd.DataFrame | None = None,
    siting_summary: Dict[str, object] | None = None,
) -> Dict[str, object]:
    """Build a compact, JSON-serializable context for grounded explanations."""
    times = sorted(optimized_df["timestamp"].unique())
    time_index = int(np.clip(time_index, 0, len(times) - 1))
    selected_time = pd.Timestamp(times[time_index])
    step = optimized_df[optimized_df["timestamp"] == selected_time].copy()

    demand_cols = [
        "h3_cell", "zone_name", "zone_type", "baseline_ev_load_kw",
        "optimized_ev_load_kw", "baseline_transformer_utilization",
        "optimized_transformer_utilization", "traffic_intensity",
        "rainfall_mm", "tariff_multiplier", "solar_generation_kw",
        "prediction_std_kw", "stress_label",
    ]
    demand_cols = [c for c in demand_cols if c in step.columns]

    top_demand = (
        step.sort_values("baseline_ev_load_kw", ascending=False)
        .head(6)[demand_cols]
        .to_dict("records")
    )
    top_risk = (
        step.sort_values("optimized_transformer_utilization", ascending=False)
        .head(6)[demand_cols]
        .to_dict("records")
    )

    aggregate = aggregate_load_timeseries(optimized_df)
    selected_agg = aggregate[aggregate["timestamp"] == selected_time].iloc[0].to_dict()
    peak_rows = aggregate.sort_values("baseline_total_load_kw", ascending=False).head(4)

    rec_records: List[Dict[str, object]] = []
    if recommendations is not None and not recommendations.empty:
        rec_cols = [
            "rank", "zone_name", "zone_type", "siting_score",
            "peak_predicted_demand_kw", "capacity_headroom_kw",
            "capacity_feasibility", "reason",
        ]
        rec_cols = [c for c in rec_cols if c in recommendations.columns]
        rec_records = recommendations.head(5)[rec_cols].to_dict("records")

    clean_metrics = {
        key: value
        for key, value in metrics.items()
        if key not in {"top_risk_zones"}
    }
    return {
        "project": "Vidyut Prajna - EV charging demand, scheduling, and infrastructure planning",
        "data_policy": "Synthetic or aggregated computed values only. No control commands and no sensitive raw data.",
        "selected_time": selected_time.isoformat(),
        "metrics": _round_record(clean_metrics, 2),
        "selected_time_aggregate": _round_record(selected_agg, 2),
        "top_predicted_demand_zones_at_selected_time": [_round_record(r, 3) for r in top_demand],
        "top_risk_zones_at_selected_time": [_round_record(r, 3) for r in top_risk],
        "highest_unmanaged_peak_times": [
            _round_record(row, 2)
            for row in peak_rows.to_dict("records")
        ],
        "station_recommendations": [_round_record(r, 2) for r in rec_records],
        "siting_summary": _round_record(siting_summary or {}, 2),
    }


def format_kw(v: float) -> str:
    """Format kilowatts for display."""
    return f"{v:,.0f} kW"


def format_pct(v: float) -> str:
    """Format percentage for display."""
    return f"{v:.1f}%"


def format_inr(v: float) -> str:
    """Format rupees for display."""
    return f"₹{v:,.0f}"
