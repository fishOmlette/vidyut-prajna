"""Utility helpers for maps, metrics, and grounded LLM context."""

from __future__ import annotations

from typing import Dict, Iterable, List

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


def build_geojson(grid_df: pd.DataFrame) -> Dict[str, object]:
    features = []
    for _, row in grid_df.iterrows():
        cell = row["h3_cell"]
        features.append(
            {
                "type": "Feature",
                "id": cell,
                "properties": {
                    "h3_cell": cell,
                    "zone_name": row.get("zone_name", "Unknown"),
                    "zone_type": row.get("zone_type", "Unknown"),
                },
                "geometry": {"type": "Polygon", "coordinates": [h3_cell_to_polygon(cell)]},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def aggregate_load_timeseries(df: pd.DataFrame) -> pd.DataFrame:
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
        # Aggregate std via root-sum-of-squares (assuming independence)
        agg_dict["prediction_std_kw"] = ("prediction_std_kw", lambda s: float(np.sqrt((s ** 2).sum())))

    return (
        df.groupby("timestamp", as_index=False)
        .agg(**agg_dict)
        .sort_values("timestamp")
    )


def stress_label(utilization: float) -> str:
    if utilization >= 1.0:
        return "Critical"
    if utilization >= 0.88:
        return "High"
    if utilization >= 0.72:
        return "Medium"
    return "Low"


def _round_record(record: Dict[str, object], ndigits: int = 2) -> Dict[str, object]:
    rounded = {}
    for k, v in record.items():
        if isinstance(v, (float, np.floating)):
            rounded[k] = round(float(v), ndigits)
        elif isinstance(v, (int, np.integer)):
            rounded[k] = int(v)
        else:
            rounded[k] = v
    return rounded


def build_llm_context(metrics: Dict[str, object], optimized_df: pd.DataFrame, time_index: int) -> Dict[str, object]:
    """Compact, JSON-serializable context for the explanation LLM.

    The context intentionally contains only computed data.  The LLM layer must not
    infer new predictions or run new optimization.
    """
    times = sorted(optimized_df["timestamp"].unique())
    time_index = int(np.clip(time_index, 0, len(times) - 1))
    selected_time = pd.Timestamp(times[time_index])
    step = optimized_df[optimized_df["timestamp"] == selected_time].copy()

    demand_cols = [
        "h3_cell", "zone_name", "zone_type",
        "baseline_ev_load_kw", "optimized_ev_load_kw",
        "baseline_transformer_utilization", "optimized_transformer_utilization",
        "traffic_intensity", "rainfall_mm", "priority_share", "deadline_steps",
    ]
    # Include optional columns if available
    for optional in ("tariff_multiplier", "solar_generation_kw", "prediction_std_kw", "v2g_potential"):
        if optional in step.columns:
            demand_cols.append(optional)

    top_demand = (
        step.sort_values("baseline_ev_load_kw", ascending=False)
        .head(6)[demand_cols]
        .to_dict("records")
    )

    risk_cols = [
        "h3_cell", "zone_name", "zone_type",
        "baseline_ev_load_kw", "optimized_ev_load_kw",
        "optimized_transformer_utilization", "transformer_capacity_kw", "stress_label",
    ]
    for optional in ("temperature_c", "v2g_potential"):
        if optional in step.columns:
            risk_cols.append(optional)

    top_risk = (
        step.sort_values("optimized_transformer_utilization", ascending=False)
        .head(6)[risk_cols]
        .to_dict("records")
    )

    aggregate = aggregate_load_timeseries(optimized_df)
    selected_agg = aggregate[aggregate["timestamp"] == selected_time].iloc[0].to_dict()
    peak_rows = aggregate.sort_values("baseline_total_load_kw", ascending=False).head(4)

    selected_agg = {k: (v.isoformat() if isinstance(v, pd.Timestamp) else v) for k, v in selected_agg.items()}

    context = {
        "project": "Vidyut Prajna - AI-driven spatio-temporal intelligence for EV charging optimization",
        "selected_time": selected_time.isoformat(),
        "llm_role_constraint": "Explain computed dashboard outputs only. Do not perform prediction or optimization.",
        "metrics": _round_record({k: v for k, v in metrics.items() if k != "top_risk_zones"}, 2),
        "selected_time_aggregate": _round_record(selected_agg, 2),
        "top_predicted_demand_zones_at_selected_time": [_round_record(r, 3) for r in top_demand],
        "top_risk_zones_at_selected_time": [_round_record(r, 3) for r in top_risk],
        "highest_unmanaged_peak_times": [
            _round_record({"timestamp": pd.Timestamp(r["timestamp"]).isoformat(), **{k: v for k, v in r.items() if k != "timestamp"}}, 2)
            for r in peak_rows.to_dict("records")
        ],
    }
    return context
