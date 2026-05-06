"""Charging infrastructure siting recommendations for Vidyut Prajna.

The scorer is intentionally transparent, but no longer simplistic: it combines
demand, growth, charger gap, graph centrality, neighboring corridor pressure,
traffic accessibility, transformer headroom, and budgeted portfolio coverage.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx


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


def _centrality_features(summary: pd.DataFrame, adjacency: Dict[str, List[str]] | None) -> pd.DataFrame:
    out = summary.copy()
    if not adjacency:
        out["degree_centrality"] = 0.0
        out["pagerank_centrality"] = 0.0
        out["betweenness_centrality"] = 0.0
        return out

    cells = set(out["h3_cell"].astype(str))
    graph = nx.Graph()
    graph.add_nodes_from(cells)
    for cell, neighbors in adjacency.items():
        if cell not in cells:
            continue
        for nbr in neighbors:
            if nbr in cells:
                graph.add_edge(cell, nbr)

    if graph.number_of_nodes() == 0:
        out["degree_centrality"] = 0.0
        out["pagerank_centrality"] = 0.0
        out["betweenness_centrality"] = 0.0
        return out

    degree = nx.degree_centrality(graph)
    pagerank = nx.pagerank(graph, alpha=0.85) if graph.number_of_edges() else {n: 0.0 for n in graph.nodes}
    betweenness = nx.betweenness_centrality(graph, normalized=True) if graph.number_of_edges() else {n: 0.0 for n in graph.nodes}
    out["degree_centrality"] = out["h3_cell"].map(degree).fillna(0.0)
    out["pagerank_centrality"] = out["h3_cell"].map(pagerank).fillna(0.0)
    out["betweenness_centrality"] = out["h3_cell"].map(betweenness).fillna(0.0)
    return out


def _portfolio_select(
    summary: pd.DataFrame,
    adjacency: Dict[str, List[str]] | None,
    top_n: int,
) -> pd.DataFrame:
    """Greedy maximum-coverage portfolio selection with redundancy control."""
    if len(summary) <= top_n:
        return summary.sort_values("siting_score", ascending=False).copy()

    demand_by_cell = dict(zip(summary["h3_cell"], summary["peak_predicted_demand_kw"]))
    score_by_idx = summary["siting_score"].to_dict()
    covered: set[str] = set()
    selected: List[int] = []
    remaining = set(summary.index)

    for _ in range(top_n):
        best_idx = None
        best_score = -float("inf")
        for idx in remaining:
            row = summary.loc[idx]
            cell = row["h3_cell"]
            neighborhood = {cell}
            if adjacency:
                neighborhood.update([n for n in adjacency.get(cell, []) if n in demand_by_cell])
            uncovered_capture = sum(demand_by_cell.get(c, 0.0) for c in neighborhood if c not in covered)
            redundancy = sum(1 for selected_idx in selected if adjacency and summary.loc[selected_idx, "h3_cell"] in adjacency.get(cell, []))
            feasibility_bonus = 5.0 if row.get("capacity_feasibility") == "Feasible on existing headroom" else -4.0
            portfolio_score = (
                float(score_by_idx[idx])
                + 0.20 * uncovered_capture
                - 6.0 * redundancy
                + feasibility_bonus
            )
            if portfolio_score > best_score:
                best_idx = idx
                best_score = portfolio_score
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
        selected_cell = summary.loc[best_idx, "h3_cell"]
        covered.add(selected_cell)
        if adjacency:
            covered.update([n for n in adjacency.get(selected_cell, []) if n in demand_by_cell])

    return summary.loc[selected].copy()


def _recommendation_reason(row: pd.Series) -> str:
    reasons = []
    if row["peak_predicted_demand_kw"] >= row["peak_predicted_demand_kw_p75"]:
        reasons.append("high forecast demand")
    if row["demand_growth_index"] >= row["growth_p75"]:
        reasons.append("fast EV adoption growth")
    if row["station_count"] <= row["station_count_p25"]:
        reasons.append("low existing charger count")
    if row.get("centrality_score", 0.0) >= 0.70:
        reasons.append("strong corridor centrality")
    if row.get("traffic_score", 0.0) >= 0.70:
        reasons.append("high-access traffic flow")
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

    agg_spec = {
        "mean_predicted_demand_kw": ("baseline_ev_load_kw", "mean"),
        "peak_predicted_demand_kw": ("baseline_ev_load_kw", "max"),
        "peak_optimized_total_kw": ("optimized_total_load_kw", "max"),
        "max_baseline_utilization": ("baseline_transformer_utilization", "max"),
        "max_optimized_utilization": ("optimized_transformer_utilization", "max"),
        "overload_hours_before": ("baseline_transformer_utilization", lambda s: int((s > 1.0).sum())),
        "overload_hours_after": ("optimized_transformer_utilization", lambda s: int((s > 1.0).sum())),
    }
    if "traffic_intensity" in optimized_df.columns:
        agg_spec["mean_traffic_intensity"] = ("traffic_intensity", "mean")
    if "congestion_index" in optimized_df.columns:
        agg_spec["mean_congestion_index"] = ("congestion_index", "mean")
    if "avg_speed_kmh" in optimized_df.columns:
        agg_spec["mean_avg_speed_kmh"] = ("avg_speed_kmh", "mean")

    summary = optimized_df.groupby(available_group_cols, as_index=False).agg(**agg_spec).reset_index(drop=True)
    summary = _centrality_features(summary, adjacency)

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
    summary["future_utilization_forecast"] = (
        summary["peak_optimized_total_kw"]
        + summary["projected_growth_kw"]
    ) / summary["transformer_capacity_kw"].clip(lower=1.0)

    summary["demand_score"] = _norm(summary["peak_predicted_demand_kw"])
    summary["growth_score"] = _norm(summary["projected_growth_kw"])
    summary["charger_gap_score"] = _norm(summary["charger_gap_index"])
    summary["neighbor_score"] = _norm(summary["neighbor_pressure_kw"])
    summary["centrality_score"] = _norm(
        0.45 * summary["pagerank_centrality"]
        + 0.35 * summary["degree_centrality"]
        + 0.20 * summary["betweenness_centrality"]
    )
    summary["stress_score"] = _norm(summary["future_utilization_forecast"].clip(upper=1.35))
    summary["headroom_score"] = _norm(summary["capacity_headroom_kw"])
    traffic_source = pd.Series(np.zeros(len(summary)), index=summary.index)
    if "mean_traffic_intensity" in summary.columns:
        traffic_source = traffic_source + 0.55 * summary["mean_traffic_intensity"].astype(float)
    if "mean_congestion_index" in summary.columns:
        traffic_source = traffic_source + 0.30 * _norm(summary["mean_congestion_index"])
    if "mean_avg_speed_kmh" in summary.columns:
        traffic_source = traffic_source + 0.15 * (1.0 - _norm(summary["mean_avg_speed_kmh"]))
    summary["traffic_score"] = _norm(traffic_source)
    summary["equity_score"] = _norm(
        summary["charger_gap_score"]
        + (summary["zone_type"].astype(str).isin(["residential", "logistics"]).astype(float) * 0.25)
    )

    summary["siting_score"] = 100.0 * (
        0.22 * summary["demand_score"]
        + 0.16 * summary["growth_score"]
        + 0.14 * summary["charger_gap_score"]
        + 0.12 * summary["neighbor_score"]
        + 0.10 * summary["centrality_score"]
        + 0.10 * summary["headroom_score"]
        + 0.08 * summary["stress_score"]
        + 0.05 * summary["traffic_score"]
        + 0.03 * summary["equity_score"]
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
    summary["utilization_forecast_after_station"] = (
        summary["peak_optimized_total_kw"] + summary["recommended_station_kw"]
    ) / summary["transformer_capacity_kw"].clip(lower=1.0)
    summary["roi_index"] = (
        summary["peak_predicted_demand_kw"]
        * (1.0 + summary["demand_growth_index"])
        * (1.0 + summary["centrality_score"])
    ) / (summary["recommended_station_kw"].clip(lower=1.0) + 1.0)

    summary["peak_predicted_demand_kw_p75"] = float(summary["peak_predicted_demand_kw"].quantile(0.75))
    summary["growth_p75"] = float(summary["demand_growth_index"].quantile(0.75))
    summary["station_count_p25"] = float(summary["station_count"].quantile(0.25))
    summary["reason"] = summary.apply(_recommendation_reason, axis=1)

    base_ranked = summary.sort_values(
        ["siting_score", "capacity_headroom_kw"],
        ascending=[False, False],
    ).reset_index(drop=True)

    top_n = max(1, min(int(top_n), len(base_ranked)))
    recommendations = _portfolio_select(base_ranked, adjacency, top_n).copy()
    remaining = base_ranked[~base_ranked["h3_cell"].isin(recommendations["h3_cell"])].copy()
    ranked = pd.concat([recommendations, remaining], ignore_index=True)
    ranked = ranked.sort_values(
        ["siting_score", "capacity_headroom_kw"],
        ascending=[False, False],
    ).reset_index(drop=True)
    recommendations = recommendations.sort_values(
        ["siting_score", "capacity_headroom_kw"],
        ascending=[False, False],
    ).reset_index(drop=True)
    recommendations.insert(0, "rank", np.arange(1, len(recommendations) + 1))
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))

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
        "recommended_mean_roi_index": float(recommendations["roi_index"].mean()),
        "uniform_mean_roi_index": float(uniform["roi_index"].mean()),
        "future_overload_risk_cells": int((summary["future_utilization_forecast"] > 1.0).sum()),
        "portfolio_method": "graph_aware_maximum_coverage",
        "top_corridor": str(recommendations["corridor_name"].mode().iloc[0]) if "corridor_name" in recommendations else "Demo corridor",
        "uniform_baseline_cells": uniform["h3_cell"].tolist(),
    }

    drop_helper_cols = [
        "peak_predicted_demand_kw_p75", "growth_p75", "station_count_p25",
    ]
    return recommendations.drop(columns=drop_helper_cols), summary_metrics
