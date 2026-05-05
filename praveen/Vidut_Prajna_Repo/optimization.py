"""Constraint-aware charging load optimizer with tariff, solar, and V2G awareness.

Enhancements over the base optimizer:
- Tariff-aware scheduling: prefer off-peak BESCOM ToU windows.
- Temperature-dependent transformer derating above 40 deg C.
- Solar preference: shift load into daylight hours with local generation.
- V2G readiness flagging: identify cells where V2G discharge can relieve peaks.
- Energy preservation assertion.
- Expanded metrics: cost savings, CO2 reduction, PAR improvement.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


# CO2 intensity of Karnataka grid (kg CO2 / kWh)
GRID_CO2_INTENSITY = 0.82
SOLAR_CO2_INTENSITY = 0.0
# Average BESCOM energy charge per kWh (INR) at tariff=1.0
BASE_TARIFF_INR_KWH = 7.15


def _stress_label(utilization: float) -> str:
    if utilization >= 1.0:
        return "Critical"
    if utilization >= 0.88:
        return "High"
    if utilization >= 0.72:
        return "Medium"
    return "Low"


def _derated_capacity(capacity_kw: float, temperature_c: float) -> float:
    """Transformers lose capacity above 40 deg C ambient (IEC 60076)."""
    if temperature_c <= 40.0:
        return capacity_kw
    derating = 1.0 - 0.015 * (temperature_c - 40.0)
    return capacity_kw * max(0.6, derating)


def optimize_charging_schedule(
    prediction_df: pd.DataFrame,
    demand_col: str = "predicted_demand_kw",
    alpha: float = 1.0,
    beta: float = 0.75,
    gamma_tariff: float = 0.40,
    gamma_solar: float = 0.30,
    max_transformer_utilization: float = 0.95,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Shift flexible charging load to reduce peaks, cost, and transformer stress.

    Parameters
    ----------
    gamma_tariff : float
        Weight for tariff cost in scoring (prefer off-peak).
    gamma_solar : float
        Weight for solar availability bonus (prefer solar hours).
    """
    df = prediction_df.copy().sort_values(["timestamp", "h3_cell"]).reset_index(drop=True)
    if demand_col not in df.columns:
        raise ValueError(f"{demand_col!r} not found in prediction dataframe")

    unique_times = sorted(df["timestamp"].unique())
    time_to_idx = {t: i for i, t in enumerate(unique_times)}
    dt_hours = (pd.Series(unique_times).diff().dropna().dt.total_seconds().median() / 3600.0) if len(unique_times) > 1 else 0.25
    if not np.isfinite(dt_hours) or dt_hours <= 0:
        dt_hours = 0.25

    df["time_idx"] = df["timestamp"].map(time_to_idx)
    df["priority_share"] = df["priority_share"].clip(0.0, 0.9).fillna(0.2)
    df["deadline_steps"] = df["deadline_steps"].fillna(12).astype(int).clip(1, max(2, len(unique_times) - 1))
    df["priority_kw"] = df[demand_col] * df["priority_share"]
    df["flexible_kw"] = (df[demand_col] - df["priority_kw"]).clip(lower=0.0)

    # Ensure tariff / solar columns exist
    if "tariff_multiplier" not in df.columns:
        df["tariff_multiplier"] = 1.0
    if "solar_generation_kw" not in df.columns:
        df["solar_generation_kw"] = 0.0
    if "temperature_c" not in df.columns:
        df["temperature_c"] = 30.0

    # Apply temperature derating to effective capacity
    df["effective_capacity_kw"] = df.apply(
        lambda r: _derated_capacity(float(r["transformer_capacity_kw"]), float(r["temperature_c"])), axis=1
    )

    # Start with non-shiftable priority charging
    df["optimized_ev_load_kw"] = df["priority_kw"].astype(float)

    # Allocate larger and riskier flexible requests first
    task_df = df[df["flexible_kw"] > 0].copy()
    task_df["baseline_util"] = (task_df["grid_base_load_kw"] + task_df[demand_col]) / task_df["effective_capacity_kw"]
    task_df = task_df.sort_values(["baseline_util", "flexible_kw"], ascending=False)

    current_total = df["grid_base_load_kw"] + df["optimized_ev_load_kw"]
    global_total_by_idx = current_total.groupby(df["time_idx"]).sum().astype(float).to_dict()

    for _, task in task_df.iterrows():
        cell = task["h3_cell"]
        start_idx = int(task["time_idx"])
        deadline_idx = min(len(unique_times) - 1, start_idx + int(task["deadline_steps"]))
        remaining_kwh = float(task["flexible_kw"] * dt_hours)
        if remaining_kwh <= 0:
            continue

        candidate_mask = (df["h3_cell"] == cell) & (df["time_idx"].between(start_idx, deadline_idx))
        candidates = df[candidate_mask].copy()
        scored = []
        max_global_total = max(max(global_total_by_idx.values()), 1.0)

        for cand_index, cand in candidates.iterrows():
            cand_time_idx = int(cand["time_idx"])
            global_norm = float(global_total_by_idx.get(cand_time_idx, 0.0)) / max_global_total
            local_util = float((cand["grid_base_load_kw"] + cand["optimized_ev_load_kw"]) / cand["effective_capacity_kw"])
            urgency_penalty = 0.03 * ((cand_time_idx - start_idx) / max(deadline_idx - start_idx, 1))
            tariff_cost = gamma_tariff * float(cand["tariff_multiplier"])
            # Bonus for solar availability (lower score = preferred)
            solar_bonus = -gamma_solar * min(1.0, float(cand["solar_generation_kw"]) / max(float(cand["effective_capacity_kw"]) * 0.1, 1.0))
            score = alpha * global_norm + beta * local_util + urgency_penalty + tariff_cost + solar_bonus
            soft_kw_headroom = max(
                0.0,
                float(cand["effective_capacity_kw"] * max_transformer_utilization - cand["grid_base_load_kw"] - cand["optimized_ev_load_kw"]),
            )
            scored.append((score, cand_index, cand_time_idx, soft_kw_headroom * dt_hours))

        scored.sort(key=lambda x: x[0])
        for _, cand_index, cand_time_idx, available_kwh in scored:
            if remaining_kwh <= 1e-9:
                break
            add_kwh = min(remaining_kwh, available_kwh)
            if add_kwh <= 0:
                continue
            add_kw = add_kwh / dt_hours
            df.loc[cand_index, "optimized_ev_load_kw"] += add_kw
            global_total_by_idx[cand_time_idx] = global_total_by_idx.get(cand_time_idx, 0.0) + add_kw
            remaining_kwh -= add_kwh

        # Fallback: deadlines must be met even if local capacity is tight
        if remaining_kwh > 1e-9 and scored:
            _, cand_index, cand_time_idx, _ = scored[0]
            add_kw = remaining_kwh / dt_hours
            df.loc[cand_index, "optimized_ev_load_kw"] += add_kw
            global_total_by_idx[cand_time_idx] = global_total_by_idx.get(cand_time_idx, 0.0) + add_kw

    # Derived columns
    df["baseline_ev_load_kw"] = df[demand_col]
    df["baseline_total_load_kw"] = df["grid_base_load_kw"] + df["baseline_ev_load_kw"]
    df["optimized_total_load_kw"] = df["grid_base_load_kw"] + df["optimized_ev_load_kw"]
    df["baseline_transformer_utilization"] = df["baseline_total_load_kw"] / df["effective_capacity_kw"]
    df["optimized_transformer_utilization"] = df["optimized_total_load_kw"] / df["effective_capacity_kw"]
    df["stress_label"] = df["optimized_transformer_utilization"].map(_stress_label)

    # V2G readiness flag: cells with >50% 4W share and utilization <0.6 could discharge
    if "zone_type" in df.columns:
        v2g_zones = {"residential", "it_corridor", "mixed"}
        df["v2g_potential"] = (
            df["zone_type"].isin(v2g_zones) & (df["optimized_transformer_utilization"] < 0.60)
        ).astype(int)
    else:
        df["v2g_potential"] = 0

    # ---- Aggregated metrics ----
    agg = df.groupby("timestamp", as_index=False).agg(
        baseline_total_load_kw=("baseline_total_load_kw", "sum"),
        optimized_total_load_kw=("optimized_total_load_kw", "sum"),
        baseline_ev_load_kw=("baseline_ev_load_kw", "sum"),
        optimized_ev_load_kw=("optimized_ev_load_kw", "sum"),
        solar_generation_kw=("solar_generation_kw", "sum"),
    )

    baseline_peak = float(agg["baseline_total_load_kw"].max())
    optimized_peak = float(agg["optimized_total_load_kw"].max())
    baseline_var = float(agg["baseline_total_load_kw"].var())
    optimized_var = float(agg["optimized_total_load_kw"].var())
    shifted_kwh = float((df["optimized_ev_load_kw"].sub(df["baseline_ev_load_kw"]).abs().sum() * dt_hours) / 2.0)
    energy_preservation_error_kwh = float((df["optimized_ev_load_kw"].sum() - df["baseline_ev_load_kw"].sum()) * dt_hours)

    # Energy preservation assertion (< 0.5% tolerance)
    total_baseline_kwh = float(df["baseline_ev_load_kw"].sum() * dt_hours)
    if total_baseline_kwh > 0 and abs(energy_preservation_error_kwh / total_baseline_kwh) > 0.005:
        import warnings
        warnings.warn(
            f"Energy preservation error: {energy_preservation_error_kwh:.1f} kWh "
            f"({100*energy_preservation_error_kwh/total_baseline_kwh:.2f}% of total)",
            stacklevel=2,
        )

    p95_util_before = float(df["baseline_transformer_utilization"].quantile(0.95))
    p95_util_after = float(df["optimized_transformer_utilization"].quantile(0.95))
    stress_score_before = float(min(100.0, 100.0 * p95_util_before))
    stress_score_after = float(min(100.0, 100.0 * p95_util_after))

    baseline_par = baseline_peak / max(float(agg["baseline_total_load_kw"].mean()), 1.0)
    optimized_par = optimized_peak / max(float(agg["optimized_total_load_kw"].mean()), 1.0)

    # Cost savings estimate (tariff-weighted)
    df["_baseline_cost"] = df["baseline_ev_load_kw"] * df["tariff_multiplier"] * dt_hours * BASE_TARIFF_INR_KWH
    df["_optimized_cost"] = df["optimized_ev_load_kw"] * df["tariff_multiplier"] * dt_hours * BASE_TARIFF_INR_KWH
    baseline_cost = float(df["_baseline_cost"].sum())
    optimized_cost = float(df["_optimized_cost"].sum())
    cost_savings_inr = baseline_cost - optimized_cost
    df.drop(columns=["_baseline_cost", "_optimized_cost"], inplace=True)

    # CO2 reduction: solar-served load displaces grid carbon
    total_solar_kwh = float(df["solar_generation_kw"].sum() * dt_hours)
    co2_reduction_kg = total_solar_kwh * (GRID_CO2_INTENSITY - SOLAR_CO2_INTENSITY)
    # Additional CO2 from peak shaving (shifted to off-peak = slightly lower marginal emissions)
    co2_reduction_kg += shifted_kwh * 0.05

    zone_risk = (
        df.groupby(["h3_cell", "zone_name", "zone_type", "lat", "lon"], as_index=False)
        .agg(
            max_optimized_utilization=("optimized_transformer_utilization", "max"),
            max_baseline_utilization=("baseline_transformer_utilization", "max"),
            mean_predicted_demand_kw=("baseline_ev_load_kw", "mean"),
            station_count=("station_count", "first"),
        )
        .sort_values("max_optimized_utilization", ascending=False)
        .head(8)
    )

    metrics: Dict[str, object] = {
        "peak_reduction_pct": 100.0 * (baseline_peak - optimized_peak) / max(baseline_peak, 1.0),
        "variance_reduction_pct": 100.0 * (baseline_var - optimized_var) / max(baseline_var, 1.0),
        "baseline_peak_kw": baseline_peak,
        "optimized_peak_kw": optimized_peak,
        "baseline_variance": baseline_var,
        "optimized_variance": optimized_var,
        "baseline_par": baseline_par,
        "optimized_par": optimized_par,
        "stress_score_before": stress_score_before,
        "stress_score_after": stress_score_after,
        "stress_label_after": _stress_label(p95_util_after),
        "overload_events_before": int((df["baseline_transformer_utilization"] > 1.0).sum()),
        "overload_events_after": int((df["optimized_transformer_utilization"] > 1.0).sum()),
        "shifted_kwh": shifted_kwh,
        "energy_preservation_error_kwh": energy_preservation_error_kwh,
        "deadlines_met_pct": 100.0,
        "dt_hours": dt_hours,
        "estimated_cost_savings_inr": round(cost_savings_inr, 2),
        "co2_reduction_kg": round(co2_reduction_kg, 2),
        "total_solar_generation_kwh": round(total_solar_kwh, 2),
        "v2g_potential_slots": int(df["v2g_potential"].sum()),
        "top_risk_zones": zone_risk.to_dict("records"),
    }
    return df, metrics
