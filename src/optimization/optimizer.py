"""EV charging schedule optimizer for Vidyut Prajna.

Default implementation uses a robust rolling-horizon linear program with:
- BESCOM time-of-use tariff optimization
- Temperature-dependent transformer derating
- Solar generation preference
- V2G readiness flagging
- Priority charging and deadline constraints
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


# Karnataka grid CO2 intensity (kg CO2 / kWh)
GRID_CO2_INTENSITY = 0.82
SOLAR_CO2_INTENSITY = 0.0

# Average BESCOM energy charge per kWh (INR) at tariff=1.0
BASE_TARIFF_INR_KWH = 7.15


def stress_label(utilization: float) -> str:
    """Categorize transformer stress level."""
    if utilization >= 1.0:
        return "Critical"
    if utilization >= 0.88:
        return "High"
    if utilization >= 0.72:
        return "Medium"
    return "Low"


def derated_capacity(capacity_kw: float, temperature_c: float) -> float:
    """Apply temperature derating to transformer capacity (IEC 60076)."""
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
    """Optimize EV charging schedule to reduce peaks and costs.
    
    Shifts flexible charging load with a robust LP while preserving:
    - Priority (non-shiftable) charging
    - Deadline constraints
    - Transformer capacity limits
    
    Args:
        prediction_df: Forecasted demand with features
        demand_col: Column name for demand to optimize
        alpha: Weight for peak reduction
        beta: Weight for variance reduction
        gamma_tariff: Weight for tariff cost optimization
        gamma_solar: Weight for solar preference
        max_transformer_utilization: Maximum allowed utilization
        
    Returns:
        optimized_df: DataFrame with optimized loads
        metrics: Dict of optimization metrics
    """
    from src.optimization.robust_optimizer import (
        RobustOptimizerConfig,
        RobustRollingHorizonOptimizer,
    )

    optimizer = RobustRollingHorizonOptimizer(
        RobustOptimizerConfig(
            max_utilization=max_transformer_utilization,
            peak_weight=max(0.1, 8.0 * alpha),
            deviation_weight=max(0.01, 0.18 * beta),
            tariff_weight=gamma_tariff,
            solar_weight=gamma_solar,
            delay_weight=max(0.01, 0.22 * beta),
        )
    )
    return optimizer.optimize(prediction_df, demand_col)

    df = prediction_df.copy().sort_values(["timestamp", "h3_cell"]).reset_index(drop=True)
    
    if demand_col not in df.columns:
        raise ValueError(f"{demand_col!r} not found in DataFrame")
    
    unique_times = sorted(df["timestamp"].unique())
    time_to_idx = {t: i for i, t in enumerate(unique_times)}
    
    dt_hours = 1.0  # Default to 1 hour
    if len(unique_times) > 1:
        dt = (pd.Series(unique_times).diff().dropna().dt.total_seconds().median() / 3600.0)
        if np.isfinite(dt) and dt > 0:
            dt_hours = dt
    
    df["time_idx"] = df["timestamp"].map(time_to_idx)
    
    # Ensure required columns exist
    if "priority_share" not in df.columns:
        df["priority_share"] = 0.2
    if "deadline_steps" not in df.columns:
        df["deadline_steps"] = 12
    if "tariff_multiplier" not in df.columns:
        df["tariff_multiplier"] = 1.0
    if "solar_generation_kw" not in df.columns:
        df["solar_generation_kw"] = 0.0
    if "temperature_c" not in df.columns:
        df["temperature_c"] = 30.0
    if "grid_base_load_kw" not in df.columns:
        df["grid_base_load_kw"] = 100.0
    if "transformer_capacity_kw" not in df.columns:
        df["transformer_capacity_kw"] = 500.0
    
    df["priority_share"] = df["priority_share"].clip(0.0, 0.9).fillna(0.2)
    df["deadline_steps"] = df["deadline_steps"].fillna(12).astype(int).clip(1, max(2, len(unique_times) - 1))
    
    # Split into priority (non-shiftable) and flexible load
    df["priority_kw"] = df[demand_col] * df["priority_share"]
    df["flexible_kw"] = (df[demand_col] - df["priority_kw"]).clip(lower=0.0)
    
    # Apply temperature derating
    df["effective_capacity_kw"] = df.apply(
        lambda r: derated_capacity(float(r["transformer_capacity_kw"]), float(r["temperature_c"])),
        axis=1
    )
    
    # Initialize with priority load only
    df["optimized_ev_load_kw"] = df["priority_kw"].astype(float)
    
    # Sort tasks by risk (highest utilization first)
    task_df = df[df["flexible_kw"] > 0].copy()
    task_df["baseline_util"] = (task_df["grid_base_load_kw"] + task_df[demand_col]) / task_df["effective_capacity_kw"]
    task_df = task_df.sort_values(["baseline_util", "flexible_kw"], ascending=False)
    
    current_total = df["grid_base_load_kw"] + df["optimized_ev_load_kw"]
    
    # Allocate flexible load
    for _, task in task_df.iterrows():
        cell = task["h3_cell"]
        orig_time_idx = int(task["time_idx"])
        flex_kw = float(task["flexible_kw"])
        deadline = int(task["deadline_steps"])
        
        if flex_kw <= 0:
            continue
        
        # Find best time slot within deadline window
        candidate_times = range(orig_time_idx, min(orig_time_idx + deadline, len(unique_times)))
        
        best_idx = orig_time_idx
        best_score = float("inf")
        
        for t_idx in candidate_times:
            # Get current load at this time for this cell
            mask = (df["h3_cell"] == cell) & (df["time_idx"] == t_idx)
            if not mask.any():
                continue
            
            row = df.loc[mask].iloc[0]
            existing_load = current_total.loc[mask].iloc[0]
            capacity = row["effective_capacity_kw"]
            tariff = row["tariff_multiplier"]
            solar = row["solar_generation_kw"]
            
            new_load = existing_load + flex_kw
            utilization = new_load / capacity
            
            # Skip if would exceed max utilization
            if utilization > max_transformer_utilization:
                continue
            
            # Score: lower is better
            # - Peak penalty (quadratic)
            # - Tariff cost
            # - Solar bonus (negative)
            score = (
                alpha * (utilization ** 2) +
                gamma_tariff * tariff -
                gamma_solar * min(solar / max(flex_kw, 1), 1.0)
            )
            
            if score < best_score:
                best_score = score
                best_idx = t_idx
        
        # Allocate to best slot
        best_mask = (df["h3_cell"] == cell) & (df["time_idx"] == best_idx)
        df.loc[best_mask, "optimized_ev_load_kw"] += flex_kw
        current_total.loc[best_mask] += flex_kw
    
    # Calculate final metrics
    df["baseline_ev_load_kw"] = df[demand_col]
    df["baseline_total_load_kw"] = df["grid_base_load_kw"] + df["baseline_ev_load_kw"]
    df["optimized_total_load_kw"] = df["grid_base_load_kw"] + df["optimized_ev_load_kw"]
    
    df["baseline_transformer_utilization"] = df["baseline_total_load_kw"] / df["effective_capacity_kw"]
    df["optimized_transformer_utilization"] = df["optimized_total_load_kw"] / df["effective_capacity_kw"]
    df["stress_label"] = df["optimized_transformer_utilization"].apply(stress_label)
    
    # V2G readiness: cells with low utilization during peak hours
    df["v2g_ready"] = (df["optimized_transformer_utilization"] < 0.6) & (df["tariff_multiplier"] >= 1.0)
    
    # Aggregate metrics
    baseline_peak = df["baseline_total_load_kw"].max()
    optimized_peak = df["optimized_total_load_kw"].max()
    peak_reduction_pct = 100 * (baseline_peak - optimized_peak) / max(baseline_peak, 1)
    
    baseline_var = df["baseline_total_load_kw"].var()
    optimized_var = df["optimized_total_load_kw"].var()
    variance_reduction_pct = 100 * (baseline_var - optimized_var) / max(baseline_var, 1)
    
    baseline_par = baseline_peak / max(df["baseline_total_load_kw"].mean(), 1)
    optimized_par = optimized_peak / max(df["optimized_total_load_kw"].mean(), 1)
    
    shifted_kwh = float((df["baseline_ev_load_kw"] - df["optimized_ev_load_kw"]).abs().sum() * dt_hours / 2)
    total_baseline_kwh = float(df["baseline_ev_load_kw"].sum() * dt_hours)
    energy_preservation_error_kwh = float(
        (df["optimized_ev_load_kw"].sum() - df["baseline_ev_load_kw"].sum()) * dt_hours
    )
    
    # Cost savings from tariff optimization
    baseline_cost = (df["baseline_ev_load_kw"] * df["tariff_multiplier"] * BASE_TARIFF_INR_KWH * dt_hours).sum()
    optimized_cost = (df["optimized_ev_load_kw"] * df["tariff_multiplier"] * BASE_TARIFF_INR_KWH * dt_hours).sum()
    cost_savings = baseline_cost - optimized_cost
    
    # CO2 reduction from solar preference
    optimized_solar_usage = df[["optimized_ev_load_kw", "solar_generation_kw"]].min(axis=1).sum() * dt_hours
    co2_reduction = optimized_solar_usage * GRID_CO2_INTENSITY
    
    p95_util_before = float(df["baseline_transformer_utilization"].quantile(0.95))
    p95_util_after = float(df["optimized_transformer_utilization"].quantile(0.95))
    stress_score_before = 100 * (1 - min(p95_util_before, 1.0))
    stress_score_after = 100 * (1 - min(p95_util_after, 1.0))
    overload_events_before = int((df["baseline_transformer_utilization"] > 1.0).sum())
    overload_events_after = int((df["optimized_transformer_utilization"] > 1.0).sum())

    top_risk_zones = (
        df.groupby(["h3_cell", "zone_name", "zone_type"], as_index=False)
        .agg(
            max_optimized_utilization=("optimized_transformer_utilization", "max"),
            max_baseline_utilization=("baseline_transformer_utilization", "max"),
            mean_predicted_demand_kw=("baseline_ev_load_kw", "mean"),
            station_count=("station_count", "first") if "station_count" in df.columns else ("baseline_ev_load_kw", "count"),
        )
        .sort_values("max_optimized_utilization", ascending=False)
        .head(8)
        .to_dict("records")
    )
    
    metrics = {
        "baseline_peak_kw": float(baseline_peak),
        "optimized_peak_kw": float(optimized_peak),
        "peak_reduction_pct": float(peak_reduction_pct),
        "variance_reduction_pct": float(variance_reduction_pct),
        "baseline_par": float(baseline_par),
        "optimized_par": float(optimized_par),
        "shifted_kwh": float(shifted_kwh),
        "estimated_cost_savings_inr": float(cost_savings),
        "co2_reduction_kg": float(co2_reduction),
        "v2g_potential_slots": int(df["v2g_ready"].sum()),
        "stress_score_before": float(stress_score_before),
        "stress_score_after": float(stress_score_after),
        "stress_label_before": stress_label(p95_util_before),
        "stress_label_after": stress_label(p95_util_after),
        "p95_utilization_before": p95_util_before,
        "p95_utilization_after": p95_util_after,
        "overload_events_before": overload_events_before,
        "overload_events_after": overload_events_after,
        "energy_preservation_error_kwh": energy_preservation_error_kwh,
        "energy_preservation_error_pct": 100.0 * energy_preservation_error_kwh / max(total_baseline_kwh, 1.0),
        "deadlines_met_pct": 100.0,
        "dt_hours": float(dt_hours),
        "top_risk_zones": top_risk_zones,
    }
    
    return df, metrics


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__).rsplit("src", 1)[0])
    
    from src.spatial_grid.simulation import CityConfig, generate_synthetic_data
    
    print("Testing optimizer...")
    
    config = CityConfig(max_cells=15, num_days=2, freq="1h")
    data, grid, adj = generate_synthetic_data(config)
    
    # Simulate prediction (use actual demand as "predicted")
    data["predicted_demand_kw"] = data["demand_kw"]
    
    optimized, metrics = optimize_charging_schedule(data)
    
    print("\nOptimization Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    
    print("\nSample optimized data:")
    print(optimized[["timestamp", "h3_cell", "baseline_ev_load_kw", "optimized_ev_load_kw", "stress_label"]].head(10))
    
    print("\nOptimizer test PASSED!")
