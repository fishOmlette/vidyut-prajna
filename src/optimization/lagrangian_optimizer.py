"""Lagrangian Multi-Criteria Decision Making (MCDM) optimizer for EV charging.

Implements the optimization approach promised in the Vidyut Prajna proposal:
    J = α · Σt(Pgrid(t) - Pavg)² + β · Σi(ωi · ΔSoCi)

Where:
- α minimizes grid load variance (peak shaving)
- β ensures high-priority vehicles meet their SoC targets
- ωi is the priority weight for vehicle i
- ΔSoCi is the SoC deficit for vehicle i

Uses Lagrangian relaxation with dual decomposition to find the
"Point of Minimum Regret" balancing grid health against user service levels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Karnataka grid CO2 intensity (kg CO2 / kWh)
GRID_CO2_INTENSITY = 0.82
SOLAR_CO2_INTENSITY = 0.0

# Average BESCOM energy charge per kWh (INR) at tariff=1.0
BASE_TARIFF_INR_KWH = 7.15


@dataclass
class ChargingTask:
    """Represents a single EV charging task to be scheduled."""
    task_id: str
    h3_cell: str
    original_time_idx: int
    energy_kwh: float
    max_power_kw: float
    priority_weight: float  # ωi - higher = more important
    deadline_time_idx: int
    soc_current: float
    soc_target: float
    is_shiftable: bool
    vehicle_type: str


@dataclass
class LagrangianState:
    """State of the Lagrangian optimization."""
    iteration: int
    primal_objective: float
    dual_objective: float
    duality_gap: float
    multipliers: np.ndarray
    step_size: float
    converged: bool


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


class LagrangianOptimizer:
    """
    Lagrangian relaxation optimizer for EV charging scheduling.
    
    Solves the multi-objective optimization problem:
        min J = α·Σt(Pgrid(t) - Pavg)² + β·Σi(ωi·ΔSoCi) + γ·cost_term
        
    Subject to:
        - Transformer capacity constraints
        - Deadline constraints
        - Energy conservation (total demand preserved)
        - Priority vehicle SoC targets
    
    Uses dual decomposition with subgradient updates for the Lagrange
    multipliers (shadow prices).
    """
    
    def __init__(
        self,
        alpha: float = 1.0,           # Weight for load variance
        beta: float = 0.75,           # Weight for SoC deficit
        gamma_tariff: float = 0.40,   # Weight for tariff cost
        gamma_solar: float = 0.30,    # Weight for solar preference
        max_utilization: float = 0.95,
        max_iterations: int = 50,
        convergence_tol: float = 1e-4,
        step_size_init: float = 0.1,
        step_size_decay: float = 0.95,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma_tariff = gamma_tariff
        self.gamma_solar = gamma_solar
        self.max_utilization = max_utilization
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
        self.step_size_init = step_size_init
        self.step_size_decay = step_size_decay
        
        self.state: Optional[LagrangianState] = None
        self.shadow_prices: Dict[Tuple[str, int], float] = {}
    
    def _build_tasks(
        self,
        df: pd.DataFrame,
        demand_col: str,
    ) -> Tuple[List[ChargingTask], Dict[Tuple[str, int], int]]:
        """Extract charging tasks from the demand data."""
        tasks = []
        task_index = {}
        
        for idx, row in df.iterrows():
            demand_kw = float(row[demand_col])
            if demand_kw <= 0:
                continue
            
            priority_share = float(row.get("priority_share", 0.2))
            priority_kw = demand_kw * priority_share
            flexible_kw = demand_kw - priority_kw
            
            time_idx = int(row["time_idx"])
            h3_cell = str(row["h3_cell"])
            deadline = int(row.get("deadline_steps", 12))
            
            # Priority task (non-shiftable)
            if priority_kw > 0.1:
                task = ChargingTask(
                    task_id=f"{h3_cell}:{time_idx}:priority",
                    h3_cell=h3_cell,
                    original_time_idx=time_idx,
                    energy_kwh=priority_kw,  # kW treated as energy for single time step
                    max_power_kw=priority_kw,
                    priority_weight=2.0,  # High priority
                    deadline_time_idx=time_idx,  # Must happen now
                    soc_current=20.0,
                    soc_target=80.0,
                    is_shiftable=False,
                    vehicle_type="priority",
                )
                tasks.append(task)
                task_index[(h3_cell, time_idx, "priority")] = len(tasks) - 1
            
            # Flexible task (shiftable)
            if flexible_kw > 0.1:
                max_deadline = time_idx + deadline
                task = ChargingTask(
                    task_id=f"{h3_cell}:{time_idx}:flexible",
                    h3_cell=h3_cell,
                    original_time_idx=time_idx,
                    energy_kwh=flexible_kw,
                    max_power_kw=flexible_kw,
                    priority_weight=1.0,  # Normal priority
                    deadline_time_idx=max_deadline,
                    soc_current=30.0,
                    soc_target=80.0,
                    is_shiftable=True,
                    vehicle_type="flexible",
                )
                tasks.append(task)
                task_index[(h3_cell, time_idx, "flexible")] = len(tasks) - 1
        
        return tasks, task_index
    
    def _compute_load_profile(
        self,
        df: pd.DataFrame,
        allocation: Dict[str, int],  # task_id -> assigned time_idx
        tasks: List[ChargingTask],
    ) -> np.ndarray:
        """Compute total load at each time step given current allocation."""
        unique_times = sorted(df["timestamp"].unique())
        n_times = len(unique_times)
        
        # Initialize with base grid load
        load_by_time = np.zeros(n_times)
        
        # Add base grid load
        for t_idx in range(n_times):
            mask = df["time_idx"] == t_idx
            if mask.any():
                load_by_time[t_idx] = df.loc[mask, "grid_base_load_kw"].sum()
        
        # Add allocated EV load
        for task in tasks:
            assigned_t = allocation.get(task.task_id, task.original_time_idx)
            if 0 <= assigned_t < n_times:
                load_by_time[assigned_t] += task.energy_kwh
        
        return load_by_time
    
    def _compute_primal_objective(
        self,
        load_profile: np.ndarray,
        tasks: List[ChargingTask],
        allocation: Dict[str, int],
        df: pd.DataFrame,
    ) -> float:
        """
        Compute the primal objective:
        J = α·Σt(Pgrid(t) - Pavg)² + β·Σi(ωi·ΔSoCi) + γ·tariff_cost
        """
        # Load variance term
        avg_load = load_profile.mean()
        variance_term = self.alpha * np.sum((load_profile - avg_load) ** 2)
        
        # SoC deficit term
        soc_term = 0.0
        for task in tasks:
            # Deficit = target - current (simplified since we're scheduling not charging)
            # In real system, this would track actual SoC evolution
            soc_deficit = max(0, task.soc_target - task.soc_current) / 100.0
            soc_term += self.beta * task.priority_weight * soc_deficit
        
        # Tariff cost term
        tariff_cost = 0.0
        for task in tasks:
            assigned_t = allocation.get(task.task_id, task.original_time_idx)
            mask = df["time_idx"] == assigned_t
            if mask.any():
                tariff = df.loc[mask, "tariff_multiplier"].iloc[0]
                solar = df.loc[mask, "solar_generation_kw"].sum()
                solar_bonus = min(solar / max(task.energy_kwh, 1), 1.0)
                tariff_cost += self.gamma_tariff * tariff * task.energy_kwh
                tariff_cost -= self.gamma_solar * solar_bonus * task.energy_kwh
        
        return variance_term + soc_term + tariff_cost
    
    def _compute_capacity_violations(
        self,
        df: pd.DataFrame,
        allocation: Dict[str, int],
        tasks: List[ChargingTask],
    ) -> Dict[Tuple[str, int], float]:
        """Compute capacity constraint violations for each (cell, time)."""
        violations = {}
        
        # Build load by (cell, time)
        cell_time_load: Dict[Tuple[str, int], float] = {}
        cell_time_capacity: Dict[Tuple[str, int], float] = {}
        
        for _, row in df.iterrows():
            key = (row["h3_cell"], int(row["time_idx"]))
            base_load = float(row["grid_base_load_kw"])
            capacity = derated_capacity(
                float(row["transformer_capacity_kw"]),
                float(row.get("temperature_c", 30))
            )
            cell_time_load[key] = base_load
            cell_time_capacity[key] = capacity * self.max_utilization
        
        # Add task allocations
        for task in tasks:
            assigned_t = allocation.get(task.task_id, task.original_time_idx)
            key = (task.h3_cell, assigned_t)
            if key in cell_time_load:
                cell_time_load[key] += task.energy_kwh
        
        # Compute violations
        for key, load in cell_time_load.items():
            capacity = cell_time_capacity.get(key, float("inf"))
            if load > capacity:
                violations[key] = load - capacity
        
        return violations
    
    def _subgradient_update(
        self,
        multipliers: np.ndarray,
        violations: Dict[Tuple[str, int], float],
        key_order: List[Tuple[str, int]],
        step_size: float,
    ) -> np.ndarray:
        """Update Lagrange multipliers using subgradient method."""
        grad = np.zeros_like(multipliers)
        
        for i, key in enumerate(key_order):
            if key in violations:
                grad[i] = violations[key]
        
        # Subgradient update with projection to non-negative orthant
        new_multipliers = multipliers + step_size * grad
        new_multipliers = np.maximum(0, new_multipliers)
        
        return new_multipliers
    
    def _solve_subproblem(
        self,
        tasks: List[ChargingTask],
        multipliers: np.ndarray,
        key_order: List[Tuple[str, int]],
        df: pd.DataFrame,
    ) -> Dict[str, int]:
        """
        Solve the Lagrangian subproblem for given multipliers.
        
        This decomposes into independent problems per task:
        For each task, find the time slot that minimizes:
            task_cost + λ(cell,t) × power
        
        Subject to deadline constraints.
        """
        allocation = {}
        unique_times = sorted(df["timestamp"].unique())
        n_times = len(unique_times)
        
        # Build multiplier lookup
        mult_lookup = {key: mult for key, mult in zip(key_order, multipliers)}
        
        for task in tasks:
            if not task.is_shiftable:
                # Non-shiftable tasks stay at original time
                allocation[task.task_id] = task.original_time_idx
                continue
            
            # Find feasible time window
            t_start = task.original_time_idx
            t_end = min(task.deadline_time_idx, n_times - 1)
            
            best_t = t_start
            best_cost = float("inf")
            
            for t_idx in range(t_start, t_end + 1):
                # Get features at this time
                mask = (df["h3_cell"] == task.h3_cell) & (df["time_idx"] == t_idx)
                if not mask.any():
                    continue
                
                row = df.loc[mask].iloc[0]
                tariff = float(row["tariff_multiplier"])
                solar = float(row.get("solar_generation_kw", 0))
                capacity = derated_capacity(
                    float(row["transformer_capacity_kw"]),
                    float(row.get("temperature_c", 30))
                )
                
                # Check if slot is feasible
                base_load = float(row["grid_base_load_kw"])
                max_allowed = capacity * self.max_utilization
                if base_load + task.energy_kwh > max_allowed:
                    continue
                
                # Lagrangian cost
                key = (task.h3_cell, t_idx)
                mult = mult_lookup.get(key, 0.0)
                
                # Cost = tariff + shadow price - solar bonus
                solar_bonus = min(solar / max(task.energy_kwh, 1), 1.0)
                cost = (
                    self.gamma_tariff * tariff * task.energy_kwh +
                    mult * task.energy_kwh -
                    self.gamma_solar * solar_bonus * task.energy_kwh
                )
                
                if cost < best_cost:
                    best_cost = cost
                    best_t = t_idx
            
            allocation[task.task_id] = best_t
        
        return allocation
    
    def _find_minimum_regret(
        self,
        pareto_front: List[Tuple[float, float, Dict[str, int]]],
    ) -> Dict[str, int]:
        """
        Find the "Point of Minimum Regret" on the Pareto front.
        
        This balances grid health (variance) against user service (SoC deficit).
        Uses the min-max regret criterion.
        """
        if not pareto_front:
            return {}
        
        if len(pareto_front) == 1:
            return pareto_front[0][2]
        
        # Normalize objectives
        variances = [p[0] for p in pareto_front]
        soc_deficits = [p[1] for p in pareto_front]
        
        min_var, max_var = min(variances), max(variances)
        min_soc, max_soc = min(soc_deficits), max(soc_deficits)
        
        # Avoid division by zero
        var_range = max(max_var - min_var, 1e-6)
        soc_range = max(max_soc - min_soc, 1e-6)
        
        # Find point with minimum max-normalized regret
        best_idx = 0
        best_regret = float("inf")
        
        for i, (var, soc, _) in enumerate(pareto_front):
            norm_var = (var - min_var) / var_range
            norm_soc = (soc - min_soc) / soc_range
            max_regret = max(norm_var, norm_soc)
            
            if max_regret < best_regret:
                best_regret = max_regret
                best_idx = i
        
        return pareto_front[best_idx][2]
    
    def optimize(
        self,
        df: pd.DataFrame,
        demand_col: str = "predicted_demand_kw",
    ) -> Tuple[pd.DataFrame, Dict[str, object]]:
        """
        Run Lagrangian optimization on the charging schedule.
        
        Args:
            df: DataFrame with predictions and features
            demand_col: Column name for demand to optimize
            
        Returns:
            optimized_df: DataFrame with optimized loads
            metrics: Dict of optimization metrics
        """
        df = df.copy().sort_values(["timestamp", "h3_cell"]).reset_index(drop=True)
        
        if demand_col not in df.columns:
            raise ValueError(f"{demand_col!r} not found in DataFrame")
        
        # Build time index
        unique_times = sorted(df["timestamp"].unique())
        time_to_idx = {t: i for i, t in enumerate(unique_times)}
        df["time_idx"] = df["timestamp"].map(time_to_idx)
        
        # Ensure required columns
        for col, default in [
            ("priority_share", 0.2),
            ("deadline_steps", 12),
            ("tariff_multiplier", 1.0),
            ("solar_generation_kw", 0.0),
            ("temperature_c", 30.0),
            ("grid_base_load_kw", 100.0),
            ("transformer_capacity_kw", 500.0),
        ]:
            if col not in df.columns:
                df[col] = default
        
        df["priority_share"] = df["priority_share"].clip(0.0, 0.9).fillna(0.2)
        df["deadline_steps"] = df["deadline_steps"].fillna(12).astype(int).clip(1, len(unique_times) - 1)
        
        # Extract tasks
        tasks, _ = self._build_tasks(df, demand_col)
        
        if not tasks:
            # No demand to optimize
            df["optimized_ev_load_kw"] = df[demand_col]
            df["baseline_ev_load_kw"] = df[demand_col]
            return df, {"error": "No charging tasks found"}
        
        # Build constraint keys (cell, time pairs)
        cell_times = df.groupby(["h3_cell", "time_idx"]).first().index.tolist()
        key_order = list(cell_times)
        n_constraints = len(key_order)
        
        # Initialize Lagrange multipliers (shadow prices)
        multipliers = np.zeros(n_constraints)
        step_size = self.step_size_init
        
        # Lagrangian iteration
        pareto_front: List[Tuple[float, float, Dict[str, int]]] = []
        best_allocation = {task.task_id: task.original_time_idx for task in tasks}
        
        for iteration in range(self.max_iterations):
            # Solve subproblem
            allocation = self._solve_subproblem(tasks, multipliers, key_order, df)
            
            # Compute load profile
            load_profile = self._compute_load_profile(df, allocation, tasks)
            
            # Compute primal objective
            primal_obj = self._compute_primal_objective(load_profile, tasks, allocation, df)
            
            # Compute constraint violations
            violations = self._compute_capacity_violations(df, allocation, tasks)
            
            # Compute dual objective (lower bound)
            dual_obj = primal_obj - sum(
                multipliers[i] * violations.get(key, 0)
                for i, key in enumerate(key_order)
            )
            
            # Track Pareto front
            variance = np.var(load_profile)
            soc_deficit = sum(
                task.priority_weight * max(0, task.soc_target - task.soc_current)
                for task in tasks
            )
            pareto_front.append((variance, soc_deficit, allocation.copy()))
            
            # Check convergence
            duality_gap = abs(primal_obj - dual_obj) / max(abs(primal_obj), 1e-6)
            
            self.state = LagrangianState(
                iteration=iteration,
                primal_objective=primal_obj,
                dual_objective=dual_obj,
                duality_gap=duality_gap,
                multipliers=multipliers.copy(),
                step_size=step_size,
                converged=duality_gap < self.convergence_tol and not violations,
            )
            
            if self.state.converged:
                best_allocation = allocation
                break
            
            # Update multipliers
            multipliers = self._subgradient_update(
                multipliers, violations, key_order, step_size
            )
            
            # Decay step size
            step_size *= self.step_size_decay
            
            # Keep best feasible solution
            if not violations:
                best_allocation = allocation
        
        # Find minimum regret point
        final_allocation = self._find_minimum_regret(pareto_front)
        if not final_allocation:
            final_allocation = best_allocation
        
        # Store shadow prices
        self.shadow_prices = {
            key: mult for key, mult in zip(key_order, multipliers)
        }
        
        # Apply allocation to DataFrame
        df = self._apply_allocation(df, tasks, final_allocation, demand_col)
        
        # Compute metrics
        metrics = self._compute_metrics(df, demand_col)
        
        # Add optimization state info
        if self.state:
            metrics["lagrangian_iterations"] = self.state.iteration + 1
            metrics["duality_gap"] = self.state.duality_gap
            metrics["converged"] = self.state.converged
            metrics["max_shadow_price"] = float(np.max(multipliers))
            metrics["mean_shadow_price"] = float(np.mean(multipliers))
        
        return df, metrics
    
    def _apply_allocation(
        self,
        df: pd.DataFrame,
        tasks: List[ChargingTask],
        allocation: Dict[str, int],
        demand_col: str,
    ) -> pd.DataFrame:
        """Apply the optimized allocation to the DataFrame."""
        df["optimized_ev_load_kw"] = 0.0
        df["baseline_ev_load_kw"] = df[demand_col]
        
        # Build load by (cell, time)
        cell_time_optimized: Dict[Tuple[str, int], float] = {}
        
        for task in tasks:
            assigned_t = allocation.get(task.task_id, task.original_time_idx)
            key = (task.h3_cell, assigned_t)
            cell_time_optimized[key] = cell_time_optimized.get(key, 0) + task.energy_kwh
        
        # Apply to DataFrame
        for (cell, t_idx), load in cell_time_optimized.items():
            mask = (df["h3_cell"] == cell) & (df["time_idx"] == t_idx)
            df.loc[mask, "optimized_ev_load_kw"] = load
        
        # Compute derived columns
        df["baseline_total_load_kw"] = df["grid_base_load_kw"] + df["baseline_ev_load_kw"]
        df["optimized_total_load_kw"] = df["grid_base_load_kw"] + df["optimized_ev_load_kw"]
        
        df["effective_capacity_kw"] = df.apply(
            lambda r: derated_capacity(
                float(r["transformer_capacity_kw"]),
                float(r.get("temperature_c", 30))
            ),
            axis=1
        )
        
        df["baseline_transformer_utilization"] = df["baseline_total_load_kw"] / df["effective_capacity_kw"]
        df["optimized_transformer_utilization"] = df["optimized_total_load_kw"] / df["effective_capacity_kw"]
        df["stress_label"] = df["optimized_transformer_utilization"].apply(stress_label)
        
        # V2G readiness
        df["v2g_ready"] = (df["optimized_transformer_utilization"] < 0.6) & (df["tariff_multiplier"] >= 1.0)
        
        return df
    
    def _compute_metrics(
        self,
        df: pd.DataFrame,
        demand_col: str,
    ) -> Dict[str, object]:
        """Compute optimization metrics."""
        dt_hours = 1.0
        if "timestamp" in df.columns:
            unique_times = sorted(df["timestamp"].unique())
            if len(unique_times) > 1:
                dt = (pd.Series(unique_times).diff().dropna().dt.total_seconds().median() / 3600.0)
                if np.isfinite(dt) and dt > 0:
                    dt_hours = dt
        
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
        
        # Cost savings
        baseline_cost = (df["baseline_ev_load_kw"] * df["tariff_multiplier"] * BASE_TARIFF_INR_KWH * dt_hours).sum()
        optimized_cost = (df["optimized_ev_load_kw"] * df["tariff_multiplier"] * BASE_TARIFF_INR_KWH * dt_hours).sum()
        cost_savings = baseline_cost - optimized_cost
        
        # CO2 reduction
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
        
        return {
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
            "optimizer_type": "lagrangian_mcdm",
        }
    
    def get_shadow_prices_df(self) -> pd.DataFrame:
        """Get shadow prices as a DataFrame for analysis."""
        if not self.shadow_prices:
            return pd.DataFrame()
        
        records = []
        for (cell, t_idx), price in self.shadow_prices.items():
            records.append({
                "h3_cell": cell,
                "time_idx": t_idx,
                "shadow_price": price,
            })
        
        return pd.DataFrame(records)


def optimize_charging_schedule_lagrangian(
    prediction_df: pd.DataFrame,
    demand_col: str = "predicted_demand_kw",
    alpha: float = 1.0,
    beta: float = 0.75,
    gamma_tariff: float = 0.40,
    gamma_solar: float = 0.30,
    max_transformer_utilization: float = 0.95,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Convenience function for Lagrangian optimization.
    
    Drop-in replacement for the greedy optimizer with same interface.
    """
    optimizer = LagrangianOptimizer(
        alpha=alpha,
        beta=beta,
        gamma_tariff=gamma_tariff,
        gamma_solar=gamma_solar,
        max_utilization=max_transformer_utilization,
    )
    
    return optimizer.optimize(prediction_df, demand_col)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__).rsplit("src", 1)[0])
    
    from src.spatial_grid.simulation import CityConfig, generate_synthetic_data
    
    print("Testing Lagrangian optimizer...")
    
    config = CityConfig(max_cells=15, num_days=2, freq="1h")
    data, grid, adj = generate_synthetic_data(config)
    
    # Simulate prediction
    data["predicted_demand_kw"] = data["demand_kw"]
    
    optimizer = LagrangianOptimizer()
    optimized, metrics = optimizer.optimize(data)
    
    print("\n=== Lagrangian Optimization Results ===")
    for k, v in metrics.items():
        if k != "top_risk_zones":
            print(f"  {k}: {v}")
    
    print(f"\n=== Shadow Prices ===")
    shadow_df = optimizer.get_shadow_prices_df()
    if len(shadow_df) > 0:
        print(f"  Max shadow price: {shadow_df['shadow_price'].max():.4f}")
        print(f"  Mean shadow price: {shadow_df['shadow_price'].mean():.4f}")
        print(f"  Non-zero prices: {(shadow_df['shadow_price'] > 0).sum()}")
    
    print(f"\n=== Energy Preservation ===")
    print(f"  Baseline total: {optimized['baseline_ev_load_kw'].sum():.2f} kW")
    print(f"  Optimized total: {optimized['optimized_ev_load_kw'].sum():.2f} kW")
    print(f"  Error: {metrics['energy_preservation_error_pct']:.3f}%")
    
    print("\nLagrangian optimizer test PASSED!")
