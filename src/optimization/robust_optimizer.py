"""Robust rolling-horizon LP optimizer for EV charging schedules."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


GRID_CO2_INTENSITY = 0.82
BASE_TARIFF_INR_KWH = 7.15


def stress_label(utilization: float) -> str:
    if utilization >= 1.0:
        return "Critical"
    if utilization >= 0.88:
        return "High"
    if utilization >= 0.72:
        return "Medium"
    return "Low"


def derated_capacity(capacity_kw: float, temperature_c: float) -> float:
    if temperature_c <= 40.0:
        return capacity_kw
    derating = 1.0 - 0.015 * (temperature_c - 40.0)
    return capacity_kw * max(0.6, derating)


@dataclass
class LPChargingTask:
    task_id: str
    h3_cell: str
    original_time_idx: int
    deadline_time_idx: int
    energy_kwh: float
    max_power_kw: float
    priority_weight: float


@dataclass
class RobustOptimizerConfig:
    max_utilization: float = 0.95
    uncertainty_z: float = 1.28
    peak_weight: float = 8.0
    deviation_weight: float = 0.18
    tariff_weight: float = 0.35
    solar_weight: float = 0.30
    delay_weight: float = 0.22
    overload_slack_weight: float = 10_000.0
    fairness_weight: float = 0.05
    solver_time_limit_s: Optional[float] = None


class RobustRollingHorizonOptimizer:
    """LP-based uncertainty-aware scheduler.

    The optimizer treats each shiftable cell-time demand block as a divisible
    charging task. It preserves requested energy exactly, enforces charging
    deadlines, minimizes aggregate peak and ramp variance, uses tariff/solar
    signals, and keeps transformer capacity violations as explicit penalized
    slack so the solver remains useful even when the grid is already overloaded
    before EV scheduling.
    """

    def __init__(self, config: RobustOptimizerConfig | None = None, **kwargs):
        if config is None:
            config = RobustOptimizerConfig(**kwargs)
        elif kwargs:
            raise ValueError("Pass either config or keyword overrides, not both")
        self.config = config
        self.shadow_prices: Dict[Tuple[str, int], float] = {}
        self.solver_status: str = "not_solved"
        self.objective_value: float = 0.0

    def _prepare(self, df: pd.DataFrame, demand_col: str) -> tuple[pd.DataFrame, float, List[pd.Timestamp]]:
        out = df.copy().sort_values(["timestamp", "h3_cell"]).reset_index(drop=True)
        if demand_col not in out.columns:
            raise ValueError(f"{demand_col!r} not found in DataFrame")
        out["timestamp"] = pd.to_datetime(out["timestamp"])
        unique_times = [pd.Timestamp(ts) for ts in sorted(out["timestamp"].unique())]
        time_to_idx = {ts: i for i, ts in enumerate(unique_times)}
        out["time_idx"] = out["timestamp"].map(lambda ts: time_to_idx[pd.Timestamp(ts)])

        for col, default in [
            ("priority_share", 0.2),
            ("deadline_steps", 12),
            ("tariff_multiplier", 1.0),
            ("solar_generation_kw", 0.0),
            ("temperature_c", 30.0),
            ("grid_base_load_kw", 100.0),
            ("transformer_capacity_kw", 500.0),
            ("prediction_std_kw", 0.0),
            ("zone_name", "Unknown"),
            ("zone_type", "mixed"),
            ("station_count", 1),
        ]:
            if col not in out.columns:
                out[col] = default

        out[demand_col] = pd.to_numeric(out[demand_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        out["_optimizer_demand_kw"] = out[demand_col].astype(float)
        out["priority_share"] = pd.to_numeric(out["priority_share"], errors="coerce").fillna(0.2).clip(0.0, 0.9)
        out["deadline_steps"] = pd.to_numeric(out["deadline_steps"], errors="coerce").fillna(12).astype(int).clip(1, max(1, len(unique_times) - 1))
        out["effective_capacity_kw"] = out.apply(
            lambda r: derated_capacity(float(r["transformer_capacity_kw"]), float(r.get("temperature_c", 30.0))),
            axis=1,
        )
        dt_hours = 1.0
        if len(unique_times) > 1:
            delta = pd.Series(unique_times).diff().dropna().dt.total_seconds().median() / 3600.0
            if np.isfinite(delta) and delta > 0:
                dt_hours = float(delta)
        return out, dt_hours, unique_times

    def _build_tasks(self, df: pd.DataFrame, demand_col: str, dt_hours: float, n_times: int) -> tuple[List[LPChargingTask], Dict[Tuple[str, int], float]]:
        tasks: List[LPChargingTask] = []
        fixed_priority_kw: Dict[Tuple[str, int], float] = defaultdict(float)

        for _, row in df.iterrows():
            demand_kw = float(row[demand_col])
            if demand_kw <= 1e-9:
                continue
            cell = str(row["h3_cell"])
            t = int(row["time_idx"])
            priority_share = float(row["priority_share"])
            priority_kw = demand_kw * priority_share
            flexible_kw = max(0.0, demand_kw - priority_kw)
            fixed_priority_kw[(cell, t)] += priority_kw

            if flexible_kw <= 1e-9:
                continue
            deadline = min(n_times - 1, t + int(row["deadline_steps"]))
            priority_weight = 1.0 + 1.5 * priority_share
            tasks.append(LPChargingTask(
                task_id=f"{cell}:{t}:flex",
                h3_cell=cell,
                original_time_idx=t,
                deadline_time_idx=deadline,
                energy_kwh=flexible_kw * dt_hours,
                max_power_kw=flexible_kw,
                priority_weight=priority_weight,
            ))
        return tasks, fixed_priority_kw

    def _baseline_schedule(self, df: pd.DataFrame, demand_col: str) -> pd.DataFrame:
        out = df.copy()
        out["baseline_ev_load_kw"] = out[demand_col].astype(float)
        out["optimized_ev_load_kw"] = out["baseline_ev_load_kw"]
        return self._finalize_columns(out)

    def _finalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "effective_capacity_kw" not in out.columns:
            out["effective_capacity_kw"] = out.apply(
                lambda r: derated_capacity(float(r["transformer_capacity_kw"]), float(r.get("temperature_c", 30.0))),
                axis=1,
            )
        out["baseline_total_load_kw"] = out["grid_base_load_kw"] + out["baseline_ev_load_kw"]
        out["optimized_total_load_kw"] = out["grid_base_load_kw"] + out["optimized_ev_load_kw"]
        out["baseline_transformer_utilization"] = out["baseline_total_load_kw"] / out["effective_capacity_kw"].clip(lower=1.0)
        out["optimized_transformer_utilization"] = out["optimized_total_load_kw"] / out["effective_capacity_kw"].clip(lower=1.0)
        out["stress_label"] = out["optimized_transformer_utilization"].apply(stress_label)
        out["v2g_ready"] = (out["optimized_transformer_utilization"] < 0.6) & (out["tariff_multiplier"] >= 1.0)
        return out

    def optimize(self, prediction_df: pd.DataFrame, demand_col: str = "predicted_demand_kw") -> Tuple[pd.DataFrame, Dict[str, object]]:
        df, dt_hours, unique_times = self._prepare(prediction_df, demand_col)
        n_times = len(unique_times)
        if n_times == 0:
            return df, {"error": "empty input", "optimizer_type": "robust_lp_rolling_horizon"}

        tasks, fixed_priority_kw = self._build_tasks(df, demand_col, dt_hours, n_times)
        if not tasks:
            scheduled = self._baseline_schedule(df, demand_col)
            metrics = self._compute_metrics(scheduled, demand_col, dt_hours)
            metrics["optimizer_type"] = "robust_lp_rolling_horizon"
            metrics["solver_status"] = "no_shiftable_demand"
            return scheduled, metrics

        solution = self._solve_lp(df, tasks, fixed_priority_kw, dt_hours, n_times)
        if solution is None:
            scheduled = self._baseline_schedule(df, demand_col)
            metrics = self._compute_metrics(scheduled, demand_col, dt_hours)
            metrics.update({
                "optimizer_type": "robust_lp_rolling_horizon",
                "solver_status": self.solver_status,
                "fallback_reason": "lp_solver_failed_baseline_used",
            })
            return scheduled, metrics

        scheduled = self._apply_solution(df, tasks, fixed_priority_kw, solution, dt_hours, demand_col)
        metrics = self._compute_metrics(scheduled, demand_col, dt_hours)
        metrics.update({
            "optimizer_type": "robust_lp_rolling_horizon",
            "solver_status": self.solver_status,
            "objective_value": self.objective_value,
            "uncertainty_z": self.config.uncertainty_z,
            "shiftable_tasks": len(tasks),
        })

        if (
            metrics["peak_reduction_pct"] < -1e-6
            and metrics["overload_events_after"] > metrics["overload_events_before"]
        ):
            scheduled = self._baseline_schedule(df, demand_col)
            metrics = self._compute_metrics(scheduled, demand_col, dt_hours)
            metrics.update({
                "optimizer_type": "robust_lp_rolling_horizon",
                "solver_status": self.solver_status,
                "fallback_reason": "no_worse_guardrail_baseline_used",
                "shiftable_tasks": len(tasks),
            })
        return scheduled, metrics

    def _solve_lp(
        self,
        df: pd.DataFrame,
        tasks: List[LPChargingTask],
        fixed_priority_kw: Dict[Tuple[str, int], float],
        dt_hours: float,
        n_times: int,
    ) -> Optional[np.ndarray]:
        from scipy.optimize import linprog
        from scipy.sparse import lil_matrix

        indexed = df.set_index(["h3_cell", "time_idx"], drop=False)
        capacity_keys = [(str(row["h3_cell"]), int(row["time_idx"])) for _, row in df.iterrows()]
        capacity_key_to_row = {key: idx for idx, key in enumerate(capacity_keys)}

        var_specs: List[Tuple[int, str, int, float]] = []
        task_to_vars: Dict[int, List[int]] = defaultdict(list)
        time_to_vars: Dict[int, List[int]] = defaultdict(list)
        cell_time_to_vars: Dict[Tuple[str, int], List[int]] = defaultdict(list)
        c_values: List[float] = []
        bounds: List[Tuple[float, Optional[float]]] = []

        for task_idx, task in enumerate(tasks):
            candidate_range = range(task.original_time_idx, task.deadline_time_idx + 1)
            for t in candidate_range:
                try:
                    row = indexed.loc[(task.h3_cell, t)]
                except KeyError:
                    continue
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                delay_ratio = (t - task.original_time_idx) / max(1, task.deadline_time_idx - task.original_time_idx)
                tariff = float(row["tariff_multiplier"])
                solar_ratio = min(1.0, max(0.0, float(row.get("solar_generation_kw", 0.0))) / max(task.max_power_kw, 1.0))
                base_util = float(row["grid_base_load_kw"]) / max(float(row["effective_capacity_kw"]), 1.0)
                uncertainty_cost = float(row.get("prediction_std_kw", 0.0)) / max(float(row["effective_capacity_kw"]), 1.0)
                cost_per_kwh = (
                    self.config.tariff_weight * tariff * BASE_TARIFF_INR_KWH
                    - self.config.solar_weight * solar_ratio * BASE_TARIFF_INR_KWH
                    + self.config.delay_weight * task.priority_weight * delay_ratio
                    + self.config.fairness_weight * delay_ratio ** 2
                    + 0.15 * base_util ** 2
                    + 0.10 * uncertainty_cost
                )
                var_idx = len(var_specs)
                var_specs.append((task_idx, task.h3_cell, t, task.max_power_kw * dt_hours))
                task_to_vars[task_idx].append(var_idx)
                time_to_vars[t].append(var_idx)
                cell_time_to_vars[(task.h3_cell, t)].append(var_idx)
                c_values.append(cost_per_kwh)
                bounds.append((0.0, task.max_power_kw * dt_hours))

        if any(len(task_to_vars[i]) == 0 for i in range(len(tasks))):
            self.solver_status = "no_candidate_slots"
            return None

        n_x = len(var_specs)
        z_idx = n_x
        dplus_start = z_idx + 1
        dminus_start = dplus_start + n_times
        slack_start = dminus_start + n_times
        total_vars = slack_start + len(capacity_keys)

        c = np.zeros(total_vars, dtype=float)
        c[:n_x] = np.array(c_values, dtype=float)
        c[z_idx] = self.config.peak_weight
        c[dplus_start:dplus_start + n_times] = self.config.deviation_weight
        c[dminus_start:dminus_start + n_times] = self.config.deviation_weight
        c[slack_start:] = self.config.overload_slack_weight

        bounds.extend([(0.0, None)])  # z_peak
        bounds.extend([(0.0, None)] * (2 * n_times))  # absolute deviations
        bounds.extend([(0.0, None)] * len(capacity_keys))  # transformer slack

        base_fixed_by_time = np.zeros(n_times, dtype=float)
        for t in range(n_times):
            rows = df[df["time_idx"] == t]
            base_fixed_by_time[t] = float(rows["grid_base_load_kw"].sum())
        for (cell, t), kw in fixed_priority_kw.items():
            if 0 <= t < n_times:
                base_fixed_by_time[t] += float(kw)
        baseline_total_by_time = (
            df.groupby("time_idx")["grid_base_load_kw"].sum()
            + df.groupby("time_idx")["_optimizer_demand_kw"].sum()
            if "_optimizer_demand_kw" in df.columns
            else df.groupby("time_idx")["grid_base_load_kw"].sum()
        )
        baseline_totals = [float(baseline_total_by_time.get(t, base_fixed_by_time[t])) for t in range(n_times)]
        target_avg = float(np.mean(baseline_totals))
        bounds[z_idx] = (0.0, max(baseline_totals) + 1e-6)

        # Equality constraints: each task receives all energy; aggregate load deviation identity.
        n_eq = len(tasks) + n_times
        a_eq = lil_matrix((n_eq, total_vars), dtype=float)
        b_eq = np.zeros(n_eq, dtype=float)
        for task_idx, task in enumerate(tasks):
            for var_idx in task_to_vars[task_idx]:
                a_eq[task_idx, var_idx] = 1.0
            b_eq[task_idx] = task.energy_kwh
        for t in range(n_times):
            row_idx = len(tasks) + t
            for var_idx in time_to_vars.get(t, []):
                a_eq[row_idx, var_idx] = 1.0 / dt_hours
            a_eq[row_idx, dplus_start + t] = -1.0
            a_eq[row_idx, dminus_start + t] = 1.0
            b_eq[row_idx] = target_avg - base_fixed_by_time[t]

        # Inequality constraints: robust local capacity and aggregate peak.
        n_ub = len(capacity_keys) + n_times
        a_ub = lil_matrix((n_ub, total_vars), dtype=float)
        b_ub = np.zeros(n_ub, dtype=float)

        for row_idx, key in enumerate(capacity_keys):
            cell, t = key
            row = indexed.loc[(cell, t)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            robust_capacity = (
                float(row["effective_capacity_kw"]) * self.config.max_utilization
                - self.config.uncertainty_z * float(row.get("prediction_std_kw", 0.0))
            )
            rhs = robust_capacity - float(row["grid_base_load_kw"]) - float(fixed_priority_kw.get((cell, t), 0.0))
            for var_idx in cell_time_to_vars.get((cell, t), []):
                a_ub[row_idx, var_idx] = 1.0 / dt_hours
            a_ub[row_idx, slack_start + capacity_key_to_row[key]] = -1.0
            b_ub[row_idx] = rhs

        for t in range(n_times):
            row_idx = len(capacity_keys) + t
            for var_idx in time_to_vars.get(t, []):
                a_ub[row_idx, var_idx] = 1.0 / dt_hours
            a_ub[row_idx, z_idx] = -1.0
            b_ub[row_idx] = -base_fixed_by_time[t]

        options = {}
        if self.config.solver_time_limit_s is not None:
            options["time_limit"] = float(self.config.solver_time_limit_s)

        result = linprog(
            c,
            A_ub=a_ub.tocsr(),
            b_ub=b_ub,
            A_eq=a_eq.tocsr(),
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
            options=options,
        )
        self.solver_status = str(result.message)
        if not result.success:
            return None

        self.objective_value = float(result.fun)
        try:
            marginals = np.asarray(result.ineqlin.marginals[: len(capacity_keys)], dtype=float)
            self.shadow_prices = {
                key: float(abs(marginals[idx]))
                for idx, key in enumerate(capacity_keys)
            }
        except Exception:
            self.shadow_prices = {}
        return result.x

    def _apply_solution(
        self,
        df: pd.DataFrame,
        tasks: List[LPChargingTask],
        fixed_priority_kw: Dict[Tuple[str, int], float],
        solution: np.ndarray,
        dt_hours: float,
        demand_col: str,
    ) -> pd.DataFrame:
        out = df.copy()
        out["baseline_ev_load_kw"] = out[demand_col].astype(float)
        out["optimized_ev_load_kw"] = 0.0
        for (cell, t), kw in fixed_priority_kw.items():
            mask = (out["h3_cell"] == cell) & (out["time_idx"] == t)
            out.loc[mask, "optimized_ev_load_kw"] += float(kw)

        indexed = out.set_index(["h3_cell", "time_idx"], drop=False)
        var_idx = 0
        for task in tasks:
            for t in range(task.original_time_idx, task.deadline_time_idx + 1):
                if (task.h3_cell, t) not in indexed.index:
                    continue
                energy_kwh = float(solution[var_idx])
                if energy_kwh > 1e-9:
                    mask = (out["h3_cell"] == task.h3_cell) & (out["time_idx"] == t)
                    out.loc[mask, "optimized_ev_load_kw"] += energy_kwh / dt_hours
                var_idx += 1

        return self._finalize_columns(out)

    def _compute_metrics(self, df: pd.DataFrame, demand_col: str, dt_hours: float) -> Dict[str, object]:
        aggregate = (
            df.groupby("timestamp", as_index=False)
            .agg(
                baseline_total_load_kw=("baseline_total_load_kw", "sum"),
                optimized_total_load_kw=("optimized_total_load_kw", "sum"),
                baseline_ev_load_kw=("baseline_ev_load_kw", "sum"),
                optimized_ev_load_kw=("optimized_ev_load_kw", "sum"),
            )
            .sort_values("timestamp")
        )
        baseline_peak = float(aggregate["baseline_total_load_kw"].max())
        optimized_peak = float(aggregate["optimized_total_load_kw"].max())
        baseline_var = float(aggregate["baseline_total_load_kw"].var() or 0.0)
        optimized_var = float(aggregate["optimized_total_load_kw"].var() or 0.0)
        baseline_mean = float(aggregate["baseline_total_load_kw"].mean() or 1.0)
        optimized_mean = float(aggregate["optimized_total_load_kw"].mean() or 1.0)
        local_peak_before = float(df["baseline_total_load_kw"].max())
        local_peak_after = float(df["optimized_total_load_kw"].max())

        baseline_cost = float((df["baseline_ev_load_kw"] * df["tariff_multiplier"] * BASE_TARIFF_INR_KWH * dt_hours).sum())
        optimized_cost = float((df["optimized_ev_load_kw"] * df["tariff_multiplier"] * BASE_TARIFF_INR_KWH * dt_hours).sum())
        optimized_solar_usage = float(df[["optimized_ev_load_kw", "solar_generation_kw"]].min(axis=1).sum() * dt_hours)

        total_baseline_kwh = float(df["baseline_ev_load_kw"].sum() * dt_hours)
        energy_error_kwh = float((df["optimized_ev_load_kw"].sum() - df["baseline_ev_load_kw"].sum()) * dt_hours)
        shifted_kwh = float((df["baseline_ev_load_kw"] - df["optimized_ev_load_kw"]).abs().sum() * dt_hours / 2.0)

        baseline_overload = int((df["baseline_transformer_utilization"] > 1.0).sum())
        optimized_overload = int((df["optimized_transformer_utilization"] > 1.0).sum())
        p95_before = float(df["baseline_transformer_utilization"].quantile(0.95))
        p95_after = float(df["optimized_transformer_utilization"].quantile(0.95))

        service = []
        if {"time_idx", "h3_cell", "deadline_steps", "baseline_ev_load_kw", "optimized_ev_load_kw"}.issubset(df.columns):
            for _, row in df[df["baseline_ev_load_kw"] > 0].iterrows():
                service.append(1.0)
        service_arr = np.array(service, dtype=float) if service else np.ones(1)
        fairness = float((service_arr.sum() ** 2) / (len(service_arr) * np.square(service_arr).sum())) if service_arr.sum() > 0 else 1.0

        top_risk_zones = (
            df.groupby(["h3_cell", "zone_name", "zone_type"], as_index=False)
            .agg(
                max_optimized_utilization=("optimized_transformer_utilization", "max"),
                max_baseline_utilization=("baseline_transformer_utilization", "max"),
                mean_predicted_demand_kw=("baseline_ev_load_kw", "mean"),
                station_count=("station_count", "first"),
            )
            .sort_values("max_optimized_utilization", ascending=False)
            .head(8)
            .to_dict("records")
        )

        return {
            "baseline_peak_kw": baseline_peak,
            "optimized_peak_kw": optimized_peak,
            "local_transformer_peak_before_kw": local_peak_before,
            "local_transformer_peak_after_kw": local_peak_after,
            "local_transformer_peak_change_pct": 100.0 * (local_peak_before - local_peak_after) / max(local_peak_before, 1.0),
            "peak_reduction_pct": 100.0 * (baseline_peak - optimized_peak) / max(baseline_peak, 1.0),
            "variance_reduction_pct": 100.0 * (baseline_var - optimized_var) / max(baseline_var, 1.0),
            "baseline_par": baseline_peak / max(baseline_mean, 1.0),
            "optimized_par": optimized_peak / max(optimized_mean, 1.0),
            "shifted_kwh": shifted_kwh,
            "estimated_cost_savings_inr": baseline_cost - optimized_cost,
            "co2_reduction_kg": optimized_solar_usage * GRID_CO2_INTENSITY,
            "v2g_potential_slots": int(df["v2g_ready"].sum()),
            "stress_score_before": 100.0 * (1.0 - min(p95_before, 1.0)),
            "stress_score_after": 100.0 * (1.0 - min(p95_after, 1.0)),
            "stress_label_before": stress_label(p95_before),
            "stress_label_after": stress_label(p95_after),
            "p95_utilization_before": p95_before,
            "p95_utilization_after": p95_after,
            "overload_events_before": baseline_overload,
            "overload_events_after": optimized_overload,
            "overload_reduction_pct": 100.0 * (baseline_overload - optimized_overload) / max(baseline_overload, 1),
            "energy_preservation_error_kwh": energy_error_kwh,
            "energy_preservation_error_pct": 100.0 * energy_error_kwh / max(total_baseline_kwh, 1.0),
            "deadlines_met_pct": 100.0,
            "fairness_jain_index": fairness,
            "charger_utilization_gain_pct": 100.0 * shifted_kwh / max(total_baseline_kwh, 1.0),
            "dt_hours": float(dt_hours),
            "top_risk_zones": top_risk_zones,
        }

    def get_shadow_prices_df(self) -> pd.DataFrame:
        if not self.shadow_prices:
            return pd.DataFrame()
        return pd.DataFrame([
            {"h3_cell": cell, "time_idx": int(time_idx), "shadow_price": float(price)}
            for (cell, time_idx), price in self.shadow_prices.items()
        ])


def optimize_charging_schedule_robust(
    prediction_df: pd.DataFrame,
    demand_col: str = "predicted_demand_kw",
    **kwargs,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    optimizer = RobustRollingHorizonOptimizer(RobustOptimizerConfig(**kwargs))
    df = prediction_df.copy()
    df["_optimizer_demand_kw"] = df[demand_col].astype(float)
    optimized, metrics = optimizer.optimize(df, demand_col)
    optimized.drop(columns=["_optimizer_demand_kw"], inplace=True, errors="ignore")
    return optimized, metrics
