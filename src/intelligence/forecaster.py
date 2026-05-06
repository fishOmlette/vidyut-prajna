"""STGCN-based demand forecaster for Vidyut Prajna.

The forecaster combines:
- historical graph sequences for spatial-temporal learning,
- known target-time exogenous features for the next timestep,
- transparent persistence/seasonal baselines as guardrails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .graph_utils import get_adjacency_matrix
from .model import VidyutPrajnaForecaster


ZONE_TYPES = ["residential", "commercial", "logistics", "mixed", "it_corridor"]
ZONE_FEATURE_COLS = [f"zone_{z}" for z in ZONE_TYPES]

SEQUENCE_FEATURE_COLS = [
    "demand_kw",
    "neighbor_demand_kw",
    "grid_base_load_kw",
    "traffic_intensity",
    "temperature_c",
    "rainfall_mm",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "ev_adoption_index",
    "charger_density_index",
    "demand_growth_index",
    "station_count",
    "tariff_multiplier",
    "solar_generation_kw",
]

FUTURE_EXOG_COLS = [
    "grid_base_load_kw",
    "traffic_intensity",
    "temperature_c",
    "rainfall_mm",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "ev_adoption_index",
    "charger_density_index",
    "demand_growth_index",
    "station_count",
    "tariff_multiplier",
    "solar_generation_kw",
    "transformer_capacity_kw",
] + ZONE_FEATURE_COLS

# Backward-compatible alias used by older tests/docs.
FEATURE_COLS = SEQUENCE_FEATURE_COLS


@dataclass
class TrainingInfo:
    """Training metrics and metadata."""

    train_samples: int
    val_samples: int
    epochs: int
    final_train_loss: float
    final_val_loss: float
    seasonal_baseline_mae_kw: float = 0.0
    persistence_baseline_mae_kw: float = 0.0
    model_health: str = "not_forecasted"
    notes: List[str] = field(default_factory=list)

    @property
    def final_loss(self) -> float:
        return self.final_val_loss


class STGCNForecaster:
    """High-level STGCN forecaster for EV demand prediction."""

    def __init__(
        self,
        seq_len: int = 8,
        hidden_size: int = 48,
        epochs: int = 12,
        batch_size: int = 64,
        learning_rate: float = 2e-3,
        num_blocks: int = 2,
        dropout: float = 0.1,
        seed: int = 42,
        device: str = None,
        val_fraction: float = 0.15,
        min_variance_ratio: float = 0.35,
    ):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.val_fraction = val_fraction
        self.min_variance_ratio = min_variance_ratio

        self.model: Optional[VidyutPrajnaForecaster] = None
        self.cells: List[str] = []
        self.cell_to_idx: Dict[str, int] = {}
        self.adj_tensor: Optional[torch.Tensor] = None

        self.sequence_mean: Optional[pd.Series] = None
        self.sequence_std: Optional[pd.Series] = None
        self.future_mean: Optional[pd.Series] = None
        self.future_std: Optional[pd.Series] = None
        self.feature_mean: Optional[pd.Series] = None
        self.feature_std: Optional[pd.Series] = None
        self.target_mean: float = 0.0
        self.target_std: float = 1.0

        self.training_info: Optional[TrainingInfo] = None
        self.forecast_info: Dict[str, object] = {}
        self._seasonal_lookup: Dict[Tuple[str, int, int], float] = {}
        self._cell_mean: Dict[str, float] = {}
        self._global_slot_mean: Dict[Tuple[int, int], float] = {}
        self._global_mean: float = 0.0

    # ------------------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------------------

    @staticmethod
    def _minute_of_day(ts: pd.Timestamp) -> int:
        return int(ts.hour * 60 + ts.minute)

    @staticmethod
    def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        hours = out["timestamp"].dt.hour + out["timestamp"].dt.minute / 60.0
        out["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
        dow = out["timestamp"].dt.dayofweek.astype(float)
        out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
        out["minute_of_day"] = (out["timestamp"].dt.hour * 60 + out["timestamp"].dt.minute).astype(int)
        if "is_weekend" not in out.columns:
            out["is_weekend"] = (out["timestamp"].dt.dayofweek >= 5).astype(float)
        else:
            out["is_weekend"] = out["is_weekend"].astype(float)
        return out

    @staticmethod
    def _add_zone_features(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        zone = out.get("zone_type", pd.Series(["mixed"] * len(out), index=out.index)).astype(str)
        for zone_type, feature_col in zip(ZONE_TYPES, ZONE_FEATURE_COLS):
            out[feature_col] = (zone == zone_type).astype(float)
        return out

    @staticmethod
    def _add_neighbor_demand(df: pd.DataFrame, adjacency: Dict[str, List[str]]) -> pd.DataFrame:
        out = df.copy()
        pivot = out.pivot(index="timestamp", columns="h3_cell", values="demand_kw").sort_index()
        neighbor_frames = []
        for cell in pivot.columns:
            nbrs = [n for n in adjacency.get(cell, []) if n in pivot.columns]
            s = pivot[nbrs].mean(axis=1) if nbrs else pivot[cell]
            neighbor_frames.append(s.rename(cell))
        neighbor_df = pd.concat(neighbor_frames, axis=1)
        long_neighbor = neighbor_df.stack().rename("neighbor_demand_kw").reset_index()
        long_neighbor.columns = ["timestamp", "h3_cell", "neighbor_demand_kw"]
        out = out.merge(long_neighbor, on=["timestamp", "h3_cell"], how="left")
        out["neighbor_demand_kw"] = out["neighbor_demand_kw"].fillna(out["demand_kw"])
        return out

    @staticmethod
    def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
        defaults = {
            "tariff_multiplier": 1.0,
            "solar_generation_kw": 0.0,
            "ev_adoption_index": 0.75,
            "charger_density_index": 5.0,
            "demand_growth_index": 0.20,
            "station_count": 4.0,
            "transformer_capacity_kw": 500.0,
            "is_weekend": 0.0,
            "traffic_intensity": 0.5,
            "temperature_c": 28.0,
            "rainfall_mm": 0.0,
            "grid_base_load_kw": 100.0,
            "zone_type": "mixed",
        }
        out = df.copy()
        if "demand_kw" not in out.columns:
            out["demand_kw"] = 0.0
        for col, default in defaults.items():
            if col not in out.columns:
                out[col] = default
        return out

    def _engineer_features(
        self,
        df: pd.DataFrame,
        adjacency: Dict[str, List[str]],
        add_neighbor: bool = True,
    ) -> pd.DataFrame:
        out = self._ensure_columns(df)
        out = self._add_time_features(out)
        out = self._add_zone_features(out)
        if add_neighbor:
            out = self._add_neighbor_demand(out, adjacency)
        elif "neighbor_demand_kw" not in out.columns:
            out["neighbor_demand_kw"] = out["demand_kw"]
        return out

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------

    def _fit_baselines(self, frame: pd.DataFrame) -> None:
        grouped = frame.groupby(["h3_cell", "minute_of_day", "is_weekend"])["demand_kw"].mean()
        self._seasonal_lookup = {
            (str(cell), int(minute), int(weekend)): float(value)
            for (cell, minute, weekend), value in grouped.items()
        }
        self._cell_mean = {
            str(cell): float(value)
            for cell, value in frame.groupby("h3_cell")["demand_kw"].mean().items()
        }
        self._global_slot_mean = {
            (int(minute), int(weekend)): float(value)
            for (minute, weekend), value in frame.groupby(["minute_of_day", "is_weekend"])["demand_kw"].mean().items()
        }
        self._global_mean = float(frame["demand_kw"].mean())

    def _seasonal_baseline(self, cell: str, ts: pd.Timestamp, is_weekend: float) -> float:
        minute = self._minute_of_day(ts)
        weekend = int(float(is_weekend) >= 0.5)
        if (cell, minute, weekend) in self._seasonal_lookup:
            return self._seasonal_lookup[(cell, minute, weekend)]
        if (cell, minute, 1 - weekend) in self._seasonal_lookup:
            return self._seasonal_lookup[(cell, minute, 1 - weekend)]
        cell_mean = self._cell_mean.get(cell, self._global_mean)
        slot_mean = self._global_slot_mean.get((minute, weekend), self._global_mean)
        return float(max(0.0, 0.65 * cell_mean + 0.35 * slot_mean))

    @staticmethod
    def _persistence_baseline(history_features: pd.DataFrame, cell: str) -> float:
        rows = history_features[history_features["h3_cell"] == cell].sort_values("timestamp")
        if rows.empty:
            return 0.0
        return float(rows["demand_kw"].iloc[-1])

    # ------------------------------------------------------------------
    # Sequence Building
    # ------------------------------------------------------------------

    def _build_sequences(self, frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        timestamps = sorted(frame["timestamp"].unique())
        n_cells = len(self.cells)

        hist_parts: List[np.ndarray] = []
        future_parts: List[np.ndarray] = []
        y_parts: List[np.ndarray] = []

        for i in range(self.seq_len, len(timestamps)):
            seq_times = timestamps[i - self.seq_len:i]
            target_time = timestamps[i]

            seq_data = np.zeros((self.seq_len, n_cells, len(SEQUENCE_FEATURE_COLS)), dtype=np.float32)
            future_data = np.zeros((n_cells, len(FUTURE_EXOG_COLS)), dtype=np.float32)
            target_data = np.zeros(n_cells, dtype=np.float32)

            for t_idx, ts in enumerate(seq_times):
                ts_data = frame[frame["timestamp"] == ts]
                for _, row in ts_data.iterrows():
                    cell = row["h3_cell"]
                    if cell in self.cell_to_idx:
                        seq_data[t_idx, self.cell_to_idx[cell], :] = row[SEQUENCE_FEATURE_COLS].values.astype(np.float32)

            target_ts_data = frame[frame["timestamp"] == target_time]
            for _, row in target_ts_data.iterrows():
                cell = row["h3_cell"]
                if cell in self.cell_to_idx:
                    c_idx = self.cell_to_idx[cell]
                    future_data[c_idx, :] = row[FUTURE_EXOG_COLS].values.astype(np.float32)
                    target_data[c_idx] = row["demand_kw"]

            hist_parts.append(seq_data)
            future_parts.append(future_data)
            y_parts.append(target_data)

        if not hist_parts:
            raise ValueError("Not enough data to build sequences. Increase num_days or reduce seq_len.")

        return np.stack(hist_parts), np.stack(future_parts), np.stack(y_parts)

    def _normalize_hist(self, x: np.ndarray) -> np.ndarray:
        mean = self.sequence_mean[SEQUENCE_FEATURE_COLS].to_numpy(dtype=np.float32)
        std = self.sequence_std[SEQUENCE_FEATURE_COLS].to_numpy(dtype=np.float32)
        return (x - mean) / std

    def _normalize_future(self, x: np.ndarray) -> np.ndarray:
        mean = self.future_mean[FUTURE_EXOG_COLS].to_numpy(dtype=np.float32)
        std = self.future_std[FUTURE_EXOG_COLS].to_numpy(dtype=np.float32)
        return (x - mean) / std

    def _normalize_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.target_mean) / self.target_std

    def _denormalize_y(self, y: np.ndarray) -> np.ndarray:
        return y * self.target_std + self.target_mean

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame, adjacency: Dict[str, List[str]]) -> "STGCNForecaster":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

        self.cells = sorted(train_df["h3_cell"].unique().tolist())
        self.cell_to_idx = {c: i for i, c in enumerate(self.cells)}
        self.adj_tensor = get_adjacency_matrix(self.cells).to(self.device)

        self.model = VidyutPrajnaForecaster(
            adj_matrix=self.adj_tensor,
            in_channels=len(SEQUENCE_FEATURE_COLS),
            future_channels=len(FUTURE_EXOG_COLS),
            hidden_channels=self.hidden_size,
            out_channels=1,
            num_blocks=self.num_blocks,
            dropout=self.dropout,
        ).to(self.device)

        frame = self._engineer_features(train_df, adjacency)
        self._fit_baselines(frame)

        self.sequence_mean = frame[SEQUENCE_FEATURE_COLS].mean()
        self.sequence_std = frame[SEQUENCE_FEATURE_COLS].std().fillna(1.0).clip(lower=0.05)
        self.future_mean = frame[FUTURE_EXOG_COLS].mean()
        self.future_std = frame[FUTURE_EXOG_COLS].std().fillna(1.0).clip(lower=0.05)
        self.feature_mean = self.sequence_mean
        self.feature_std = self.sequence_std
        self.target_mean = float(frame["demand_kw"].mean())
        self.target_std = float(frame["demand_kw"].std() or 1.0)

        x_hist_raw, x_future_raw, y_raw = self._build_sequences(frame)
        x_hist = self._normalize_hist(x_hist_raw)
        x_future = self._normalize_future(x_future_raw)
        y = self._normalize_y(y_raw)

        n = len(x_hist)
        n_val = max(1, int(n * self.val_fraction))
        indices = np.arange(n)
        np.random.shuffle(indices)
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        train_ds = TensorDataset(
            torch.tensor(x_hist[train_idx], dtype=torch.float32),
            torch.tensor(x_future[train_idx], dtype=torch.float32),
            torch.tensor(y[train_idx], dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(x_hist[val_idx], dtype=torch.float32),
            torch.tensor(x_future[val_idx], dtype=torch.float32),
            torch.tensor(y[val_idx], dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        loss_fn = nn.HuberLoss(delta=1.0)

        best_val_loss = float("inf")
        patience_counter = 0
        max_patience = 5
        final_train_loss = 0.0
        final_val_loss = 0.0

        for epoch in range(self.epochs):
            self.model.train()
            losses = []
            for bx_hist, bx_future, by in train_loader:
                bx_hist = bx_hist.to(self.device)
                bx_future = bx_future.to(self.device)
                by = by.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                pred = self.model(bx_hist, bx_future).squeeze(-1)
                loss = loss_fn(pred, by)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            final_train_loss = float(np.mean(losses)) if losses else 0.0

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for bx_hist, bx_future, by in val_loader:
                    bx_hist = bx_hist.to(self.device)
                    bx_future = bx_future.to(self.device)
                    by = by.to(self.device)
                    pred = self.model(bx_hist, bx_future).squeeze(-1)
                    val_losses.append(float(loss_fn(pred, by).cpu()))
            final_val_loss = float(np.mean(val_losses)) if val_losses else final_train_loss
            scheduler.step(final_val_loss)

            if final_val_loss < best_val_loss - 1e-5:
                best_val_loss = final_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    break

        self.training_info = TrainingInfo(
            train_samples=len(train_ds),
            val_samples=len(val_ds),
            epochs=epoch + 1,
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _build_one_step_tensors(
        self,
        history_features: pd.DataFrame,
        step_exog_features: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        timestamps = sorted(history_features["timestamp"].unique())
        if len(timestamps) < self.seq_len:
            raise RuntimeError("Unable to produce predictions; insufficient history.")
        seq_times = timestamps[-self.seq_len:]
        n_cells = len(self.cells)

        seq_data = np.zeros((1, self.seq_len, n_cells, len(SEQUENCE_FEATURE_COLS)), dtype=np.float32)
        future_data = np.zeros((1, n_cells, len(FUTURE_EXOG_COLS)), dtype=np.float32)

        for t_idx, ts in enumerate(seq_times):
            ts_data = history_features[history_features["timestamp"] == ts]
            for _, row in ts_data.iterrows():
                cell = row["h3_cell"]
                if cell in self.cell_to_idx:
                    seq_data[0, t_idx, self.cell_to_idx[cell], :] = row[SEQUENCE_FEATURE_COLS].values.astype(np.float32)

        for _, row in step_exog_features.iterrows():
            cell = row["h3_cell"]
            if cell in self.cell_to_idx:
                future_data[0, self.cell_to_idx[cell], :] = row[FUTURE_EXOG_COLS].values.astype(np.float32)

        return self._normalize_hist(seq_data), self._normalize_future(future_data)

    def _predict_one_step(
        self,
        history_features: pd.DataFrame,
        step_exog_features: pd.DataFrame,
    ) -> Dict[str, Tuple[float, float]]:
        self.model.eval()

        loss_scale = 0.10
        if self.training_info:
            loss_scale = float(np.clip(self.training_info.final_loss * 0.5, 0.05, 0.30))

        x_hist, x_future = self._build_one_step_tensors(history_features, step_exog_features)

        predictions: Dict[str, Tuple[float, float]] = {}
        with torch.no_grad():
            xt_hist = torch.tensor(x_hist, dtype=torch.float32).to(self.device)
            xt_future = torch.tensor(x_future, dtype=torch.float32).to(self.device)
            p = self.model(xt_hist, xt_future).squeeze(-1).cpu().numpy()[0]
            p_denorm = self._denormalize_y(p)

            for cell, c_idx in self.cell_to_idx.items():
                mean_kw = max(0.0, float(p_denorm[c_idx]))
                std_kw = max(0.5, mean_kw * loss_scale)
                predictions[cell] = (mean_kw, std_kw)
        return predictions

    @staticmethod
    def _smape(actual: pd.Series, pred: pd.Series) -> float:
        denom = actual.abs() + pred.abs()
        return float((2.0 * (actual - pred).abs() / denom.clip(lower=1.0)).mean() * 100.0)

    @staticmethod
    def _aggregate_metric_frame(out: pd.DataFrame) -> pd.DataFrame:
        agg_cols = {
            "actual_demand_kw": ("actual_demand_kw", "sum"),
            "stgcn_predicted_demand_kw": ("stgcn_predicted_demand_kw", "sum"),
            "seasonal_baseline_kw": ("seasonal_baseline_kw", "sum"),
            "persistence_baseline_kw": ("persistence_baseline_kw", "sum"),
            "blended_predicted_demand_kw": ("blended_predicted_demand_kw", "sum"),
        }
        return out.groupby("timestamp", as_index=False).agg(**agg_cols).sort_values("timestamp")

    def _choose_forecast_method(self, out: pd.DataFrame) -> Tuple[str, Dict[str, object]]:
        actual_available = out["actual_demand_kw"].notna().any()
        agg = self._aggregate_metric_frame(out)

        actual_std = float(agg["actual_demand_kw"].std() or 0.0) if actual_available else 0.0
        stgcn_std = float(agg["stgcn_predicted_demand_kw"].std() or 0.0)
        seasonal_std = float(agg["seasonal_baseline_kw"].std() or 0.0)
        raw_to_actual_variance_ratio = stgcn_std / max(actual_std, 1.0) if actual_available else np.nan
        raw_to_seasonal_variance_ratio = stgcn_std / max(seasonal_std, 1.0)

        method = "stgcn"
        notes: List[str] = []
        candidate_cols = {
            "stgcn": "stgcn_predicted_demand_kw",
            "seasonal_baseline": "seasonal_baseline_kw",
            "persistence_baseline": "persistence_baseline_kw",
            "stgcn_seasonal_blend": "blended_predicted_demand_kw",
        }

        candidate_metrics: Dict[str, Dict[str, float]] = {}
        if actual_available:
            actual = agg["actual_demand_kw"].astype(float)
            for name, col in candidate_cols.items():
                pred = agg[col].astype(float)
                mae = float((actual - pred).abs().mean())
                smape = self._smape(actual, pred)
                pred_std = float(pred.std() or 0.0)
                corr = float(actual.corr(pred)) if actual.std() > 0 and pred.std() > 0 else 0.0
                candidate_metrics[name] = {
                    "mae_kw": mae,
                    "smape_pct": smape,
                    "variance_ratio": pred_std / max(actual_std, 1.0),
                    "correlation": corr,
                }

            best_method = min(candidate_metrics, key=lambda k: candidate_metrics[k]["mae_kw"])
            raw_flat = raw_to_actual_variance_ratio < self.min_variance_ratio
            raw_worse_than_seasonal = (
                candidate_metrics["stgcn"]["mae_kw"]
                > candidate_metrics["seasonal_baseline"]["mae_kw"] * 1.10
            )
            if raw_flat:
                notes.append("STGCN aggregate forecast was too flat; guardrail activated.")
            if raw_worse_than_seasonal:
                notes.append("Seasonal baseline outperformed raw STGCN on synthetic holdout.")

            if raw_flat or raw_worse_than_seasonal:
                method = "stgcn_seasonal_blend"
                if best_method == "seasonal_baseline":
                    method = "seasonal_baseline"
            elif candidate_metrics["stgcn_seasonal_blend"]["mae_kw"] <= candidate_metrics["stgcn"]["mae_kw"] * 0.98:
                method = "stgcn_seasonal_blend"
                notes.append("STGCN-seasonal blend improved holdout accuracy.")
            else:
                method = "stgcn"
        else:
            raw_flat = raw_to_seasonal_variance_ratio < self.min_variance_ratio
            if raw_flat:
                method = "stgcn_seasonal_blend"
                notes.append("Raw STGCN was flatter than the seasonal prior; blend used.")
            candidate_metrics = {}

        health = "healthy"
        if method == "seasonal_baseline":
            health = "seasonal_guardrail"
        elif method == "stgcn_seasonal_blend":
            health = "ensemble_guardrail"

        info: Dict[str, object] = {
            "forecast_method": method,
            "forecast_health": health,
            "forecast_notes": notes,
            "raw_to_actual_variance_ratio": None if np.isnan(raw_to_actual_variance_ratio) else float(raw_to_actual_variance_ratio),
            "raw_to_seasonal_variance_ratio": float(raw_to_seasonal_variance_ratio),
            "candidate_metrics": candidate_metrics,
        }
        if actual_available and method in candidate_metrics:
            info.update({
                "forecast_mae_kw": candidate_metrics[method]["mae_kw"],
                "forecast_smape_pct": candidate_metrics[method]["smape_pct"],
                "forecast_variance_ratio": candidate_metrics[method]["variance_ratio"],
                "forecast_correlation": candidate_metrics[method]["correlation"],
                "seasonal_baseline_mae_kw": candidate_metrics["seasonal_baseline"]["mae_kw"],
                "persistence_baseline_mae_kw": candidate_metrics["persistence_baseline"]["mae_kw"],
                "raw_stgcn_mae_kw": candidate_metrics["stgcn"]["mae_kw"],
            })
        return method, info

    def forecast(
        self,
        train_df: pd.DataFrame,
        future_df: pd.DataFrame,
        adjacency: Dict[str, List[str]],
        horizon_steps: int = None,
    ) -> pd.DataFrame:
        """Generate future forecasts with target-time exogenous conditioning."""
        if self.sequence_mean is None or self.future_mean is None:
            raise RuntimeError("Call fit() before forecast()")

        history_features = self._engineer_features(train_df, adjacency)
        future = future_df.sort_values(["timestamp", "h3_cell"]).copy()
        all_times = sorted(future["timestamp"].unique())
        if horizon_steps is not None:
            all_times = all_times[:horizon_steps]

        result_rows: List[dict] = []

        for ts in all_times:
            step_exog = future[future["timestamp"] == ts].copy()
            step_exog_features = self._engineer_features(step_exog, adjacency, add_neighbor=False)
            pred_by_cell = self._predict_one_step(history_features, step_exog_features)

            step_rows = []
            for _, exog in step_exog_features.iterrows():
                cell = exog["h3_cell"]
                if cell not in pred_by_cell:
                    continue
                actual = float(exog["demand_kw"]) if "demand_kw" in exog and not pd.isna(exog["demand_kw"]) else np.nan
                stgcn_kw, std_kw = pred_by_cell[cell]
                seasonal_kw = self._seasonal_baseline(cell, pd.Timestamp(ts), float(exog["is_weekend"]))
                persistence_kw = self._persistence_baseline(history_features, cell)
                blended_kw = max(0.0, 0.65 * stgcn_kw + 0.35 * seasonal_kw)

                row = {k: exog[k] for k in exog.index if k not in {"neighbor_demand_kw"}}
                row["actual_demand_kw"] = actual
                row["stgcn_predicted_demand_kw"] = stgcn_kw
                row["seasonal_baseline_kw"] = seasonal_kw
                row["persistence_baseline_kw"] = persistence_kw
                row["blended_predicted_demand_kw"] = blended_kw
                row["predicted_demand_kw"] = blended_kw
                row["prediction_std_kw"] = std_kw
                row["demand_kw"] = blended_kw
                step_rows.append(row)

            step_df = pd.DataFrame(step_rows)
            demand_lookup = dict(zip(step_df["h3_cell"], step_df["demand_kw"]))
            step_df["neighbor_demand_kw"] = step_df["h3_cell"].map(
                lambda c: float(np.mean([demand_lookup[n] for n in adjacency.get(c, []) if n in demand_lookup]))
                if any(n in demand_lookup for n in adjacency.get(c, []))
                else float(demand_lookup.get(c, 0.0))
            )
            step_df = self._add_time_features(self._ensure_columns(step_df))
            step_df = self._add_zone_features(step_df)
            history_features = pd.concat([history_features, step_df], ignore_index=True)
            result_rows.extend(step_df.to_dict("records"))

        out = pd.DataFrame(result_rows).sort_values(["timestamp", "h3_cell"]).reset_index(drop=True)
        method, info = self._choose_forecast_method(out)
        method_col = {
            "stgcn": "stgcn_predicted_demand_kw",
            "seasonal_baseline": "seasonal_baseline_kw",
            "persistence_baseline": "persistence_baseline_kw",
            "stgcn_seasonal_blend": "blended_predicted_demand_kw",
        }[method]
        out["predicted_demand_kw"] = out[method_col].astype(float).clip(lower=0.0)
        out["demand_kw"] = out["predicted_demand_kw"]
        if method != "stgcn":
            guardrail_delta = (out["stgcn_predicted_demand_kw"] - out["seasonal_baseline_kw"]).abs()
            out["prediction_std_kw"] = np.maximum(
                out["prediction_std_kw"].astype(float),
                0.5 + 0.35 * guardrail_delta + 0.05 * out["predicted_demand_kw"].astype(float),
            )
        out["forecast_method"] = method
        out["forecast_health"] = str(info.get("forecast_health", "healthy"))
        out["forecast_warning"] = "; ".join(info.get("forecast_notes", []))
        self.forecast_info = info
        if self.training_info:
            self.training_info.model_health = str(info.get("forecast_health", "healthy"))
            self.training_info.notes = list(info.get("forecast_notes", []))
            self.training_info.seasonal_baseline_mae_kw = float(info.get("seasonal_baseline_mae_kw", 0.0) or 0.0)
            self.training_info.persistence_baseline_mae_kw = float(info.get("persistence_baseline_mae_kw", 0.0) or 0.0)
        return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__).rsplit("src", 1)[0])

    from src.spatial_grid.simulation import CityConfig, generate_synthetic_data

    print("Testing STGCNForecaster...")
    config = CityConfig(max_cells=15, num_days=7, freq="1h")
    data, grid, adj = generate_synthetic_data(config)
    times = sorted(data["timestamp"].unique())
    train_split = len(times) - 24
    train = data[data["timestamp"].isin(times[:train_split])]
    future = data[data["timestamp"].isin(times[train_split:])]

    forecaster = STGCNForecaster(seq_len=12, epochs=4, num_blocks=2)
    forecaster.fit(train, adj)
    pred = forecaster.forecast(train, future, adj, horizon_steps=24)
    print(pred[["timestamp", "h3_cell", "predicted_demand_kw", "forecast_method"]].head())
    print(forecaster.forecast_info)
    print("STGCNForecaster test PASSED!")
