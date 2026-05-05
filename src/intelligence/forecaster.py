"""STGCN-based demand forecaster for Vidyut Prajna.

Provides a high-level API for training and forecasting EV charging demand
using Spatio-Temporal Graph Convolutional Networks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import STGCNBlock, VidyutPrajnaForecaster
from .graph_utils import get_adjacency_matrix


# Features used for prediction
FEATURE_COLS = [
    "demand_kw",
    "traffic_intensity",
    "temperature_c",
    "rainfall_mm",
    "grid_base_load_kw",
    "neighbor_demand_kw",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "ev_adoption_index",
    "charger_density_index",
    "tariff_multiplier",
    "solar_generation_kw",
]


@dataclass
class TrainingInfo:
    """Training metrics and metadata."""
    train_samples: int
    val_samples: int
    epochs: int
    final_train_loss: float
    final_val_loss: float
    
    @property
    def final_loss(self) -> float:
        return self.final_val_loss


class STGCNForecaster:
    """High-level STGCN forecaster for EV demand prediction.
    
    Wraps the VidyutPrajnaForecaster model with data processing,
    training, and inference capabilities.
    """
    
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
        
        self.model: Optional[VidyutPrajnaForecaster] = None
        self.cells: List[str] = []
        self.cell_to_idx: Dict[str, int] = {}
        self.adj_tensor: Optional[torch.Tensor] = None
        
        self.feature_mean: Optional[pd.Series] = None
        self.feature_std: Optional[pd.Series] = None
        self.target_mean: float = 0.0
        self.target_std: float = 1.0
        self.training_info: Optional[TrainingInfo] = None
    
    # ------------------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------------------
    
    @staticmethod
    def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical time features."""
        out = df.copy()
        hours = out["timestamp"].dt.hour + out["timestamp"].dt.minute / 60.0
        out["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
        dow = out["timestamp"].dt.dayofweek.astype(float)
        out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
        if "is_weekend" not in out.columns:
            out["is_weekend"] = (out["timestamp"].dt.dayofweek >= 5).astype(float)
        else:
            out["is_weekend"] = out["is_weekend"].astype(float)
        return out
    
    @staticmethod
    def _add_neighbor_demand(df: pd.DataFrame, adjacency: Dict[str, List[str]]) -> pd.DataFrame:
        """Add average neighbor demand feature."""
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
        """Fill missing feature columns with defaults."""
        defaults = {
            "tariff_multiplier": 1.0,
            "solar_generation_kw": 0.0,
            "ev_adoption_index": 0.75,
            "charger_density_index": 5.0,
            "is_weekend": 0.0,
            "traffic_intensity": 0.5,
            "temperature_c": 28.0,
            "rainfall_mm": 0.0,
            "grid_base_load_kw": 100.0,
        }
        for col, default in defaults.items():
            if col not in df.columns:
                df[col] = default
        return df
    
    def _engineer_features(self, df: pd.DataFrame, adjacency: Dict[str, List[str]]) -> pd.DataFrame:
        """Full feature engineering pipeline."""
        out = self._ensure_columns(df.copy())
        out = self._add_neighbor_demand(out, adjacency)
        out = self._add_time_features(out)
        return out
    
    # ------------------------------------------------------------------
    # Sequence Building
    # ------------------------------------------------------------------
    
    def _build_sequences(self, frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Build sequences with spatial structure: (batch, time, nodes, features)."""
        timestamps = sorted(frame["timestamp"].unique())
        n_cells = len(self.cells)
        
        X_parts: List[np.ndarray] = []
        y_parts: List[np.ndarray] = []
        
        for i in range(self.seq_len, len(timestamps)):
            seq_times = timestamps[i - self.seq_len:i]
            target_time = timestamps[i]
            
            seq_data = np.zeros((self.seq_len, n_cells, len(FEATURE_COLS)), dtype=np.float32)
            target_data = np.zeros(n_cells, dtype=np.float32)
            
            for t_idx, ts in enumerate(seq_times):
                ts_data = frame[frame["timestamp"] == ts]
                for _, row in ts_data.iterrows():
                    cell = row["h3_cell"]
                    if cell in self.cell_to_idx:
                        c_idx = self.cell_to_idx[cell]
                        seq_data[t_idx, c_idx, :] = row[FEATURE_COLS].values.astype(np.float32)
            
            target_ts_data = frame[frame["timestamp"] == target_time]
            for _, row in target_ts_data.iterrows():
                cell = row["h3_cell"]
                if cell in self.cell_to_idx:
                    c_idx = self.cell_to_idx[cell]
                    target_data[c_idx] = row["demand_kw"]
            
            X_parts.append(seq_data)
            y_parts.append(target_data)
        
        if not X_parts:
            raise ValueError("Not enough data to build sequences. Increase num_days or reduce seq_len.")
        
        return np.stack(X_parts), np.stack(y_parts)
    
    def _normalize_X(self, X: np.ndarray) -> np.ndarray:
        """Normalize input features."""
        mean = self.feature_mean[FEATURE_COLS].to_numpy(dtype=np.float32)
        std = self.feature_std[FEATURE_COLS].to_numpy(dtype=np.float32)
        return (X - mean) / std
    
    def _normalize_y(self, y: np.ndarray) -> np.ndarray:
        """Normalize targets."""
        return (y - self.target_mean) / self.target_std
    
    def _denormalize_y(self, y: np.ndarray) -> np.ndarray:
        """Denormalize predictions."""
        return y * self.target_std + self.target_mean
    
    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    
    def fit(self, train_df: pd.DataFrame, adjacency: Dict[str, List[str]]) -> "STGCNForecaster":
        """Train the STGCN model on historical data."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        # Setup cell indexing
        self.cells = sorted(train_df["h3_cell"].unique().tolist())
        self.cell_to_idx = {c: i for i, c in enumerate(self.cells)}
        n_cells = len(self.cells)
        
        # Build adjacency matrix
        self.adj_tensor = get_adjacency_matrix(self.cells).to(self.device)
        
        # Initialize model
        self.model = VidyutPrajnaForecaster(
            adj_matrix=self.adj_tensor,
            in_channels=len(FEATURE_COLS),
            hidden_channels=self.hidden_size,
            out_channels=1,
            num_blocks=self.num_blocks,
            dropout=self.dropout,
        ).to(self.device)
        
        # Feature engineering
        frame = self._engineer_features(train_df, adjacency)
        self.feature_mean = frame[FEATURE_COLS].mean()
        self.feature_std = frame[FEATURE_COLS].std().fillna(1.0).clip(lower=0.5)
        self.target_mean = float(frame["demand_kw"].mean())
        self.target_std = float(frame["demand_kw"].std() or 1.0)
        
        # Build sequences
        X_raw, y_raw = self._build_sequences(frame)
        X = self._normalize_X(X_raw)
        y = self._normalize_y(y_raw)
        
        # Train/val split
        n = len(X)
        n_val = max(1, int(n * self.val_fraction))
        indices = np.arange(n)
        np.random.shuffle(indices)
        val_idx, train_idx = indices[:n_val], indices[n_val:]
        
        train_ds = TensorDataset(
            torch.tensor(X[train_idx], dtype=torch.float32),
            torch.tensor(y[train_idx], dtype=torch.float32)
        )
        val_ds = TensorDataset(
            torch.tensor(X[val_idx], dtype=torch.float32),
            torch.tensor(y[val_idx], dtype=torch.float32)
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
            # Train
            self.model.train()
            losses = []
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                pred = self.model(bx).squeeze(-1)
                loss = loss_fn(pred, by)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            final_train_loss = float(np.mean(losses))
            
            # Validate
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    pred = self.model(bx).squeeze(-1)
                    val_losses.append(float(loss_fn(pred, by).cpu()))
            final_val_loss = float(np.mean(val_losses)) if val_losses else final_train_loss
            scheduler.step(final_val_loss)
            
            # Early stopping
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
    
    def _predict_one_step(self, history_features: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        """Predict next timestep for all cells. Returns {cell: (mean, std)}."""
        self.model.eval()
        
        # Uncertainty scaling from training loss
        loss_scale = 0.10
        if self.training_info:
            loss_scale = float(np.clip(self.training_info.final_loss * 0.5, 0.05, 0.30))
        
        timestamps = sorted(history_features["timestamp"].unique())
        if len(timestamps) < self.seq_len:
            return {}
        
        seq_times = timestamps[-self.seq_len:]
        n_cells = len(self.cells)
        
        seq_data = np.zeros((1, self.seq_len, n_cells, len(FEATURE_COLS)), dtype=np.float32)
        
        for t_idx, ts in enumerate(seq_times):
            ts_data = history_features[history_features["timestamp"] == ts]
            for _, row in ts_data.iterrows():
                cell = row["h3_cell"]
                if cell in self.cell_to_idx:
                    c_idx = self.cell_to_idx[cell]
                    seq_data[0, t_idx, c_idx, :] = row[FEATURE_COLS].values.astype(np.float32)
        
        X = self._normalize_X(seq_data)
        
        predictions: Dict[str, Tuple[float, float]] = {}
        with torch.no_grad():
            xt = torch.tensor(X, dtype=torch.float32).to(self.device)
            p = self.model(xt).squeeze(-1).cpu().numpy()[0]
            p_denorm = self._denormalize_y(p)
            
            for cell, c_idx in self.cell_to_idx.items():
                mean_kw = max(0.0, float(p_denorm[c_idx]))
                std_kw = max(0.5, mean_kw * loss_scale)
                predictions[cell] = (mean_kw, std_kw)
        
        return predictions
    
    def forecast(
        self,
        train_df: pd.DataFrame,
        future_df: pd.DataFrame,
        adjacency: Dict[str, List[str]],
        horizon_steps: int = None,
    ) -> pd.DataFrame:
        """Generate forecasts for future timesteps.
        
        Args:
            train_df: Historical data used for context
            future_df: Future exogenous features (weather, tariff, etc.)
            adjacency: H3 cell adjacency dict
            horizon_steps: Number of steps to forecast (default: all in future_df)
            
        Returns:
            DataFrame with predictions and uncertainty estimates
        """
        if self.feature_mean is None:
            raise RuntimeError("Call fit() before forecast()")
        
        history_features = self._engineer_features(train_df, adjacency)
        future = future_df.sort_values(["timestamp", "h3_cell"]).copy()
        all_times = sorted(future["timestamp"].unique())
        
        if horizon_steps is not None:
            all_times = all_times[:horizon_steps]
        
        result_rows: List[dict] = []
        
        for ts in all_times:
            step_exog = future[future["timestamp"] == ts].copy()
            pred_by_cell = self._predict_one_step(history_features)
            
            if not pred_by_cell:
                raise RuntimeError("Unable to produce predictions; insufficient history.")
            
            step_rows = []
            for _, exog in step_exog.iterrows():
                cell = exog["h3_cell"]
                if cell not in pred_by_cell:
                    continue
                
                actual = float(exog["demand_kw"]) if "demand_kw" in exog and not pd.isna(exog["demand_kw"]) else np.nan
                row = {k: exog[k] for k in exog.index if k not in {"neighbor_demand_kw"}}
                mean_kw, std_kw = pred_by_cell[cell]
                
                row["actual_demand_kw"] = actual
                row["predicted_demand_kw"] = mean_kw
                row["prediction_std_kw"] = std_kw
                row["demand_kw"] = mean_kw
                step_rows.append(row)
            
            step_df = pd.DataFrame(step_rows)
            demand_lookup = dict(zip(step_df["h3_cell"], step_df["demand_kw"]))
            step_df["neighbor_demand_kw"] = step_df["h3_cell"].map(
                lambda c: float(np.mean([demand_lookup[n] for n in adjacency.get(c, []) if n in demand_lookup]))
                if any(n in demand_lookup for n in adjacency.get(c, []))
                else float(demand_lookup.get(c, 0))
            )
            step_df = self._add_time_features(self._ensure_columns(step_df))
            history_features = pd.concat([history_features, step_df], ignore_index=True)
            result_rows.extend(step_df.to_dict("records"))
        
        out = pd.DataFrame(result_rows).sort_values(["timestamp", "h3_cell"]).reset_index(drop=True)
        return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__).rsplit("src", 1)[0])
    
    from src.spatial_grid.simulation import CityConfig, generate_synthetic_data
    
    print("Testing STGCNForecaster...")
    
    config = CityConfig(max_cells=15, num_days=3, freq="1h")
    data, grid, adj = generate_synthetic_data(config)
    
    times = sorted(data["timestamp"].unique())
    train_split = int(len(times) * 0.7)
    train = data[data["timestamp"].isin(times[:train_split])]
    future = data[data["timestamp"].isin(times[train_split:train_split + 12])]
    
    print(f"Training on {len(train)} rows, forecasting {len(future)} rows")
    
    forecaster = STGCNForecaster(seq_len=6, epochs=5, num_blocks=2)
    forecaster.fit(train, adj)
    
    print(f"\nTraining info:")
    print(f"  Epochs: {forecaster.training_info.epochs}")
    print(f"  Train loss: {forecaster.training_info.final_train_loss:.4f}")
    print(f"  Val loss: {forecaster.training_info.final_val_loss:.4f}")
    
    pred = forecaster.forecast(train, future, adj, horizon_steps=6)
    print(f"\nForecast: {len(pred)} rows")
    print(pred[["timestamp", "h3_cell", "predicted_demand_kw", "prediction_std_kw"]].head(10))
    
    print("\nSTGCNForecaster test PASSED!")
