"""Forecasting feedback loop for continuous improvement.

Implements a feedback mechanism that:
1. Tracks forecast accuracy over time
2. Identifies systematic biases (under/over-estimation)
3. Detects concept drift
4. Triggers retraining when needed
5. Stores feedback metrics for analysis

This ensures the STGCN model improves over time as more data becomes available.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ForecastMetrics:
    """Metrics for a single forecast evaluation."""
    timestamp: datetime
    horizon_hours: int
    mae: float
    rmse: float
    mape: float
    bias: float  # Positive = overestimation, Negative = underestimation
    variance_ratio: float  # predicted_var / actual_var
    peak_error_pct: float
    samples: int
    by_zone: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_hour: Dict[int, Dict[str, float]] = field(default_factory=dict)


@dataclass
class DriftDetectionResult:
    """Result of concept drift detection."""
    drift_detected: bool
    drift_score: float  # 0-1, higher = more drift
    drift_type: str  # "gradual", "sudden", "none"
    affected_zones: List[str]
    affected_hours: List[int]
    recommendation: str  # "retrain", "fine_tune", "monitor"


@dataclass
class FeedbackLoopConfig:
    """Configuration for the feedback loop."""
    history_window_days: int = 30
    min_samples_for_analysis: int = 100
    retrain_threshold_mae_increase: float = 0.20  # 20% increase triggers retrain
    drift_detection_window_hours: int = 72
    bias_correction_threshold: float = 0.10  # 10% systematic bias
    storage_path: str = "data/feedback"


class ForecastFeedbackLoop:
    """
    Feedback loop for continuous forecasting improvement.
    
    This component sits between the forecaster and the dashboard,
    collecting actual vs predicted values and triggering model
    updates when needed.
    """
    
    def __init__(self, config: FeedbackLoopConfig = None):
        self.config = config or FeedbackLoopConfig()
        self.metrics_history: List[ForecastMetrics] = []
        self.bias_corrections: Dict[str, float] = {}  # zone -> correction factor
        self.hour_corrections: Dict[int, float] = {}  # hour -> correction factor
        self.last_retrain_timestamp: Optional[datetime] = None
        
        # Ensure storage path exists
        os.makedirs(self.config.storage_path, exist_ok=True)
    
    def evaluate_forecast(
        self,
        actual_df: pd.DataFrame,
        predicted_df: pd.DataFrame,
        actual_col: str = "demand_kw",
        predicted_col: str = "predicted_demand_kw",
        cell_col: str = "h3_cell",
        time_col: str = "timestamp",
        zone_col: str = "zone_type",
    ) -> ForecastMetrics:
        """
        Evaluate forecast accuracy and compute metrics.
        
        Args:
            actual_df: DataFrame with actual values
            predicted_df: DataFrame with predictions
            
        Returns:
            ForecastMetrics with detailed error analysis
        """
        # Merge actual and predicted
        merged = predicted_df.merge(
            actual_df[[cell_col, time_col, actual_col]].rename(columns={actual_col: "actual"}),
            on=[cell_col, time_col],
            how="inner",
        )
        
        if len(merged) == 0:
            return ForecastMetrics(
                timestamp=datetime.now(),
                horizon_hours=0,
                mae=float("nan"),
                rmse=float("nan"),
                mape=float("nan"),
                bias=0.0,
                variance_ratio=1.0,
                peak_error_pct=0.0,
                samples=0,
            )
        
        merged["predicted"] = merged[predicted_col]
        merged["error"] = merged["predicted"] - merged["actual"]
        merged["abs_error"] = merged["error"].abs()
        merged["sq_error"] = merged["error"] ** 2
        merged["pct_error"] = merged["abs_error"] / merged["actual"].clip(lower=1)
        
        # Global metrics
        mae = merged["abs_error"].mean()
        rmse = math.sqrt(merged["sq_error"].mean())
        mape = merged["pct_error"].mean() * 100
        bias = merged["error"].mean()
        
        # Variance ratio (predicted variability vs actual)
        pred_var = merged["predicted"].var()
        actual_var = merged["actual"].var()
        variance_ratio = pred_var / max(actual_var, 1e-6)
        
        # Peak error
        actual_peak = merged["actual"].max()
        pred_at_peak_time = merged.loc[merged["actual"].idxmax(), "predicted"]
        peak_error_pct = abs(actual_peak - pred_at_peak_time) / max(actual_peak, 1) * 100
        
        # By zone
        by_zone = {}
        if zone_col in merged.columns:
            for zone in merged[zone_col].unique():
                zone_df = merged[merged[zone_col] == zone]
                by_zone[zone] = {
                    "mae": float(zone_df["abs_error"].mean()),
                    "bias": float(zone_df["error"].mean()),
                    "samples": len(zone_df),
                }
        
        # By hour
        by_hour = {}
        if time_col in merged.columns:
            merged["hour"] = pd.to_datetime(merged[time_col]).dt.hour
            for hour in range(24):
                hour_df = merged[merged["hour"] == hour]
                if len(hour_df) > 0:
                    by_hour[hour] = {
                        "mae": float(hour_df["abs_error"].mean()),
                        "bias": float(hour_df["error"].mean()),
                        "samples": len(hour_df),
                    }
        
        # Compute horizon
        timestamps = pd.to_datetime(merged[time_col])
        horizon_hours = int((timestamps.max() - timestamps.min()).total_seconds() / 3600)
        
        metrics = ForecastMetrics(
            timestamp=datetime.now(),
            horizon_hours=horizon_hours,
            mae=float(mae),
            rmse=float(rmse),
            mape=float(mape),
            bias=float(bias),
            variance_ratio=float(variance_ratio),
            peak_error_pct=float(peak_error_pct),
            samples=len(merged),
            by_zone=by_zone,
            by_hour=by_hour,
        )
        
        self.metrics_history.append(metrics)
        self._save_metrics(metrics)
        
        return metrics
    
    def detect_concept_drift(self) -> DriftDetectionResult:
        """
        Detect concept drift by comparing recent vs historical performance.
        
        Uses multiple indicators:
        - MAE trend
        - Bias shift
        - Variance ratio change
        """
        if len(self.metrics_history) < 2:
            return DriftDetectionResult(
                drift_detected=False,
                drift_score=0.0,
                drift_type="none",
                affected_zones=[],
                affected_hours=[],
                recommendation="monitor",
            )
        
        # Split into recent vs historical
        window = min(len(self.metrics_history), self.config.drift_detection_window_hours // 24)
        recent = self.metrics_history[-window:]
        historical = self.metrics_history[:-window] if len(self.metrics_history) > window else recent
        
        # MAE trend
        recent_mae = np.mean([m.mae for m in recent])
        historical_mae = np.mean([m.mae for m in historical]) if historical else recent_mae
        mae_ratio = recent_mae / max(historical_mae, 1e-6)
        
        # Bias shift
        recent_bias = np.mean([m.bias for m in recent])
        historical_bias = np.mean([m.bias for m in historical]) if historical else 0
        bias_shift = abs(recent_bias - historical_bias)
        
        # Variance ratio change
        recent_var = np.mean([m.variance_ratio for m in recent])
        historical_var = np.mean([m.variance_ratio for m in historical]) if historical else 1.0
        var_change = abs(recent_var - historical_var)
        
        # Composite drift score
        drift_score = (
            0.4 * max(0, mae_ratio - 1) +
            0.3 * min(1, bias_shift / 10) +
            0.3 * min(1, var_change)
        )
        
        # Identify affected zones
        affected_zones = []
        for m in recent:
            for zone, stats in m.by_zone.items():
                historical_zone_bias = np.mean([
                    hist.by_zone.get(zone, {}).get("bias", 0)
                    for hist in historical if zone in hist.by_zone
                ]) if historical else 0
                if abs(stats["bias"] - historical_zone_bias) > self.config.bias_correction_threshold * 100:
                    if zone not in affected_zones:
                        affected_zones.append(zone)
        
        # Identify affected hours
        affected_hours = []
        for m in recent:
            for hour, stats in m.by_hour.items():
                historical_hour_bias = np.mean([
                    hist.by_hour.get(hour, {}).get("bias", 0)
                    for hist in historical if hour in hist.by_hour
                ]) if historical else 0
                if abs(stats["bias"] - historical_hour_bias) > self.config.bias_correction_threshold * 100:
                    if hour not in affected_hours:
                        affected_hours.append(hour)
        
        # Determine drift type and recommendation
        drift_detected = drift_score > 0.15
        
        if drift_score > 0.4:
            drift_type = "sudden"
            recommendation = "retrain"
        elif drift_score > 0.15:
            drift_type = "gradual"
            recommendation = "fine_tune"
        else:
            drift_type = "none"
            recommendation = "monitor"
        
        return DriftDetectionResult(
            drift_detected=drift_detected,
            drift_score=float(drift_score),
            drift_type=drift_type,
            affected_zones=affected_zones,
            affected_hours=affected_hours,
            recommendation=recommendation,
        )
    
    def compute_bias_corrections(self) -> Tuple[Dict[str, float], Dict[int, float]]:
        """
        Compute bias correction factors for zones and hours.
        
        These can be applied to predictions to reduce systematic errors.
        """
        if len(self.metrics_history) < 3:
            return {}, {}
        
        # Use recent metrics
        window = min(len(self.metrics_history), 7)
        recent = self.metrics_history[-window:]
        
        # Zone corrections
        zone_biases: Dict[str, List[float]] = {}
        for m in recent:
            for zone, stats in m.by_zone.items():
                if zone not in zone_biases:
                    zone_biases[zone] = []
                zone_biases[zone].append(stats["bias"])
        
        zone_corrections = {}
        for zone, biases in zone_biases.items():
            avg_bias = np.mean(biases)
            if abs(avg_bias) > self.config.bias_correction_threshold:
                zone_corrections[zone] = -avg_bias
        
        # Hour corrections
        hour_biases: Dict[int, List[float]] = {}
        for m in recent:
            for hour, stats in m.by_hour.items():
                if hour not in hour_biases:
                    hour_biases[hour] = []
                hour_biases[hour].append(stats["bias"])
        
        hour_corrections = {}
        for hour, biases in hour_biases.items():
            avg_bias = np.mean(biases)
            if abs(avg_bias) > self.config.bias_correction_threshold:
                hour_corrections[hour] = -avg_bias
        
        self.bias_corrections = zone_corrections
        self.hour_corrections = hour_corrections
        
        return zone_corrections, hour_corrections
    
    def apply_corrections(
        self,
        df: pd.DataFrame,
        predicted_col: str = "predicted_demand_kw",
        zone_col: str = "zone_type",
        time_col: str = "timestamp",
    ) -> pd.DataFrame:
        """
        Apply bias corrections to predictions.
        """
        df = df.copy()
        
        if not self.bias_corrections and not self.hour_corrections:
            return df
        
        # Apply zone corrections
        if self.bias_corrections and zone_col in df.columns:
            for zone, correction in self.bias_corrections.items():
                mask = df[zone_col] == zone
                df.loc[mask, predicted_col] += correction
        
        # Apply hour corrections
        if self.hour_corrections and time_col in df.columns:
            df["_hour"] = pd.to_datetime(df[time_col]).dt.hour
            for hour, correction in self.hour_corrections.items():
                mask = df["_hour"] == hour
                df.loc[mask, predicted_col] += correction
            df.drop("_hour", axis=1, inplace=True)
        
        # Ensure non-negative
        df[predicted_col] = df[predicted_col].clip(lower=0)
        
        return df
    
    def should_retrain(self) -> Tuple[bool, str]:
        """
        Determine if the model should be retrained.
        
        Returns:
            (should_retrain, reason)
        """
        if len(self.metrics_history) < 5:
            return False, "Insufficient history"
        
        # Check MAE trend
        recent = self.metrics_history[-5:]
        baseline = self.metrics_history[:5] if len(self.metrics_history) > 10 else self.metrics_history[:len(self.metrics_history)//2]
        
        recent_mae = np.mean([m.mae for m in recent])
        baseline_mae = np.mean([m.mae for m in baseline])
        
        mae_increase = (recent_mae - baseline_mae) / max(baseline_mae, 1e-6)
        
        if mae_increase > self.config.retrain_threshold_mae_increase:
            return True, f"MAE increased by {mae_increase*100:.1f}%"
        
        # Check drift
        drift = self.detect_concept_drift()
        if drift.recommendation == "retrain":
            return True, f"Concept drift detected: {drift.drift_type}"
        
        # Check bias
        if self.metrics_history[-1].bias > 50:  # 50 kW systematic overestimation
            return True, f"High systematic bias: {self.metrics_history[-1].bias:.1f} kW"
        
        return False, "Model performance acceptable"
    
    def get_performance_report(self) -> Dict:
        """Generate a performance report for the dashboard."""
        if not self.metrics_history:
            return {"status": "no_data", "message": "No forecast evaluations yet"}
        
        recent = self.metrics_history[-min(len(self.metrics_history), 7):]
        
        return {
            "status": "active",
            "evaluations": len(self.metrics_history),
            "recent_metrics": {
                "mae_kw": round(np.mean([m.mae for m in recent]), 2),
                "rmse_kw": round(np.mean([m.rmse for m in recent]), 2),
                "mape_pct": round(np.mean([m.mape for m in recent]), 2),
                "bias_kw": round(np.mean([m.bias for m in recent]), 2),
                "peak_error_pct": round(np.mean([m.peak_error_pct for m in recent]), 2),
            },
            "drift_status": self.detect_concept_drift().__dict__,
            "bias_corrections": {
                "zone": self.bias_corrections,
                "hour": {k: round(v, 2) for k, v in self.hour_corrections.items()},
            },
            "retrain_check": {
                "should_retrain": self.should_retrain()[0],
                "reason": self.should_retrain()[1],
            },
            "last_updated": datetime.now().isoformat(),
        }
    
    def _save_metrics(self, metrics: ForecastMetrics):
        """Save metrics to storage."""
        filepath = Path(self.config.storage_path) / "metrics_history.jsonl"
        
        record = {
            "timestamp": metrics.timestamp.isoformat(),
            "horizon_hours": metrics.horizon_hours,
            "mae": metrics.mae,
            "rmse": metrics.rmse,
            "mape": metrics.mape,
            "bias": metrics.bias,
            "variance_ratio": metrics.variance_ratio,
            "peak_error_pct": metrics.peak_error_pct,
            "samples": metrics.samples,
            "by_zone": metrics.by_zone,
            "by_hour": {str(k): v for k, v in metrics.by_hour.items()},
        }
        
        with open(filepath, "a") as f:
            f.write(json.dumps(record) + "\n")
    
    def load_history(self):
        """Load metrics history from storage."""
        filepath = Path(self.config.storage_path) / "metrics_history.jsonl"
        
        if not filepath.exists():
            return
        
        self.metrics_history = []
        with open(filepath, "r") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    metrics = ForecastMetrics(
                        timestamp=datetime.fromisoformat(record["timestamp"]),
                        horizon_hours=record["horizon_hours"],
                        mae=record["mae"],
                        rmse=record["rmse"],
                        mape=record["mape"],
                        bias=record["bias"],
                        variance_ratio=record["variance_ratio"],
                        peak_error_pct=record["peak_error_pct"],
                        samples=record["samples"],
                        by_zone=record.get("by_zone", {}),
                        by_hour={int(k): v for k, v in record.get("by_hour", {}).items()},
                    )
                    self.metrics_history.append(metrics)


class OnlineForecasterWrapper:
    """
    Wrapper that adds feedback loop to any forecaster.
    
    Automatically tracks performance and applies corrections.
    """
    
    def __init__(self, forecaster, feedback_config: FeedbackLoopConfig = None):
        self.forecaster = forecaster
        self.feedback_loop = ForecastFeedbackLoop(feedback_config)
        self.feedback_loop.load_history()
    
    def fit(self, train_df: pd.DataFrame, adjacency: Dict, **kwargs):
        """Fit the underlying forecaster."""
        return self.forecaster.fit(train_df, adjacency, **kwargs)
    
    def forecast(
        self,
        train_df: pd.DataFrame,
        future_df: pd.DataFrame,
        adjacency: Dict,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generate forecast with automatic bias correction.
        """
        # Get base forecast
        pred_df = self.forecaster.forecast(train_df, future_df, adjacency, **kwargs)
        
        # Apply learned corrections
        self.feedback_loop.compute_bias_corrections()
        pred_df = self.feedback_loop.apply_corrections(pred_df)
        
        return pred_df
    
    def update_with_actuals(
        self,
        actual_df: pd.DataFrame,
        predicted_df: pd.DataFrame,
    ) -> ForecastMetrics:
        """
        Update feedback loop with actual values.
        
        Call this when actual demand becomes available.
        """
        metrics = self.feedback_loop.evaluate_forecast(actual_df, predicted_df)
        
        # Check if retraining needed
        should_retrain, reason = self.feedback_loop.should_retrain()
        if should_retrain:
            print(f"[FeedbackLoop] Retraining recommended: {reason}")
        
        return metrics
    
    def get_performance_report(self) -> Dict:
        """Get performance report for dashboard."""
        return self.feedback_loop.get_performance_report()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__).rsplit("src", 1)[0])
    
    from src.spatial_grid.simulation import CityConfig, generate_synthetic_data
    
    print("Testing feedback loop...")
    
    config = CityConfig(max_cells=15, num_days=5, freq="1h")
    data, grid, adj = generate_synthetic_data(config)
    
    # Simulate predictions with some error
    data["predicted_demand_kw"] = data["demand_kw"] * (1 + np.random.normal(0, 0.15, len(data)))
    data["predicted_demand_kw"] = data["predicted_demand_kw"].clip(lower=0)
    
    # Create feedback loop
    feedback = ForecastFeedbackLoop()
    
    # Split data into chunks and evaluate
    timestamps = sorted(data["timestamp"].unique())
    chunk_size = len(timestamps) // 5
    
    for i in range(5):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        chunk_times = timestamps[start_idx:end_idx]
        
        chunk_actual = data[data["timestamp"].isin(chunk_times)]
        chunk_pred = chunk_actual.copy()
        
        # Add some bias to later chunks to simulate drift
        if i >= 3:
            chunk_pred["predicted_demand_kw"] += 20  # Systematic overestimation
        
        metrics = feedback.evaluate_forecast(chunk_actual, chunk_pred)
        print(f"\n=== Chunk {i+1} Metrics ===")
        print(f"  MAE: {metrics.mae:.2f} kW")
        print(f"  Bias: {metrics.bias:.2f} kW")
        print(f"  MAPE: {metrics.mape:.2f}%")
    
    # Check for drift
    drift = feedback.detect_concept_drift()
    print(f"\n=== Drift Detection ===")
    print(f"  Detected: {drift.drift_detected}")
    print(f"  Score: {drift.drift_score:.3f}")
    print(f"  Type: {drift.drift_type}")
    print(f"  Recommendation: {drift.recommendation}")
    
    # Compute corrections
    zone_corr, hour_corr = feedback.compute_bias_corrections()
    print(f"\n=== Bias Corrections ===")
    print(f"  Zone corrections: {zone_corr}")
    print(f"  Hour corrections: {dict(list(hour_corr.items())[:5])}...")
    
    # Check retrain
    should_retrain, reason = feedback.should_retrain()
    print(f"\n=== Retrain Check ===")
    print(f"  Should retrain: {should_retrain}")
    print(f"  Reason: {reason}")
    
    # Performance report
    report = feedback.get_performance_report()
    print(f"\n=== Performance Report ===")
    print(f"  Status: {report['status']}")
    print(f"  Recent MAE: {report['recent_metrics']['mae_kw']} kW")
    
    print("\nFeedback loop test PASSED!")
