"""Vidyut Prajna dashboard - Plotly Dash planning console.

Run:
    python -m src.dashboard.app
Or:
    python main.py

Then open: http://127.0.0.1:8050
"""

from __future__ import annotations

import os
import sys
import hashlib
import json
from typing import Dict, List
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - dotenv is optional at import time
    def load_dotenv(*_: object, **__: object) -> bool:
        return False


load_dotenv()

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html
from dash.exceptions import PreventUpdate

from src.dashboard.llm_interface import VidyutLLM
from src.dashboard.utils import (
    aggregate_load_timeseries,
    build_geojson,
    build_llm_context,
    format_inr,
    format_kw,
    format_pct,
)
from src.intelligence.forecaster import STGCNForecaster
from src.optimization.optimizer import optimize_charging_schedule
from src.optimization.siting import recommend_station_locations
from src.spatial_grid.simulation import CityConfig, generate_synthetic_data


DEFAULT_CENTER = {"lat": 12.9716, "lon": 77.5946}


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_bool(name: str, default: bool = True) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}


def bootstrap_demo() -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]], pd.DataFrame, Dict[str, object], pd.DataFrame, Dict[str, object], pd.DataFrame]:
    """Generate data, train model, forecast, optimize, and rank station sites."""
    print("Initializing Vidyut Prajna planning console...")

    fast_dashboard = env_bool("DASHBOARD_FAST", True)
    max_cells_default = min(env_int("MAX_CELLS", 54), env_int("DASHBOARD_MAX_CELLS", 36)) if fast_dashboard else env_int("MAX_CELLS", 54)
    num_days_default = min(env_int("NUM_DAYS", 7), env_int("DASHBOARD_NUM_DAYS", 5)) if fast_dashboard else env_int("NUM_DAYS", 7)

    config = CityConfig(
        h3_resolution=env_int("H3_RESOLUTION", 8),
        max_cells=max_cells_default,
        num_days=num_days_default,
        freq=os.getenv("FREQ", "1h"),
        seed=env_int("SEED", 42),
        start=os.getenv("SIM_START", "2026-05-01"),
        scenario=os.getenv("SCENARIO", "orr_whitefield"),
    )
    station_budget = env_int("STATION_BUDGET", 8)
    station_kw = float(os.getenv("STATION_KW", "22.0"))
    requested_epochs = env_int("EPOCHS", 10)
    dashboard_epochs = min(requested_epochs, env_int("DASHBOARD_EPOCHS", 3)) if fast_dashboard else requested_epochs

    cache_payload = {
        "version": "dashboard_v5_aggregate_peak_metrics",
        "fast_dashboard": fast_dashboard,
        "config": config.__dict__,
        "train_steps": os.getenv("TRAIN_STEPS", ""),
        "forecast_steps": os.getenv("FORECAST_STEPS", ""),
        "seq_len": os.getenv("SEQ_LEN", ""),
        "hidden_size": os.getenv("HIDDEN_SIZE", "48"),
        "epochs": dashboard_epochs,
        "stgcn_blocks": os.getenv("STGCN_BLOCKS", "2"),
        "station_budget": station_budget,
        "station_kw": station_kw,
    }
    cache_key = hashlib.sha256(json.dumps(cache_payload, sort_keys=True, default=str).encode()).hexdigest()[:16]
    cache_dir = Path(PROJECT_ROOT) / "data" / "cache"
    cache_path = cache_dir / f"dashboard_{cache_key}.pkl"
    if env_bool("DASHBOARD_CACHE", True) and cache_path.exists():
        try:
            print(f"Loading cached dashboard scenario: {cache_path.name}")
            cached = pd.read_pickle(cache_path)
            if len(cached) >= 8 and isinstance(cached[4], dict) and "local_transformer_peak_before_kw" in cached[4]:
                return cached
            print("Cached dashboard scenario is from an older metric schema; rebuilding.")
        except Exception as exc:
            print(f"Cache read failed; rebuilding scenario ({exc}).")

    print(
        f"Generating synthetic data: {config.scenario}, "
        f"{config.max_cells} cells, {config.num_days} days..."
    )
    raw_df, grid_df, adjacency = generate_synthetic_data(config)

    unique_times = sorted(raw_df["timestamp"].unique())
    default_horizon_steps = min(24, max(4, len(unique_times) // 4))
    default_train_steps = max(24, len(unique_times) - default_horizon_steps)
    train_steps = min(env_int("TRAIN_STEPS", default_train_steps), len(unique_times) - 4)
    horizon_steps = min(env_int("FORECAST_STEPS", default_horizon_steps), len(unique_times) - train_steps)
    if horizon_steps <= 0:
        raise RuntimeError("Not enough simulated data. Increase NUM_DAYS or reduce TRAIN_STEPS.")

    train_times = unique_times[:train_steps]
    future_times = unique_times[train_steps:train_steps + horizon_steps]
    train_df = raw_df[raw_df["timestamp"].isin(train_times)].copy()
    future_df = raw_df[raw_df["timestamp"].isin(future_times)].copy()

    seq_len = min(env_int("SEQ_LEN", 12), max(3, len(train_times) // 3))
    print(f"Training STGCN model: {len(train_df)} samples, seq_len={seq_len}...")
    forecaster = STGCNForecaster(
        seq_len=seq_len,
        hidden_size=env_int("HIDDEN_SIZE", 48),
        epochs=dashboard_epochs,
        num_blocks=env_int("STGCN_BLOCKS", 2),
        seed=config.seed,
    )
    forecaster.fit(train_df, adjacency)

    print(f"Forecasting {horizon_steps} steps...")
    pred_df = forecaster.forecast(train_df, future_df, adjacency, horizon_steps=horizon_steps)
    forecast_info = forecaster.forecast_info.copy()

    print("Optimizing charging schedule...")
    optimized_df, metrics = optimize_charging_schedule(pred_df)
    metrics.update(forecast_info)

    if forecaster.training_info:
        metrics["training_samples"] = forecaster.training_info.train_samples
        metrics["training_final_loss"] = forecaster.training_info.final_loss
        metrics["training_epochs"] = forecaster.training_info.epochs

    if {"actual_demand_kw", "predicted_demand_kw"}.issubset(pred_df.columns) and "forecast_mae_kw" not in metrics:
        actual = pred_df["actual_demand_kw"].astype(float)
        predicted = pred_df["predicted_demand_kw"].astype(float)
        metrics["forecast_mae_kw"] = float((actual - predicted).abs().mean())
    if "prediction_std_kw" in pred_df.columns:
        metrics["mean_prediction_std_kw"] = float(pred_df["prediction_std_kw"].mean())

    print("Ranking infrastructure locations...")
    recommendations, siting_summary = recommend_station_locations(
        optimized_df,
        adjacency=adjacency,
        top_n=station_budget,
        station_kw=station_kw,
    )
    siting_all, _ = recommend_station_locations(
        optimized_df,
        adjacency=adjacency,
        top_n=len(grid_df),
        station_kw=station_kw,
    )

    print("Dashboard ready.")
    result = raw_df, grid_df, adjacency, optimized_df, metrics, recommendations, siting_summary, siting_all
    if env_bool("DASHBOARD_CACHE", True):
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            pd.to_pickle(result, cache_path)
            print(f"Cached dashboard scenario: {cache_path.name}")
        except Exception as exc:
            print(f"Cache write failed; continuing without cache ({exc}).")
    return result


RAW_DF, GRID_DF, ADJACENCY, OPTIMIZED_DF, METRICS, RECOMMENDATIONS, SITING_SUMMARY, SITING_ALL = bootstrap_demo()
GEOJSON = build_geojson(GRID_DF)
AGG_TS = aggregate_load_timeseries(OPTIMIZED_DF)
TIME_VALUES = sorted(OPTIMIZED_DF["timestamp"].unique())
MAP_CENTER = {
    "lat": float(GRID_DF["lat"].mean()) if "lat" in GRID_DF else DEFAULT_CENTER["lat"],
    "lon": float(GRID_DF["lon"].mean()) if "lon" in GRID_DF else DEFAULT_CENTER["lon"],
}
LLM = VidyutLLM()
SITING_BY_CELL = dict(zip(SITING_ALL["h3_cell"], SITING_ALL["siting_score"]))

GRAPH_CONFIG = {"displayModeBar": False, "responsive": True}
SYSTEM_POSTURE = {
    "Demand forecast": "Time and H3-zone EV demand prediction",
    "Charging schedule": "Robust LP recommendation under transformer constraints",
    "Station siting": "Graph-aware portfolio planning vs uniform baseline",
    "Data privacy": "Synthetic or masked aggregates only",
    "Deployment": "Read-only decision-support sidecar",
    "Sensitive data": "No hosted LLM dependence required",
}


def make_slider_marks() -> Dict[int, str]:
    marks: Dict[int, str] = {}
    step = max(1, len(TIME_VALUES) // 8)
    for idx in range(0, len(TIME_VALUES), step):
        marks[idx] = pd.Timestamp(TIME_VALUES[idx]).strftime("%H:%M")
    marks[len(TIME_VALUES) - 1] = pd.Timestamp(TIME_VALUES[-1]).strftime("%H:%M")
    return marks


def selected_time(time_index: int) -> pd.Timestamp:
    return pd.Timestamp(TIME_VALUES[int(np.clip(time_index, 0, len(TIME_VALUES) - 1))])


def info_tip(text: str, icon: str = "ⓘ") -> html.Span:
    """Create an info icon with hover tooltip."""
    return html.Span(
        icon,
        className="info-tip",
        title=text,
    )


def _plot_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#ffffff",
        font=dict(color="#172033", family="Inter, Segoe UI, Arial, sans-serif", size=11),
        margin={"l": 50, "r": 50, "t": 40, "b": 40},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
            font=dict(size=10),
            itemsizing="constant",
        ),
        title=None,
        hovermode="x unified",
        transition={"duration": 0},
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eef2f7", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eef2f7", zeroline=False)
    return fig


def make_map_figure(time_index: int, mode: str) -> go.Figure:
    ts = selected_time(time_index)
    frame = OPTIMIZED_DF[OPTIMIZED_DF["timestamp"] == ts].copy()
    frame["siting_score"] = frame["h3_cell"].map(SITING_BY_CELL).fillna(0.0)

    if mode == "optimized":
        z = frame["optimized_ev_load_kw"]
        title = "Optimized EV load (kW)"
        colorscale = "Viridis"
    elif mode == "stress":
        z = 100 * frame["optimized_transformer_utilization"]
        title = "Transformer utilization (%)"
        colorscale = "RdYlGn_r"
    elif mode == "siting":
        z = frame["siting_score"]
        title = "Station siting priority score"
        colorscale = "Magma"
    else:
        z = frame["baseline_ev_load_kw"]
        title = "Predicted unmanaged EV demand (kW)"
        colorscale = "YlOrRd"

    custom = np.stack(
        [
            frame["zone_name"].astype(str),
            frame["zone_type"].astype(str),
            frame["baseline_ev_load_kw"].round(1),
            frame["optimized_ev_load_kw"].round(1),
            (100 * frame["optimized_transformer_utilization"]).round(1),
            frame["siting_score"].round(1),
            frame["stress_label"].astype(str),
        ],
        axis=-1,
    )

    fig = go.Figure(
        go.Choroplethmapbox(
            geojson=GEOJSON,
            locations=frame["h3_cell"],
            z=z,
            featureidkey="properties.h3_cell",
            colorscale=colorscale,
            marker_opacity=0.72,
            marker_line_width=0.7,
            marker_line_color="#ffffff",
            colorbar_title=title,
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                "Predicted EV: %{customdata[2]} kW<br>"
                "Optimized EV: %{customdata[3]} kW<br>"
                "Utilization: %{customdata[4]}%<br>"
                "Siting score: %{customdata[5]}<br>"
                "Stress: %{customdata[6]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"{title} - {ts.strftime('%Y-%m-%d %H:%M')}",
        mapbox_style="carto-positron",
        mapbox_zoom=10.4,
        mapbox_center=MAP_CENTER,
        height=500,
        margin={"l": 0, "r": 0, "t": 42, "b": 0},
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(color="#172033", family="Inter, Segoe UI, Arial, sans-serif"),
        transition={"duration": 0},
    )
    return fig


def make_load_figure(time_index: int) -> go.Figure:
    ts = selected_time(time_index)
    fig = go.Figure()

    if "prediction_std_kw" in AGG_TS.columns:
        upper = AGG_TS["optimized_total_load_kw"] + 1.96 * AGG_TS["prediction_std_kw"]
        lower = (AGG_TS["optimized_total_load_kw"] - 1.96 * AGG_TS["prediction_std_kw"]).clip(lower=0)
        fig.add_trace(
            go.Scatter(
                x=pd.concat([AGG_TS["timestamp"], AGG_TS["timestamp"][::-1]]),
                y=pd.concat([upper, lower[::-1]]),
                fill="toself",
                fillcolor="rgba(37, 99, 235, 0.10)",
                line=dict(color="rgba(37, 99, 235, 0)"),
                name="95% forecast band",
            )
        )

    fig.add_trace(go.Scatter(x=AGG_TS["timestamp"], y=AGG_TS["baseline_total_load_kw"], name="Unmanaged total load", line=dict(color="#dc2626", width=2)))
    fig.add_trace(go.Scatter(x=AGG_TS["timestamp"], y=AGG_TS["optimized_total_load_kw"], name="Optimized total load", line=dict(color="#059669", width=2.5)))
    fig.add_trace(go.Scatter(x=AGG_TS["timestamp"], y=AGG_TS["safe_capacity_kw"], name="95% capacity envelope", line=dict(color="#d97706", width=1.8, dash="dash")))
    if "solar_generation_kw" in AGG_TS.columns:
        fig.add_trace(go.Scatter(x=AGG_TS["timestamp"], y=AGG_TS["solar_generation_kw"], name="Solar generation", line=dict(color="#0284c7", width=1.7, dash="dot")))
    fig.add_trace(go.Scatter(
        x=[ts.isoformat(), ts.isoformat()], y=[0, 1],
        mode="lines", yaxis="y2",
        line=dict(color="#374151", dash="dot", width=1.5),
        showlegend=False, hoverinfo="skip",
    ))
    fig.update_layout(
        title="Unmanaged vs optimized aggregate load",
        xaxis_title="Forecast horizon", yaxis_title="kW",
        yaxis2=dict(range=[0, 1], overlaying="y", visible=False, showgrid=False),
    )
    return _plot_layout(fig, height=380)


def make_forecast_accuracy_figure(time_index: int) -> go.Figure:
    ts = selected_time(time_index)
    fig = go.Figure()
    if "actual_ev_load_kw" in AGG_TS.columns:
        fig.add_trace(go.Scatter(x=AGG_TS["timestamp"], y=AGG_TS["actual_ev_load_kw"], name="Synthetic actual EV load", line=dict(color="#7c3aed", width=2)))
    if "stgcn_predicted_demand_kw" in AGG_TS.columns:
        fig.add_trace(go.Scatter(x=AGG_TS["timestamp"], y=AGG_TS["stgcn_predicted_demand_kw"], name="Raw STGCN", line=dict(color="#f97316", width=1.7, dash="dash")))
    if "seasonal_baseline_kw" in AGG_TS.columns:
        fig.add_trace(go.Scatter(x=AGG_TS["timestamp"], y=AGG_TS["seasonal_baseline_kw"], name="Seasonal baseline", line=dict(color="#64748b", width=1.7, dash="dot")))
    fig.add_trace(go.Scatter(x=AGG_TS["timestamp"], y=AGG_TS["baseline_ev_load_kw"], name="Selected forecast", line=dict(color="#059669", width=2.6)))
    if "prediction_std_kw" in AGG_TS.columns:
        fig.add_trace(go.Scatter(x=AGG_TS["timestamp"], y=AGG_TS["prediction_std_kw"], name="Aggregate uncertainty", line=dict(color="#64748b", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(
        x=[ts.isoformat(), ts.isoformat()], y=[0, 1],
        mode="lines", yaxis="y2",
        line=dict(color="#374151", dash="dot", width=1.5),
        showlegend=False, hoverinfo="skip",
    ))
    fig.update_layout(
        title="Forecast quality on synthetic holdout",
        xaxis_title="Time", yaxis_title="kW",
        yaxis2=dict(range=[0, 1], overlaying="y", visible=False, showgrid=False),
    )
    return _plot_layout(fig, height=330)


def forecast_health_panel() -> html.Div:
    notes = METRICS.get("forecast_notes", [])
    if isinstance(notes, str):
        notes = [notes]
    if not notes:
        notes = ["Forecast passed variance and baseline guardrail checks."]
    return html.Div(
        [
            html.Div(f"Forecast method: {METRICS.get('forecast_method', 'n/a')}", className="summary-line"),
            html.Div(f"Health: {METRICS.get('forecast_health', 'n/a')}", className="summary-line"),
            html.Div(f"sMAPE: {float(METRICS.get('forecast_smape_pct', 0.0)):,.1f}%", className="summary-line"),
            html.Div(f"Variance ratio: {float(METRICS.get('forecast_variance_ratio', 0.0)):,.2f}", className="summary-line"),
            html.Div(f"Correlation: {float(METRICS.get('forecast_correlation', 0.0)):,.2f}", className="summary-line"),
            html.Div(" | ".join(notes), className="callout"),
        ],
        className="summary-box",
    )


def make_scheduling_figure(time_index: int) -> go.Figure:
    ts = selected_time(time_index)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=AGG_TS["timestamp"], y=AGG_TS["baseline_ev_load_kw"], name="Unmanaged EV charging", line=dict(color="#dc2626", width=2)))
    fig.add_trace(go.Scatter(x=AGG_TS["timestamp"], y=AGG_TS["optimized_ev_load_kw"], name="Optimized EV charging", line=dict(color="#059669", width=2.5)))
    fig.add_trace(go.Scatter(x=AGG_TS["timestamp"], y=AGG_TS["solar_generation_kw"], name="Solar generation", line=dict(color="#0284c7", width=1.8, dash="dot")))
    tariff = OPTIMIZED_DF.groupby("timestamp", as_index=False)["tariff_multiplier"].mean().sort_values("timestamp")
    fig.add_trace(go.Scatter(x=tariff["timestamp"], y=tariff["tariff_multiplier"], name="Tariff multiplier", yaxis="y2", line=dict(color="#9333ea", width=1.8, dash="dash")))
    fig.add_trace(go.Scatter(
        x=[ts.isoformat(), ts.isoformat()], y=[0, 1],
        mode="lines", yaxis="y3",
        line=dict(color="#374151", dash="dot", width=1.5),
        showlegend=False, hoverinfo="skip",
    ))
    fig.update_layout(
        title="Charging schedule, tariff signal, and solar alignment",
        xaxis_title="Forecast horizon",
        yaxis=dict(title="EV or solar kW"),
        yaxis2=dict(title="Tariff", overlaying="y", side="right", rangemode="tozero"),
        yaxis3=dict(range=[0, 1], overlaying="y", visible=False, showgrid=False),
    )
    return _plot_layout(fig, height=410)


METRIC_TOOLTIPS = {
    "Peak impact": "Percentage reduction in aggregate peak load after optimization. Lower peaks reduce transformer stress and defer infrastructure upgrades.",
    "Overload reduction": "Number of cell-hours where transformer utilization exceeded 100% capacity, before vs after optimization.",
    "Shifted energy": "Total kWh of flexible EV charging moved to off-peak or solar-rich periods while meeting deadline constraints.",
    "Cost signal": "Estimated savings from tariff optimization and solar alignment. Based on BESCOM ToU rates and avoided peak charges.",
    "Forecast health": "Model diagnostic: method used, prediction accuracy (MAE), and guardrail check status.",
    "Siting lift": "Improvement in demand capture from graph-aware station placement vs uniform/random allocation.",
}


def metric_card(title: str, value: str, detail: str, tone: str = "neutral") -> html.Div:
    tooltip = METRIC_TOOLTIPS.get(title, "")
    return html.Div(
        [
            html.Div(
                [html.Span(title), info_tip(tooltip) if tooltip else None],
                className="metric-title",
            ),
            html.Div(value, className=f"metric-value {tone}"),
            html.Div(detail, className="metric-detail"),
        ],
        className="metric-card",
    )


def posture_strip() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div("Problem Fit", className="strip-kicker"),
                    html.Div("BESCOM decision-support layer covering prediction, scheduling, siting, grid constraints, and explainability.", className="strip-copy"),
                ],
                className="strip-intro",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(label, className="posture-label"),
                            html.Span(value, className="posture-value"),
                        ],
                        className="posture-chip",
                    )
                    for label, value in SYSTEM_POSTURE.items()
                ],
                className="posture-grid",
            ),
        ],
        className="posture-strip",
    )


def metrics_row() -> html.Div:
    forecast_method = str(METRICS.get("forecast_method", "n/a")).replace("_", " ")
    forecast_health = str(METRICS.get("forecast_health", "n/a")).replace("_", " ")
    overload_before = float(METRICS.get("overload_events_before", 0))
    overload_after = float(METRICS.get("overload_events_after", 0))
    overload_delta = 100.0 * (overload_before - overload_after) / max(overload_before, 1.0)
    return html.Div(
        [
            metric_card("Peak impact", format_pct(float(METRICS["peak_reduction_pct"])), f"{format_kw(METRICS['baseline_peak_kw'])} to {format_kw(METRICS['optimized_peak_kw'])}", "good"),
            metric_card("Overload reduction", format_pct(overload_delta), f"{int(overload_before)} to {int(overload_after)} transformer-hours", "warn" if overload_after else "good"),
            metric_card("Shifted energy", f"{float(METRICS['shifted_kwh']):,.1f} kWh", f"Deadlines met: {format_pct(float(METRICS.get('deadlines_met_pct', 100.0)))}", "good"),
            metric_card("Cost signal", format_inr(float(METRICS["estimated_cost_savings_inr"])), "Tariff and solar aligned schedule", "good"),
            metric_card("Forecast health", forecast_health.title(), f"{forecast_method}; MAE {float(METRICS.get('forecast_mae_kw', 0.0)):,.1f} kW", "neutral"),
            metric_card("Siting lift", format_pct(float(SITING_SUMMARY["capture_improvement_pct"])), "Demand capture vs uniform placement", "good"),
        ],
        className="metric-grid",
    )


def global_time_controls() -> html.Div:
    """Shared time slider and playback controls visible across all tabs."""
    return html.Div(
        [
            html.Div(
                [
                    html.Button("Simulate", id="play-button", n_clicks=0, className="control-button"),
                    html.Div(id="current-time-display", className="time-display"),
                ],
                className="time-control-row",
            ),
            dcc.Slider(
                id="time-slider",
                min=0,
                max=len(TIME_VALUES) - 1,
                value=0,
                marks=make_slider_marks(),
                step=1,
                tooltip={"placement": "bottom"},
            ),
            dcc.Interval(id="playback", interval=900, n_intervals=0, disabled=True),
            dcc.Store(id="playing", data=False),
        ],
        className="global-time-controls",
    )


def table_from_records(records: List[Dict[str, object]], columns: List[tuple[str, str]]) -> html.Table:
    return html.Table(
        [
            html.Thead(html.Tr([html.Th(label) for _, label in columns])),
            html.Tbody(
                [
                    html.Tr([html.Td(record.get(key, "")) for key, _ in columns])
                    for record in records
                ]
            ),
        ],
        className="data-table",
    )


def risk_table() -> html.Table:
    records = []
    for item in METRICS.get("top_risk_zones", [])[:6]:
        records.append(
            {
                "zone": item.get("zone_name"),
                "type": item.get("zone_type"),
                "before": f"{100 * float(item.get('max_baseline_utilization', 0.0)):.1f}%",
                "after": f"{100 * float(item.get('max_optimized_utilization', 0.0)):.1f}%",
                "demand": format_kw(float(item.get("mean_predicted_demand_kw", 0.0))),
            }
        )
    return html.Div(
        table_from_records(
            records,
            [("zone", "Zone"), ("type", "Type"), ("before", "Before"), ("after", "After"), ("demand", "Mean EV Demand")],
        ),
        className="table-scroll",
    )


def recommendation_table() -> html.Table:
    records = []
    for _, row in RECOMMENDATIONS.iterrows():
        records.append(
            {
                "rank": int(row["rank"]),
                "zone": row["zone_name"],
                "type": row["zone_type"],
                "score": f"{float(row['siting_score']):.1f}",
                "demand": format_kw(float(row["peak_predicted_demand_kw"])),
                "headroom": format_kw(float(row["capacity_headroom_kw"])),
                "fit": row["capacity_feasibility"],
                "reason": row["reason"],
            }
        )
    return html.Div(
        table_from_records(
            records,
            [
                ("rank", "#"),
                ("zone", "Zone"),
                ("type", "Type"),
                ("score", "Score"),
                ("demand", "Peak Demand"),
                ("headroom", "Headroom"),
                ("fit", "Capacity Fit"),
                ("reason", "Planner Reason"),
            ],
        ),
        className="table-scroll",
    )


def selected_summary(time_index: int) -> html.Div:
    context = build_llm_context(METRICS, OPTIMIZED_DF, time_index, RECOMMENDATIONS, SITING_SUMMARY)
    top = context["top_predicted_demand_zones_at_selected_time"][0]
    risk = context["top_risk_zones_at_selected_time"][0]
    agg = context["selected_time_aggregate"]
    return html.Div(
        [
            html.Div(f"Selected time: {pd.Timestamp(context['selected_time']).strftime('%Y-%m-%d %H:%M')}", className="summary-line"),
            html.Div(f"Highest demand: {top['zone_name']} ({top['zone_type']}) at {top['baseline_ev_load_kw']} kW unmanaged EV load.", className="summary-line"),
            html.Div(f"Highest risk: {risk['zone_name']} at {100 * float(risk['optimized_transformer_utilization']):.1f}% optimized utilization.", className="summary-line"),
            html.Div(f"Aggregate load: unmanaged {agg['baseline_total_load_kw']} kW, optimized {agg['optimized_total_load_kw']} kW.", className="summary-line"),
        ],
        className="summary-box",
    )


def overview_tab() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H2("Operational Snapshot"),
                                    html.Div("Transformer stress and EV demand at the selected forecast hour.", className="panel-subtitle"),
                                ],
                                className="panel-heading",
                            ),
                            dcc.Graph(id="overview-map", config=GRAPH_CONFIG),
                        ],
                        className="panel wide-panel map-panel",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H2("Top Risk Zones"),
                                    html.Div("Prioritized by optimized transformer utilization.", className="panel-subtitle"),
                                ],
                                className="panel-heading",
                            ),
                            risk_table(),
                        ],
                        className="panel",
                    ),
                ],
                className="two-column",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H2("Load Reduction Baseline"),
                                    html.Div("Unmanaged charging compared with the recommended sidecar schedule.", className="panel-subtitle"),
                                ],
                                className="panel-heading",
                            ),
                            dcc.Graph(id="load-figure", config=GRAPH_CONFIG),
                        ],
                        className="panel",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H2("Planner Assurance"),
                                    html.Div("Non-invasive by design; recommendations stay explainable and actionable.", className="panel-subtitle"),
                                ],
                                className="panel-heading",
                            ),
                            html.Div("No feeder, DTR, SCADA, or charger-control system is modified.", className="callout success"),
                            html.Div(
                                [
                                    html.Div([html.Span("Scenario"), html.Strong(str(GRID_DF["corridor_name"].iloc[0]))], className="fact-row"),
                                    html.Div([html.Span("H3 zones"), html.Strong(str(len(GRID_DF)))], className="fact-row"),
                                    html.Div([html.Span("Forecast horizon"), html.Strong(f"{len(TIME_VALUES)} steps")], className="fact-row"),
                    html.Div([html.Span("Optimizer"), html.Strong(str(METRICS.get("optimizer_type", "robust_lp")).replace("_", " "))], className="fact-row"),
                    html.Div([html.Span("Local peak change"), html.Strong(format_pct(float(METRICS.get("local_transformer_peak_change_pct", 0.0))))], className="fact-row"),
                    html.Div([html.Span("Forecast method"), html.Strong(str(METRICS.get("forecast_method", "n/a")).replace("_", " "))], className="fact-row"),
                ],
                className="fact-list",
            ),
                        ],
                        className="panel",
                    ),
                ],
                className="two-column",
            ),
        ],
        className="tab-content",
    )


def forecast_tab() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    dcc.RadioItems(
                        id="map-mode",
                        options=[
                            {"label": "Predicted demand", "value": "predicted"},
                            {"label": "Optimized load", "value": "optimized"},
                            {"label": "Transformer stress", "value": "stress"},
                            {"label": "Siting priority", "value": "siting"},
                        ],
                        value="predicted",
                        inline=True,
                        className="radio-row",
                    ),
                ],
                className="control-row",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H2("Spatial Forecast"),
                                    html.Div("Switch layers to inspect demand, optimized load, transformer stress, or siting priority.", className="panel-subtitle"),
                                ],
                                className="panel-heading",
                            ),
                            dcc.Graph(id="forecast-map", config=GRAPH_CONFIG),
                            html.Div(id="selected-summary"),
                        ],
                        className="panel wide-panel map-panel",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H2("Forecast Reliability"),
                                    html.Div("Selected forecast, raw model, seasonal baseline, and uncertainty.", className="panel-subtitle"),
                                ],
                                className="panel-heading",
                            ),
                            dcc.Graph(id="forecast-accuracy", config=GRAPH_CONFIG),
                            forecast_health_panel(),
                        ],
                        className="panel",
                    ),
                ],
                className="two-column",
            ),
        ],
        className="tab-content",
    )


def scheduling_tab() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Schedule Optimization"),
                            html.Div("Recommendation balances peak load, tariff, solar availability, deadlines, and transformer headroom.", className="panel-subtitle"),
                        ],
                        className="panel-heading",
                    ),
                    dcc.Graph(id="schedule-figure", config=GRAPH_CONFIG),
                ],
                className="panel wide-panel",
            ),
            html.Div(
                [
                    html.Div("Scheduling objective: reduce corridor peak and transformer overload events while preserving priority charging and deadlines. Local transformer peaks are tracked separately from aggregate peak impact.", className="callout"),
                    html.Div(f"Peak-to-average ratio: {float(METRICS['baseline_par']):.2f} unmanaged to {float(METRICS['optimized_par']):.2f} optimized.", className="summary-line"),
                    html.Div(f"Local transformer peak: {format_kw(float(METRICS.get('local_transformer_peak_before_kw', 0.0)))} to {format_kw(float(METRICS.get('local_transformer_peak_after_kw', 0.0)))}.", className="summary-line"),
                    html.Div(f"Energy preservation error: {float(METRICS['energy_preservation_error_kwh']):.3f} kWh ({float(METRICS['energy_preservation_error_pct']):.3f}%).", className="summary-line"),
                    html.Div(f"Deadline feasibility: {format_pct(float(METRICS['deadlines_met_pct']))}.", className="summary-line"),
                ],
                className="panel",
            ),
        ],
        className="tab-content",
    )


def infrastructure_tab() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    metric_card("Station budget", str(SITING_SUMMARY["station_budget"]), f"{SITING_SUMMARY['station_kw']} kW planning increment", "neutral"),
                    metric_card("Recommended capture", format_kw(float(SITING_SUMMARY["recommended_captured_peak_kw"])), "Peak demand covered by ranked sites", "good"),
                    metric_card("Uniform capture", format_kw(float(SITING_SUMMARY["uniform_captured_peak_kw"])), "Baseline equal-spread placement", "warn"),
                    metric_card("Feasible sites", format_pct(float(SITING_SUMMARY["recommended_feasible_pct"])), "Existing transformer headroom", "good"),
                ],
                className="metric-grid compact infrastructure-metrics",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H2("Siting Priority Map"),
                                    html.Div("Graph-aware portfolio score with demand, growth, charger gap, centrality, traffic, and headroom.", className="panel-subtitle"),
                                ],
                                className="panel-heading",
                            ),
                            dcc.Graph(id="siting-map", config=GRAPH_CONFIG),
                        ],
                        className="panel wide-panel map-panel",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H2("Station Recommendations"),
                                    html.Div("Compared against uniform placement baseline.", className="panel-subtitle"),
                                ],
                                className="panel-heading",
                            ),
                            recommendation_table(),
                        ],
                        className="panel",
                    ),
                ],
                className="two-column",
            ),
        ],
        className="tab-content",
    )


def data_sources_tab() -> html.Div:
    sources = [
        ("Synthetic telemetry", "EV demand, base grid load, traffic, weather, solar generation, and tariff signals."),
        ("H3 spatial grid", "Adjacent hex cells along the configured Bengaluru corridor with graph neighbors for STGCN."),
        ("Forecasting layer", "STGCN baseline plus competition Graph-TFT implementation for probabilistic multi-horizon forecasting."),
        ("Optimizer", "Robust LP sidecar scheduler with priority-share, deadline, tariff, solar, uncertainty, and transformer-capacity constraints."),
        ("Infrastructure scoring", "Demand growth, charger gap, neighbor pressure, graph centrality, traffic, stress, and available headroom."),
        ("LLM context", "Only compact computed metrics and synthetic aggregate rows are sent when an API key is configured."),
    ]
    return html.Div(
        [
            html.Div(
                [
                    html.Div([html.Div(name, className="source-title"), html.Div(desc, className="source-detail")], className="source-row")
                    for name, desc in sources
                ],
                className="panel wide-panel",
            )
        ],
        className="tab-content",
    )


def problem_fit_tab() -> html.Div:
    criteria = [
        ("Predict EV charging demand by time and location", "Covered", "Forecast map and holdout metrics show H3-zone demand across the forecast horizon."),
        ("Recommend optimal charging times", "Covered", "Scheduling tab compares unmanaged and optimized EV load with tariff and solar signals."),
        ("Align charging with grid conditions", "Covered", "Robust optimizer uses transformer derating, utilization limits, uncertainty bands, and priority shares."),
        ("Identify high-demand zones", "Covered", "Risk and siting maps rank demand, stress, growth, and charger gaps by corridor cell."),
        ("Recommend new charging locations", "Covered", "Infrastructure tab selects a graph-aware station portfolio and compares it to uniform placement."),
        ("Use masked or synthetic data", "Covered", "Demo data is synthetic; enhanced path includes k-anonymity masking for OCPP-style sessions."),
        ("Explainable and actionable outputs", "Covered", "Each table row includes planner-facing metrics, capacity fit, and recommendation reasons."),
        ("No hosted LLM dependence on sensitive data", "Covered", "The assistant works locally by default and only receives aggregate computed context if enabled."),
    ]
    risk_items = [
        ("Data gaps", "Use synthetic twin, missing-value defaults, seasonal baselines, and feedback-loop drift checks."),
        ("Behavior adoption", "Keep schedules as planner recommendations; measure uptake in shadow mode before pilots."),
        ("Forecast drift", "Compare neural forecasts against seasonal/persistence guardrails and retrain by corridor."),
        ("Grid infeasibility", "Expose overload slack/headroom instead of hiding impossible transformer constraints."),
    ]
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Problem Statement Coverage"),
                            html.Div("A judge-facing checklist mapped directly to BESCOM's stated evaluation needs.", className="panel-subtitle"),
                        ],
                        className="panel-heading",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(item[0], className="check-title"),
                                    html.Div(item[1], className="check-status"),
                                    html.Div(item[2], className="check-detail"),
                                ],
                                className="check-row",
                            )
                            for item in criteria
                        ],
                        className="checklist",
                    ),
                ],
                className="panel wide-panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Risk Handling"),
                            html.Div("Known practical risks and mitigation built into the architecture.", className="panel-subtitle"),
                        ],
                        className="panel-heading",
                    ),
                    html.Div(
                        [
                            html.Div([html.Strong(name), html.Span(mitigation)], className="risk-row")
                            for name, mitigation in risk_items
                        ],
                        className="risk-list",
                    ),
                ],
                className="panel",
            ),
        ],
        className="two-column tab-content",
    )


def sidebar() -> html.Div:
    """Fixed left navigation panel with emoji icons."""
    nav_items = [
        ("nav-overview",       "📊", "Overview"),
        ("nav-forecast",       "📈", "Demand Forecast"),
        ("nav-scheduling",     "⚡", "Scheduling"),
        ("nav-infrastructure", "🏗️", "Infrastructure"),
        ("nav-assistant",      "🤖", "Assistant"),
    ]
    ref_items = [
        ("nav-data",      "💾", "Data Sources"),
        ("nav-about",     "ℹ️", "About"),
    ]
    return html.Div(
        [
            html.Div(
                [
                    html.Span("⚡", className="brand-icon"),
                    html.Div(
                        [
                            html.Div("Vidyut Prajna", className="brand-name"),
                            html.Div("BESCOM EV Intelligence", className="brand-sub"),
                        ]
                    ),
                ],
                className="brand-block",
            ),
            html.Div("Views", className="nav-section-label"),
            *[
                html.Button(
                    [html.Span(icon, className="nav-icon"), html.Span(label)],
                    id=btn_id, n_clicks=0, className="nav-item",
                )
                for btn_id, icon, label in nav_items
            ],
            html.Div("Reference", className="nav-section-label"),
            *[
                html.Button(
                    [html.Span(icon, className="nav-icon"), html.Span(label)],
                    id=btn_id, n_clicks=0, className="nav-item",
                )
                for btn_id, icon, label in ref_items
            ],
        ],
        className="sidebar",
    )


def assistant_page() -> html.Div:
    """Full-page assistant with chat interface."""
    llm_status = f"Connected: {LLM.model}" if LLM.enabled else "Using local fallback (no API key)"
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Explanation Assistant"),
                            html.Div(
                                [
                                    "Ask questions about the forecast, scheduling decisions, or infrastructure recommendations. ",
                                    info_tip("The assistant uses computed dashboard context only. It does not have access to raw data or external information."),
                                ],
                                className="panel-subtitle",
                            ),
                        ],
                        className="panel-heading",
                    ),
                    html.Div(
                        [
                            html.Span("🔗 " + llm_status, className="llm-status-badge"),
                        ],
                        className="assistant-status",
                    ),
                    html.Div(id="chat-window", className="chat-window"),
                    html.Div(
                        [
                            dcc.Input(
                                id="chat-input",
                                placeholder="Ask about demand patterns, scheduling logic, zone risks, or siting recommendations...",
                                type="text",
                                className="chat-input",
                                debounce=False,
                            ),
                            html.Button("Send", id="chat-send", n_clicks=0, className="chat-send-button"),
                        ],
                        className="chat-controls",
                    ),
                    dcc.Store(id="chat-history", data=[]),
                    dcc.Store(id="chat-loading", data=False),
                ],
                className="panel assistant-panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Example Questions"),
                            html.Div("Try these to explore the system.", className="panel-subtitle"),
                        ],
                        className="panel-heading",
                    ),
                    html.Div(
                        [
                            html.Div("Which zones have the highest transformer stress?", className="example-q"),
                            html.Div("Why was charging shifted to 14:00-16:00?", className="example-q"),
                            html.Div("What factors drive the siting recommendations?", className="example-q"),
                            html.Div("How much peak load reduction did we achieve?", className="example-q"),
                            html.Div("Explain the forecast uncertainty bands.", className="example-q"),
                        ],
                        className="example-list",
                    ),
                ],
                className="panel",
            ),
        ],
        className="two-column tab-content",
    )


def topbar() -> html.Div:
    """Slim sticky bar: page title (left) + time controls (right)."""
    return html.Div(
        [
            html.Div(
                [html.Div(id="page-title", className="page-title")],
                className="topbar-left",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Sidecar Mode", className="badge-title"),
                            html.Span("No grid-control writes", className="badge-detail"),
                        ],
                        className="mode-badge",
                    ),
                    html.Button("Simulate", id="play-button", n_clicks=0, className="control-button"),
                    html.Div(id="current-time-display", className="time-display"),
                    dcc.Slider(
                        id="time-slider",
                        min=0,
                        max=len(TIME_VALUES) - 1,
                        value=0,
                        marks=make_slider_marks(),
                        step=1,
                        tooltip={"placement": "bottom"},
                        className="topbar-slider",
                    ),
                    dcc.Interval(id="playback", interval=900, n_intervals=0, disabled=True),
                    dcc.Store(id="playing", data=False),
                ],
                className="topbar-right",
            ),
        ],
        className="topbar",
    )


def kpi_strip() -> html.Div:
    """Collapsible KPI metric card strip."""
    return html.Div(
        [
            html.Button("v Metrics", id="kpi-toggle", className="kpi-toggle", n_clicks=0),
            html.Div(metrics_row(), id="kpi-content", className="kpi-content"),
            dcc.Store(id="kpi-collapsed", data=False),
        ],
        className="kpi-strip",
    )


def about_page() -> html.Div:
    """About page: posture chips + system facts + problem fit checklist + risk handling."""
    criteria = [
        ("Predict EV charging demand by time and location", "Covered", "Forecast map and holdout metrics show H3-zone demand across the forecast horizon."),
        ("Recommend optimal charging times", "Covered", "Scheduling tab compares unmanaged and optimized EV load with tariff and solar signals."),
        ("Align charging with grid conditions", "Covered", "Robust optimizer uses transformer derating, utilization limits, uncertainty bands, and priority shares."),
        ("Identify high-demand zones", "Covered", "Risk and siting maps rank demand, stress, growth, and charger gaps by corridor cell."),
        ("Recommend new charging locations", "Covered", "Infrastructure tab selects a graph-aware station portfolio and compares it to uniform placement."),
        ("Use masked or synthetic data", "Covered", "Demo data is synthetic; enhanced path includes k-anonymity masking for OCPP-style sessions."),
        ("Explainable and actionable outputs", "Covered", "Each table row includes planner-facing metrics, capacity fit, and recommendation reasons."),
        ("No hosted LLM dependence on sensitive data", "Covered", "The assistant works locally by default and only receives aggregate computed context if enabled."),
    ]
    risk_items = [
        ("Data gaps", "Use synthetic twin, missing-value defaults, seasonal baselines, and feedback-loop drift checks."),
        ("Behavior adoption", "Keep schedules as planner recommendations; measure uptake in shadow mode before pilots."),
        ("Forecast drift", "Compare neural forecasts against seasonal/persistence guardrails and retrain by corridor."),
        ("Grid infeasibility", "Expose overload slack/headroom instead of hiding impossible transformer constraints."),
    ]
    return html.Div(
        [
            # Posture strip
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Problem Fit", className="strip-kicker"),
                            html.Div("BESCOM decision-support layer covering prediction, scheduling, siting, grid constraints, and explainability.", className="strip-copy"),
                        ],
                        className="strip-intro",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [html.Span(label, className="posture-label"), html.Span(value, className="posture-value")],
                                className="posture-chip",
                            )
                            for label, value in SYSTEM_POSTURE.items()
                        ],
                        className="posture-grid",
                    ),
                ],
                className="posture-strip panel wide-panel",
                style={"marginBottom": "14px"},
            ),
            # Problem statement + risk
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H2("Problem Statement Coverage"),
                                    html.Div("A judge-facing checklist mapped directly to BESCOM's stated evaluation needs.", className="panel-subtitle"),
                                ],
                                className="panel-heading",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [html.Div(c[0], className="check-title"), html.Div(c[1], className="check-status"), html.Div(c[2], className="check-detail")],
                                        className="check-row",
                                    )
                                    for c in criteria
                                ],
                                className="checklist",
                            ),
                        ],
                        className="panel wide-panel",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H2("Risk Handling"),
                                    html.Div("Known practical risks and mitigation built into the architecture.", className="panel-subtitle"),
                                ],
                                className="panel-heading",
                            ),
                            html.Div(
                                [html.Div([html.Strong(n), html.Span(m)], className="risk-row") for n, m in risk_items],
                                className="risk-list",
                            ),
                        ],
                        className="panel",
                    ),
                ],
                className="two-column",
            ),
        ],
        className="tab-content",
    )


app = Dash(__name__, title="Vidyut Prajna", suppress_callback_exceptions=True)
server = app.server

_VIEWS = ["overview", "forecast", "scheduling", "infrastructure", "data", "assistant", "about"]
_VIEW_TITLES = {
    "overview": "Overview",
    "forecast": "Demand Forecast",
    "scheduling": "Schedule Optimization",
    "infrastructure": "Infrastructure Planning",
    "data": "Data Sources",
    "assistant": "Assistant",
    "about": "About",
}

app.layout = html.Div(
    [
        sidebar(),
        html.Div(
            [
                topbar(),
                kpi_strip(),
                html.Div(
                    [
                        html.Div(overview_tab(),        id="page-overview",        className="page active"),
                        html.Div(forecast_tab(),        id="page-forecast",        className="page"),
                        html.Div(scheduling_tab(),      id="page-scheduling",      className="page"),
                        html.Div(infrastructure_tab(),  id="page-infrastructure",  className="page"),
                        html.Div(data_sources_tab(),    id="page-data",            className="page"),
                        html.Div(assistant_page(),      id="page-assistant",       className="page"),
                        html.Div(about_page(),          id="page-about",           className="page"),
                    ],
                    className="pages-container",
                ),
                dcc.Store(id="active-view", data="overview"),
            ],
            className="main-content",
        ),
    ],
    className="app-shell",
)


# â”€â”€ Navigation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.callback(
    Output("page-overview",        "className"),
    Output("page-forecast",        "className"),
    Output("page-scheduling",      "className"),
    Output("page-infrastructure",  "className"),
    Output("page-data",            "className"),
    Output("page-assistant",       "className"),
    Output("page-about",           "className"),
    Output("page-title",           "children"),
    Output("active-view",          "data"),
    Input("nav-overview",          "n_clicks"),
    Input("nav-forecast",          "n_clicks"),
    Input("nav-scheduling",        "n_clicks"),
    Input("nav-infrastructure",    "n_clicks"),
    Input("nav-data",              "n_clicks"),
    Input("nav-assistant",         "n_clicks"),
    Input("nav-about",             "n_clicks"),
    State("active-view",           "data"),
    prevent_initial_call=True,
)
def navigate(*args):
    current = args[-1]
    trigger = ctx.triggered_id or "nav-" + current
    view = trigger.replace("nav-", "") if trigger else current
    classes = ["page active" if v == view else "page" for v in _VIEWS]
    return (*classes, _VIEW_TITLES.get(view, "Overview"), view)


@app.callback(
    Output("nav-overview",        "className"),
    Output("nav-forecast",        "className"),
    Output("nav-scheduling",      "className"),
    Output("nav-infrastructure",  "className"),
    Output("nav-data",            "className"),
    Output("nav-assistant",       "className"),
    Output("nav-about",           "className"),
    Input("active-view", "data"),
)
def update_nav_classes(active_view: str):
    return ["nav-item active" if v == active_view else "nav-item" for v in _VIEWS]


# â”€â”€ KPI strip toggle â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.callback(
    Output("kpi-content",  "style"),
    Output("kpi-toggle",   "children"),
    Output("kpi-collapsed", "data"),
    Input("kpi-toggle",    "n_clicks"),
    State("kpi-collapsed", "data"),
    prevent_initial_call=True,
)
def toggle_kpi(_: int, collapsed: bool):
    new = not bool(collapsed)
    style = {"display": "none"} if new else {"display": "block"}
    label = "^ Hide Metrics" if not new else "v Show Metrics"
    return style, label, new


# â”€â”€ Playback â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.callback(
    Output("playing", "data"),
    Output("playback", "disabled"),
    Output("play-button", "children"),
    Input("play-button", "n_clicks"),
    State("playing", "data"),
    prevent_initial_call=True,
)
def toggle_play(_: int, playing: bool) -> tuple[bool, bool, str]:
    new_state = not bool(playing)
    return new_state, not new_state, "Stop" if new_state else "Simulate"


@app.callback(Output("time-slider", "value"), Input("playback", "n_intervals"), State("time-slider", "value"))
def advance_time(_: int, current_value: int) -> int:
    return (int(current_value) + 1) % len(TIME_VALUES)


@app.callback(Output("current-time-display", "children"), Input("time-slider", "value"))
def update_time_display(time_index: int) -> str:
    ts = selected_time(time_index)
    return ts.strftime('%H:%M')


# ---- Charts -----------------------------------------------------------------------------

@app.callback(
    Output("overview-map",    "figure"),
    Output("forecast-map",    "figure"),
    Output("load-figure",     "figure"),
    Output("forecast-accuracy", "figure"),
    Output("schedule-figure", "figure"),
    Output("siting-map",      "figure"),
    Output("selected-summary", "children"),
    Input("time-slider", "value"),
    Input("map-mode",    "value"),
    Input("active-view", "data"),
)
def update_dashboard(time_index: int, map_mode: str, _active_view: str):
    return (
        make_map_figure(time_index, "stress"),
        make_map_figure(time_index, map_mode),
        make_load_figure(time_index),
        make_forecast_accuracy_figure(time_index),
        make_scheduling_figure(time_index),
        make_map_figure(time_index, "siting"),
        selected_summary(time_index),
    )



# ── Chat ──────────────────────────────────────────────────────────────────────

@app.callback(
    Output("chat-history", "data"),
    Output("chat-input",   "value"),
    Output("chat-loading", "data"),
    Input("chat-send",     "n_clicks"),
    State("chat-input",    "value"),
    State("chat-history",  "data"),
    State("time-slider",   "value"),
    prevent_initial_call=True,
)
def handle_chat(_: int, question: str | None, history: list | None, time_index: int):
    if not question or not question.strip():
        raise PreventUpdate
    context = build_llm_context(METRICS, OPTIMIZED_DF, int(time_index), RECOMMENDATIONS, SITING_SUMMARY)
    answer = LLM.answer(question.strip(), context)
    history = history or []
    history.append({"role": "user", "content": question.strip()})
    history.append({"role": "assistant", "content": answer})
    return history, "", False


@app.callback(Output("chat-window", "children"), Input("chat-history", "data"), Input("chat-loading", "data"))
def render_chat(history: list | None, loading: bool):
    if not history and not loading:
        return html.Div(
            [
                html.Div("👋 Welcome to the Assistant", className="empty-chat-title"),
                html.Div("Ask about demand patterns, scheduling decisions, infrastructure risks, or siting recommendations.", className="empty-chat-sub"),
            ],
            className="empty-chat",
        )
    bubbles = []
    for msg in (history or [])[-20:]:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        if role == "user":
            bubbles.append(html.Div(
                [html.Span("You", className="bubble-label"), html.Div(content)],
                className="chat-bubble user",
            ))
        else:
            bubbles.append(html.Div(
                [html.Span("Assistant", className="bubble-label"), dcc.Markdown(content)],
                className="chat-bubble assistant",
            ))
    if loading:
        bubbles.append(html.Div(
            [html.Span("Assistant", className="bubble-label"), html.Div(className="loading-spinner")],
            className="chat-bubble assistant loading",
        ))
    return bubbles



app.index_string = """
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>Vidyut Prajna</title>
    {%favicon%}
    {%css%}
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
        :root {
            --bg: #f1f5f9;
            --panel: #ffffff;
            --panel-alt: #f8fafc;
            --text: #0f172a;
            --muted: #64748b;
            --border: #e2e8f0;
            --blue: #2563eb;
            --green: #059669;
            --green-dark: #047857;
            --amber: #d97706;
            --red: #dc2626;
            --violet: #7c3aed;
            --sidebar-bg: #0f172a;
            --sidebar-border: rgba(255,255,255,0.07);
            --sidebar-text: #e2e8f0;
            --sidebar-active-bg: rgba(5,150,105,0.13);
            --sidebar-active-color: #34d399;
            --sidebar-hover-bg: rgba(255,255,255,0.05);
        }
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        html, body { height: 100%; overflow: hidden; }
        body {
            font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
            font-size: 14px;
            color: var(--text);
            background: var(--bg);
        }

        /* â”€â”€ Shell â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
        .app-shell {
            display: flex;
            height: 100vh;
            overflow: hidden;
        }

        /* â”€â”€ Sidebar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
        .sidebar {
            width: 252px;
            flex-shrink: 0;
            height: 100vh;
            background: var(--sidebar-bg);
            display: flex;
            flex-direction: column;
            overflow-y: auto;
            overflow-x: hidden;
            scrollbar-width: thin;
            scrollbar-color: rgba(255,255,255,0.1) transparent;
        }
        .brand-block {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 18px 16px 16px;
            border-bottom: 1px solid var(--sidebar-border);
            flex-shrink: 0;
        }
        .brand-icon { font-size: 26px; line-height: 1; }
        .brand-name {
            color: #f1f5f9;
            font-size: 16px;
            font-weight: 900;
            letter-spacing: -0.01em;
            line-height: 1.1;
        }
        .brand-sub {
            color: #94a3b8;
            font-size: 10px;
            font-weight: 600;
            letter-spacing: 0.04em;
            margin-top: 2px;
        }
        .nav-section-label {
            display: block;
            color: #cbd5e1;
            font-size: 10px;
            font-weight: 900;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            padding: 14px 16px 5px;
        }
        .nav-item {
            display: flex;
            align-items: center;
            gap: 9px;
            width: 100%;
            background: none;
            border: none;
            border-left: 3px solid transparent;
            color: var(--sidebar-text);
            font-size: 13px;
            font-weight: 600;
            padding: 10px 16px 10px 13px;
            cursor: pointer;
            text-align: left;
            font-family: inherit;
            transition: background 0.13s, color 0.13s, border-color 0.13s;
        }
        .nav-item:hover { background: var(--sidebar-hover-bg); color: #cbd5e1; }
        .nav-item.active {
            background: var(--sidebar-active-bg);
            border-left-color: var(--green);
            color: var(--sidebar-active-color);
            font-weight: 700;
        }
        .nav-icon { font-size: 16px; width: 24px; height: 24px; text-align: center; flex-shrink: 0;
            background: rgba(255,255,255,0.08); border-radius: 6px; display: flex; align-items: center;
            justify-content: center; }
        .nav-item.active .nav-icon { background: rgba(52,211,153,0.2); }
        .brand-icon { font-size: 28px; line-height: 1; flex-shrink: 0; }

        /* Sidebar assistant panel */
        .sidebar-assistant {
            margin-top: auto;
            border-top: 1px solid var(--sidebar-border);
            padding: 12px 12px 14px;
            flex-shrink: 0;
        }
        .llm-status-mini {
            color: #34d399;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .sidebar-chat-window {
            min-height: 110px;
            max-height: 180px;
            overflow-y: auto;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 7px;
            padding: 8px;
            margin-bottom: 8px;
            scrollbar-width: thin;
        }
        .sidebar-chat-window .chat-bubble {
            font-size: 11.5px;
            padding: 6px 9px;
            max-width: 100%;
            margin: 5px 0;
            line-height: 1.4;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.1);
            color: #cbd5e1;
        }
        .sidebar-chat-window .chat-bubble.user {
            background: rgba(37,99,235,0.15);
            border-color: rgba(37,99,235,0.25);
            color: #bfdbfe;
            margin-left: auto;
        }
        .sidebar-chat-window .chat-bubble.assistant {
            background: rgba(5,150,105,0.1);
            border-color: rgba(5,150,105,0.22);
            color: #6ee7b7;
        }
        .sidebar-chat-controls { display: flex; gap: 6px; }
        .chat-input-mini {
            flex: 1;
            min-width: 0;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 6px;
            color: #e2e8f0;
            padding: 7px 10px;
            font-size: 12px;
            font-family: inherit;
            outline: none;
        }
        .chat-input-mini::placeholder { color: #475569; }
        .chat-input-mini:focus { border-color: rgba(52,211,153,0.4); }
        .chat-send-mini {
            background: var(--green);
            border: none;
            border-radius: 6px;
            color: white;
            padding: 7px 11px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 900;
            flex-shrink: 0;
            transition: background 0.13s;
        }
        .chat-send-mini:hover { background: var(--green-dark); }
        .empty-chat { color: #475569; font-size: 11.5px; line-height: 1.4; }

        /* ── Assistant Page ─────────────────────────────────────────────── */
        .assistant-panel {
            flex: 2;
            display: flex;
            flex-direction: column;
            min-height: 600px;
        }
        .assistant-status {
            margin-bottom: 16px;
        }
        .llm-status-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            background: linear-gradient(135deg, rgba(5,150,105,0.1), rgba(37,99,235,0.08));
            border: 1px solid rgba(5,150,105,0.2);
            border-radius: 8px;
            font-size: 12px;
            color: var(--green);
            font-weight: 600;
        }
        .chat-window {
            flex: 1;
            min-height: 350px;
            max-height: 500px;
            overflow-y: auto;
            background: var(--panel-alt);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .chat-bubble {
            max-width: 85%;
            padding: 12px 16px;
            border-radius: 12px;
            line-height: 1.5;
            font-size: 14px;
        }
        .chat-bubble.user {
            align-self: flex-end;
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            color: white;
            border-bottom-right-radius: 4px;
        }
        .chat-bubble.assistant {
            align-self: flex-start;
            background: white;
            border: 1px solid var(--border);
            color: var(--text);
            border-bottom-left-radius: 4px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        }
        .chat-bubble.assistant p { margin: 0 0 8px; }
        .chat-bubble.assistant p:last-child { margin-bottom: 0; }
        .chat-bubble.assistant ul, .chat-bubble.assistant ol { margin: 8px 0; padding-left: 20px; }
        .chat-bubble.assistant li { margin: 4px 0; }
        .chat-bubble.assistant code { background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-size: 13px; }
        .chat-bubble.assistant pre { background: #f1f5f9; padding: 12px; border-radius: 8px; overflow-x: auto; margin: 8px 0; }
        .chat-bubble.assistant pre code { background: none; padding: 0; }
        .bubble-label {
            display: block;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 4px;
            opacity: 0.7;
        }
        .chat-bubble.user .bubble-label { color: rgba(255,255,255,0.8); }
        .chat-bubble.assistant .bubble-label { color: var(--green); }
        .chat-controls {
            display: flex;
            gap: 12px;
            align-items: stretch;
        }
        .chat-input {
            flex: 1;
            padding: 16px 18px;
            min-height: 56px;
            background: white;
            border: 2px solid var(--border);
            border-radius: 10px;
            font-size: 15px;
            font-family: inherit;
            color: var(--text);
            outline: none;
            transition: border-color 0.2s;
        }
        .chat-input::placeholder { color: var(--muted); }
        .chat-input:focus { border-color: var(--green); }
        .chat-send-button {
            background: linear-gradient(135deg, var(--green), var(--green-dark));
            border: none;
            color: white;
            padding: 14px 28px;
            border-radius: 10px;
            font-size: 14px;
            font-weight: 700;
            font-family: inherit;
            cursor: pointer;
            transition: transform 0.1s, box-shadow 0.2s;
        }
        .chat-send-button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(5,150,105,0.3);
        }
        .empty-chat {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            text-align: center;
            padding: 40px;
        }
        .empty-chat-title {
            font-size: 20px;
            font-weight: 700;
            color: var(--text);
            margin-bottom: 8px;
        }
        .empty-chat-sub {
            color: var(--muted);
            max-width: 400px;
            line-height: 1.6;
        }
        .example-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .example-q {
            background: var(--panel-alt);
            border: 1px solid var(--border);
            border-left: 3px solid var(--green);
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 13px;
            color: var(--text);
            cursor: pointer;
            transition: background 0.15s, border-color 0.15s;
        }
        .example-q:hover {
            background: #ecfdf5;
            border-color: var(--green);
        }
        .loading-spinner {
            width: 20px;
            height: 20px;
            border: 3px solid var(--border);
            border-top-color: var(--green);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* â”€â”€ Main content â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
        .main-content {
            flex: 1;
            min-width: 0;
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            background:
                radial-gradient(circle at 15% -10%, rgba(59,130,246,0.10), transparent 30rem),
                radial-gradient(circle at 90% 5%,  rgba(16,185,129,0.08), transparent 24rem),
                var(--bg);
        }

        /* â”€â”€ Topbar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
        .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 14px;
            padding: 10px 20px;
            background: rgba(255,255,255,0.96);
            border-bottom: 1px solid var(--border);
            backdrop-filter: blur(12px);
            flex-shrink: 0;
            box-shadow: 0 2px 10px rgba(15,23,42,0.06);
            z-index: 10;
        }
        .topbar-left { display: flex; align-items: center; gap: 10px; min-width: 0; }
        .page-title {
            font-size: 17px;
            font-weight: 900;
            color: var(--text);
            white-space: nowrap;
        }
        .topbar-right {
            display: flex;
            align-items: center;
            gap: 10px;
            flex: 1;
            justify-content: flex-end;
        }
        .topbar-slider {
            flex: 1;
            max-width: 500px;
            min-width: 200px;
        }
        /* Override Dash slider margin in topbar */
        .topbar-right .rc-slider { margin: 0 !important; }

        .time-display {
            white-space: nowrap;
            font-size: 13px;
            color: var(--muted);
            font-weight: 700;
            font-variant-numeric: tabular-nums;
            min-width: 48px;
        }
        .mode-badge {
            white-space: nowrap;
            color: #065f46;
            background: linear-gradient(180deg, #ecfdf5, #d1fae5);
            border: 1px solid #86efac;
            border-radius: 6px;
            padding: 5px 9px;
            font-size: 11px;
        }
        .badge-title { font-weight: 900; display: block; }
        .badge-detail { font-size: 10px; color: #047857; }
        .control-button {
            border: 1px solid var(--green-dark);
            background: var(--green);
            color: white;
            border-radius: 6px;
            padding: 8px 14px;
            cursor: pointer;
            font-weight: 800;
            font-size: 13px;
            font-family: inherit;
            white-space: nowrap;
            transition: background 0.13s;
        }
        .control-button:hover { background: var(--green-dark); }

        /* â”€â”€ KPI strip â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
        .kpi-strip {
            background: rgba(255,255,255,0.90);
            border-bottom: 1px solid var(--border);
            padding: 4px 20px 0;
            flex-shrink: 0;
        }
        .kpi-toggle {
            background: none;
            border: none;
            color: var(--muted);
            font-size: 11px;
            font-weight: 700;
            font-family: inherit;
            cursor: pointer;
            padding: 6px 0;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .kpi-toggle:hover { color: var(--text); }
        .kpi-content { padding-bottom: 10px; }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(6, minmax(0,1fr));
            gap: 10px;
            margin: 6px 0 0;
        }
        .metric-grid.compact { grid-template-columns: repeat(4, minmax(0,1fr)); }
        .metric-card {
            background: rgba(255,255,255,0.94);
            border: 1px solid rgba(148,163,184,0.30);
            border-radius: 8px;
            padding: 11px 13px;
            min-height: 86px;
            box-shadow: 0 6px 18px rgba(15,23,42,0.05);
            transition: transform 0.15s, box-shadow 0.15s;
        }
        .metric-card:hover { transform: translateY(-1px); box-shadow: 0 12px 28px rgba(15,23,42,0.09); }
        .metric-title { color: var(--muted); font-size: 10px; text-transform: uppercase; font-weight: 800; letter-spacing: 0.08em; }
        .metric-value { margin-top: 6px; font-size: 21px; font-weight: 900; line-height: 1.1; color: var(--text); }
        .metric-value.good { color: var(--green); }
        .metric-value.warn { color: var(--amber); }
        .metric-detail { margin-top: 5px; color: var(--muted); font-size: 11px; line-height: 1.35; }

        /* â”€â”€ Pages container â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
        .pages-container {
            flex: 1;
            overflow-y: auto;
            overflow-x: hidden;
            padding: 14px 20px 28px;
        }
        .page { display: none; }
        .page.active { display: block; }

        /* â”€â”€ Panels â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
        .panel {
            background: rgba(255,255,255,0.94);
            border: 1px solid rgba(148,163,184,0.28);
            border-radius: 8px;
            padding: 14px;
            box-shadow: 0 8px 24px rgba(15,23,42,0.06);
            min-width: 0;
        }
        .wide-panel { min-width: 0; }
        .map-panel { padding-bottom: 8px; }
        .panel-heading {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 14px;
            margin-bottom: 10px;
        }
        .panel-subtitle { color: var(--muted); font-size: 12px; line-height: 1.35; max-width: 560px; }
        h2 { color: var(--text); font-size: 15px; font-weight: 900; margin-bottom: 2px; }
        .tab-content { padding-top: 0; }
        .two-column {
            display: grid;
            grid-template-columns: minmax(0,1.35fr) minmax(320px,0.65fr);
            gap: 12px;
            margin-bottom: 12px;
            align-items: start;
        }

        /* â”€â”€ Control row (forecast tab radio) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
        .control-row {
            display: flex;
            align-items: center;
            gap: 16px;
            background: rgba(255,255,255,0.94);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 10px;
            box-shadow: 0 4px 12px rgba(15,23,42,0.04);
        }
        .radio-row label {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 7px 10px;
            margin-right: 6px;
            font-weight: 700;
            font-size: 13px;
            color: #374151;
        }

        /* â”€â”€ Summary / callout boxes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
        .summary-box, .callout, .fact-list, .check-row, .risk-row {
            background: var(--panel-alt);
            border: 1px solid var(--border);
            border-radius: 8px;
        }
        .summary-box, .callout { padding: 11px 13px; margin-top: 10px; }
        .summary-line { color: #374151; font-size: 13px; margin: 7px 0; line-height: 1.4; }
        .callout.success { background: #ecfdf5; border-color: #86efac; color: #065f46; font-weight: 800; }
        .fact-list { margin-top: 10px; padding: 4px 12px; }
        .fact-row {
            display: flex; justify-content: space-between; gap: 14px;
            padding: 9px 0; border-bottom: 1px solid var(--border);
            color: var(--muted); font-size: 13px;
        }
        .fact-row:last-child { border-bottom: 0; }
        .fact-row strong { color: var(--text); text-align: right; }

        /* â”€â”€ Tables â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
        .table-scroll { width: 100%; overflow-x: auto; border-radius: 8px; }
        .data-table { width: 100%; border-collapse: collapse; font-size: 12px; min-width: 680px; }
        .data-table th {
            text-align: left; color: #374151; background: #f3f4f6;
            border-bottom: 1px solid var(--border);
            padding: 8px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
            position: sticky; top: 0; z-index: 1;
        }
        .data-table td { border-bottom: 1px solid #e5e7eb; padding: 8px; vertical-align: top; }
        .data-table tr:hover td { background: #f8fafc; }

        /* â”€â”€ Checklist / risk â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
        .checklist, .risk-list { display: grid; gap: 8px; }
        .check-row {
            display: grid;
            grid-template-columns: minmax(200px,0.8fr) 82px 1fr;
            gap: 12px; align-items: center; padding: 9px 12px;
        }
        .check-title { color: var(--text); font-weight: 800; font-size: 13px; }
        .check-status {
            color: #065f46; background: #d1fae5; border: 1px solid #86efac;
            border-radius: 6px; padding: 4px 7px; text-align: center; font-weight: 900; font-size: 11px;
        }
        .check-detail, .risk-row span { color: #475569; font-size: 13px; line-height: 1.36; }
        .risk-row { display: grid; gap: 4px; padding: 10px 12px; }
        .risk-row strong { color: var(--text); font-size: 13px; }

        /* â”€â”€ Posture strip (on About page) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
        .posture-strip {
            display: grid;
            grid-template-columns: minmax(240px,0.4fr) 1fr;
            gap: 14px;
            align-items: stretch;
        }
        .strip-intro { border-radius: 8px; padding: 12px 14px; }
        .strip-kicker { color: #0f766e; font-size: 11px; font-weight: 900; text-transform: uppercase; letter-spacing: 0.08em; }
        .strip-copy { color: #334155; font-size: 13px; line-height: 1.42; margin-top: 5px; }
        .posture-grid { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 8px; }
        .posture-chip { display: grid; gap: 3px; border-radius: 8px; padding: 9px 11px; min-height: 58px;
            background: rgba(255,255,255,0.94); border: 1px solid rgba(148,163,184,0.28);
            box-shadow: 0 6px 16px rgba(15,23,42,0.05); }
        .posture-label { color: var(--muted); font-size: 10px; font-weight: 900; text-transform: uppercase; letter-spacing: 0.06em; }
        .posture-value { color: var(--text); font-size: 12px; line-height: 1.28; font-weight: 650; }

        /* â”€â”€ Data sources â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
        .source-row {
            display: grid; grid-template-columns: 200px 1fr;
            gap: 16px; border-bottom: 1px solid #e5e7eb; padding: 11px 0;
        }
        .source-title { font-weight: 800; }
        .source-detail { color: var(--muted); }

        /* â”€â”€ Infrastructure â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
        .infrastructure-metrics { position: relative; z-index: 0; margin-bottom: 12px; }

        /* â”€â”€ Plotly â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
        .js-plotly-plot, .dash-graph { min-width: 0; }

        /* â”€â”€ Responsive â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
        @media (max-width: 1280px) {
            .metric-grid { grid-template-columns: repeat(3, minmax(0,1fr)); }
            .metric-grid.compact { grid-template-columns: repeat(2, minmax(0,1fr)); }
            .two-column { grid-template-columns: 1fr; }
            .posture-strip { grid-template-columns: 1fr; }
            .posture-grid { grid-template-columns: repeat(2, minmax(0,1fr)); }
            .topbar-slider { width: 140px !important; }
        }
        @media (max-width: 900px) {
            .sidebar { width: 52px; }
            .brand-name, .brand-sub, .nav-item span:last-child,
            .nav-section-label, .sidebar-assistant { display: none; }
            .nav-item { justify-content: center; padding: 12px 0; }
            .nav-icon { width: auto; font-size: 18px; }
            .topbar-slider { display: none; }
        }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>
"""



def run_server(debug: bool = False, port: int = 8050):
    app.run(debug=debug, host=os.getenv("HOST", "127.0.0.1"), port=port)


if __name__ == "__main__":
    run_server(
        debug=os.getenv("DEBUG", "false").lower() == "true",
        port=env_int("PORT", 8050),
    )
