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
from typing import Dict, List

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
from dash import Dash, Input, Output, State, dcc, html
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


def bootstrap_demo() -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]], pd.DataFrame, Dict[str, object], pd.DataFrame, Dict[str, object]]:
    """Generate data, train model, forecast, optimize, and rank station sites."""
    print("Initializing Vidyut Prajna planning console...")

    config = CityConfig(
        h3_resolution=env_int("H3_RESOLUTION", 8),
        max_cells=env_int("MAX_CELLS", 54),
        num_days=env_int("NUM_DAYS", 7),
        freq=os.getenv("FREQ", "1h"),
        seed=env_int("SEED", 42),
        start=os.getenv("SIM_START", "2026-05-01"),
        scenario=os.getenv("SCENARIO", "orr_whitefield"),
    )

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
        epochs=env_int("EPOCHS", 10),
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
    station_budget = env_int("STATION_BUDGET", 8)
    recommendations, siting_summary = recommend_station_locations(
        optimized_df,
        adjacency=adjacency,
        top_n=station_budget,
        station_kw=float(os.getenv("STATION_KW", "22.0")),
    )

    print("Dashboard ready.")
    return raw_df, grid_df, adjacency, optimized_df, metrics, recommendations, siting_summary


RAW_DF, GRID_DF, ADJACENCY, OPTIMIZED_DF, METRICS, RECOMMENDATIONS, SITING_SUMMARY = bootstrap_demo()
GEOJSON = build_geojson(GRID_DF)
AGG_TS = aggregate_load_timeseries(OPTIMIZED_DF)
TIME_VALUES = sorted(OPTIMIZED_DF["timestamp"].unique())
MAP_CENTER = {
    "lat": float(GRID_DF["lat"].mean()) if "lat" in GRID_DF else DEFAULT_CENTER["lat"],
    "lon": float(GRID_DF["lon"].mean()) if "lon" in GRID_DF else DEFAULT_CENTER["lon"],
}
LLM = VidyutLLM()
SITING_ALL, _ = recommend_station_locations(
    OPTIMIZED_DF,
    adjacency=ADJACENCY,
    top_n=len(GRID_DF),
    station_kw=float(os.getenv("STATION_KW", "22.0")),
)
SITING_BY_CELL = dict(zip(SITING_ALL["h3_cell"], SITING_ALL["siting_score"]))


def make_slider_marks() -> Dict[int, str]:
    marks: Dict[int, str] = {}
    step = max(1, len(TIME_VALUES) // 8)
    for idx in range(0, len(TIME_VALUES), step):
        marks[idx] = pd.Timestamp(TIME_VALUES[idx]).strftime("%H:%M")
    marks[len(TIME_VALUES) - 1] = pd.Timestamp(TIME_VALUES[-1]).strftime("%H:%M")
    return marks


def selected_time(time_index: int) -> pd.Timestamp:
    return pd.Timestamp(TIME_VALUES[int(np.clip(time_index, 0, len(TIME_VALUES) - 1))])


def _plot_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#1f2937", family="Inter, Segoe UI, Arial, sans-serif"),
        margin={"l": 55, "r": 25, "t": 48, "b": 45},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb")
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb")
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
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#1f2937", family="Inter, Segoe UI, Arial, sans-serif"),
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
    fig.add_vline(x=ts, line_color="#374151", line_dash="dot", line_width=1.5)
    fig.update_layout(title="Unmanaged vs optimized aggregate load", xaxis_title="Forecast horizon", yaxis_title="kW")
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
    fig.add_vline(x=ts, line_color="#374151", line_dash="dot", line_width=1.5)
    fig.update_layout(title="Forecast quality on synthetic holdout", xaxis_title="Time", yaxis_title="kW")
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
    fig.add_vline(x=ts, line_color="#374151", line_dash="dot", line_width=1.5)
    fig.update_layout(
        title="Charging schedule, tariff signal, and solar alignment",
        xaxis_title="Forecast horizon",
        yaxis=dict(title="EV or solar kW"),
        yaxis2=dict(title="Tariff", overlaying="y", side="right", rangemode="tozero"),
    )
    return _plot_layout(fig, height=410)


def metric_card(title: str, value: str, detail: str, tone: str = "neutral") -> html.Div:
    return html.Div(
        [
            html.Div(title, className="metric-title"),
            html.Div(value, className=f"metric-value {tone}"),
            html.Div(detail, className="metric-detail"),
        ],
        className="metric-card",
    )


def metrics_row() -> html.Div:
    forecast_method = str(METRICS.get("forecast_method", "n/a")).replace("_", " ")
    forecast_health = str(METRICS.get("forecast_health", "n/a")).replace("_", " ")
    return html.Div(
        [
            metric_card("Peak reduction", format_pct(float(METRICS["peak_reduction_pct"])), f"{format_kw(METRICS['baseline_peak_kw'])} to {format_kw(METRICS['optimized_peak_kw'])}", "good"),
            metric_card("Overload events", f"{METRICS['overload_events_before']} -> {METRICS['overload_events_after']}", "Transformer-hours above 100%", "warn"),
            metric_card("Shifted energy", f"{float(METRICS['shifted_kwh']):,.1f} kWh", "Within priority/deadline windows", "good"),
            metric_card("Cost savings", format_inr(float(METRICS["estimated_cost_savings_inr"])), "Tariff-aware scheduling", "good"),
            metric_card("Forecast health", forecast_health.title(), f"{forecast_method}; MAE {float(METRICS.get('forecast_mae_kw', 0.0)):,.1f} kW", "neutral"),
            metric_card("Station baseline", format_pct(float(SITING_SUMMARY["capture_improvement_pct"])), "Demand capture lift vs uniform", "good"),
        ],
        className="metric-grid",
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
    return table_from_records(
        records,
        [("zone", "Zone"), ("type", "Type"), ("before", "Before"), ("after", "After"), ("demand", "Mean EV Demand")],
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
    return table_from_records(
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
                    html.Div([html.H2("Operational Snapshot"), dcc.Graph(id="overview-map")], className="panel wide-panel"),
                    html.Div([html.H2("Top Post-Optimization Risk Zones"), risk_table()], className="panel"),
                ],
                className="two-column",
            ),
            html.Div(
                [
                    html.Div([html.H2("Load Reduction Baseline"), dcc.Graph(id="load-figure")], className="panel"),
                    html.Div(
                        [
                            html.H2("Decision-Support Mode"),
                            html.Div("Sidecar observation only: no feeder, DTR, SCADA, or charger-control system is modified.", className="callout"),
                            html.Div(f"Scenario: {GRID_DF['corridor_name'].iloc[0]}", className="summary-line"),
                            html.Div(f"Cells: {len(GRID_DF)} adjacent H3 zones. Forecast horizon: {len(TIME_VALUES)} time steps.", className="summary-line"),
                            html.Div(f"Model: STGCN with {METRICS.get('training_samples', 'n/a')} training sequences over {METRICS.get('training_epochs', 'n/a')} epochs.", className="summary-line"),
                            html.Div(f"Forecast method: {METRICS.get('forecast_method', 'n/a')}", className="summary-line"),
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
                    html.Button("Play", id="play-button", n_clicks=0, className="control-button"),
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
            dcc.Slider(id="time-slider", min=0, max=len(TIME_VALUES) - 1, value=0, marks=make_slider_marks(), step=1, tooltip={"placement": "bottom"}),
            dcc.Interval(id="playback", interval=900, n_intervals=0, disabled=True),
            dcc.Store(id="playing", data=False),
            html.Div(
                [
                    html.Div([dcc.Graph(id="forecast-map"), html.Div(id="selected-summary")], className="panel wide-panel"),
                    html.Div([html.H2("Actual vs Forecast"), dcc.Graph(id="forecast-accuracy"), forecast_health_panel()], className="panel"),
                ],
                className="two-column",
            ),
        ],
        className="tab-content",
    )


def scheduling_tab() -> html.Div:
    return html.Div(
        [
            html.Div([html.H2("Schedule Optimization"), dcc.Graph(id="schedule-figure")], className="panel wide-panel"),
            html.Div(
                [
                    html.Div("Scheduling objective: shift flexible EV demand away from transformer stress and on-peak tariff periods while preserving priority charging.", className="callout"),
                    html.Div(f"Peak-to-average ratio: {float(METRICS['baseline_par']):.2f} unmanaged to {float(METRICS['optimized_par']):.2f} optimized.", className="summary-line"),
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
                className="metric-grid compact",
            ),
            html.Div(
                [
                    html.Div([html.H2("Siting Priority Map"), dcc.Graph(id="siting-map")], className="panel wide-panel"),
                    html.Div([html.H2("Ranked Station Recommendations"), recommendation_table()], className="panel"),
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
        ("STGCN forecast", "Graph convolution captures neighboring-cell pressure; LSTM captures temporal demand patterns."),
        ("Optimizer", "Priority-share, deadline, tariff, solar, and transformer-capacity constraints."),
        ("Infrastructure scoring", "Demand growth, charger gap, neighbor pressure, stress, and available headroom."),
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


def explanation_tab() -> html.Div:
    status = f"Hosted LLM enabled: {LLM.model}" if LLM.enabled else "Hosted LLM disabled; using deterministic local fallback."
    return html.Div(
        [
            html.Div(
                [
                    html.H2("Planner Explanation Assistant"),
                    html.Div(status, className="llm-status"),
                    html.Div(id="chat-window", className="chat-window"),
                    html.Div(
                        [
                            dcc.Input(id="chat-input", placeholder="Ask about demand, stress, scheduling, or station recommendations...", type="text", className="chat-input"),
                            html.Button("Send", id="chat-send", n_clicks=0, className="control-button"),
                        ],
                        className="chat-controls",
                    ),
                    dcc.Store(id="chat-history", data=[]),
                ],
                className="panel wide-panel",
            )
        ],
        className="tab-content",
    )


app = Dash(__name__, title="Vidyut Prajna")
server = app.server

app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.H1("Vidyut Prajna"),
                        html.Div("AI for EV charging optimization and infrastructure planning by BESCOM", className="subtitle"),
                    ],
                    className="header-copy",
                ),
                html.Div("Read-only decision support", className="mode-badge"),
            ],
            className="header",
        ),
        metrics_row(),
        dcc.Tabs(
            id="main-tabs",
            value="overview",
            className="tabs",
            children=[
                dcc.Tab(label="Overview", value="overview", children=overview_tab()),
                dcc.Tab(label="Demand Forecast", value="forecast", children=forecast_tab()),
                dcc.Tab(label="Scheduling Optimization", value="scheduling", children=scheduling_tab()),
                dcc.Tab(label="Infrastructure Planning", value="infrastructure", children=infrastructure_tab()),
                dcc.Tab(label="Data Sources", value="data", children=data_sources_tab()),
                dcc.Tab(label="Explanation Assistant", value="explain", children=explanation_tab()),
            ],
        ),
    ],
    className="app-shell",
)


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
    return new_state, not new_state, "Pause" if new_state else "Play"


@app.callback(Output("time-slider", "value"), Input("playback", "n_intervals"), State("time-slider", "value"))
def advance_time(_: int, current_value: int) -> int:
    return (int(current_value) + 1) % len(TIME_VALUES)


@app.callback(
    Output("overview-map", "figure"),
    Output("forecast-map", "figure"),
    Output("load-figure", "figure"),
    Output("forecast-accuracy", "figure"),
    Output("schedule-figure", "figure"),
    Output("siting-map", "figure"),
    Output("selected-summary", "children"),
    Input("time-slider", "value"),
    Input("map-mode", "value"),
)
def update_dashboard(time_index: int, map_mode: str):
    return (
        make_map_figure(time_index, "stress"),
        make_map_figure(time_index, map_mode),
        make_load_figure(time_index),
        make_forecast_accuracy_figure(time_index),
        make_scheduling_figure(time_index),
        make_map_figure(time_index, "siting"),
        selected_summary(time_index),
    )


@app.callback(
    Output("chat-history", "data"),
    Output("chat-input", "value"),
    Input("chat-send", "n_clicks"),
    State("chat-input", "value"),
    State("chat-history", "data"),
    State("time-slider", "value"),
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
    return history, ""


@app.callback(Output("chat-window", "children"), Input("chat-history", "data"))
def render_chat(history: list | None):
    if not history:
        return html.Div("Ask about peak demand, high-risk zones, schedule shifts, or station siting recommendations.", className="empty-chat")
    return [
        html.Div(dcc.Markdown(msg.get("content", "")), className=f"chat-bubble {msg.get('role', 'assistant')}")
        for msg in history[-10:]
    ]


app.index_string = """
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>Vidyut Prajna</title>
    {%favicon%}
    {%css%}
    <style>
        :root {
            --bg: #f3f4f6;
            --panel: #ffffff;
            --panel-alt: #f9fafb;
            --text: #111827;
            --muted: #6b7280;
            --border: #d1d5db;
            --blue: #2563eb;
            --green: #059669;
            --amber: #d97706;
            --red: #dc2626;
            --violet: #7c3aed;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            background: var(--bg);
            color: var(--text);
            font-family: Inter, Segoe UI, Arial, sans-serif;
        }
        .app-shell {
            max-width: 1720px;
            margin: 0 auto;
            padding: 20px 24px 32px;
        }
        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            padding: 20px 22px;
            background: #ffffff;
            border: 1px solid var(--border);
            border-left: 5px solid var(--green);
            border-radius: 8px;
        }
        h1 { margin: 0; font-size: 28px; letter-spacing: 0; }
        h2 { margin: 0 0 14px; font-size: 17px; letter-spacing: 0; }
        .subtitle { color: var(--muted); margin-top: 4px; font-size: 14px; }
        .mode-badge {
            white-space: nowrap;
            color: #065f46;
            background: #d1fae5;
            border: 1px solid #a7f3d0;
            border-radius: 6px;
            padding: 8px 10px;
            font-weight: 700;
            font-size: 13px;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 12px;
            margin: 14px 0;
        }
        .metric-grid.compact { grid-template-columns: repeat(4, minmax(0, 1fr)); }
        .metric-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 14px;
            min-height: 94px;
        }
        .metric-title { color: var(--muted); font-size: 11px; text-transform: uppercase; font-weight: 800; letter-spacing: 0.08em; }
        .metric-value { margin-top: 8px; font-size: 24px; font-weight: 800; line-height: 1.1; }
        .metric-value.good { color: var(--green); }
        .metric-value.warn { color: var(--amber); }
        .metric-detail { margin-top: 6px; color: var(--muted); font-size: 12px; line-height: 1.35; }
        .tabs .tab { border-color: var(--border) !important; padding: 12px 14px !important; font-weight: 700; }
        .tabs .tab--selected { border-top: 3px solid var(--green) !important; color: var(--green) !important; }
        .tab-content { padding-top: 14px; }
        .two-column {
            display: grid;
            grid-template-columns: minmax(0, 1.3fr) minmax(360px, 0.7fr);
            gap: 14px;
            margin-bottom: 14px;
            align-items: start;
        }
        .panel {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            overflow: hidden;
        }
        .wide-panel { min-width: 0; }
        .control-row {
            display: flex;
            align-items: center;
            gap: 16px;
            justify-content: space-between;
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 14px;
            margin-bottom: 12px;
        }
        .control-button {
            border: 1px solid #047857;
            background: var(--green);
            color: white;
            border-radius: 6px;
            padding: 9px 15px;
            cursor: pointer;
            font-weight: 800;
        }
        .radio-row label { margin-right: 14px; font-size: 13px; color: #374151; }
        .summary-box, .callout {
            background: var(--panel-alt);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 14px;
            margin-top: 12px;
        }
        .summary-line { color: #374151; font-size: 13px; margin: 8px 0; line-height: 1.4; }
        .data-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }
        .data-table th {
            text-align: left;
            color: #374151;
            background: #f3f4f6;
            border-bottom: 1px solid var(--border);
            padding: 9px 8px;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .data-table td {
            border-bottom: 1px solid #e5e7eb;
            padding: 9px 8px;
            vertical-align: top;
        }
        .source-row {
            display: grid;
            grid-template-columns: 220px 1fr;
            gap: 16px;
            border-bottom: 1px solid #e5e7eb;
            padding: 12px 0;
        }
        .source-title { font-weight: 800; }
        .source-detail { color: var(--muted); }
        .llm-status { color: var(--green); font-weight: 800; margin-bottom: 12px; }
        .chat-window {
            min-height: 260px;
            max-height: 430px;
            overflow-y: auto;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: var(--panel-alt);
            padding: 12px;
        }
        .chat-controls { display: flex; gap: 10px; margin-top: 12px; }
        .chat-input {
            flex: 1;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 10px 12px;
            font-size: 14px;
        }
        .chat-bubble {
            max-width: 82%;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px 12px;
            margin: 8px 0;
            background: white;
            line-height: 1.45;
            font-size: 13px;
        }
        .chat-bubble.user {
            margin-left: auto;
            border-color: #bfdbfe;
            background: #eff6ff;
        }
        .chat-bubble.assistant {
            border-color: #bbf7d0;
            background: #f0fdf4;
        }
        .empty-chat { color: var(--muted); font-size: 13px; }
        @media (max-width: 1300px) {
            .metric-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
            .metric-grid.compact { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .two-column { grid-template-columns: 1fr; }
        }
        @media (max-width: 720px) {
            .app-shell { padding: 12px; }
            .header, .control-row, .chat-controls { flex-direction: column; align-items: stretch; }
            .metric-grid, .metric-grid.compact { grid-template-columns: 1fr; }
            .source-row { grid-template-columns: 1fr; }
            .mode-badge { white-space: normal; }
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
    app.run(debug=debug, port=port)


if __name__ == "__main__":
    run_server(
        debug=os.getenv("DEBUG", "false").lower() == "true",
        port=env_int("PORT", 8050),
    )
