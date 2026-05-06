"""Vidyut Prajna Professional Dashboard - MSN Weather Inspired Design

A professional-grade planning console for EV charging optimization
with enhanced data simulation, Lagrangian optimization, and real-time
forecasting feedback.

Run:
    python -m src.dashboard.enhanced_app
Or:
    python main.py --enhanced

Then open: http://127.0.0.1:8050
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*_: object, **__: object) -> bool:
        return False

load_dotenv()

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, Input, Output, State, dcc, html, callback_context
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
from src.optimization.siting import recommend_station_locations
from src.spatial_grid.simulation import CityConfig, generate_synthetic_data

# Import enhanced modules
try:
    from src.spatial_grid.enhanced_simulation import generate_enhanced_synthetic_data
    from src.optimization.lagrangian_optimizer import optimize_charging_schedule_lagrangian, LagrangianOptimizer
    from src.intelligence.kde_arrivals import SpatioTemporalKDE, add_kde_features
    from src.intelligence.feedback_loop import ForecastFeedbackLoop, OnlineForecasterWrapper
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False
    from src.optimization.optimizer import optimize_charging_schedule

# Configuration
DEFAULT_CENTER = {"lat": 12.9716, "lon": 77.5946}


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).lower() in ("true", "1", "yes")


# ============================================================================
# BOOTSTRAP FUNCTIONS
# ============================================================================

def bootstrap_enhanced_demo() -> tuple:
    """Generate data with enhanced simulation and Lagrangian optimization."""
    print("=" * 60)
    print("VIDYUT PRAJNA - Enhanced Planning Console")
    print("=" * 60)
    
    config = CityConfig(
        h3_resolution=env_int("H3_RESOLUTION", 8),
        max_cells=env_int("MAX_CELLS", 54),
        num_days=env_int("NUM_DAYS", 7),
        freq=os.getenv("FREQ", "1h"),
        seed=env_int("SEED", 42),
        start=os.getenv("SIM_START", "2026-05-01"),
        scenario=os.getenv("SCENARIO", "orr_whitefield"),
    )
    
    # Generate enhanced synthetic data
    print(f"\n[1/6] Generating enhanced synthetic data...")
    print(f"      Scenario: {config.scenario}")
    print(f"      Cells: {config.max_cells}, Days: {config.num_days}")
    
    if ENHANCED_AVAILABLE and env_bool("USE_ENHANCED_SIM", True):
        raw_df, grid_df, adjacency, ocpp_sessions, dtrs = generate_enhanced_synthetic_data(
            config,
            include_ocpp=True,
            include_gig_fleet=True,
            apply_anonymization=True,
        )
        print(f"      ✓ Enhanced simulation with gig fleet & OCPP data")
        print(f"      ✓ Generated {len(ocpp_sessions)} OCPP sessions")
        print(f"      ✓ Generated {len(dtrs)} DTR specifications")
    else:
        raw_df, grid_df, adjacency = generate_synthetic_data(config)
        ocpp_sessions, dtrs = [], []
        print(f"      ✓ Standard simulation")
    
    # Prepare train/test split
    unique_times = sorted(raw_df["timestamp"].unique())
    default_horizon_steps = min(24, max(4, len(unique_times) // 4))
    default_train_steps = max(24, len(unique_times) - default_horizon_steps)
    train_steps = min(env_int("TRAIN_STEPS", default_train_steps), len(unique_times) - 4)
    horizon_steps = min(env_int("FORECAST_STEPS", default_horizon_steps), len(unique_times) - train_steps)
    
    if horizon_steps <= 0:
        raise RuntimeError("Not enough simulated data. Increase NUM_DAYS.")
    
    train_times = unique_times[:train_steps]
    future_times = unique_times[train_steps:train_steps + horizon_steps]
    train_df = raw_df[raw_df["timestamp"].isin(train_times)].copy()
    future_df = raw_df[raw_df["timestamp"].isin(future_times)].copy()
    
    # Add KDE features if available
    print(f"\n[2/6] Computing KDE arrival features...")
    if ENHANCED_AVAILABLE and env_bool("USE_KDE", True):
        train_df = add_kde_features(train_df, adjacency)
        future_df = add_kde_features(future_df, adjacency, historical_df=train_df)
        print(f"      ✓ Added KDE arrival rate predictions")
        print(f"      ✓ Added neighbor demand propagation")
    else:
        print(f"      - KDE features disabled")
    
    # Train STGCN
    seq_len = min(env_int("SEQ_LEN", 12), max(3, len(train_times) // 3))
    print(f"\n[3/6] Training STGCN model...")
    print(f"      Samples: {len(train_df)}, Seq length: {seq_len}")
    
    forecaster = STGCNForecaster(
        seq_len=seq_len,
        hidden_size=env_int("HIDDEN_SIZE", 48),
        epochs=env_int("EPOCHS", 10),
        num_blocks=env_int("STGCN_BLOCKS", 2),
        seed=config.seed,
    )
    
    # Wrap with feedback loop if available
    if ENHANCED_AVAILABLE and env_bool("USE_FEEDBACK", True):
        wrapped_forecaster = OnlineForecasterWrapper(forecaster)
        wrapped_forecaster.fit(train_df, adjacency)
        pred_df = wrapped_forecaster.forecast(train_df, future_df, adjacency, horizon_steps=horizon_steps)
        feedback_report = wrapped_forecaster.get_performance_report()
        print(f"      ✓ STGCN with feedback loop wrapper")
    else:
        forecaster.fit(train_df, adjacency)
        pred_df = forecaster.forecast(train_df, future_df, adjacency, horizon_steps=horizon_steps)
        feedback_report = {}
        print(f"      ✓ Standard STGCN training")
    
    forecast_info = forecaster.forecast_info.copy()
    
    # Optimize with Lagrangian
    print(f"\n[4/6] Optimizing charging schedule...")
    if ENHANCED_AVAILABLE and env_bool("USE_LAGRANGIAN", True):
        optimizer = LagrangianOptimizer()
        optimized_df, metrics = optimizer.optimize(pred_df)
        shadow_prices = optimizer.get_shadow_prices_df()
        print(f"      ✓ Lagrangian MCDM optimization")
        print(f"      ✓ Iterations: {metrics.get('lagrangian_iterations', 'n/a')}")
        print(f"      ✓ Converged: {metrics.get('converged', 'n/a')}")
        metrics["shadow_prices_summary"] = {
            "max": float(shadow_prices["shadow_price"].max()) if len(shadow_prices) > 0 else 0,
            "mean": float(shadow_prices["shadow_price"].mean()) if len(shadow_prices) > 0 else 0,
            "non_zero": int((shadow_prices["shadow_price"] > 0).sum()) if len(shadow_prices) > 0 else 0,
        }
    else:
        optimized_df, metrics = optimize_charging_schedule(pred_df)
        shadow_prices = pd.DataFrame()
        print(f"      ✓ Greedy optimization")
    
    metrics.update(forecast_info)
    metrics["feedback_loop"] = feedback_report
    
    if forecaster.training_info:
        metrics["training_samples"] = forecaster.training_info.train_samples
        metrics["training_final_loss"] = forecaster.training_info.final_loss
        metrics["training_epochs"] = forecaster.training_info.epochs
    
    # Siting recommendations
    print(f"\n[5/6] Computing siting recommendations...")
    station_budget = env_int("STATION_BUDGET", 8)
    recommendations, siting_summary = recommend_station_locations(
        optimized_df,
        adjacency=adjacency,
        top_n=station_budget,
        station_kw=float(os.getenv("STATION_KW", "22.0")),
    )
    print(f"      ✓ Ranked {len(recommendations)} candidate locations")
    
    # Final summary
    print(f"\n[6/6] Dashboard ready!")
    print(f"      Peak reduction: {metrics['peak_reduction_pct']:.1f}%")
    print(f"      Cost savings: ₹{metrics['estimated_cost_savings_inr']:,.0f}")
    print(f"      CO2 reduction: {metrics['co2_reduction_kg']:.1f} kg")
    print("=" * 60)
    
    return (
        raw_df, grid_df, adjacency, optimized_df, metrics,
        recommendations, siting_summary, ocpp_sessions, dtrs, shadow_prices
    )


# Bootstrap data
(
    RAW_DF, GRID_DF, ADJACENCY, OPTIMIZED_DF, METRICS,
    RECOMMENDATIONS, SITING_SUMMARY, OCPP_SESSIONS, DTR_SPECS, SHADOW_PRICES
) = bootstrap_enhanced_demo()

GEOJSON = build_geojson(GRID_DF)
AGG_TS = aggregate_load_timeseries(OPTIMIZED_DF)
TIME_VALUES = sorted(OPTIMIZED_DF["timestamp"].unique())
MAP_CENTER = {
    "lat": float(GRID_DF["lat"].mean()) if "lat" in GRID_DF else DEFAULT_CENTER["lat"],
    "lon": float(GRID_DF["lon"].mean()) if "lon" in GRID_DF else DEFAULT_CENTER["lon"],
}
LLM = VidyutLLM()

# Pre-compute siting scores for all cells
SITING_ALL, _ = recommend_station_locations(
    OPTIMIZED_DF,
    adjacency=ADJACENCY,
    top_n=len(GRID_DF),
    station_kw=float(os.getenv("STATION_KW", "22.0")),
)
SITING_BY_CELL = dict(zip(SITING_ALL["h3_cell"], SITING_ALL["siting_score"]))


# ============================================================================
# DASH APPLICATION
# ============================================================================

app = Dash(
    __name__,
    title="Vidyut Prajna - EV Charging Intelligence",
    assets_folder="assets",
    suppress_callback_exceptions=True,
)

server = app.server


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def make_slider_marks() -> Dict[int, str]:
    marks: Dict[int, str] = {}
    step = max(1, len(TIME_VALUES) // 8)
    for idx in range(0, len(TIME_VALUES), step):
        marks[idx] = pd.Timestamp(TIME_VALUES[idx]).strftime("%H:%M")
    marks[len(TIME_VALUES) - 1] = pd.Timestamp(TIME_VALUES[-1]).strftime("%H:%M")
    return marks


def selected_time(time_index: int) -> pd.Timestamp:
    return pd.Timestamp(TIME_VALUES[int(np.clip(time_index, 0, len(TIME_VALUES) - 1))])


def _plot_layout(fig: go.Figure, height: int = 360, dark: bool = False) -> go.Figure:
    bg_color = "#1f2937" if dark else "#ffffff"
    grid_color = "#374151" if dark else "#e5e7eb"
    font_color = "#f9fafb" if dark else "#1f2937"
    
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=bg_color,
        font=dict(color=font_color, family="Inter, Segoe UI, Arial, sans-serif"),
        margin={"l": 55, "r": 25, "t": 48, "b": 45},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor=grid_color)
    fig.update_yaxes(showgrid=True, gridcolor=grid_color)
    return fig


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def make_map_figure(time_index: int, mode: str) -> go.Figure:
    ts = selected_time(time_index)
    frame = OPTIMIZED_DF[OPTIMIZED_DF["timestamp"] == ts].copy()
    frame["siting_score"] = frame["h3_cell"].map(SITING_BY_CELL).fillna(0.0)

    color_configs = {
        "predicted": ("baseline_ev_load_kw", "Predicted EV Demand (kW)", "YlOrRd"),
        "optimized": ("optimized_ev_load_kw", "Optimized EV Load (kW)", "Viridis"),
        "stress": (None, "Transformer Utilization (%)", "RdYlGn_r"),
        "siting": ("siting_score", "Siting Priority Score", "Magma"),
    }
    
    col, title, colorscale = color_configs.get(mode, color_configs["predicted"])
    
    if mode == "stress":
        z = 100 * frame["optimized_transformer_utilization"]
    else:
        z = frame[col]

    custom = np.stack([
        frame["zone_name"].astype(str),
        frame["zone_type"].astype(str),
        frame["baseline_ev_load_kw"].round(1),
        frame["optimized_ev_load_kw"].round(1),
        (100 * frame["optimized_transformer_utilization"]).round(1),
        frame["siting_score"].round(1),
        frame["stress_label"].astype(str),
    ], axis=-1)

    fig = go.Figure(go.Choroplethmapbox(
        geojson=GEOJSON,
        locations=frame["h3_cell"],
        z=z,
        featureidkey="properties.h3_cell",
        colorscale=colorscale,
        marker_opacity=0.75,
        marker_line_width=1,
        marker_line_color="rgba(255,255,255,0.6)",
        colorbar=dict(
            title=title,
            thickness=15,
            len=0.7,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
        ),
        customdata=custom,
        hovertemplate=(
            "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
            "Predicted: %{customdata[2]} kW<br>"
            "Optimized: %{customdata[3]} kW<br>"
            "Utilization: %{customdata[4]}%<br>"
            "Siting Score: %{customdata[5]}<br>"
            "Status: %{customdata[6]}<extra></extra>"
        ),
    ))
    
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=10.5,
        mapbox_center=MAP_CENTER,
        height=520,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_load_comparison_figure() -> go.Figure:
    fig = go.Figure()

    # Confidence band
    if "prediction_std_kw" in AGG_TS.columns:
        upper = AGG_TS["optimized_total_load_kw"] + 1.96 * AGG_TS["prediction_std_kw"]
        lower = (AGG_TS["optimized_total_load_kw"] - 1.96 * AGG_TS["prediction_std_kw"]).clip(lower=0)
        fig.add_trace(go.Scatter(
            x=pd.concat([AGG_TS["timestamp"], AGG_TS["timestamp"][::-1]]),
            y=pd.concat([upper, lower[::-1]]),
            fill="toself",
            fillcolor="rgba(16, 185, 129, 0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% Confidence Band",
            showlegend=True,
        ))

    # Traces
    fig.add_trace(go.Scatter(
        x=AGG_TS["timestamp"],
        y=AGG_TS["baseline_total_load_kw"],
        name="Unmanaged Load",
        line=dict(color="#ef4444", width=2.5),
        mode="lines",
    ))
    
    fig.add_trace(go.Scatter(
        x=AGG_TS["timestamp"],
        y=AGG_TS["optimized_total_load_kw"],
        name="Optimized Load",
        line=dict(color="#10b981", width=3),
        mode="lines",
    ))
    
    fig.add_trace(go.Scatter(
        x=AGG_TS["timestamp"],
        y=AGG_TS["safe_capacity_kw"],
        name="95% Capacity Limit",
        line=dict(color="#f59e0b", width=2, dash="dash"),
        mode="lines",
    ))
    
    if "solar_generation_kw" in AGG_TS.columns:
        fig.add_trace(go.Scatter(
            x=AGG_TS["timestamp"],
            y=AGG_TS["solar_generation_kw"],
            name="Solar Generation",
            line=dict(color="#0ea5e9", width=2, dash="dot"),
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(14, 165, 233, 0.1)",
        ))

    fig.update_layout(
        title=dict(text="Grid Load Comparison", font=dict(size=16, weight="bold")),
        xaxis_title="Forecast Horizon",
        yaxis_title="Power (kW)",
        hovermode="x unified",
    )
    
    return _plot_layout(fig, height=400)


def make_scheduling_figure() -> go.Figure:
    fig = go.Figure()
    
    # EV charging traces
    fig.add_trace(go.Scatter(
        x=AGG_TS["timestamp"],
        y=AGG_TS["baseline_ev_load_kw"],
        name="Unmanaged EV",
        line=dict(color="#ef4444", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(239, 68, 68, 0.1)",
    ))
    
    fig.add_trace(go.Scatter(
        x=AGG_TS["timestamp"],
        y=AGG_TS["optimized_ev_load_kw"],
        name="Optimized EV",
        line=dict(color="#10b981", width=3),
        fill="tozeroy",
        fillcolor="rgba(16, 185, 129, 0.15)",
    ))
    
    fig.add_trace(go.Scatter(
        x=AGG_TS["timestamp"],
        y=AGG_TS["solar_generation_kw"],
        name="Solar Available",
        line=dict(color="#0ea5e9", width=2, dash="dot"),
    ))
    
    # Tariff on secondary axis
    tariff = OPTIMIZED_DF.groupby("timestamp", as_index=False)["tariff_multiplier"].mean().sort_values("timestamp")
    fig.add_trace(go.Scatter(
        x=tariff["timestamp"],
        y=tariff["tariff_multiplier"],
        name="Tariff Signal",
        yaxis="y2",
        line=dict(color="#8b5cf6", width=2, dash="dash"),
    ))

    fig.update_layout(
        title=dict(text="Charging Schedule Optimization", font=dict(size=16, weight="bold")),
        xaxis_title="Time",
        yaxis=dict(title="Power (kW)", side="left"),
        yaxis2=dict(title="Tariff Multiplier", overlaying="y", side="right", rangemode="tozero"),
        hovermode="x unified",
    )
    
    return _plot_layout(fig, height=420)


def make_weather_impact_figure() -> go.Figure:
    """Show weather impact on demand (MSN Weather inspired)."""
    if "temperature_c" not in OPTIMIZED_DF.columns:
        return go.Figure()
    
    hourly = OPTIMIZED_DF.groupby("timestamp").agg({
        "temperature_c": "mean",
        "demand_kw": "sum",
        "rainfall_mm": "mean" if "rainfall_mm" in OPTIMIZED_DF.columns else lambda x: 0,
    }).reset_index()
    
    fig = go.Figure()
    
    # Temperature
    fig.add_trace(go.Scatter(
        x=hourly["timestamp"],
        y=hourly["temperature_c"],
        name="Temperature (°C)",
        line=dict(color="#f59e0b", width=2),
        yaxis="y",
    ))
    
    # Demand
    fig.add_trace(go.Scatter(
        x=hourly["timestamp"],
        y=hourly["demand_kw"],
        name="EV Demand (kW)",
        line=dict(color="#3b82f6", width=2.5),
        yaxis="y2",
        fill="tozeroy",
        fillcolor="rgba(59, 130, 246, 0.1)",
    ))
    
    # Rainfall bars
    if "rainfall_mm" in hourly.columns and hourly["rainfall_mm"].sum() > 0:
        fig.add_trace(go.Bar(
            x=hourly["timestamp"],
            y=hourly["rainfall_mm"],
            name="Rainfall (mm)",
            marker_color="rgba(14, 165, 233, 0.4)",
            yaxis="y3",
        ))
    
    fig.update_layout(
        title=dict(text="Weather & Demand Correlation", font=dict(size=16, weight="bold")),
        yaxis=dict(title="Temperature (°C)", side="left", showgrid=False),
        yaxis2=dict(title="Demand (kW)", overlaying="y", side="right"),
        yaxis3=dict(title="Rainfall", overlaying="y", side="right", position=0.95, showgrid=False) if "rainfall_mm" in hourly.columns else {},
        hovermode="x unified",
    )
    
    return _plot_layout(fig, height=350)


# ============================================================================
# COMPONENT FUNCTIONS
# ============================================================================

def metric_card(title: str, value: str, detail: str, tone: str = "neutral", icon: str = "") -> html.Div:
    return html.Div([
        html.Div([
            html.Span(icon, className="metric-icon") if icon else None,
            html.Span(title, className="metric-title"),
        ], className="metric-header"),
        html.Div(value, className=f"metric-value {tone}"),
        html.Div(detail, className="metric-detail"),
    ], className="metric-card animate-in")


def status_badge(text: str, status: str = "live") -> html.Span:
    return html.Span(text, className=f"status-badge {status}")


def create_header() -> html.Div:
    return html.Div([
        html.Div([
            html.Div("⚡", className="logo"),
            html.Div([
                html.H1("Vidyut Prajna", className="header-title"),
                html.Div("EV Charging Intelligence Platform", className="header-subtitle"),
            ]),
        ], className="header-left"),
        html.Div([
            status_badge("Live Simulation", "live"),
            html.Span(
                f"{datetime.now().strftime('%d %b %Y, %H:%M')}",
                className="header-time"
            ),
        ], className="header-right"),
    ], className="header")


def create_metrics_row() -> html.Div:
    forecast_method = str(METRICS.get("forecast_method", "STGCN")).replace("_", " ").title()
    optimizer_type = str(METRICS.get("optimizer_type", "greedy")).replace("_", " ").title()
    
    cards = [
        metric_card(
            "Peak Reduction",
            format_pct(float(METRICS["peak_reduction_pct"])),
            f"{format_kw(METRICS['baseline_peak_kw'])} → {format_kw(METRICS['optimized_peak_kw'])}",
            "good", "📉"
        ),
        metric_card(
            "Overload Events",
            f"{METRICS['overload_events_before']} → {METRICS['overload_events_after']}",
            "Transformer-hours above 100%",
            "warn" if METRICS['overload_events_after'] > 0 else "good",
            "⚠️"
        ),
        metric_card(
            "Energy Shifted",
            f"{float(METRICS['shifted_kwh']):,.0f} kWh",
            "Within priority/deadline windows",
            "good", "🔄"
        ),
        metric_card(
            "Cost Savings",
            format_inr(float(METRICS["estimated_cost_savings_inr"])),
            "Tariff-optimized scheduling",
            "good", "💰"
        ),
        metric_card(
            "CO₂ Reduction",
            f"{float(METRICS['co2_reduction_kg']):,.1f} kg",
            "Solar-aligned charging",
            "good", "🌱"
        ),
        metric_card(
            "Forecast Model",
            forecast_method,
            f"MAE: {float(METRICS.get('forecast_mae_kw', 0)):,.1f} kW",
            "neutral", "🧠"
        ),
    ]
    
    return html.Div(cards, className="metric-grid")


def create_risk_table() -> html.Div:
    records = METRICS.get("top_risk_zones", [])[:6]
    
    rows = []
    for item in records:
        before = 100 * float(item.get("max_baseline_utilization", 0))
        after = 100 * float(item.get("max_optimized_utilization", 0))
        
        status_class = "critical" if after >= 100 else "warn" if after >= 88 else "good"
        
        rows.append(html.Tr([
            html.Td(item.get("zone_name", ""), className="font-medium"),
            html.Td(item.get("zone_type", "").title()),
            html.Td(f"{before:.1f}%"),
            html.Td([
                html.Span(f"{after:.1f}%", className=f"status-badge {status_class}")
            ]),
            html.Td(format_kw(float(item.get("mean_predicted_demand_kw", 0)))),
        ]))
    
    return html.Table([
        html.Thead(html.Tr([
            html.Th("Zone"),
            html.Th("Type"),
            html.Th("Before"),
            html.Th("After"),
            html.Th("Avg Demand"),
        ])),
        html.Tbody(rows),
    ], className="data-table")


def create_recommendations_table() -> html.Div:
    rows = []
    for _, row in RECOMMENDATIONS.iterrows():
        score = float(row["siting_score"])
        score_class = "good" if score >= 70 else "warn" if score >= 50 else "neutral"
        
        rows.append(html.Tr([
            html.Td(f"#{int(row['rank'])}", className="font-bold"),
            html.Td(row["zone_name"], className="font-medium"),
            html.Td(row["zone_type"].title()),
            html.Td([
                html.Span(f"{score:.0f}", className=f"status-badge {score_class}")
            ]),
            html.Td(format_kw(float(row["peak_predicted_demand_kw"]))),
            html.Td(format_kw(float(row["capacity_headroom_kw"]))),
            html.Td(row["capacity_feasibility"]),
        ]))
    
    return html.Table([
        html.Thead(html.Tr([
            html.Th("Rank"),
            html.Th("Zone"),
            html.Th("Type"),
            html.Th("Score"),
            html.Th("Peak Demand"),
            html.Th("Headroom"),
            html.Th("Feasibility"),
        ])),
        html.Tbody(rows),
    ], className="data-table")


# ============================================================================
# TAB CONTENT
# ============================================================================

def overview_tab() -> html.Div:
    return html.Div([
        create_metrics_row(),
        html.Div([
            html.Div([
                html.H2("Grid Overview"),
                dcc.Graph(
                    id="overview-map",
                    figure=make_map_figure(0, "stress"),
                    config={"displayModeBar": False},
                ),
            ], className="panel wide-panel"),
            html.Div([
                html.H2("Risk Assessment"),
                create_risk_table(),
                html.Div([
                    html.Div(f"Scenario: {GRID_DF['corridor_name'].iloc[0]}", className="summary-line"),
                    html.Div(f"H3 Cells: {len(GRID_DF)} | Time Steps: {len(TIME_VALUES)}", className="summary-line"),
                    html.Div(f"Optimizer: {METRICS.get('optimizer_type', 'greedy').title()}", className="summary-line"),
                ], className="summary-box"),
            ], className="panel"),
        ], className="two-column"),
        html.Div([
            html.Div([
                html.H2("Load Profile Comparison"),
                dcc.Graph(
                    id="load-comparison",
                    figure=make_load_comparison_figure(),
                    config={"displayModeBar": False},
                ),
            ], className="panel"),
        ]),
    ], className="tab-content animate-in")


def forecast_tab() -> html.Div:
    return html.Div([
        html.Div([
            html.Div([
                html.Button("▶ Play", id="play-button", n_clicks=0, className="control-button"),
                dcc.RadioItems(
                    id="map-mode",
                    options=[
                        {"label": "Predicted Demand", "value": "predicted"},
                        {"label": "Optimized Load", "value": "optimized"},
                        {"label": "Grid Stress", "value": "stress"},
                        {"label": "Siting Priority", "value": "siting"},
                    ],
                    value="predicted",
                    inline=True,
                    className="radio-row",
                ),
            ], className="control-row"),
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
        ], className="panel"),
        
        html.Div([
            html.Div([
                html.H2("Spatial Demand Forecast"),
                html.Div(id="selected-time-display", className="callout"),
                dcc.Graph(id="forecast-map", config={"displayModeBar": False}),
            ], className="panel wide-panel"),
            html.Div([
                html.H2("Forecast Performance"),
                dcc.Graph(id="forecast-metrics", config={"displayModeBar": False}),
                html.Div([
                    html.Div(f"Method: {METRICS.get('forecast_method', 'n/a')}", className="summary-line"),
                    html.Div(f"Health: {METRICS.get('forecast_health', 'n/a')}", className="summary-line"),
                    html.Div(f"MAE: {float(METRICS.get('forecast_mae_kw', 0)):,.1f} kW", className="summary-line"),
                    html.Div(f"sMAPE: {float(METRICS.get('forecast_smape_pct', 0)):,.1f}%", className="summary-line"),
                ], className="summary-box"),
            ], className="panel"),
        ], className="two-column"),
    ], className="tab-content animate-in")


def scheduling_tab() -> html.Div:
    return html.Div([
        html.Div([
            html.Div([
                html.H2("Charging Schedule Optimization"),
                dcc.Graph(
                    figure=make_scheduling_figure(),
                    config={"displayModeBar": False},
                ),
            ], className="panel wide-panel"),
            html.Div([
                html.H2("Optimization Results"),
                html.Div([
                    html.Div([
                        html.Span("Variance Reduction", className="metric-title"),
                        html.Span(format_pct(float(METRICS["variance_reduction_pct"])), className="metric-value good"),
                    ]),
                    html.Div([
                        html.Span("PAR Improvement", className="metric-title"),
                        html.Span(f"{METRICS['baseline_par']:.2f} → {METRICS['optimized_par']:.2f}", className="metric-value good"),
                    ]),
                    html.Div([
                        html.Span("V2G Ready Slots", className="metric-title"),
                        html.Span(f"{METRICS['v2g_potential_slots']}", className="metric-value neutral"),
                    ]),
                    html.Div([
                        html.Span("Deadlines Met", className="metric-title"),
                        html.Span(f"{METRICS['deadlines_met_pct']:.0f}%", className="metric-value good"),
                    ]),
                ], className="summary-box"),
                html.Div([
                    html.H3("Optimizer Details"),
                    html.Div(f"Type: {METRICS.get('optimizer_type', 'greedy').title()}", className="summary-line"),
                    html.Div(f"Iterations: {METRICS.get('lagrangian_iterations', 'n/a')}", className="summary-line"),
                    html.Div(f"Converged: {METRICS.get('converged', 'n/a')}", className="summary-line"),
                    html.Div(f"Energy Error: {METRICS['energy_preservation_error_pct']:.3f}%", className="summary-line"),
                ], className="summary-box"),
            ], className="panel"),
        ], className="two-column"),
    ], className="tab-content animate-in")


def infrastructure_tab() -> html.Div:
    return html.Div([
        html.Div([
            html.Div([
                html.H2("Station Siting Recommendations"),
                dcc.Graph(
                    id="siting-map",
                    figure=make_map_figure(0, "siting"),
                    config={"displayModeBar": False},
                ),
            ], className="panel wide-panel"),
            html.Div([
                html.H2("Siting Analysis"),
                html.Div([
                    html.Div([
                        html.Span("Capture Improvement", className="metric-title"),
                        html.Span(format_pct(float(SITING_SUMMARY["capture_improvement_pct"])), className="metric-value good"),
                    ]),
                    html.Div([
                        html.Span("vs Uniform Baseline", className="metric-title"),
                        html.Span(f"{float(SITING_SUMMARY.get('recommended_demand_capture_pct', 0)):.1f}% vs {float(SITING_SUMMARY.get('uniform_demand_capture_pct', 0)):.1f}%", className="metric-value"),
                    ]),
                ], className="summary-box"),
                html.Div("Scoring weights: Demand (30%), Growth (20%), Charger Gap (16%), Neighbor Pressure (14%), Stress (12%), Headroom (8%)", className="callout"),
            ], className="panel"),
        ], className="two-column"),
        html.Div([
            html.Div([
                html.H2("Recommended Locations"),
                create_recommendations_table(),
            ], className="panel"),
        ]),
    ], className="tab-content animate-in")


def weather_tab() -> html.Div:
    return html.Div([
        html.Div([
            html.Div([
                html.H2("Weather Impact Analysis"),
                dcc.Graph(
                    figure=make_weather_impact_figure(),
                    config={"displayModeBar": False},
                ),
            ], className="panel"),
        ]),
        html.Div([
            html.Div([
                html.H2("Environmental Conditions"),
                html.Div([
                    html.Div(f"Avg Temperature: {OPTIMIZED_DF['temperature_c'].mean():.1f}°C" if 'temperature_c' in OPTIMIZED_DF.columns else "N/A", className="summary-line"),
                    html.Div(f"Total Rainfall: {OPTIMIZED_DF.get('rainfall_mm', pd.Series([0])).sum():.1f}mm" if 'rainfall_mm' in OPTIMIZED_DF.columns else "N/A", className="summary-line"),
                    html.Div(f"Solar Generation: {OPTIMIZED_DF['solar_generation_kw'].sum():.0f} kWh" if 'solar_generation_kw' in OPTIMIZED_DF.columns else "N/A", className="summary-line"),
                ], className="summary-box"),
            ], className="panel"),
            html.Div([
                html.H2("Data Sources"),
                html.Div([
                    html.Div("✓ Enhanced monsoon weather simulation", className="summary-line"),
                    html.Div("✓ Gig fleet (Swiggy/Zomato) demand", className="summary-line"),
                    html.Div("✓ OCPP charging session simulation", className="summary-line"),
                    html.Div("✓ DTR topology modeling", className="summary-line"),
                    html.Div("✓ K-anonymity privacy masking", className="summary-line"),
                ], className="summary-box"),
            ], className="panel"),
        ], className="two-column"),
    ], className="tab-content animate-in")


# ============================================================================
# MAIN LAYOUT
# ============================================================================

app.layout = html.Div([
    create_header(),
    
    dcc.Tabs(
        id="main-tabs",
        value="overview",
        children=[
            dcc.Tab(label="Overview", value="overview"),
            dcc.Tab(label="Demand Forecast", value="forecast"),
            dcc.Tab(label="Scheduling", value="scheduling"),
            dcc.Tab(label="Infrastructure", value="infrastructure"),
            dcc.Tab(label="Weather & Data", value="weather"),
        ],
        className="tabs-container",
    ),
    
    html.Div(id="tab-content"),
    
], className="app-container")


# ============================================================================
# CALLBACKS
# ============================================================================

@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "value"),
)
def render_tab_content(tab: str):
    if tab == "overview":
        return overview_tab()
    elif tab == "forecast":
        return forecast_tab()
    elif tab == "scheduling":
        return scheduling_tab()
    elif tab == "infrastructure":
        return infrastructure_tab()
    elif tab == "weather":
        return weather_tab()
    return overview_tab()


@app.callback(
    Output("forecast-map", "figure"),
    Output("selected-time-display", "children"),
    Input("time-slider", "value"),
    Input("map-mode", "value"),
)
def update_forecast_map(time_index: int, mode: str):
    ts = selected_time(time_index)
    display_text = f"Showing: {ts.strftime('%A, %d %B %Y at %H:%M')}"
    return make_map_figure(time_index, mode), display_text


@app.callback(
    Output("playing", "data"),
    Output("play-button", "children"),
    Input("play-button", "n_clicks"),
    State("playing", "data"),
)
def toggle_playback(n_clicks: int, playing: bool):
    if n_clicks is None or n_clicks == 0:
        raise PreventUpdate
    new_state = not playing
    button_text = "⏸ Pause" if new_state else "▶ Play"
    return new_state, button_text


@app.callback(
    Output("playback", "disabled"),
    Input("playing", "data"),
)
def control_interval(playing: bool):
    return not playing


@app.callback(
    Output("time-slider", "value"),
    Input("playback", "n_intervals"),
    State("time-slider", "value"),
    State("playing", "data"),
)
def advance_time(n_intervals: int, current_value: int, playing: bool):
    if not playing:
        raise PreventUpdate
    return (current_value + 1) % len(TIME_VALUES)


@app.callback(
    Output("forecast-metrics", "figure"),
    Input("time-slider", "value"),
)
def update_forecast_metrics(time_index: int):
    ts = selected_time(time_index)
    
    fig = go.Figure()
    
    if "actual_demand_kw" in AGG_TS.columns:
        fig.add_trace(go.Scatter(
            x=AGG_TS["timestamp"],
            y=AGG_TS["actual_demand_kw"] if "actual_demand_kw" in AGG_TS.columns else AGG_TS["baseline_ev_load_kw"],
            name="Actual",
            line=dict(color="#7c3aed", width=2),
        ))
    
    fig.add_trace(go.Scatter(
        x=AGG_TS["timestamp"],
        y=AGG_TS["baseline_ev_load_kw"],
        name="Forecast",
        line=dict(color="#10b981", width=2.5),
    ))
    
    fig.add_vline(x=ts, line_color="#374151", line_dash="dot", line_width=1.5)
    
    fig.update_layout(
        title="Forecast vs Actual",
        xaxis_title="Time",
        yaxis_title="Demand (kW)",
        hovermode="x unified",
    )
    
    return _plot_layout(fig, height=300)


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    port = env_int("PORT", 8050)
    debug = env_bool("DEBUG", True)
    
    print(f"\n🚀 Starting Vidyut Prajna Dashboard at http://127.0.0.1:{port}")
    app.run(debug=debug, port=port)
