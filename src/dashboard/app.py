"""Vidyut Prajna Dashboard - Plotly Dash application.

Run:
    python -m src.dashboard.app
Or:
    python main.py
    
Then open: http://127.0.0.1:8050
"""

from __future__ import annotations

import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

from src.spatial_grid.simulation import CityConfig, generate_synthetic_data
from src.intelligence.forecaster import STGCNForecaster
from src.optimization.optimizer import optimize_charging_schedule
from .utils import aggregate_load_timeseries, build_geojson, format_kw, format_pct


BENGALURU_CENTER = {"lat": 12.9716, "lon": 77.5946}


def bootstrap_demo() -> tuple:
    """Generate data, train model, forecast, and optimize."""
    print("Initializing Vidyut Prajna...")
    
    config = CityConfig(
        h3_resolution=int(os.getenv("H3_RESOLUTION", "9")),
        max_cells=int(os.getenv("MAX_CELLS", "30")),
        num_days=int(os.getenv("NUM_DAYS", "5")),
        freq=os.getenv("FREQ", "1h"),
        seed=int(os.getenv("SEED", "42")),
        start=os.getenv("SIM_START", "2026-05-01"),
    )
    
    print(f"Generating synthetic data: {config.max_cells} cells, {config.num_days} days...")
    raw_df, grid_df, adjacency = generate_synthetic_data(config)
    
    unique_times = sorted(raw_df["timestamp"].unique())
    train_steps = int(len(unique_times) * 0.7)
    horizon_steps = min(24, len(unique_times) - train_steps)
    
    train_times = unique_times[:train_steps]
    future_times = unique_times[train_steps:train_steps + horizon_steps]
    train_df = raw_df[raw_df["timestamp"].isin(train_times)].copy()
    future_df = raw_df[raw_df["timestamp"].isin(future_times)].copy()
    
    print(f"Training STGCN model: {len(train_df)} samples...")
    forecaster = STGCNForecaster(
        seq_len=int(os.getenv("SEQ_LEN", "6")),
        hidden_size=int(os.getenv("HIDDEN_SIZE", "32")),
        epochs=int(os.getenv("EPOCHS", "8")),
        num_blocks=int(os.getenv("STGCN_BLOCKS", "2")),
        seed=config.seed,
    )
    forecaster.fit(train_df, adjacency)
    
    print(f"Forecasting {horizon_steps} steps...")
    pred_df = forecaster.forecast(train_df, future_df, adjacency, horizon_steps=horizon_steps)
    
    print("Optimizing charging schedule...")
    optimized_df, metrics = optimize_charging_schedule(pred_df)
    
    if forecaster.training_info:
        metrics["training_samples"] = forecaster.training_info.train_samples
        metrics["training_final_loss"] = forecaster.training_info.final_loss
    
    print("Dashboard ready!")
    return raw_df, grid_df, adjacency, optimized_df, metrics


# Initialize data
RAW_DF, GRID_DF, ADJACENCY, OPTIMIZED_DF, METRICS = bootstrap_demo()
GEOJSON = build_geojson(GRID_DF)
AGG_TS = aggregate_load_timeseries(OPTIMIZED_DF)
TIME_VALUES = sorted(OPTIMIZED_DF["timestamp"].unique())


def make_slider_marks() -> Dict[int, str]:
    """Create time slider marks."""
    marks = {}
    step = max(1, len(TIME_VALUES) // 8)
    for idx in range(0, len(TIME_VALUES), step):
        marks[idx] = pd.Timestamp(TIME_VALUES[idx]).strftime("%H:%M")
    marks[len(TIME_VALUES) - 1] = pd.Timestamp(TIME_VALUES[-1]).strftime("%H:%M")
    return marks


def make_map_figure(time_index: int, mode: str) -> go.Figure:
    """Create choropleth map."""
    selected_time = TIME_VALUES[int(np.clip(time_index, 0, len(TIME_VALUES) - 1))]
    frame = OPTIMIZED_DF[OPTIMIZED_DF["timestamp"] == selected_time].copy()
    
    if mode == "optimized":
        z = frame["optimized_ev_load_kw"]
        title = "Optimized EV load (kW)"
        colorscale = "Viridis"
    elif mode == "stress":
        z = 100 * frame["optimized_transformer_utilization"]
        title = "Transformer utilization (%)"
        colorscale = "RdYlGn_r"
    else:
        z = frame["baseline_ev_load_kw"]
        title = "Predicted EV demand (kW)"
        colorscale = "YlOrRd"
    
    custom = np.stack([
        frame["zone_name"].astype(str),
        frame["zone_type"].astype(str),
        frame["baseline_ev_load_kw"].round(1),
        frame["optimized_ev_load_kw"].round(1),
        (100 * frame["optimized_transformer_utilization"]).round(1),
        frame["stress_label"].astype(str),
    ], axis=-1)
    
    fig = go.Figure(go.Choroplethmapbox(
        geojson=GEOJSON,
        locations=frame["h3_cell"],
        z=z,
        featureidkey="properties.h3_cell",
        colorscale=colorscale,
        marker_opacity=0.7,
        marker_line_width=0.5,
        colorbar_title=title,
        customdata=custom,
        hovertemplate=(
            "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
            "Predicted: %{customdata[2]} kW<br>"
            "Optimized: %{customdata[3]} kW<br>"
            "Util: %{customdata[4]}%<br>"
            "Stress: %{customdata[5]}<extra></extra>"
        ),
    ))
    
    fig.update_layout(
        title=f"H3 Grid — {pd.Timestamp(selected_time).strftime('%Y-%m-%d %H:%M')}",
        mapbox_style="carto-positron",
        mapbox_zoom=10,
        mapbox_center=BENGALURU_CENTER,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        height=450,
    )
    return fig


def make_load_figure() -> go.Figure:
    """Create aggregate load time series plot."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=AGG_TS["timestamp"],
        y=AGG_TS["baseline_total_load_kw"],
        name="Baseline Load",
        line=dict(color="#E74C3C", width=2, dash="dot"),
    ))
    
    fig.add_trace(go.Scatter(
        x=AGG_TS["timestamp"],
        y=AGG_TS["optimized_total_load_kw"],
        name="Optimized Load",
        line=dict(color="#27AE60", width=2),
        fill="tonexty",
        fillcolor="rgba(39, 174, 96, 0.2)",
    ))
    
    fig.add_trace(go.Scatter(
        x=AGG_TS["timestamp"],
        y=AGG_TS["safe_capacity_kw"],
        name="Safe Capacity (95%)",
        line=dict(color="#F39C12", width=1.5, dash="dash"),
    ))
    
    if "solar_generation_kw" in AGG_TS.columns:
        fig.add_trace(go.Scatter(
            x=AGG_TS["timestamp"],
            y=AGG_TS["solar_generation_kw"],
            name="Solar Generation",
            line=dict(color="#3498DB", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(52, 152, 219, 0.15)",
        ))
    
    fig.update_layout(
        title="Aggregate Load Profile",
        xaxis_title="Time",
        yaxis_title="Power (kW)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
        height=350,
    )
    return fig


def card(title: str, value: str, subtitle: str) -> html.Div:
    """Create a metric card."""
    return html.Div([
        html.H4(title, style={"margin": "0", "color": "#666", "fontSize": "12px"}),
        html.H2(value, style={"margin": "5px 0", "color": "#2C3E50"}),
        html.P(subtitle, style={"margin": "0", "color": "#999", "fontSize": "11px"}),
    ], style={
        "backgroundColor": "white",
        "padding": "15px",
        "borderRadius": "8px",
        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
        "textAlign": "center",
        "minWidth": "120px",
    })


# Create Dash app
app = Dash(__name__, title="Vidyut Prajna")

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Vidyut Prajna", style={"margin": "0", "color": "#2C3E50"}),
        html.P("AI-Driven EV Charging Optimization for Bengaluru", 
               style={"margin": "5px 0 0 0", "color": "#7F8C8D"}),
    ], style={"padding": "20px", "backgroundColor": "#ECF0F1"}),
    
    # Metrics row
    html.Div([
        card("Peak Reduction", format_pct(METRICS["peak_reduction_pct"]),
             f"{format_kw(METRICS['baseline_peak_kw'])} → {format_kw(METRICS['optimized_peak_kw'])}"),
        card("Variance Reduction", format_pct(METRICS["variance_reduction_pct"]),
             f"PAR: {METRICS['baseline_par']:.2f} → {METRICS['optimized_par']:.2f}"),
        card("Grid Stress", METRICS["stress_label_after"],
             f"Score: {METRICS['stress_score_after']:.1f}/100"),
        card("Shifted Energy", f"{METRICS['shifted_kwh']:.1f} kWh",
             "Within deadline constraints"),
        card("Cost Savings", f"₹{METRICS['estimated_cost_savings_inr']:,.0f}",
             "Tariff-optimized"),
        card("CO₂ Reduction", f"{METRICS['co2_reduction_kg']:.1f} kg",
             f"V2G slots: {METRICS['v2g_potential_slots']}"),
    ], style={
        "display": "flex",
        "gap": "15px",
        "padding": "20px",
        "overflowX": "auto",
        "backgroundColor": "#F8F9FA",
    }),
    
    # Main content
    html.Div([
        # Left: Map
        html.Div([
            html.Div([
                html.Label("View Mode:", style={"marginRight": "10px"}),
                dcc.RadioItems(
                    id="map-mode",
                    options=[
                        {"label": "Predicted", "value": "predicted"},
                        {"label": "Optimized", "value": "optimized"},
                        {"label": "Stress", "value": "stress"},
                    ],
                    value="optimized",
                    inline=True,
                    style={"display": "inline-block"},
                ),
            ], style={"marginBottom": "10px"}),
            dcc.Graph(id="map-figure", figure=make_map_figure(0, "optimized")),
            html.Div([
                html.Label("Time:"),
                dcc.Slider(
                    id="time-slider",
                    min=0,
                    max=len(TIME_VALUES) - 1,
                    value=0,
                    marks=make_slider_marks(),
                    step=1,
                ),
            ], style={"marginTop": "10px"}),
        ], style={"flex": "1", "minWidth": "400px"}),
        
        # Right: Load chart
        html.Div([
            dcc.Graph(id="load-figure", figure=make_load_figure()),
        ], style={"flex": "1", "minWidth": "400px"}),
    ], style={
        "display": "flex",
        "gap": "20px",
        "padding": "20px",
        "flexWrap": "wrap",
    }),
    
    # Footer
    html.Div([
        html.P([
            "Vidyut Prajna — STGCN-based EV charging optimization. ",
            f"Model trained on {METRICS.get('training_samples', 'N/A')} samples.",
        ], style={"margin": "0", "color": "#95A5A6", "fontSize": "12px"}),
    ], style={"padding": "10px 20px", "backgroundColor": "#2C3E50", "textAlign": "center"}),
    
], style={"fontFamily": "Arial, sans-serif", "backgroundColor": "#F5F5F5", "minHeight": "100vh"})


@app.callback(
    Output("map-figure", "figure"),
    [Input("time-slider", "value"), Input("map-mode", "value")],
)
def update_map(time_index: int, mode: str) -> go.Figure:
    return make_map_figure(time_index, mode)


def run_server(debug: bool = False, port: int = 8050):
    """Run the dashboard server."""
    app.run(debug=debug, port=port)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8050"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    run_server(debug=debug, port=port)
