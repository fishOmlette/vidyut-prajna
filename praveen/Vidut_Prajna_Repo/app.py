"""Plotly Dash prototype for Vidyut Prajna.

Run:
    python app.py
Then open:
    http://127.0.0.1:8050
"""

from __future__ import annotations

import os
from typing import Dict, List

import dash
from dash import Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from data_simulation import CityConfig, generate_synthetic_city_data
from llm_interface import VidyutLLM
from model_stgcn import SpatioTemporalForecaster  # STGCN architecture for better spatial modeling
from optimization import optimize_charging_schedule
from utils import aggregate_load_timeseries, build_geojson, build_llm_context

load_dotenv()

BENGALURU_CENTER = {"lat": 12.9716, "lon": 77.5946}


def bootstrap_demo() -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]], pd.DataFrame, Dict[str, object]]:
    """Generate data, train model, forecast horizon, and optimize schedule."""
    config = CityConfig(
        h3_resolution=int(os.getenv("H3_RESOLUTION", "8")),
        max_cells=int(os.getenv("MAX_CELLS", "72")),
        num_days=int(os.getenv("NUM_DAYS", "3")),
        freq=os.getenv("FREQ", "15min"),
        seed=int(os.getenv("SEED", "42")),
        start=os.getenv("SIM_START", "2026-01-01"),
    )
    raw_df, grid_df, adjacency = generate_synthetic_city_data(config)

    unique_times = sorted(raw_df["timestamp"].unique())
    steps_per_day = int(pd.Timedelta(days=1) / pd.Timedelta(config.freq))
    train_steps = min(int(os.getenv("TRAIN_STEPS", str(2 * steps_per_day))), len(unique_times) - 12)
    horizon_steps = min(int(os.getenv("FORECAST_STEPS", str(steps_per_day))), len(unique_times) - train_steps)
    if horizon_steps <= 0:
        raise RuntimeError("Not enough simulated time steps. Increase NUM_DAYS or reduce TRAIN_STEPS.")

    train_times = unique_times[:train_steps]
    future_times = unique_times[train_steps : train_steps + horizon_steps]
    train_df = raw_df[raw_df["timestamp"].isin(train_times)].copy()
    future_df = raw_df[raw_df["timestamp"].isin(future_times)].copy()

    forecaster = SpatioTemporalForecaster(
        seq_len=int(os.getenv("SEQ_LEN", "8")),
        hidden_size=int(os.getenv("HIDDEN_SIZE", "48")),
        epochs=int(os.getenv("EPOCHS", "12")),
        seed=config.seed,
        num_blocks=int(os.getenv("STGCN_BLOCKS", "2")),  # STGCN spatial-temporal blocks
    )
    forecaster.fit(train_df, adjacency)
    pred_df = forecaster.forecast(train_df, future_df, adjacency, horizon_steps=horizon_steps)
    optimized_df, metrics = optimize_charging_schedule(pred_df)
    metrics["training_samples"] = forecaster.training_info.train_samples if forecaster.training_info else None
    metrics["training_final_loss"] = forecaster.training_info.final_loss if forecaster.training_info else None
    return raw_df, grid_df, adjacency, optimized_df, metrics


RAW_DF, GRID_DF, ADJACENCY, OPTIMIZED_DF, METRICS = bootstrap_demo()
GEOJSON = build_geojson(GRID_DF)
AGG_TS = aggregate_load_timeseries(OPTIMIZED_DF)
TIME_VALUES = sorted(OPTIMIZED_DF["timestamp"].unique())
LLM = VidyutLLM()


def fmt_kw(v: float) -> str:
    return f"{v:,.0f} kW"


def fmt_pct(v: float) -> str:
    return f"{v:.1f}%"


def card(title: str, value: str, subtitle: str, card_id: str = "") -> html.Div:
    return html.Div(
        [html.Div(title, className="metric-title"), html.Div(value, className="metric-value"), html.Div(subtitle, className="metric-subtitle")],
        className="metric-card",
        id=card_id or None,
    )


def metric_cards() -> html.Div:
    cost = METRICS.get("estimated_cost_savings_inr", 0)
    co2 = METRICS.get("co2_reduction_kg", 0)
    v2g = METRICS.get("v2g_potential_slots", 0)
    return html.Div(
        [
            card("Peak reduction", fmt_pct(float(METRICS["peak_reduction_pct"])),
                 f"{fmt_kw(float(METRICS['baseline_peak_kw']))} → {fmt_kw(float(METRICS['optimized_peak_kw']))}",
                 "card-peak"),
            card("Variance reduction", fmt_pct(float(METRICS["variance_reduction_pct"])),
                 f"PAR: {float(METRICS['baseline_par']):.2f} → {float(METRICS['optimized_par']):.2f}",
                 "card-variance"),
            card("Grid stress", str(METRICS["stress_label_after"]),
                 f"P95 score {float(METRICS['stress_score_after']):.1f}/100",
                 "card-stress"),
            card("Shifted energy", f"{float(METRICS['shifted_kwh']):,.1f} kWh",
                 "Within priority/deadline constraints",
                 "card-shifted"),
            card("Cost savings", f"₹{float(cost):,.0f}",
                 "Tariff-optimized scheduling",
                 "card-cost"),
            card("CO₂ reduction", f"{float(co2):,.1f} kg",
                 f"V2G-ready slots: {v2g}",
                 "card-co2"),
        ],
        className="metric-grid",
    )


def make_slider_marks() -> Dict[int, str]:
    marks: Dict[int, str] = {}
    step = max(1, len(TIME_VALUES) // 8)
    for idx in range(0, len(TIME_VALUES), step):
        marks[idx] = pd.Timestamp(TIME_VALUES[idx]).strftime("%H:%M")
    marks[len(TIME_VALUES) - 1] = pd.Timestamp(TIME_VALUES[-1]).strftime("%H:%M")
    return marks


def make_map_figure(time_index: int, mode: str) -> go.Figure:
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
        title = "Predicted unmanaged EV demand (kW)"
        colorscale = "YlOrRd"

    custom = np.stack(
        [
            frame["zone_name"].astype(str),
            frame["zone_type"].astype(str),
            frame["baseline_ev_load_kw"].round(1),
            frame["optimized_ev_load_kw"].round(1),
            (100 * frame["optimized_transformer_utilization"]).round(1),
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
            marker_opacity=0.68,
            marker_line_width=0.7,
            colorbar_title=title,
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                "Predicted demand: %{customdata[2]} kW<br>"
                "Optimized EV load: %{customdata[3]} kW<br>"
                "Transformer util.: %{customdata[4]}%<br>"
                "Stress: %{customdata[5]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"H3 demand heatmap — {pd.Timestamp(selected_time).strftime('%Y-%m-%d %H:%M')}",
        mapbox_style="carto-positron",
        mapbox_zoom=10.1,
        mapbox_center=BENGALURU_CENTER,
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_load_figure(time_index: int) -> go.Figure:
    selected_time = TIME_VALUES[int(np.clip(time_index, 0, len(TIME_VALUES) - 1))]
    fig = go.Figure()

    # Confidence band (if available)
    if "prediction_std_kw" in AGG_TS.columns:
        upper = AGG_TS["optimized_total_load_kw"] + 1.96 * AGG_TS["prediction_std_kw"]
        lower = (AGG_TS["optimized_total_load_kw"] - 1.96 * AGG_TS["prediction_std_kw"]).clip(lower=0)
        fig.add_trace(go.Scatter(
            x=pd.concat([AGG_TS["timestamp"], AGG_TS["timestamp"][::-1]]),
            y=pd.concat([upper, lower[::-1]]),
            fill="toself", fillcolor="rgba(99, 110, 250, 0.10)",
            line=dict(color="rgba(99, 110, 250, 0)"),
            showlegend=True, name="95% confidence",
        ))

    fig.add_trace(go.Scatter(
        x=AGG_TS["timestamp"], y=AGG_TS["baseline_total_load_kw"],
        mode="lines", name="Unmanaged total load",
        line=dict(color="#e74c3c", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=AGG_TS["timestamp"], y=AGG_TS["optimized_total_load_kw"],
        mode="lines", name="Optimized total load",
        line=dict(color="#2ecc71", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=AGG_TS["timestamp"], y=AGG_TS["safe_capacity_kw"],
        mode="lines", name="Soft capacity envelope",
        line=dict(dash="dash", color="#95a5a6", width=1.5),
    ))
    # Solar generation overlay
    if "solar_generation_kw" in AGG_TS.columns:
        fig.add_trace(go.Scatter(
            x=AGG_TS["timestamp"], y=AGG_TS["solar_generation_kw"],
            mode="lines", name="Solar generation",
            line=dict(color="#f39c12", width=1.5, dash="dot"),
            yaxis="y",
        ))

    fig.add_vline(x=selected_time, line_dash="dot", line_color="#636efa")
    fig.update_layout(
        title="Demand vs optimised load (with solar & confidence)",
        xaxis_title="Forecast horizon",
        yaxis_title="Aggregated load (kW)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin={"l": 55, "r": 20, "t": 45, "b": 45},
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def selected_summary(time_index: int) -> html.Div:
    context = build_llm_context(METRICS, OPTIMIZED_DF, time_index)
    top = context["top_predicted_demand_zones_at_selected_time"][0]
    risk = context["top_risk_zones_at_selected_time"][0]
    agg = context["selected_time_aggregate"]
    return html.Div(
        [
            html.Div(f"Selected time: {pd.Timestamp(context['selected_time']).strftime('%Y-%m-%d %H:%M')}", className="summary-line"),
            html.Div(
                f"Highest demand: {top['zone_name']} ({top['zone_type']}) — {top['baseline_ev_load_kw']} kW predicted unmanaged EV load.",
                className="summary-line",
            ),
            html.Div(
                f"Highest post-opt risk: {risk['zone_name']} — utilization {100 * float(risk['optimized_transformer_utilization']):.1f}%.",
                className="summary-line",
            ),
            html.Div(
                f"Aggregate: unmanaged {agg['baseline_total_load_kw']} kW, optimised {agg['optimized_total_load_kw']} kW.",
                className="summary-line",
            ),
        ],
        className="summary-box",
    )


app: Dash = dash.Dash(__name__)
server = app.server

app.layout = html.Div(
    [
        html.Div(
            [
                html.H1("Vidyut Prajna — AI-Driven EV Charging Intelligence"),
                html.P(
                    "Spatio-temporal demand forecasting, tariff-aware optimisation, transformer-stress management, and planner explanations for Bengaluru.",
                    className="tagline",
                ),
            ],
            className="header",
        ),
        metric_cards(),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Button("▶ Play", id="play-button", n_clicks=0, className="play-button"),
                                dcc.RadioItems(
                                    id="map-mode",
                                    options=[
                                        {"label": "Predicted demand", "value": "predicted"},
                                        {"label": "Optimized EV load", "value": "optimized"},
                                        {"label": "Transformer stress", "value": "stress"},
                                    ],
                                    value="predicted",
                                    inline=True,
                                    className="radio-row",
                                ),
                            ],
                            className="control-row",
                        ),
                        dcc.Slider(
                            id="time-slider",
                            min=0,
                            max=len(TIME_VALUES) - 1,
                            value=0,
                            step=1,
                            marks=make_slider_marks(),
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                        dcc.Interval(id="playback", interval=900, n_intervals=0, disabled=True),
                        dcc.Store(id="playing", data=False),
                        dcc.Graph(id="h3-map"),
                    ],
                    className="panel map-panel",
                ),
                html.Div(
                    [
                        dcc.Graph(id="load-chart"),
                        html.Div(id="selected-summary"),
                    ],
                    className="panel",
                ),
            ],
            className="dashboard-grid",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H2("Planner chat — LLM explanation layer"),
                        html.P(
                            "Ask: Which areas will have peak demand tonight? Why is this zone high risk? What is the benefit of optimisation?",
                            className="hint",
                        ),
                        html.Div(
                            f"LLM status: {'Gemini API enabled (' + LLM.model + ')' if LLM.enabled else 'local fallback; set GEMINI_API_KEY for API answers'}",
                            className="llm-status",
                        ),
                        html.Div(id="chat-window", className="chat-window"),
                        dcc.Input(
                            id="chat-input",
                            placeholder="Ask about the displayed forecast or optimization...",
                            type="text",
                            className="chat-input",
                        ),
                        html.Button("Send", id="chat-send", n_clicks=0, className="send-button"),
                        dcc.Store(id="chat-history", data=[]),
                    ],
                    className="panel chat-panel",
                )
            ]
        ),
    ],
    className="app",
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
    return new_state, not new_state, "⏸ Pause" if new_state else "▶ Play"


@app.callback(Output("time-slider", "value"), Input("playback", "n_intervals"), State("time-slider", "value"))
def advance_time(_: int, current_value: int) -> int:
    return (int(current_value) + 1) % len(TIME_VALUES)


@app.callback(
    Output("h3-map", "figure"),
    Output("load-chart", "figure"),
    Output("selected-summary", "children"),
    Input("time-slider", "value"),
    Input("map-mode", "value"),
)
def update_dashboard(time_index: int, map_mode: str):
    return make_map_figure(time_index, map_mode), make_load_figure(time_index), selected_summary(time_index)


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
    context = build_llm_context(METRICS, OPTIMIZED_DF, int(time_index))
    answer = LLM.answer(question.strip(), context)
    history = history or []
    history.append({"role": "user", "content": question.strip()})
    history.append({"role": "assistant", "content": answer})
    return history, ""


@app.callback(Output("chat-window", "children"), Input("chat-history", "data"))
def render_chat(history: list | None):
    if not history:
        return html.Div("No messages yet. Try asking about peak demand, high-risk zones, or optimization benefits.", className="empty-chat")
    bubbles = []
    for msg in history[-10:]:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        # Use dcc.Markdown to properly render formatting like bold/italics
        bubbles.append(
            html.Div(
                dcc.Markdown(content, mathjax=True),
                className=f"chat-bubble {role}"
            )
        )
    return bubbles


app.index_string = """
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>Vidyut Prajna — EV Charging Intelligence</title>
    <meta name="description" content="AI-driven EV charging demand forecasting, tariff-aware optimization, and infrastructure planning for Bengaluru.">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
    {%favicon%}
    {%css%}
    <style>
        :root {
            --bg: #0b1120;
            --surface: rgba(17, 25, 45, 0.85);
            --surface-hover: rgba(25, 35, 60, 0.95);
            --text: #e2e8f0;
            --text-muted: #8b99b0;
            --accent: #38bdf8;
            --accent2: #22d3ee;
            --green: #34d399;
            --amber: #fbbf24;
            --red: #f87171;
            --border: rgba(99, 110, 250, 0.12);
            --glow: rgba(56, 189, 248, 0.15);
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background: var(--bg);
            color: var(--text);
            -webkit-font-smoothing: antialiased;
        }
        .app { padding: 22px 28px; max-width: 1600px; margin: 0 auto; }

        /* Header */
        .header {
            background: linear-gradient(135deg, #0c2746 0%, #0e3d5e 40%, #145c63 70%, #0f766e 100%);
            color: white;
            padding: 28px 32px;
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(20, 92, 99, 0.25), inset 0 1px 0 rgba(255,255,255,0.08);
            position: relative;
            overflow: hidden;
        }
        .header::before {
            content: '';
            position: absolute; top: 0; left: 0; right: 0; bottom: 0;
            background: radial-gradient(ellipse at 30% 50%, rgba(56,189,248,0.08) 0%, transparent 70%);
            pointer-events: none;
        }
        h1 { margin: 0 0 8px 0; font-size: 28px; font-weight: 800; letter-spacing: -0.02em; }
        h2 { margin-top: 0; font-size: 20px; font-weight: 700; color: var(--accent); }
        .tagline, .hint { opacity: 0.82; margin: 0; font-size: 14px; }

        /* Metric cards */
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 14px;
            margin: 20px 0;
        }
        .metric-card {
            background: var(--surface);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 18px 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 28px rgba(56, 189, 248, 0.15);
            border-color: var(--accent);
        }
        .metric-title { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted); font-weight: 600; }
        .metric-value { font-size: 26px; font-weight: 800; margin: 8px 0 4px; color: var(--accent); }
        .metric-subtitle { color: var(--text-muted); font-size: 12px; }

        /* Dashboard grid */
        .dashboard-grid { display: grid; grid-template-columns: 1.35fr 1fr; gap: 18px; align-items: start; }
        .panel {
            background: var(--surface);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 18px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }
        .control-row { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 10px; }
        .play-button, .send-button {
            border: 0;
            background: linear-gradient(135deg, var(--accent), var(--accent2));
            color: #0b1120;
            border-radius: 10px;
            padding: 10px 18px;
            cursor: pointer;
            font-weight: 700;
            font-size: 14px;
            transition: opacity 0.2s, transform 0.15s;
        }
        .play-button:hover, .send-button:hover { opacity: 0.9; transform: scale(1.03); }
        .radio-row label { margin-right: 14px; color: var(--text); font-size: 13px; }
        .summary-box { background: rgba(56,189,248,0.06); border: 1px solid var(--border); border-radius: 14px; padding: 12px 14px; }
        .summary-line { margin: 6px 0; color: var(--text); font-size: 13px; }

        /* Chat */
        .chat-panel { margin-top: 18px; }
        .llm-status { margin: 10px 0; color: var(--green); font-weight: 700; font-size: 13px; }
        .chat-window {
            min-height: 160px; max-height: 380px; overflow-y: auto;
            background: rgba(11, 17, 32, 0.6);
            border: 1px solid var(--border);
            border-radius: 14px; padding: 14px; margin: 12px 0;
        }
        .empty-chat { color: var(--text-muted); font-size: 13px; }
        .chat-bubble {
            padding: 10px 14px; border-radius: 14px; margin: 8px 0;
            white-space: pre-wrap; line-height: 1.5; font-size: 13px;
        }
        .chat-bubble.user {
            background: rgba(56, 189, 248, 0.12);
            border: 1px solid rgba(56, 189, 248, 0.2);
            margin-left: 14%;
        }
        .chat-bubble.assistant {
            background: var(--surface);
            border: 1px solid var(--border);
            margin-right: 14%;
        }
        .chat-input {
            width: calc(100% - 92px);
            padding: 12px 14px;
            border-radius: 10px;
            border: 1px solid var(--border);
            background: rgba(11, 17, 32, 0.7);
            color: var(--text);
            margin-right: 8px;
            font-size: 14px;
            outline: none;
            transition: border-color 0.2s;
        }
        .chat-input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px var(--glow); }

        /* Plotly override */
        .js-plotly-plot .plotly .modebar { right: 8px !important; }

        /* Responsive */
        @media (max-width: 1200px) { .metric-grid { grid-template-columns: repeat(3, 1fr); } }
        @media (max-width: 900px) { .metric-grid { grid-template-columns: repeat(2, 1fr); } .dashboard-grid { grid-template-columns: 1fr; } }
        @media (max-width: 600px) { .metric-grid { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=int(os.getenv("PORT", "8050")))
