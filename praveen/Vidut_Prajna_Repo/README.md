# Vidyut Prajna — AI-Driven Spatio-Temporal Intelligence for EV Charging Optimization

A Plotly Dash prototype for EV charging demand forecasting and infrastructure-planning support in Bengaluru.

## What this prototype does

- Simulates traffic, weather, EV charging demand (multi-class fleet: 2W/3W/4W/bus), and transformer base load across ~20 Bengaluru neighbourhoods.
- Divides Bengaluru into H3 hex cells with zone types: residential, commercial, logistics, IT-corridor, and mixed.
- Trains a graph-aware 2-layer GRU forecaster with temporal attention, using H3 neighbour demand, day-of-week, weekend flag, tariff signal, solar generation, and weather/traffic features.
- Runs a tariff-aware, temperature-derated, solar-preferring optimizer that shifts flexible charging load while preserving priority charging and deadline windows.
- Computes cost savings (BESCOM ToU), CO₂ reduction, V2G readiness, and confidence intervals (MC-dropout).
- Visualises H3 demand, optimised load, transformer stress, aggregate load curves with solar overlay and confidence bands.
- Adds a Gemini-powered LLM chat layer (via OpenAI-compatible endpoint) that explains computed outputs only.

The LLM does **not** forecast or optimize. It receives structured computed context from the dashboard and explains it.

## Project structure

```text
vidyut_prajna_prototype/
├── app.py                 # Dash dashboard
├── data_simulation.py     # Synthetic Bengaluru data generator
├── model.py               # 2-layer GRU + temporal attention forecaster
├── optimization.py        # Tariff/solar/temperature-aware optimizer
├── llm_interface.py       # Gemini LLM via OpenAI-compatible endpoint
├── utils.py               # Map, metrics, and LLM context helpers
├── requirements.txt
├── .env.example
└── README.md
```

## Local setup

```bash
cd vidyut_prajna_prototype
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env           # Windows: copy .env.example .env
```

Edit `.env`:

```bash
GEMINI_API_KEY=your-gemini-api-key-here
GEMINI_MODEL=gemini-2.5-flash
```

Run:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:8050
```

## Running without an API key

The app still runs. The chat panel uses a deterministic local fallback summary built from dashboard metrics. Add `GEMINI_API_KEY` to enable the Gemini LLM explanation layer.

## Useful environment variables

```bash
H3_RESOLUTION=8        # Use 9 for smaller production-like H3 cells, but more cells may be slower.
MAX_CELLS=72           # Keep small for demos.
NUM_DAYS=3             # 2 days training + 1 day forecast by default.
TRAIN_STEPS=192        # 2 days at 15-minute cadence.
FORECAST_STEPS=96      # 1 day at 15-minute cadence.
EPOCHS=12              # Increase for better fit, reduce for faster startup.
HIDDEN_SIZE=48         # GRU hidden dimension.
```

## Notes

- All data is synthetic.
- No control commands are sent to grid hardware.
- The optimizer is a sidecar planning tool, not a SCADA control loop.
- The model is an STGCN-inspired simplified equivalent: H3 graph aggregation + 2-layer GRU + temporal attention.
- BESCOM ToU tariff windows and Karnataka grid CO₂ intensity are modelled for cost/carbon estimates.
