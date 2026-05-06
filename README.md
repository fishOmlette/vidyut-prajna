# Vidyut Prajna

AI-driven EV charging optimization and infrastructure-planning support for Bengaluru.

## What The Prototype Shows

- Contiguous H3 corridor simulation for Bengaluru instead of scattered citywide samples.
- EV charging demand forecasting with both an STGCN baseline and a competition-grade probabilistic Graph Temporal Fusion Transformer.
- Future-conditioned intelligence: the STGCN sees target-time exogenous signals and is guarded by persistence/seasonal baselines to avoid flat mean forecasts.
- Robust rolling-horizon LP scheduling that preserves energy/deadlines, models uncertainty, and exposes transformer shadow prices.
- Graph-aware infrastructure portfolio planning that balances demand, centrality, transformer headroom, traffic, ROI, and uniform-placement baselines.
- Optional grounded planner chat that explains computed dashboard outputs only. The app runs without an API key using a deterministic local fallback.

## Dashboard Sections

- **Overview**: peak reduction, overload events, cost savings, model uncertainty, risk zones, and read-only sidecar mode.
- **Demand Forecast**: adjacent H3 map, timeline playback, confidence bands, and actual-vs-predicted synthetic holdout.
- **Scheduling Optimization**: unmanaged vs optimized EV charging, tariff signal, solar alignment, and energy preservation.
- **Infrastructure Planning**: ranked station recommendations, capacity feasibility, and comparison with uniform placement.
- **Data Sources**: synthetic telemetry, H3 graph, STGCN features, optimizer constraints, and LLM context policy.
- **Explanation Assistant**: optional hosted LLM with synthetic/aggregated computed context only.

## Quick Start

```bash
python -m pip install -r requirements.txt
copy .env.example .env
python main.py
```

Open:

```text
http://127.0.0.1:8050
```

Run tests:

```bash
python main.py --test
```

Run a compact competition benchmark:

```bash
python main.py --competition-demo
```

Generate synthetic data only:

```bash
python main.py --simulate
```

## Environment Variables

Root configuration lives in `.env.example`.

```bash
SCENARIO=orr_whitefield
H3_RESOLUTION=8
MAX_CELLS=54
NUM_DAYS=7
FREQ=1h
FORECAST_STEPS=24
SEQ_LEN=12
HIDDEN_SIZE=48
STGCN_BLOCKS=2
EPOCHS=10
STATION_BUDGET=8
PORT=8050
```

Optional explanation assistant:

```bash
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
```

No hosted LLM is required. If no API key is configured, the assistant uses local deterministic explanations from the same computed context.

## Architecture

- `src/spatial_grid`: contiguous Bengaluru scenario generation, H3 cells, adjacency, synthetic telemetry.
- `src/intelligence`: STGCN baseline plus probabilistic graph temporal fusion forecasting.
- `src/optimization`: robust LP scheduler, Lagrangian baseline, and graph-aware station siting.
- `src/evaluation`: forecast, scheduling, and siting benchmark harness.
- `src/dashboard`: Dash planning console, grounded explanation assistant, and UI helpers.
- `praveen`: reference implementation only; `src/` is the canonical app.

## Constraints Covered

- No modification to existing distribution systems.
- Synthetic or aggregated computed data only.
- Explainable demand, scheduling, and siting recommendations.
- Grid constraints: transformer capacity, temperature derating, overload events, priority charging, deadlines, tariff, and solar availability.
- Baselines: unmanaged charging vs optimized scheduling, and uniform infrastructure placement vs ranked recommendations.

Detailed redesign: `docs/COMPETITION_ARCHITECTURE.md`.
