# Vidyut Prajna

AI-Driven Spatio-Temporal Intelligence for EV Charging Optimization in Bengaluru.

## Overview

Vidyut Prajna uses **STGCN (Spatio-Temporal Graph Convolutional Networks)** to forecast EV charging demand across Bengaluru's H3 hexagonal grid and optimize charging schedules to:

- Reduce peak grid load
- Minimize transformer stress
- Optimize for BESCOM time-of-use tariffs
- Maximize solar generation utilization
- Enable V2G readiness

## Project Structure

```
vidyut-prajna/
├── main.py                     # Entry point
├── requirements.in             # Dependencies
├── src/
│   ├── spatial_grid/           # H3 grid & data simulation
│   │   ├── simulation.py       # Bengaluru city simulation
│   │   ├── generator.py        # Basic grid generation
│   │   └── visualizer.py       # Visualization utilities
│   ├── intelligence/           # Forecasting models
│   │   ├── model.py            # STGCN architecture
│   │   ├── forecaster.py       # High-level forecaster API
│   │   └── graph_utils.py      # Graph utilities
│   ├── optimization/           # Load optimization
│   │   └── optimizer.py        # Constraint-aware optimizer
│   └── dashboard/              # Web interface
│       ├── app.py              # Dash application
│       └── utils.py            # Dashboard utilities
├── data/
│   ├── raw/                    # Raw simulation outputs
│   └── processed/              # Processed data
└── praveen/                    # Praveen's reference implementation
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.in

# Or with uv
uv pip install -r requirements.in

# Run the dashboard
python main.py

# Run tests
python main.py --test

# Generate simulation data only
python main.py --simulate
```

Then open: http://127.0.0.1:8050

## Architecture

### STGCN Model

The forecasting model combines:
- **Graph Convolution**: Captures spatial dependencies across H3 neighboring cells
- **LSTM**: Captures temporal patterns (hourly, daily, weekly)
- **BatchNorm + Dropout**: Regularization for better generalization

### Optimization

The optimizer shifts flexible EV charging load while respecting:
- Priority (non-shiftable) charging requirements
- Deadline constraints
- Transformer capacity limits (with temperature derating)
- BESCOM ToU tariff windows
- Solar generation availability

### Data Simulation

Generates realistic Bengaluru digital twin with:
- 20 neighborhood archetypes (residential, commercial, IT corridor, logistics, mixed)
- Multi-class EV fleet (2W, 3W, 4W, bus)
- Weather patterns (temperature, rainfall)
- Traffic intensity profiles
- BESCOM tariff signals
- Rooftop solar generation

## Environment Variables

```bash
H3_RESOLUTION=9       # H3 resolution (9 = ~100m)
MAX_CELLS=30          # Number of H3 cells
NUM_DAYS=5            # Simulation days
FREQ=1h               # Time frequency
EPOCHS=8              # Training epochs
STGCN_BLOCKS=2        # Number of STGCN blocks
PORT=8050             # Dashboard port
```

## License

See [LICENSE](LICENSE)