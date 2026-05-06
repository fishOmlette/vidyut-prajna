# Vidyut Prajna (विद्युत् प्रज्ञा)

**AI for EV Charging Optimization & Infrastructure Planning**

*BESCOM Hackathon - Theme 9*

**Team:** Adil Farooq (Lead), Aashrith Reddy, Meher Praveen

---

## Executive Summary

As Bengaluru accelerates its Electric Vehicle (EV) transition, BESCOM faces the **Coincident Peak Challenge**: the high-probability overlap of vehicle charging with existing residential/commercial peak windows. Unmanaged charging creates localized thermal stress on distribution transformers (DTRs), threatening grid stability.

**Vidyut Prajna** is a non-invasive "sidecar" decision-support layer. It treats the city as a living energy organism, using **Graph-based Neural Networks** to predict demand and **Lagrangian Optimization** to balance load. Our solution ensures that the EV revolution supports, rather than stresses, the Bengaluru grid.

---

## Problem Understanding

| Challenge | Impact |
|-----------|--------|
| Grid failures at 11kV/440V transformer level | Local outages even when city-wide load is stable |
| Non-uniform demand patterns | Evening residential spikes in HSR Layout differ from daytime logistics surges in Peenya |
| Infrastructure planning gaps | New station siting not always data-driven |

**Constraints:**
- No modification to existing SCADA or distribution hardware
- Works as a data-driven overlay (read-only sidecar mode)
- Uses masked or synthetic data where required
- Outputs must be explainable and actionable

---

## Solution Architecture

### Part A: EV Charging Demand & Scheduling

**Spatio-Temporal Graph Convolutional Networks (STGCN)**
- Partitions Bengaluru into hexagonal cells using Uber's H3 Index (Resolution 8)
- H3 hexagons provide uniform neighbor distances, mathematically vital for accurate Kernel Density Estimation (KDE)
- Learns how traffic bottlenecks on ORR act as leading indicators for charging spikes in adjacent cells

**Lagrangian-based Multi-Criteria Decision Making (MCDM)**

$$J = \alpha \cdot \sum_t (P_{grid}(t) - P_{avg})^2 + \beta \cdot \sum_i (\omega_i \cdot \Delta SoC_i)$$

Where:
- α minimizes load variance (peak shaving)
- β ensures high-priority vehicles meet their SoC targets
- The system finds the "Point of Minimum Regret" balancing grid health against user service levels

### Part B: Infrastructure Location Planning

The siting engine identifies "Future-Proof" corridors by fusing:
- **Gig Fleet Mapping**: Congregation points for 2W/3W delivery fleets (Swiggy/Zomato)
- **Transformer Capacity**: Real-time and historical spare capacity in local DTRs
- **Demand Growth**: Hexagonal cells where demand outpaces current charger density

---

## Technical Stack

| Component | Technology |
|-----------|------------|
| Spatial Grid | H3 Hexagonal Index (Resolution 8) |
| Forecasting | STGCN + Graph Temporal Fusion Transformer |
| Optimization | Lagrangian MCDM + Robust LP |
| Arrival Estimation | Kernel Density Estimation (KDE) |
| Dashboard | Plotly Dash |
| Data Privacy | K-anonymity masking |

---

## Dashboard Tabs

| Tab | Description |
|-----|-------------|
| **Overview** | Peak reduction, overload events, cost savings, risk zones |
| **Demand Forecast** | H3 map, timeline playback, confidence bands, actual vs predicted |
| **Scheduling** | Unmanaged vs optimized charging, tariff signals, solar alignment |
| **Infrastructure** | Ranked station recommendations, capacity feasibility, uniform baseline comparison |
| **Data Sources** | Synthetic telemetry, STGCN features, optimizer constraints |
| **Problem Fit** | Judge-facing checklist mapped to BESCOM requirements |
| **Explanation** | Optional LLM assistant (works offline with deterministic fallback) |

---

## Data Sources (Simulated)

- **Swiggy/Zomato Fleet**: 15 congregation points with 2W/3W delivery vehicles
- **OCPP Sessions**: Synthetic charging session telemetry
- **Monsoon Weather**: May monsoon patterns for Bengaluru
- **DTR Topology**: Distribution transformer health and capacity
- **BESCOM Tariff**: Time-of-use tariff multipliers
- **Solar Generation**: Rooftop solar availability

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment template
copy .env.example .env      # Windows
cp .env.example .env        # Linux/Mac

# Run the dashboard
python main.py
```

Open: http://127.0.0.1:8050

### Other Commands

```bash
# Run test suite
python main.py --test

# Generate simulation data only
python main.py --simulate
```

---

## Project Structure

```
vidyut-prajna/
├── src/
│   ├── spatial_grid/       # H3 grid, Bengaluru simulation, gig fleet, OCPP, weather
│   ├── intelligence/       # STGCN, Graph-TFT, KDE arrivals, feedback loop
│   ├── optimization/       # Lagrangian MCDM, Robust LP, station siting
│   ├── evaluation/         # Benchmarking harness
│   └── dashboard/          # Plotly Dash console, LLM interface
├── tests/                  # Test suite
├── scripts/                # Standalone utilities
├── data/                   # Runtime data (cache, processed)
├── docs/                   # Documentation
├── main.py                 # Entry point
└── requirements.txt
```

---

## Evaluation Metrics

| Metric | Target |
|--------|--------|
| Transformer overload reduction | 15-20% |
| Peak-to-Average Ratio (PAR) | Lower than unmanaged baseline |
| Charger utilization | Better than uniform placement |
| Deadline feasibility | >95% |
| Energy preservation | <0.5% error |

---

## Constraints Covered

- ✅ No modification to existing distribution systems
- ✅ Works as a decision-support layer (sidecar mode)
- ✅ Uses synthetic data with K-anonymity masking
- ✅ Outputs are explainable and actionable
- ✅ Considers grid constraints (transformer capacity, derating, deadlines)
- ✅ No hosted LLM dependence on sensitive data
- ✅ Comparison against baselines (unmanaged charging, uniform placement)

---

## Environment Variables

See [.env.example](.env.example) for full configuration. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `SCENARIO` | `orr_whitefield` | Bengaluru corridor scenario |
| `H3_RESOLUTION` | `8` | H3 hex resolution (~461m edge) |
| `MAX_CELLS` | `54` | Max H3 cells in simulation |
| `EPOCHS` | `10` | STGCN training epochs |
| `STATION_BUDGET` | `8` | Stations to recommend |
| `GEMINI_API_KEY` | *(empty)* | Optional LLM for explanations |

---

## License

MIT License - See [LICENSE](LICENSE)
