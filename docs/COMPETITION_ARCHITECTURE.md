# Vidyut Prajna Competition Architecture

## 1. Why The Original Design Was Too Weak

The concept paper correctly identifies the grid risk: EV demand is local, temporal,
and constrained by transformer headroom rather than city-wide energy balance. The
initial STGCN + Lagrangian design is a good prototype, but it is not competition
grade for four reasons.

1. **Forecasting is under-specified.** A shallow graph convolution plus LSTM tends
   to smooth demand into a mean profile. It does not model known future inputs
   across the whole horizon, produces weak uncertainty estimates, and struggles
   when charging behavior shifts because it learns a static graph.
2. **Optimization is not a real constrained scheduler.** Lagrangian relaxation can
   be useful for decomposition, but the current implementation behaves like an
   iterative heuristic with guardrails. It can converge poorly, has no explicit
   robust feasibility margin, and does not expose hard service guarantees.
3. **Siting was a scalar ranker.** Infrastructure planning is a budgeted portfolio
   problem. Picking top independent scores can cluster stations in adjacent cells
   and waste coverage.
4. **Evaluation was narrow.** A winning system must compare forecast quality,
   overload reduction, PAR, utilization, fairness, latency, and station ROI
   against several baselines.

The redesigned system keeps the non-invasive BESCOM sidecar constraint, but
replaces the fragile core with probabilistic forecasting, robust optimization,
graph-aware portfolio siting, and benchmarkable experiment plumbing.

## 2. End-To-End Pipeline

```text
Masked telemetry
  OCPP sessions | H3 cell aggregates | DTR topology | traffic | weather | tariffs | solar
        |
        v
Privacy and quality layer
  k-anonymity, H3 generalization, missingness flags, robust scaling, drift checks
        |
        v
Feature store by (timestamp, h3_cell)
  lagged demand, rolling demand, KDE arrivals, neighbor pressure, DTR headroom,
  tariff, solar, weather, traffic, zone embeddings, charger density, growth
        |
        v
Hybrid graph builder
  H3 adjacency + learned corridor attention + centrality + transformer topology
        |
        v
Probabilistic Graph Temporal Fusion Transformer
  p10 / p50 / p90 forecasts for EV demand over H future steps
        |
        v
Robust rolling-horizon LP optimizer
  deadline-feasible schedules, peak shaving, overload slack, tariff and solar alignment
        |
        v
Graph-aware siting portfolio engine
  station budget -> ranked cells with ROI, feasibility, coverage, and explanations
        |
        v
Planner dashboard / reports
  decision support only; no direct control commands to BESCOM assets
```

## 3. Forecasting Architecture

Implemented in:

- `src/intelligence/competition_model.py`
- `src/intelligence/competition_forecaster.py`

### Tensor Contract

Let:

- `B`: batch size
- `L`: history length
- `H`: forecast horizon
- `N`: H3 cells
- `Fh`: historical features
- `Ff`: known-future exogenous features
- `Q`: quantiles, default `[0.1, 0.5, 0.9]`

Inputs:

```text
history_x: (B, L, N, Fh)
future_x:  (B, H, N, Ff)
```

Output:

```text
forecast:  (B, H, N, Q)
```

### Model Diagram

```text
history_x ──> Variable Selection Network ──> H3-biased Dynamic Graph Attention
                                                   |
                                                   v
                                             GRU encoder
                                                   |
future_x  ──> Variable Selection Network ──────────┘
                                                   |
                                                   v
                        node-wise temporal multi-head self-attention
                                                   |
                                                   v
                              future graph attention refinement
                                                   |
                                                   v
                               quantile head: p10 / p50 / p90
```

### Attention Mechanics

The graph layer computes multi-head node attention:

```text
A_h(i,j) = softmax_j((Q_i K_j^T / sqrt(d)) + log(pi_ij) + b_hij)
```

where `pi_ij` is the H3 topology prior and `b_hij` is a learned edge bias. This
lets nearby H3 cells dominate by default while still allowing long-range corridor
relations when the data supports them.

Temporal attention is applied per node across the concatenated history and
known-future tokens. This is stronger than one-step autoregression because the
model sees the full tariff/weather/traffic trajectory for the horizon at once.

### Loss

The training objective is:

```text
L = Pinball(y, q_hat) + 0.15 * Huber(y, median_hat)
    + 0.01 * mean(|median_hat[t] - median_hat[t-1]|)
```

The pinball term gives calibrated quantiles, Huber improves median accuracy under
spikes, and the smoothness term discourages unrealistic horizon jitter without
flattening real peaks.

### Features

Historical features include demand, neighbor demand, base load, weather, traffic,
solar, tariff, charger density, EV adoption, station count, growth index, rolling
lags, KDE arrival estimates, congestion, gig-fleet demand, and capacity headroom.

Known-future features include exogenous values available to a planner: tariff,
solar, weather forecast, traffic forecast, zone type, transformer capacity,
charger density, station count, growth, and expected arrival rates.

### Explainability

The variable selection networks emit learned feature weights. The wrapper exposes:

```python
forecaster.explain_global_feature_importance(top_n=20)
```

Forecast rows also include p10/p90 bands, selected forecast method, seasonal
baseline, persistence baseline, and guardrail notes.

## 4. Robust Optimization Engine

Implemented in:

- `src/optimization/robust_optimizer.py`
- `src/optimization/optimizer.py`

The new default scheduler is a robust LP. It can run every planning interval as a
rolling-horizon MPC layer. It never sends direct hardware commands; it emits a
planner-approved schedule recommendation.

### Decision Variables

For each flexible charging task `i` and feasible time slot `t`:

```text
x_it = energy assigned to task i at time t, in kWh
z    = aggregate peak load upper bound
d+_t, d-_t = absolute deviation variables around target average load
s_ct = transformer overload slack for cell c at time t
```

### Objective

```text
min  w_peak z
   + w_dev sum_t(d+_t + d-_t)
   + sum_i,t x_it * (
       w_tariff * tariff_t
     - w_solar  * solar_bonus_ct
     + w_delay  * priority_i * delay_ratio_it
     + w_stress * base_util_ct^2
     + w_uncert * sigma_ct / capacity_ct
   )
   + w_slack sum_c,t s_ct
```

### Constraints

Energy conservation:

```text
sum_t x_it = requested_energy_i
```

Deadline feasibility:

```text
x_it exists only for arrival_i <= t <= deadline_i
```

Per-slot charging power:

```text
0 <= x_it <= max_power_i * delta_t
```

Robust transformer headroom:

```text
base_ct + priority_ct + sum_i x_it / delta_t
  <= utilization_limit * derated_capacity_ct - z_uncert * sigma_ct + s_ct
```

Peak bound:

```text
base_t + priority_t + sum_i x_it / delta_t <= z
```

Load smoothing:

```text
load_t - target_average = d+_t - d-_t
```

This formulation gives hard service guarantees and explicit, explainable
violations when the grid is already overloaded before flexible EV scheduling.

## 5. Infrastructure Planning Engine

Implemented in:

- `src/optimization/siting.py`

The engine scores each H3 cell and then selects a budgeted portfolio.

Cell score drivers:

- forecast peak demand
- future growth
- charger gap
- neighbor pressure
- graph centrality: degree, PageRank, betweenness
- transformer headroom
- future utilization risk
- traffic accessibility
- equity/underserved zone adjustment

The portfolio stage is a greedy maximum-coverage selector. It adds neighbor
coverage value and penalizes adjacent redundancy, so a station budget spreads
across useful corridors instead of blindly clustering in one hotspot.

Outputs include:

- `siting_score`
- `roi_index`
- `capacity_feasibility`
- `recommended_station_kw`
- `future_utilization_forecast`
- human-readable `reason`
- uniform-placement baseline comparison

## 6. Evaluation Strategy

Implemented in:

- `src/evaluation/benchmarks.py`

Forecast metrics:

- MAE
- RMSE
- MAPE
- sMAPE
- peak error
- correlation
- prediction interval coverage for p10/p90

Scheduling metrics:

- peak reduction percent
- PAR reduction
- transformer overload reduction
- p95 utilization reduction
- cost savings
- solar alignment / CO2 proxy
- energy preservation error
- deadlines met
- Jain fairness index
- solver status and shadow prices

Infrastructure metrics:

- captured peak demand
- feasible station percentage
- ROI index
- uniform-placement improvement
- future overload risk cells
- portfolio coverage method

Baselines:

- unmanaged charging
- vanilla/guarded STGCN forecast
- Lagrangian MCDM scheduler
- robust LP scheduler
- uniform placement

## 7. Reproducibility And Scaling

The code supports CPU or CUDA through PyTorch. Training uses deterministic seeds,
AdamW, clipped gradients, Huber/quantile losses, and contiguous time validation.
The LP uses SciPy HiGHS with sparse matrices, so the memory footprint scales with
actual feasible task-slot pairs rather than dense tensor products.

For city-scale deployment:

- shard H3 corridors by feeder/DTR neighborhood
- train graph models with mixed precision and distributed data parallel
- run robust LP per feeder group with overlap on corridor boundaries
- log forecasts, optimizer status, shadow prices, and planner decisions
- compare shadow-mode recommendations against actual telemetry before pilot use

## 8. Pseudocode

Forecasting:

```text
fit(train):
  engineer features by (time, h3_cell)
  build H3 graph prior
  create windows history[L] -> future[H]
  train GraphTemporalFusionTransformer with quantile loss
  store feature importances and scalers

forecast(history, future):
  engineer history and known future features
  emit p10/p50/p90 for all H cells and time slots
  compare against seasonal/persistence guardrails when actual holdout exists
  return calibrated forecast frame
```

Scheduling:

```text
optimize(forecast):
  split each demand block into fixed priority load and flexible task
  create LP variables for feasible deadline slots
  add energy, deadline, power, transformer, peak, and deviation constraints
  solve sparse LP
  emit optimized load, shadow prices, overload slack, and fairness metrics
```

Siting:

```text
rank_cells(optimized_forecast):
  aggregate demand, utilization, growth, charger gap, traffic, centrality
  compute transparent weighted siting score
  greedily select budgeted portfolio with neighbor coverage and redundancy penalty
  compare against uniform placement
```
