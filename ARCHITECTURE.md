# Architecture

```
data/workload_data.csv --> sanitize --> forecasting --> autoscaling simulator --> metrics --> results/*.csv, *.png
                                             |                  ^                                   |
                                             +-- failure injection (spike, bad data)               v
                                                                                           app.py (Streamlit)
```

## Modules

- `src/pipeline.py` - all logic, importable and runnable.
  - Data: `load_workload`, `build_workload_from_task_usage`, `synthetic_workload`, `sanitize`, `download`
  - Forecasting: `naive_forecast`, `holt_forecast`, `rmse`, `mape`, `forecast_metrics_by_split`
  - Simulation: `SimConfig`, `simulate`, `summarize`, `run_strategies`
  - Robustness: `inject_spike`, `inject_bad_data`, `failure_injection`, `repeated_checks`
  - Output: `make_graphs`, `run_all`
- `run.py` - runs `run_all`, then launches the dashboard.
- `app.py` - Streamlit dashboard; recomputes with sidebar settings and reads the results CSVs.

## Design decisions

- **No lookahead.** Capacity for interval *t* uses only data up to *t-1*; forecasts are rolling one-step-ahead.
- **One sizing rule for all strategies** so they differ only in the demand estimate and (for Threshold) the trigger band.
- **Sanitize before simulating.** NaN, inf and negative values are interpolated; the count is reported.
- **Determinism.** No randomness outside the seeded synthetic trace; repeated runs confirm identical output.
- **Known simplification.** No provisioning delay, so scaling takes effect instantly.
