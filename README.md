# Workload Forecasting for Autoscaling

A Python project that forecasts computing workload and simulates automatic resource allocation. It compares a traditional threshold-based autoscaler with forecast-based autoscalers.

## Objectives

- Analyze historical computing workload and forecast future CPU demand
- Compare Naive and Holt forecasting models
- Implement threshold-based, Naive-forecast, Holt-forecast and hybrid autoscaling
- Evaluate SLA overload and wasted capacity
- Test behaviour under sudden spikes and invalid/missing data
- Evaluate forecasts across multiple training splits and verify reproducibility

## Technologies

Python, NumPy, Pandas, Matplotlib, Statsmodels, Streamlit, Certifi

## Dataset

The intended source is Google Cluster Data (https://github.com/google/cluster-data). CPU usage records are aggregated into five-minute intervals and normalised so peak demand is 100 units.

`data/workload_data.csv` ships with a **synthetic stand-in trace** (76 intervals: trend, two seasonal cycles, noise, bursts, two idle intervals) so the project runs out of the box. To use the real trace, download a `task_usage` file and run:

```bash
python run.py --trace path/to/part-00000-of-00500.csv.gz
```

The loader expects the Google `task_usage` schema (column 0 = start time in microseconds, column 5 = mean CPU rate). It has not been tested against the real files, so check the column layout of the version you download.

## Forecasting models

1. **Naive** - next value equals the last observed value.
2. **Holt** - Holt linear exponential smoothing (statsmodels), refit at every step.

Forecasts are rolling one-step-ahead over the test portion of each split (50/60/70/80 % training).

## Autoscaling strategies

Capacity for interval *t* is decided using only information up to *t-1*. Sizing rule: `servers = ceil(estimate / (server_capacity x target_utilisation))`, clipped to [min, max].

| Strategy | Estimate used |
|---|---|
| Threshold | Reactive: resize using the last observation, only when utilisation leaves the 40-80 % band |
| Naive | Naive forecast, every interval |
| Holt | Holt forecast, every interval |
| Hybrid | max(Holt forecast, last observation) |

Defaults (editable in `SimConfig` or the dashboard sidebar): capacity 10 units/server, target utilisation 70 %, 1-30 servers.

## Failure injection

1. **Normal workload**
2. **Injected spike** - 3x demand for 2 intervals in the middle of the test period
3. **Bad data (sanitized)** - NaN, negative and infinite values injected, then repaired by interpolation before simulation

## Metrics

- RMSE and MAPE (MAPE excludes zero-demand intervals; the number excluded is reported)
- `sla_overload_pct` - share of intervals where demand exceeded capacity
- `unmet_demand_pct` - demand not served, as a share of total demand
- `wasted_capacity_pct` - idle capacity as a share of provisioned capacity
- `avg_servers`, `scaling_actions`

## Project structure

```
Workload-Autoscaling/
|-- data/
|   |-- workload_data.csv
|-- results/
|   |-- forecast_metrics_by_split.csv
|   |-- autoscaling_summary.csv
|   |-- autoscaling_traces.csv
|   |-- failure_injection_results.csv
|   |-- repeated_forecast_summary.csv
|   |-- repeated_autoscaling_summary.csv
|   |-- historical_workload.png
|   |-- forecast_comparison.png
|   |-- autoscaling_comparison.png
|-- src/
|   |-- pipeline.py
|-- app.py
|-- run.py
|-- requirements.txt
|-- README.md
|-- ARCHITECTURE.md
```

## Installation

```bash
git clone https://github.com/shambhavi-chandrawat/Workload-Forecasting-for-Autoscaling.git
cd Workload-Forecasting-for-Autoscaling
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Execution

```bash
python run.py                  # pipeline, then dashboard
python run.py --no-dashboard   # pipeline only
python src/pipeline.py         # pipeline only
streamlit run app.py           # dashboard only (reads results/)
```

## Limitations

- Simulated environment; no real cloud infrastructure is provisioned.
- The workload is short (76 aggregated intervals), so metric differences between models are noisy, and later splits have very little test data.
- Historical traces may not represent modern cloud workloads.
- Repeated runs use identical inputs: they verify determinism, not independent statistical variation.
- MAPE excludes zero-demand intervals.
- No provisioning delay is modelled, which removes much of the real-world advantage of forecasting ahead of demand.
- The shipped data is synthetic until a real trace is supplied.

## Future scope

Larger datasets, ARIMA and machine-learning models, provisioning delay, real-time monitoring, cloud integration, adaptive scaling policies.
