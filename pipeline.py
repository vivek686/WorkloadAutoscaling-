"""Workload forecasting and autoscaling simulation pipeline.

Steps:
  1. Load (or build) a 5-minute aggregated CPU workload series.
  2. Forecast one step ahead with Naive and Holt models over several train/test splits.
  3. Simulate four autoscaling strategies and score SLA overload / wasted capacity.
  4. Re-run under failure injection (spike, bad data).
  5. Repeat the experiment to verify reproducibility.
  6. Write CSVs and graphs to results/.

Run directly:  python src/pipeline.py
"""
from __future__ import annotations

import argparse
import ssl
import urllib.request
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
WORKLOAD_CSV = DATA_DIR / "workload_data.csv"

INTERVAL_SECONDS = 300          # five-minute aggregation
N_INTERVALS = 76                # length of the selected trace window
SPLITS = (0.5, 0.6, 0.7, 0.8)   # training fractions
DEFAULT_SPLIT = 0.7
N_REPEATS = 5

STRATEGIES = ("Threshold", "Naive", "Holt", "Hybrid")
MODELS = ("Naive", "Holt")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class SimConfig:
    server_capacity: float = 10.0    # CPU units one server can serve per interval
    min_servers: int = 1
    max_servers: int = 30
    target_util: float = 0.70        # utilisation the sizing rule aims for
    scale_up_threshold: float = 0.80   # reactive policy: resize when util above
    scale_down_threshold: float = 0.40  # reactive policy: resize when util below


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def download(url: str, dest: Path) -> Path:
    """Download a file using certifi's CA bundle."""
    import certifi

    ctx = ssl.create_default_context(cafile=certifi.where())
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, context=ctx, timeout=60) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest


def build_workload_from_task_usage(path: str | Path, n_intervals: int = N_INTERVALS) -> pd.DataFrame:
    """Aggregate a Google cluster `task_usage` CSV(.gz) into 5-minute CPU demand.

    Expects the Google cluster-data task_usage schema where column 0 is the
    start time (microseconds) and column 5 is the mean CPU rate. Demand per
    interval is the sum of CPU rates, normalised so the peak equals 100.
    """
    df = pd.read_csv(path, header=None, usecols=[0, 5], names=["start_time", "cpu_rate"])
    df["interval"] = (df["start_time"] / 1e6 // INTERVAL_SECONDS).astype(int)
    agg = df.groupby("interval")["cpu_rate"].sum()
    agg = agg.reindex(range(int(agg.index.min()), int(agg.index.min()) + n_intervals), fill_value=0.0)
    demand = agg.to_numpy(dtype=float)
    if demand.max() > 0:
        demand = demand / demand.max() * 100.0
    return _frame(demand, source=f"google-task-usage:{Path(path).name}")


def synthetic_workload(n: int = N_INTERVALS, seed: int = 42) -> pd.DataFrame:
    """Stand-in trace: trend + two seasonalities + noise + bursts + idle gaps."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = 45 + 0.15 * t + 18 * np.sin(2 * np.pi * t / 24) + 7 * np.sin(2 * np.pi * t / 8)
    y = y + rng.normal(0, 4, n)
    y[30:32] *= 1.4
    y = np.clip(y, 0, None)
    y[[5, 41]] = 0.0   # idle intervals (excluded from MAPE)
    return _frame(y, source="synthetic")


def _frame(demand: np.ndarray, source: str) -> pd.DataFrame:
    n = len(demand)
    return pd.DataFrame({
        "interval": np.arange(n),
        "seconds": np.arange(n) * INTERVAL_SECONDS,
        "cpu_demand": np.round(demand, 4),
        "source": source,
    })


def load_workload(trace: str | None = None, refresh: bool = False) -> pd.DataFrame:
    """Return the workload frame, building and caching it if needed."""
    if trace:
        df = build_workload_from_task_usage(trace)
        DATA_DIR.mkdir(exist_ok=True)
        df.to_csv(WORKLOAD_CSV, index=False)
        return df
    if WORKLOAD_CSV.exists() and not refresh:
        return pd.read_csv(WORKLOAD_CSV)
    DATA_DIR.mkdir(exist_ok=True)
    df = synthetic_workload()
    df.to_csv(WORKLOAD_CSV, index=False)
    return df


def sanitize(values) -> tuple[np.ndarray, int]:
    """Replace NaN / inf / negative values by linear interpolation.

    Returns (clean_array, number_of_values_fixed).
    """
    s = pd.Series(np.asarray(values, dtype=float))
    s = s.replace([np.inf, -np.inf], np.nan)
    s[s < 0] = np.nan
    n_bad = int(s.isna().sum())
    s = s.interpolate(limit_direction="both")
    if s.isna().any():     # an all-invalid series
        s = s.fillna(0.0)
    return s.to_numpy(), n_bad


# --------------------------------------------------------------------------- #
# Forecasting (rolling one-step-ahead)
# --------------------------------------------------------------------------- #
def naive_forecast(y: np.ndarray, start: int) -> np.ndarray:
    return np.array([y[t - 1] for t in range(start, len(y))], dtype=float)


def holt_forecast(y: np.ndarray, start: int) -> np.ndarray:
    from statsmodels.tsa.holtwinters import Holt

    out = []
    for t in range(start, len(y)):
        hist = y[:t]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = Holt(hist, initialization_method="estimated").fit()
                f = float(np.asarray(fit.forecast(1))[0])
            if not np.isfinite(f):
                raise ValueError("non-finite forecast")
        except Exception:
            f = float(hist[-1])
        out.append(max(f, 0.0))
    return np.array(out, dtype=float)


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def mape(actual: np.ndarray, pred: np.ndarray) -> tuple[float, int]:
    """MAPE excluding zero-demand intervals. Returns (mape_pct, n_excluded)."""
    mask = actual > 0
    excluded = int((~mask).sum())
    if not mask.any():
        return float("nan"), excluded
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100), excluded


def forecast_metrics_by_split(y: np.ndarray, splits=SPLITS) -> pd.DataFrame:
    rows = []
    for frac in splits:
        start = int(round(len(y) * frac))
        actual = y[start:]
        for name, fn in (("Naive", naive_forecast), ("Holt", holt_forecast)):
            pred = fn(y, start)
            m, excl = mape(actual, pred)
            rows.append({
                "train_split": frac, "train_size": start, "test_size": len(actual),
                "model": name, "rmse": round(rmse(actual, pred), 4),
                "mape_pct": round(m, 4), "mape_zero_intervals_excluded": excl,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Autoscaling simulation
# --------------------------------------------------------------------------- #
def _size(demand_estimate: float, cfg: SimConfig) -> int:
    n = int(np.ceil(max(demand_estimate, 0.0) / (cfg.server_capacity * cfg.target_util)))
    return int(np.clip(n, cfg.min_servers, cfg.max_servers))


def simulate(strategy: str, demand: np.ndarray, start: int, cfg: SimConfig,
             fc_naive: np.ndarray, fc_holt: np.ndarray) -> pd.DataFrame:
    """Simulate one strategy over demand[start:].

    Capacity for interval t is decided using only information up to t-1
    (last observation and/or a one-step forecast). No provisioning delay.
    Strategies:
      Threshold - reactive: resize to target utilisation only when the last
                  interval's utilisation left the [down, up] band.
      Naive     - size for the Naive forecast (= last observation) every interval.
      Holt      - size for the Holt forecast every interval.
      Hybrid    - size for max(Holt forecast, last observation).
    """
    servers = _size(demand[start - 1], cfg)
    rows = []
    for i, t in enumerate(range(start, len(demand))):
        last = demand[t - 1]
        if strategy == "Threshold":
            util = last / (servers * cfg.server_capacity)
            if util > cfg.scale_up_threshold or util < cfg.scale_down_threshold:
                servers = _size(last, cfg)
        elif strategy == "Naive":
            servers = _size(fc_naive[i], cfg)
        elif strategy == "Holt":
            servers = _size(fc_holt[i], cfg)
        elif strategy == "Hybrid":
            servers = _size(max(fc_holt[i], last), cfg)
        else:
            raise ValueError(strategy)
        capacity = servers * cfg.server_capacity
        rows.append({
            "interval": t, "strategy": strategy, "demand": demand[t],
            "servers": servers, "capacity": capacity,
            "overload": demand[t] > capacity,
            "unmet": max(demand[t] - capacity, 0.0),
            "wasted": max(capacity - demand[t], 0.0),
        })
    return pd.DataFrame(rows)


def summarize(trace: pd.DataFrame) -> dict:
    return {
        "sla_overload_pct": round(float(trace["overload"].mean() * 100), 4),
        "unmet_demand_pct": round(float(trace["unmet"].sum() / max(trace["demand"].sum(), 1e-9) * 100), 4),
        "wasted_capacity_pct": round(float(trace["wasted"].sum() / max(trace["capacity"].sum(), 1e-9) * 100), 4),
        "avg_servers": round(float(trace["servers"].mean()), 4),
        "scaling_actions": int((trace["servers"].diff().fillna(0) != 0).sum()),
    }


def run_strategies(y: np.ndarray, split: float, cfg: SimConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (summary_df, per-interval traces, forecasts) for all strategies."""
    start = int(round(len(y) * split))
    fc = {"Naive": naive_forecast(y, start), "Holt": holt_forecast(y, start)}
    traces, rows = [], []
    for s in STRATEGIES:
        tr = simulate(s, y, start, cfg, fc["Naive"], fc["Holt"])
        traces.append(tr)
        rows.append({"strategy": s, **summarize(tr)})
    return pd.DataFrame(rows), pd.concat(traces, ignore_index=True), fc


# --------------------------------------------------------------------------- #
# Failure injection
# --------------------------------------------------------------------------- #
def inject_spike(y: np.ndarray, split: float, factor: float = 3.0, width: int = 2) -> np.ndarray:
    z = y.copy()
    start = int(round(len(y) * split))
    mid = start + (len(y) - start) // 2
    z[mid:mid + width] *= factor
    return z


def inject_bad_data(y: np.ndarray) -> np.ndarray:
    z = y.copy()
    n = len(z)
    for idx, val in zip((int(n * .13), int(n * .33), int(n * .66), int(n * .79)),
                        (np.nan, -50.0, np.inf, np.nan)):
        z[idx] = val
    return z


def failure_injection(y: np.ndarray, split: float, cfg: SimConfig) -> pd.DataFrame:
    scenarios = {
        "Normal Workload": (y, 0),
        "Injected Workload Spike": (inject_spike(y, split), 0),
    }
    bad = inject_bad_data(y)
    clean, n_fixed = sanitize(bad)
    scenarios["Bad Data (Sanitized)"] = (clean, n_fixed)

    rows = []
    for name, (series, n_fixed) in scenarios.items():
        summary, _, _ = run_strategies(series, split, cfg)
        summary.insert(0, "scenario", name)
        summary["invalid_values_fixed"] = n_fixed
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- #
# Experiments, reproducibility
# --------------------------------------------------------------------------- #
def run_experiments(y: np.ndarray, cfg: SimConfig, split: float = DEFAULT_SPLIT):
    fm = forecast_metrics_by_split(y)
    summary, traces, fc = run_strategies(y, split, cfg)
    return fm, summary, traces, fc


def repeated_checks(y: np.ndarray, cfg: SimConfig, first: tuple, n_repeats: int = N_REPEATS):
    """Re-run identical experiments and check results do not change.

    Inputs are identical each run, so this verifies determinism, not
    independent statistical variation.
    """
    fms, sums = [first[0]], [first[1]]
    for _ in range(n_repeats - 1):
        fm, summary, _, _ = run_experiments(y, cfg)
        fms.append(fm)
        sums.append(summary)

    fm_all = pd.concat(fms, keys=range(n_repeats), names=["run"]).reset_index(level=0)
    rep_fm = (fm_all.groupby(["train_split", "model"])
              .agg(runs=("run", "nunique"), mean_rmse=("rmse", "mean"), std_rmse=("rmse", "std"),
                   mean_mape_pct=("mape_pct", "mean"), std_mape_pct=("mape_pct", "std"))
              .reset_index())
    rep_fm["identical_across_runs"] = (rep_fm["std_rmse"].fillna(0) == 0) & (rep_fm["std_mape_pct"].fillna(0) == 0)

    sum_all = pd.concat(sums, keys=range(n_repeats), names=["run"]).reset_index(level=0)
    metrics = ["sla_overload_pct", "unmet_demand_pct", "wasted_capacity_pct", "avg_servers"]
    agg = {"runs": ("run", "nunique")}
    for m in metrics:
        agg[f"mean_{m}"] = (m, "mean")
        agg[f"std_{m}"] = (m, "std")
    rep_sum = sum_all.groupby("strategy").agg(**agg).reset_index()
    std_cols = [f"std_{m}" for m in metrics]
    rep_sum["identical_across_runs"] = (rep_sum[std_cols].fillna(0) == 0).all(axis=1)
    return rep_fm.round(6), rep_sum.round(6)


# --------------------------------------------------------------------------- #
# Graphs
# --------------------------------------------------------------------------- #
PALETTE = {"Naive": "#D55E00", "Holt": "#0072B2", "Threshold": "#6B6B6B", "Hybrid": "#009E73",
           "Actual": "#222222", "Capacity": "#0072B2"}


def _style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def make_graphs(y: np.ndarray, split: float, traces: pd.DataFrame, fc: dict, out: Path = RESULTS_DIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(exist_ok=True)
    start = int(round(len(y) * split))
    x = np.arange(len(y))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, color=PALETTE["Actual"], linewidth=1.6, label="CPU demand")
    ax.axvline(start - 0.5, color="#999999", linestyle="--", linewidth=1)
    ax.text(start - 0.5, ax.get_ylim()[1] * 0.97, " test period", fontsize=8, va="top", color="#555555")
    _style(ax, "Historical workload (5-minute intervals)", "CPU demand (units)")
    ax.set_xlabel("Interval")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "historical_workload.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    xt = np.arange(start, len(y))
    ax.plot(xt, y[start:], color=PALETTE["Actual"], linewidth=1.8, label="Actual")
    for m in MODELS:
        ax.plot(xt, fc[m], color=PALETTE[m], linewidth=1.4, label=f"{m} forecast")
    _style(ax, "One-step-ahead forecasts (test period)", "CPU demand (units)")
    ax.set_xlabel("Interval")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "forecast_comparison.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True, sharey=True)
    for ax, s in zip(axes.ravel(), STRATEGIES):
        tr = traces[traces["strategy"] == s]
        ax.plot(tr["interval"], tr["demand"], color=PALETTE["Actual"], linewidth=1.4, label="Demand")
        ax.step(tr["interval"], tr["capacity"], where="mid", color=PALETTE[s], linewidth=1.6, label="Capacity")
        _style(ax, s, "CPU units")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    for ax in axes[1]:
        ax.set_xlabel("Interval")
    fig.suptitle("Autoscaling: provisioned capacity vs demand", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(out / "autoscaling_comparison.png", dpi=150); plt.close(fig)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_all(trace: str | None = None, refresh: bool = False, cfg: SimConfig | None = None,
            n_repeats: int = N_REPEATS, verbose: bool = True) -> dict:
    cfg = cfg or SimConfig()
    RESULTS_DIR.mkdir(exist_ok=True)
    log = print if verbose else (lambda *a, **k: None)

    df = load_workload(trace=trace, refresh=refresh)
    y, n_bad = sanitize(df["cpu_demand"].to_numpy())
    log(f"Workload: {len(y)} intervals from '{df['source'].iloc[0]}' ({n_bad} invalid values fixed)")

    log("Forecasting and simulating autoscaling ...")
    first = run_experiments(y, cfg)
    fm, summary, traces, fc = first
    fm.to_csv(RESULTS_DIR / "forecast_metrics_by_split.csv", index=False)
    summary.to_csv(RESULTS_DIR / "autoscaling_summary.csv", index=False)

    log("Failure injection ...")
    fi = failure_injection(y, DEFAULT_SPLIT, cfg)
    fi.to_csv(RESULTS_DIR / "failure_injection_results.csv", index=False)

    log(f"Repeated reproducibility checks ({n_repeats} runs) ...")
    rep_fm, rep_sum = repeated_checks(y, cfg, first, n_repeats)
    rep_fm.to_csv(RESULTS_DIR / "repeated_forecast_summary.csv", index=False)
    rep_sum.to_csv(RESULTS_DIR / "repeated_autoscaling_summary.csv", index=False)

    log("Graphs ...")
    make_graphs(y, DEFAULT_SPLIT, traces, fc)
    traces.to_csv(RESULTS_DIR / "autoscaling_traces.csv", index=False)

    log(f"Done. Results in {RESULTS_DIR}")
    return {"workload": df, "forecast_metrics": fm, "autoscaling": summary, "failure": fi,
            "repeat_forecast": rep_fm, "repeat_autoscaling": rep_sum, "config": asdict(cfg)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", help="Path to a Google cluster task_usage CSV(.gz) to aggregate")
    p.add_argument("--trace-url", help="URL of a task_usage file to download first")
    p.add_argument("--refresh", action="store_true", help="Rebuild data/workload_data.csv")
    a = p.parse_args()
    trace = a.trace
    if a.trace_url:
        trace = str(download(a.trace_url, DATA_DIR / "task_usage_download.csv.gz"))
    run_all(trace=trace, refresh=a.refresh)


if __name__ == "__main__":
    main()
