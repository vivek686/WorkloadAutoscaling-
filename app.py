"""Streamlit dashboard: streamlit run app.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import pipeline as pl  # noqa: E402

st.set_page_config(page_title="Workload Forecasting for Autoscaling", layout="wide")
st.title("Workload Forecasting for Autoscaling")
st.caption("Simulated autoscaling: threshold-based vs forecast-based. No real infrastructure is provisioned.")

# ---- sidebar -------------------------------------------------------------- #
st.sidebar.header("Simulation settings")
split = st.sidebar.select_slider("Training split", options=list(pl.SPLITS), value=pl.DEFAULT_SPLIT)
cfg = pl.SimConfig(
    server_capacity=st.sidebar.number_input("Server capacity (CPU units)", 1.0, 100.0, 10.0, 1.0),
    target_util=st.sidebar.slider("Target utilisation", 0.3, 0.95, 0.70, 0.05),
    scale_up_threshold=st.sidebar.slider("Scale-up threshold (Threshold policy)", 0.5, 1.0, 0.80, 0.05),
    scale_down_threshold=st.sidebar.slider("Scale-down threshold (Threshold policy)", 0.1, 0.7, 0.40, 0.05),
)
scenario = st.sidebar.selectbox("Scenario", ["Normal Workload", "Injected Workload Spike", "Bad Data (Sanitized)"])


@st.cache_data(show_spinner="Running simulation ...")
def compute(split, cfg_tuple, scenario):
    cfg = pl.SimConfig(*cfg_tuple)
    df = pl.load_workload()
    y, _ = pl.sanitize(df["cpu_demand"].to_numpy())
    n_fixed = 0
    if scenario == "Injected Workload Spike":
        y = pl.inject_spike(y, split)
    elif scenario == "Bad Data (Sanitized)":
        y, n_fixed = pl.sanitize(pl.inject_bad_data(y))
    summary, traces, fc = pl.run_strategies(y, split, cfg)
    return y, summary, traces, fc, n_fixed, pl.forecast_metrics_by_split(y)


cfg_tuple = (cfg.server_capacity, cfg.min_servers, cfg.max_servers, cfg.target_util,
             cfg.scale_up_threshold, cfg.scale_down_threshold)
y, summary, traces, fc, n_fixed, fm = compute(split, cfg_tuple, scenario)
start = int(round(len(y) * split))
if n_fixed:
    st.info(f"{n_fixed} invalid values (NaN / inf / negative) were sanitized before simulation.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Workload", "Forecasting", "Autoscaling", "Failure injection", "Reproducibility"])

with tab1:
    st.subheader("Historical workload")
    st.line_chart(pd.DataFrame({"CPU demand": y}))
    st.caption(f"{len(y)} five-minute intervals. Test period starts at interval {start}.")

with tab2:
    st.subheader("One-step-ahead forecasts")
    st.line_chart(pd.DataFrame({"Actual": y[start:], "Naive": fc["Naive"], "Holt": fc["Holt"]},
                               index=np.arange(start, len(y))))
    st.dataframe(fm, width="stretch", hide_index=True)
    st.caption("MAPE excludes intervals with zero demand.")

with tab3:
    st.subheader("Strategy comparison")
    st.dataframe(summary, width="stretch", hide_index=True)
    chosen = st.multiselect("Show capacity for", list(pl.STRATEGIES), default=["Threshold", "Holt"])
    chart = pd.DataFrame({"Demand": y[start:]}, index=np.arange(start, len(y)))
    for s in chosen:
        chart[f"{s} capacity"] = traces[traces["strategy"] == s].set_index("interval")["capacity"]
    st.line_chart(chart)

with tab4:
    st.subheader("Failure injection results")
    p = pl.RESULTS_DIR / "failure_injection_results.csv"
    if p.exists():
        st.dataframe(pd.read_csv(p), width="stretch", hide_index=True)
    else:
        st.warning("Run `python run.py` or `python src/pipeline.py` first.")

with tab5:
    st.subheader("Repeated runs (identical inputs)")
    st.caption("These verify determinism, not independent statistical variation.")
    for name in ("repeated_forecast_summary.csv", "repeated_autoscaling_summary.csv"):
        p = pl.RESULTS_DIR / name
        if p.exists():
            st.markdown(f"**{name}**")
            st.dataframe(pd.read_csv(p), width="stretch", hide_index=True)
