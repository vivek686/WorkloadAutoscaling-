"""Run the full project: pipeline, then the Streamlit dashboard.

    python run.py                    # pipeline + dashboard
    python run.py --no-dashboard     # pipeline only
    python run.py --trace FILE.csv.gz  # build workload from a Google task_usage file
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import pipeline  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace", help="Google cluster task_usage CSV(.gz) to aggregate")
    p.add_argument("--refresh", action="store_true", help="Rebuild data/workload_data.csv")
    p.add_argument("--no-dashboard", action="store_true", help="Skip launching Streamlit")
    a = p.parse_args()

    pipeline.run_all(trace=a.trace, refresh=a.refresh)

    if not a.no_dashboard:
        print("Launching dashboard ...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py")], check=False)


if __name__ == "__main__":
    main()
