"""Reproducible, non-gating benchmark for dataset loading and analysis.

Example (PowerShell):
  .\.venv_regular\Scripts\python.exe scripts\benchmark_dataset.py --rows 200000

Timings vary by hardware; this script reports measurements rather than enforcing
fragile wall-clock thresholds in CI.
"""
import argparse
import io
import json
import time
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_analyzer import analyze_data
from data_profiler import profile_dataset


def timed(function):
    started = time.perf_counter()
    result = function()
    return result, time.perf_counter() - started


def make_dataset(rows):
    return pd.DataFrame({
        "Transaction ID": range(rows),
        "Sales": [(i % 997) * 1.25 for i in range(rows)],
        "Quantity": [(i % 12) + 1 for i in range(rows)],
        "Region": [["East", "West", "North", "South"][i % 4] for i in range(rows)],
        "Category": [["Office", "Technology", "Furniture"][i % 3] for i in range(rows)],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000, help="Synthetic rows to benchmark")
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be positive")

    source, build_seconds = timed(lambda: make_dataset(args.rows))
    csv_bytes, serialize_seconds = timed(lambda: source.to_csv(index=False).encode("utf-8"))
    loaded, load_seconds = timed(lambda: pd.read_csv(io.BytesIO(csv_bytes)))
    profile, profile_seconds = timed(lambda: profile_dataset(loaded))
    grouped, group_seconds = timed(lambda: analyze_data(
        loaded, {"intent": "group_count", "group_by": "Region"}
    ))
    averaged, average_seconds = timed(lambda: analyze_data(
        loaded, {"intent": "average", "column": "Sales"}
    ))

    report = {
        "rows": int(loaded.shape[0]),
        "columns": int(loaded.shape[1]),
        "csv_size_mb": round(len(csv_bytes) / (1024 * 1024), 2),
        "dataframe_memory_mb": round(float(loaded.memory_usage(deep=True).sum()) / (1024 * 1024), 2),
        "profile_rows_reported": int(profile["rows"]),
        "group_count_success": bool(grouped.get("success")),
        "average_success": bool(averaged.get("success")),
        "average_sales": float(averaged["value"]) if averaged.get("success") else None,
        "seconds": {
            "build_synthetic_data": round(build_seconds, 4),
            "serialize_csv": round(serialize_seconds, 4),
            "parse_csv": round(load_seconds, 4),
            "profile_dataset": round(profile_seconds, 4),
            "group_count": round(group_seconds, 4),
            "average": round(average_seconds, 4),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
