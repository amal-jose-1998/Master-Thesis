"""Reporting helpers for latency benchmark output."""

import csv
from pathlib import Path
import numpy as np

from ..utils.latency_utils import summarize_arr
from ..utils.latency_utils import summary_stats


def make_csv_row(label: str, scenario, res: dict, n_streams: int):
    stats_total = summary_stats(res.get("subsequent_total_ms", np.asarray([], dtype=np.float64)))

    return {
        "device": label.lower(),
        "scenario": scenario,
        "n_veh": int(n_streams),
        "warmup_updates": res.get("warmup_updates", np.nan),
        "warmup_frames": res.get("warmup_frames", np.nan),
        "window_W": res.get("window_W", np.nan),
        "window_stride": res.get("window_stride", np.nan),
        "first_prediction_total_ms": float(res.get("first_prediction_total_ms", np.nan)),
        "subsequent_mean_ms": stats_total["mean"],
        "subsequent_p50_ms": stats_total["p50"],
        "subsequent_p90_ms": stats_total["p90"],
        "subsequent_p99_ms": stats_total["p99"],
    }


def write_results_csv(csv_path: Path, rows):
    fieldnames = [
        "device",
        "scenario",
        "n_veh",
        "warmup_updates",
        "warmup_frames",
        "window_W",
        "window_stride",
        "first_prediction_total_ms",
        "subsequent_mean_ms",
        "subsequent_p50_ms",
        "subsequent_p90_ms",
        "subsequent_p99_ms",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\n[benchmark] Wrote CSV summary: {csv_path}")


def print_block(title, res):
    print(f"\n{title}")
    for key in (
        "warmup_updates",
        "warmup_frames",
        "window_W",
        "window_stride",
        "first_prediction_total_ms",
    ):
        if key in res:
            print(f"  {key}: {res[key]}")

    if "subsequent_total_ms" in res:
        summarize_arr(res["subsequent_total_ms"], "  subsequent total")
