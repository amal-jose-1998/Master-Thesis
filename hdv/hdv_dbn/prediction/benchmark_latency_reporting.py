"""Reporting helpers for latency benchmark output."""

import csv
from pathlib import Path
import numpy as np

from ..utils.latency_utils import summarize_arr
from ..utils.latency_utils import summary_stats


def make_csv_row(label, scenario, res, n_streams):
    stats_total = summary_stats(res.get("subsequent_total_ms", np.asarray([], dtype=np.float64)))
    stats_win = summary_stats(res.get("subsequent_windowize_ms", np.asarray([], dtype=np.float64)))
    stats_prep = summary_stats(res.get("subsequent_prep_ms", np.asarray([], dtype=np.float64)))
    stats_inf = summary_stats(res.get("subsequent_infer_ms", np.asarray([], dtype=np.float64)))

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
        "subsequent_windowize_mean_ms": stats_win["mean"],
        "subsequent_windowize_p50_ms": stats_win["p50"],
        "subsequent_windowize_p90_ms": stats_win["p90"],
        "subsequent_windowize_p99_ms": stats_win["p99"],
        "subsequent_prep_mean_ms": stats_prep["mean"],
        "subsequent_prep_p50_ms": stats_prep["p50"],
        "subsequent_prep_p90_ms": stats_prep["p90"],
        "subsequent_prep_p99_ms": stats_prep["p99"],
        "subsequent_infer_mean_ms": stats_inf["mean"],
        "subsequent_infer_p50_ms": stats_inf["p50"],
        "subsequent_infer_p90_ms": stats_inf["p90"],
        "subsequent_infer_p99_ms": stats_inf["p99"],
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
        "subsequent_windowize_mean_ms",
        "subsequent_windowize_p50_ms",
        "subsequent_windowize_p90_ms",
        "subsequent_windowize_p99_ms",
        "subsequent_prep_mean_ms",
        "subsequent_prep_p50_ms",
        "subsequent_prep_p90_ms",
        "subsequent_prep_p99_ms",
        "subsequent_infer_mean_ms",
        "subsequent_infer_p50_ms",
        "subsequent_infer_p90_ms",
        "subsequent_infer_p99_ms",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\n[benchmark] Wrote CSV summary: {csv_path}")


def print_block(title, res, is_e2e):
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

    if is_e2e:
        if "subsequent_windowize_ms" in res:
            summarize_arr(res["subsequent_windowize_ms"], "  subsequent windowize")
        if "subsequent_prep_ms" in res:
            summarize_arr(res["subsequent_prep_ms"], "  subsequent prep")
        if "subsequent_infer_ms" in res:
            summarize_arr(res["subsequent_infer_ms"], "  subsequent infer (update+predict)")
