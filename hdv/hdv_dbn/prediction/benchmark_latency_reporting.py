"""Reporting helpers for latency benchmark output."""

import csv
from pathlib import Path

from ..utils.latency_utils import summarize_arr


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
        "first_predict_call_ms",
        "first_prediction_per_vehicle_ms",
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
        "per_frame_windowize_mean_ms",
        "per_frame_windowize_p50_ms",
        "per_frame_windowize_p90_ms",
        "per_frame_windowize_p99_ms",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\n[benchmark] Wrote CSV summary: {csv_path}")


def print_block(title: str, res: dict, is_e2e: bool):
    print(f"\n{title}")
    for key in (
        "warmup_updates",
        "warmup_frames",
        "window_W",
        "window_stride",
        "first_prediction_total_ms",
        "first_predict_call_ms",
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
        if "per_frame_windowize_ms" in res:
            summarize_arr(res["per_frame_windowize_ms"], "  per-frame windowize (all frames)")
