"""
Lane-change count analysis utilities.

This module computes lane-change counts per trajectory and provides simple
histogram plots. It also includes a validation routine against highD tracksMeta
(numLaneChanges).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# Core computations
# =========================================================

def count_lane_changes(lane_id):
    """
    Count lane changes in a lane-id sequence.

    Parameters
    ----------
    lane_id : array-like
        Lane id per frame (length T).

    Returns
    -------
    int
        Number of indices t where lane_id[t] != lane_id[t-1].
    """
    if lane_id is None:
        return 0
    lane_id = np.asarray(lane_id)
    if lane_id.size <= 1:
        return 0
    return int(np.sum(lane_id[1:] != lane_id[:-1]))

def lane_change_ternary(lane_id):
    """
    Per-frame ternary lane-change signal:
        lc[t] ∈ {-1, 0, +1}
    using sign(lane_id[t] - lane_id[t-1]), with lc[0] = 0.
    """
    if lane_id is None:
        return np.zeros((0,), dtype=np.int64)

    lane_id = np.asarray(lane_id, dtype=np.int64)
    if lane_id.size <= 1:
        return np.zeros_like(lane_id, dtype=np.int64)

    dlane = np.diff(lane_id, prepend=lane_id[0])
    lc = np.sign(dlane).astype(np.int64)
    lc[dlane == 0] = 0
    return lc


def compute_lane_change_ternary_sequences(trajs, lane_key="lane_id"):
    """
    Return list of lc arrays (one per trajectory).
    """
    out = []
    for tr in trajs:
        lane = _as_1d_int(tr.get(lane_key, None))
        if lane is None:
            lane = _as_1d_int(tr.get("laneId", None))
        if lane is None or lane.size == 0:
            continue
        out.append(lane_change_ternary(lane))
    return out


def _save_bar_counts(counts_dict, out_path, title, xlabel, ylabel="count"):
    """
    Save a simple bar plot for discrete counts dict like {-1: n, 0: n, +1: n}.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    keys = [-1, 0, 1]
    vals = [int(counts_dict.get(k, 0)) for k in keys]
    labels = ["-1 (left)", "0 (none)", "+1 (right)"]

    plt.figure()
    plt.bar(labels, vals, edgecolor="black", alpha=0.8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_lane_change_ternary_plots(out_dir, trajs, lane_key="lane_id", bins=40):
    """
    Save plots for ternary per-frame lane change signal lc ∈ {-1,0,+1}.

    Outputs:
      - lc_value_counts_overall.png  (bar: left/none/right over all frames)
      - lc_nonzero_fraction_hist.png (hist: fraction of frames with lc != 0 per trajectory)
      - lanechange_ternary_summary.json
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lc_seqs = compute_lane_change_ternary_sequences(trajs, lane_key=lane_key)
    if not lc_seqs:
        return

    # overall counts across all frames
    all_lc = np.concatenate(lc_seqs, axis=0)
    uniq, cnt = np.unique(all_lc, return_counts=True)
    counts = {int(k): int(v) for k, v in zip(uniq, cnt)}

    _save_bar_counts(
        counts,
        out_dir / "lc_value_counts_overall.png",
        title="Lane-change ternary values across all frames",
        xlabel="lc value",
        ylabel="frame count",
    )

    # per-trajectory "how often lane changes happen" as a fraction of frames
    frac = []
    for lc in lc_seqs:
        if lc.size == 0:
            continue
        frac.append(float(np.mean(lc != 0)))
    frac = np.asarray(frac, dtype=float)

    _save_hist(
        frac,
        out_dir / "lc_nonzero_fraction_hist.png",
        title="Fraction of frames with lane change (lc != 0)",
        xlabel="fraction per trajectory",
        bins=bins,
    )

    summary = {
        "total_trajectories": int(len(lc_seqs)),
        "overall_counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "nonzero_fraction": {
            "mean": float(np.mean(frac)) if frac.size else None,
            "median": float(np.median(frac)) if frac.size else None,
            "p90": float(np.quantile(frac, 0.90)) if frac.size else None,
        },
    }
    (out_dir / "lanechange_ternary_summary.json").write_text(json.dumps(summary, indent=2))

def _as_1d_int(x):
    """
    Convert to a 1D int array. Returns None if unusable.

    Parameters
    ----------
    x : array-like | None

    Returns
    -------
    np.ndarray | None
        1D int array or None.
    """
    if x is None:
        return None
    a = np.asarray(x)
    if a.ndim != 1:
        return None
    return a.astype(np.int64, copy=False)


def compute_lane_change_counts(trajs, lane_key="lane_id"):
    """
    Compute lane-change count for each trajectory.

    Parameters
    ----------
    trajs : iterable of dict
        Per-trajectory dicts.
    lane_key : str
        Key for lane-id sequence. If missing, "laneId" is tried.

    Returns
    -------
    np.ndarray
        Counts per usable trajectory (length N).
    """
    lane_changes = []
    for tr in trajs:
        lane = _as_1d_int(tr.get(lane_key, None))
        if lane is None:
            lane = _as_1d_int(tr.get("laneId", None))
        if lane is None or lane.size == 0:
            continue
        lane_changes.append(count_lane_changes(lane))
    return np.asarray(lane_changes, dtype=np.int64)


# =========================================================
# Plot helpers
# =========================================================

def _finite_1d(x):
    """
    Return finite values as a 1D float array.

    Parameters
    ----------
    x : array-like | None

    Returns
    -------
    np.ndarray
        Finite float values.
    """
    if x is None:
        return np.array([], dtype=float)
    a = np.asarray(x).astype(float, copy=False).ravel()
    return a[np.isfinite(a)]


def _save_hist(values, out_path, title, xlabel, ylabel="count", bins=60):
    """
    Save a histogram plot to disk.

    Parameters
    ----------
    values : array-like
        Data to histogram.
    out_path : str | Path
        Output png.
    title : str
        Plot title.
    xlabel : str
        X label.
    ylabel : str
        Y label.
    bins : int
        Histogram bins.
    """
    v = _finite_1d(values)
    if v.size == 0:
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.hist(v, bins=bins, edgecolor="black", alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _lane_change_frequency_dict(lc):
    """
    Compute a frequency dict for show/recording.

    Parameters
    ----------
    lc : array-like
        Lane-change counts per trajectory.

    Returns
    -------
    dict
        {"total_trajectories": N, "counts": {"0": c0, "1": c1, ...}}
    """
    lc = np.asarray(lc, dtype=np.int64)
    if lc.size == 0:
        return {"total_trajectories": 0, "counts": {}}

    uniq, cnt = np.unique(lc, return_counts=True)
    return {
        "total_trajectories": int(lc.size),
        "counts": {str(int(k)): int(v) for k, v in zip(uniq, cnt)},
    }


# =========================================================
# Public API
# =========================================================

def save_lane_change_plots(out_dir, trajs, lane_key="lane_id", bins=40):
    """
    Compute lane-change counts and save plots.

    Parameters
    ----------
    out_dir : str | Path
        Output directory.
    trajs : iterable of dict
        Per-trajectory dicts.
    lane_key : str
        Lane sequence key. "laneId" is auto-tried if missing.
    bins : int
        Histogram bins.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lc = compute_lane_change_counts(trajs, lane_key=lane_key)

    _save_hist(
        lc,
        out_dir / "lane_changes_per_trajectory_hist.png",
        title="Lane changes per trajectory",
        xlabel="Number of lane changes",
        bins=bins,
    )

    freq = _lane_change_frequency_dict(lc)
    (out_dir / "lanechange_frequency_computed.json").write_text(json.dumps(freq, indent=2))


def validate_lane_changes_against_meta(out_dir, trajs, df_tracks_meta, lane_key="lane_id", meta_col="numLaneChanges"):
    """
    Validate computed lane-change counts against tracksMeta.

    Parameters
    ----------
    out_dir : str | Path
        Output directory for JSON summary.
    trajs : iterable of dict
        Trajectory dicts containing recording_id and id/vehicle_id.
    df_tracks_meta : pd.DataFrame
        tracksMeta table. Must include recording_id, id, and meta_col.
    lane_key : str
        Key name for lane id sequence.
    meta_col : str
        tracksMeta lane-change count column (default: numLaneChanges).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for tr in trajs:
        rid = tr.get("recording_id", None)
        vid = tr.get("vehicle_id", tr.get("id", None))
        if rid is None or vid is None:
            continue

        lane = _as_1d_int(tr.get(lane_key, None))
        if lane is None:
            lane = _as_1d_int(tr.get("laneId", None))
        if lane is None or lane.size == 0:
            continue

        rows.append({
            "recording_id": str(rid),
            "id": int(vid),
            "computed_numLaneChanges": int(count_lane_changes(lane)),
        })

    df_comp = pd.DataFrame(rows)
    if df_comp.empty:
        summary = {
            "status": "no_computed_rows",
            "num_computed": 0,
            "num_matched": 0,
            "match_rate_over_computed": None,
            "exact_match_rate": None,
            "lane_change_frequency_meta": None,
        }
        (out_dir / "lanechange_validation_summary.json").write_text(json.dumps(summary, indent=2))
        return

    meta = df_tracks_meta.copy()
    if "recording_id" not in meta.columns:
        raise KeyError("df_tracks_meta must contain 'recording_id'.")
    if "id" not in meta.columns:
        raise KeyError("df_tracks_meta must contain 'id'.")
    if meta_col not in meta.columns:
        raise KeyError(f"df_tracks_meta missing required column '{meta_col}'.")

    meta_small = meta[["recording_id", "id", meta_col]].copy()
    meta_small["recording_id"] = meta_small["recording_id"].astype(str)
    meta_small["id"] = pd.to_numeric(meta_small["id"], errors="coerce")
    meta_small[meta_col] = pd.to_numeric(meta_small[meta_col], errors="coerce")
    meta_small = meta_small.dropna(subset=["id", meta_col]).copy()
    meta_small["id"] = meta_small["id"].astype(int)
    meta_small[meta_col] = meta_small[meta_col].astype(int)

    meta_freq = _lane_change_frequency_dict(meta_small[meta_col].to_numpy(dtype=np.int64))

    merged = df_comp.merge(meta_small, on=["recording_id", "id"], how="inner")
    merged = merged.rename(columns={meta_col: "meta_numLaneChanges"})

    if merged.empty:
        summary = {
            "status": "no_matches_after_join",
            "num_computed": int(len(df_comp)),
            "num_matched": 0,
            "match_rate_over_computed": 0.0,
            "exact_match_rate": None,
            "lane_change_frequency_meta": meta_freq,
        }
        (out_dir / "lanechange_validation_summary.json").write_text(json.dumps(summary, indent=2))
        return

    merged["diff"] = merged["computed_numLaneChanges"] - merged["meta_numLaneChanges"]
    n_comp = int(len(df_comp))
    n_match = int(len(merged))
    exact = int((merged["diff"] == 0).sum())

    summary = {
        "status": "ok",
        "num_computed": n_comp,
        "num_matched": n_match,
        "match_rate_over_computed": float(n_match / n_comp) if n_comp else None,
        "exact_match_rate": float(exact / n_match) if n_match else None,
        "lane_change_frequency_meta": meta_freq,
    }

    (out_dir / "lanechange_validation_summary.json").write_text(json.dumps(summary, indent=2))
