import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import json

from pathlib import Path
# -----------------------------
# Core computations
# -----------------------------
def count_lane_changes(lane_id):
    """
    Count lane changes in a lane-id sequence.

    Parameters
    lane_id : np.ndarray, shape (T,)
        Lane id per frame.

    Returns
    int
        Number of indices t where lane_id[t] != lane_id[t-1].
    """
    if lane_id is None or lane_id.size <= 1:
        return 0
    return int(np.sum(lane_id[1:] != lane_id[:-1]))

def _as_1d_int(x):
    """
    Convert to 1D int array or return None if unusable.

    Parameters
    x : array-like or None

    Returns
    np.ndarray, shape (N,), dtype int64, or None
    
    """
    if x is None:
        return None
    a = np.asarray(x)
    if a.ndim != 1:
        return None
    return a.astype(np.int64, copy=False)

def compute_lane_change_arrays(trajs, lane_key="lane_id"):
    """
    Extract two arrays from trajectories:
    - lane_changes_per_traj: (N,)
    - dwell_times_frames: (M,)

    Parameters
    trajs : iterable of dict
    lane_key : str

    Returns
    (lc, dwell)
        lc: np.ndarray[int64], shape (N,)
        dwell: np.ndarray[int64], shape (M,)
    """
    lane_changes = []

    for tr in trajs:
        lane = _as_1d_int(tr.get(lane_key, None))
        if lane is None:
            lane = _as_1d_int(tr.get("laneId", None))  # highD fallback
        if lane is None or lane.size <= 0:
            continue

        lane_changes.append(count_lane_changes(lane))

    lc = np.asarray(lane_changes, dtype=np.int64)
    return lc

# -----------------------------
# Plot helpers
# -----------------------------
def _finite_1d(x):
    """
    Return finite values as 1D float array.

    Parameters
    x : array-like or None

    Returns
    np.ndarray, shape (N,), dtype float
    """
    if x is None:
        return np.array([], dtype=float)
    a = np.asarray(x).astype(float, copy=False).ravel()
    return a[np.isfinite(a)]

def _save_hist(values, out_path, title, xlabel, ylabel="count", bins=60):
    v = _finite_1d(values)
    if v.size == 0:
        return

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
    Compute lane-change frequency dict.
    Parameters
    lc : array-like, shape (N,)
        Number of lane changes per trajectory.
    Returns
    dict
        Return frequency dict: how many trajectories have k lane changes.
    """
    lc = np.asarray(lc, dtype=np.int64)
    if lc.size == 0:
        return {"total_trajectories": 0, "counts": {}}

    uniq, cnt = np.unique(lc, return_counts=True)
    return {
        "total_trajectories": int(lc.size),
        "counts": {str(int(k)): int(v) for k, v in zip(uniq, cnt)},
    }

# -----------------------------
# Public API
# -----------------------------
def save_lane_change_plots(out_dir, trajs, lane_key="lane_id", bins=40):
    """
    Compute lane-change arrays and save plots.

    Parameters
    out_dir : str | Path
        Output directory for PNG files.
    trajs : iterable of dict
        Per-trajectory dicts.
    lane_key : str
        Lane id key (default "lane_id"). Fallback key "laneId" is auto-tried.
    frame_rate : float | None
        If provided, also plot dwell times in seconds (dwell_frames / frame_rate).
        For highD, frame_rate is typically 25.0 Hz.
    config : LaneChangePlotConfig | None
        Plot style/limits configuration.
    bins : int
        Number of histogram bins.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lc = compute_lane_change_arrays(trajs, lane_key=lane_key)

    _save_hist(
        lc,
        out_dir / "lane_changes_per_trajectory_hist.png",
        title="Lane changes per trajectory",
        xlabel="Number of lane changes",
        bins=bins
    )
    
def validate_lane_changes_against_meta(out_dir, trajs, df_tracks_meta, lane_key="lane_id", meta_col="numLaneChanges", make_plots=True):
    """
    Validate lane-change counts computed from lane-id sequences against tracksMeta.

    Parameters
    out_dir : str | Path
        Output directory for validation artifacts.
    trajs : iterable[dict]
        Trajectory dict 
    df_tracks_meta : pd.DataFrame
        Must contain columns:
          - recording_id
          - id
          - meta_col (default: 'numLaneChanges')
    lane_key : str
        Trajectory lane sequence key (default: 'lane_id').
    meta_col : str
        tracksMeta lane-change count column (default: 'numLaneChanges').
    make_plots : bool
        If True, save a diff histogram and scatter plot.
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
            "exact_match_rate": None,
            "lane_change_frequency_meta": None
        }
        (out_dir / "lanechange_validation_summary.json").write_text(json.dumps(summary, indent=2))

    #Prepare meta table
    meta = df_tracks_meta.copy()

    if "recording_id" not in meta.columns:
        raise KeyError("df_tracks_meta must contain 'recording_id' column (added in load_tracks_and_metadata).")
    if "id" not in meta.columns:
        raise KeyError("df_tracks_meta must contain 'id' column.")
    if meta_col not in meta.columns:
        raise KeyError(f"df_tracks_meta missing required column '{meta_col}'.")

    meta_small = meta[["recording_id", "id", meta_col]].copy()
    meta_small["recording_id"] = meta_small["recording_id"].astype(str)
    meta_small["id"] = pd.to_numeric(meta_small["id"], errors="coerce").astype("Int64")
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
            "lane_change_frequency_meta": meta_freq
        }
        (out_dir / "lanechange_validation_summary.json").write_text(json.dumps(summary, indent=2))
        
    merged["diff"] = merged["computed_numLaneChanges"] - merged["meta_numLaneChanges"]
    merged["abs_diff"] = merged["diff"].abs()

    n_comp = int(len(df_comp))
    n_match = int(len(merged))
    exact = int((merged["diff"] == 0).sum()) if n_match else 0

    summary = {
        "status": "ok",
        "num_computed": n_comp,
        "num_matched": n_match,
        "match_rate_over_computed": float(n_match / n_comp) if n_comp else None,
        "exact_match_rate": float(exact / n_match) if n_match else None,
        "lane_change_frequency_meta": meta_freq
    }

    if make_plots and n_match:
        # diff histogram
        plt.figure()
        plt.hist(merged["diff"].to_numpy(), bins=31, edgecolor="black", alpha=0.7)
        plt.xlabel("computed - meta")
        plt.ylabel("count")
        plt.title(f"Lane-change count difference (N={n_match}, exact={exact})")
        plt.tight_layout()
        plt.savefig(out_dir / "lanechange_diff_hist.png", dpi=200)
        plt.close()

    # write summary json
    (out_dir / "lanechange_validation_summary.json").write_text(json.dumps(summary, indent=2))
