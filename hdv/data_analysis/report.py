"""
Standalone analysis entry point for highD *_tracks.csv files.

This script:
1) loads (or builds) the highD Feather cache (vehicle-centric),
2) runs kinematics analysis,
3) runs lane-change analysis (+ side context),
4) runs lane-pose report,
5) runs feasibility reports:
   - LC window feasibility (using lc directly)
   - Longitudinal window feasibility (using ax)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import math

# ---------------------------------------------------------
# Ensure project root (containing 'hdv/') is on PYTHONPATH
# ---------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
for parent in [_THIS_FILE] + list(_THIS_FILE.parents):
    if (parent / "hdv").is_dir():
        sys.path.insert(0, str(parent))
        break
else:
    raise ImportError("Could not locate project root containing 'hdv/' directory.")

from hdv.hdv_dbn.datasets.highd_loader import load_highd_folder

from kinematic_summary import run_kinematics_analysis
from lane_change_summary import run_lane_change_analysis
from lane_pose_analysis import run_lane_pose_report, LanePoseReportConfig
from window_feasibility import save_window_feasibility_report, WindowFeasibilityConfig
from longitudinal_window_feasibility import (
    save_longitudinal_window_feasibility_report,
    LongitudinalFeasibilityConfig,
)

# =========================================================
# Feather-first loading
# =========================================================
def load_highd_cached_or_build(tracks_dir, max_recordings=None, force_rebuild=False):
    """
    Priority:
      1) Use Feather cache if present
      2) Else build via load_highd_folder (which also writes the Feather)
    """
    tracks_dir = Path(tracks_dir)
    feather_path = (
        tracks_dir / "highd_all_with_meta.feather"
        if max_recordings is None
        else tracks_dir / f"highd_first_{int(max_recordings)}_with_meta.feather"
    )

    if (not force_rebuild) and feather_path.exists():
        print(f"[report] Using Feather cache: {feather_path}")
        return pd.read_feather(feather_path)

    print("[report] Feather cache missing (or force_rebuild=True). Building via load_highd_folder(...)")
    df = load_highd_folder(
        tracks_dir,
        force_rebuild=force_rebuild,
        max_recordings=max_recordings,
        apply_vehicle_centric=True,   
        flip_lateral=True,            
        flip_positions=False,
    )
    return df

# =========================================================
# Feasibility trajectory builder
# =========================================================

def build_trajs_for_feasibility(df, show_progress=True):
    """
    Convert the Feather dataframe into a list of per-trajectory dicts used by
    the feasibility modules.

    Each trajectory dict includes:
      - T   : length in frames
      - lc  : lane-change indicator sequence (preferred by window_feasibility.py)
      - ax  : longitudinal acceleration sequence (used by longitudinal feasibility)
      - frame : frame indices (optional; helpful for debugging)

    Parameters
    df : pandas.DataFrame
        Feather dataframe.
    show_progress : bool
        If True, show a tqdm progress bar over trajectories.

    Returns
    list of dict
        Per-trajectory dictionaries.
    """
    required = ["recording_id", "vehicle_id", "frame", "lc", "ax"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"[report] build_trajs_for_feasibility missing columns: {missing}")

    gb = df.groupby(["recording_id", "vehicle_id"], sort=False)
    items = list(gb)

    it = tqdm(items, desc="[report] build trajs", unit="traj", disable=(not show_progress))

    trajs = []
    for (_, _), g in it:
        g = g.sort_values("frame", kind="mergesort")

        frame = g["frame"].to_numpy(dtype=np.int64, copy=False)
        T = int(frame.size)
        if T <= 0:
            continue

        lc = pd.to_numeric(g["lc"], errors="coerce").round()
        lc = lc.fillna(0).astype(np.int64).to_numpy(copy=False)

        ax = pd.to_numeric(g["ax"], errors="coerce").to_numpy(dtype=np.float64, copy=False)

        # Basic consistency check
        if lc.size != T or ax.size != T:
            continue

        trajs.append({
            "T": T,
            "frame": frame,
            "lc": lc,
            "ax": ax,
        })

    return trajs


# =========================================================
# Percentile-based longitudinal thresholds
# =========================================================
def compute_longitudinal_thresholds_from_df(df, ax_col="ax"):
    """
    Compute percentile-based longitudinal action thresholds from vehicle-centric ax.

    Returns
    dict with keys:
      a_brake, a_accel, p10, p90
    """
    ax = pd.to_numeric(df[ax_col], errors="coerce").dropna()
    if ax.empty:
        raise ValueError("[report] ax column is empty; cannot compute thresholds.")

    p10 = float(ax.quantile(0.10))
    p90 = float(ax.quantile(0.90))

    return {
        "p10": p10,
        "p90": p90,
        "a_brake": abs(p10),
        "a_accel": p90,
    }

# =========================================================
# Main report
# =========================================================
def run_report(tracks_dir, out_dir, max_recordings=None, show_progress=True):
    """
    Run the dataset analysis and write outputs to disk.
    
    Parameters
    tracks_dir : str or pathlib.Path
        Directory containing highD files and/or Feather cache.
    out_dir : str or pathlib.Path
        Output directory for report products.
    max_recordings : int or None
        Optional subset size.
    show_progress : bool
        If True, show a single report-level progress bar.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(total=7, desc="[report] progress", disable=(not show_progress))

    try:
        # ---- Step 1: Load ----
        df = load_highd_cached_or_build(tracks_dir=tracks_dir, max_recordings=max_recordings, force_rebuild=False)
        pbar.update(1)

        # ---- Step 2: Kinematics ----
        run_kinematics_analysis(
            df=df,
            out_dir=out_dir / "kinematics",
            signals=("vx", "ax", "vy", "ay"),
            direction_col="meta_drivingDirection",
            class_col="meta_class",
            recording_col="recording_id",
            vehicle_col="vehicle_id",
            frame_col="frame",
            min_T_std=10,
        )
        pbar.update(1)

        # ---- Step 3: Lane-change ----
        run_lane_change_analysis(
            df=df,
            out_dir=out_dir / "lane_change",
            recording_col="recording_id",
            vehicle_col="vehicle_id",
            frame_col="frame",
            lc_col="lc",
            direction_col="meta_drivingDirection",
            class_col="meta_class",
            run_side_context=True,
        )
        pbar.update(1)

        # ---- Step 4: Lane-pose ----
        # Frame-based fractions (default). If you later want "one vote per vehicle", set count_unique_vehicles=True.
        cfg = LanePoseReportConfig(
            lane_pos_col="lane_pos",
            class_col="meta_class",
            direction_col="meta_drivingDirection",
            recording_col="recording_id",
            vehicle_col="vehicle_id",
            count_unique_vehicles=False,
            dpi=160,
        )
        run_lane_pose_report(df=df, out_dir=out_dir / "lane_pose", cfg=cfg)
        pbar.update(1)
        
        # ---- Step 5: Build traj dicts for feasibility ----
        trajs = build_trajs_for_feasibility(df, show_progress=show_progress)
        pbar.update(1)

        # ---- Step 6: Window feasibility (lane-change) ----
        cfg_w = WindowFeasibilityConfig(window_len=150, stride=10, min_gap_frames=10, lc_key="lc", lane_key="lane_id",)
        save_window_feasibility_report(out_dir=out_dir / "window_feasibility_lc", trajs=trajs, cfg=cfg_w)
        pbar.update(1)

        # ---- Step 7: Window feasibility (longitudinal) ----
        thr = compute_longitudinal_thresholds_from_df(df)
        W = 150
        min_frames = int(math.ceil(0.05 * W))
        cfg_long = LongitudinalFeasibilityConfig(
            window_len=W,
            stride=10,
            a_brake=thr["a_brake"],
            a_accel=thr["a_accel"],
            min_event_frames=5,
            min_gap_frames=10,
            min_frames_in_window=min_frames,
        )
        print(
            f"[report] Longitudinal thresholds: "
            f"a_brake={cfg_long.a_brake:.3f}, "
            f"a_accel={cfg_long.a_accel:.3f}"
        )
        save_longitudinal_window_feasibility_report(
            out_dir=out_dir / "window_feasibility_longitudinal",
            trajs=trajs,
            cfg=cfg_long,
        )
        pbar.update(1)
        
    finally:
        pbar.close()

if __name__ == "__main__":
    TRACKS_DIR = Path(r"C:\Users\amalj\OneDrive\Desktop\Master's Thesis\Implementation\hdv\data\highd")
    OUT_DIR = Path(__file__).resolve().parent / "data_analysis" / "highd_report"

    print("[report] Starting analysis...")
    print("[report] tracks_dir:", TRACKS_DIR)
    print("[report] out_dir:", OUT_DIR)

    run_report(
        tracks_dir=TRACKS_DIR,
        out_dir=OUT_DIR,
        max_recordings=5,
        show_progress=True,
    )

    print(f"[report] Analysis completed. Results written to:\n  {OUT_DIR}")
