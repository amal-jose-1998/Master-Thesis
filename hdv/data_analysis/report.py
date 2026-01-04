"""
Standalone analysis entry point for highD *_tracks.csv files.

This script:
1) loads tracks + tracksMeta + recordingMeta,
2) runs kinematic plots,
3) runs lane-change plots + meta validation + context plots,
4) runs lane-pose report + debug plots (representative violation examples).

Notes
-----
- This script bootstraps the project root so it can import the `hdv/` package when
  executed as a standalone file.
"""

import sys
import json
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


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


from hdv.hdv_dbn.datasets.highd_loader import add_lane_position_feature

from kinematic_summary import (
    plot_global_kinematics,
    plot_per_vehicle_variability,
    plot_per_recording_kinematics,
    plot_signal_by_direction,
    plot_signal_by_class,
)
from lane_change_analysis import save_lane_change_plots, validate_lane_changes_against_meta
from lane_change_context import (
    LaneChangeContextConfig,
    compute_lane_change_context,
    save_lane_change_context_plots,
)
from lane_pose_analysis import (
    LanePoseReportConfig,
    run_lane_pose_report,
    plot_lane_pose_debug,
)

# =========================================================
# I/O schema (minimal and stable)
# =========================================================

TRACKS_COLS = [
    "id", "frame", "laneId",
    "x", "y",
    "width", "height",
    "xVelocity", "yVelocity", "xAcceleration", "yAcceleration",
]
TRACKS_META_COLS = [
    "id", "numFrames", "class", "drivingDirection",
    "minXVelocity", "maxXVelocity", "meanXVelocity", "numLaneChanges",
]
RECORDING_META_COLS = [
    "numVehicles", "numCars", "numTrucks",
    "upperLaneMarkings", "lowerLaneMarkings",
]


# =========================================================
# Loading
# =========================================================

def load_highd_bundle(tracks_dir, pattern="*_tracks.csv", max_files=None, show_progress=True):
    """
    Load highD tracks + tracksMeta + recordingMeta.

    Returns
    -------
    (df_tracks, df_tracks_meta, df_recording_meta)
    """
    tracks_dir = Path(tracks_dir)
    files = sorted(tracks_dir.glob(pattern))
    if max_files is not None:
        files = files[: int(max_files)]
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {tracks_dir}")

    it = tqdm(files, desc="Loading highD", unit="file") if show_progress else files

    dfs_tracks, dfs_tm, dfs_rm = [], [], []
    for p in it:
        rec_id = p.stem.split("_", 1)[0]

        df = pd.read_csv(p, usecols=TRACKS_COLS)
        df["x_center"] = df["x"].astype(float) + 0.5 * df["width"].astype(float)
        df["y_center"] = df["y"].astype(float) + 0.5 * df["height"].astype(float)
        df["recording_id"] = rec_id
        dfs_tracks.append(df)

        tm_path = p.parent / f"{rec_id}_tracksMeta.csv"
        if tm_path.exists():
            tm = pd.read_csv(tm_path, usecols=TRACKS_META_COLS)
            tm["recording_id"] = rec_id
            dfs_tm.append(tm)

        rm_path = p.parent / f"{rec_id}_recordingMeta.csv"
        if rm_path.exists():
            rm = pd.read_csv(rm_path, usecols=RECORDING_META_COLS)
            rm["recording_id"] = rec_id
            dfs_rm.append(rm)

    df_all = pd.concat(dfs_tracks, ignore_index=True, copy=False)
    df_tracks_meta = pd.concat(dfs_tm, ignore_index=True) if dfs_tm else pd.DataFrame()
    df_recording_meta = pd.concat(dfs_rm, ignore_index=True) if dfs_rm else pd.DataFrame()

    return df_all, df_tracks_meta, df_recording_meta


def build_traj_dicts(df_all, columns, id_col="id", frame_col="frame", include_frame=True, show_progress=True):
    """
    Convert long-table tracks into a list of per-vehicle trajectory dicts.
    """
    required = {"recording_id", id_col, frame_col}
    missing = required - set(df_all.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    cols = []
    if include_frame and frame_col not in columns:
        cols.append(frame_col)
    cols.extend([c for c in columns if c != "recording_id"])
    cols = [c for c in cols if c in df_all.columns]

    grouped = df_all.groupby(["recording_id", id_col], sort=False)
    iterator = grouped
    if show_progress:
        iterator = tqdm(grouped, total=grouped.ngroups, desc="Building trajectories", unit="traj")

    trajs = []
    for (rec_id, vid), g in iterator:
        g = g.sort_values(frame_col)
        tr = {"recording_id": rec_id, "vehicle_id": int(vid), "T": int(len(g))}
        for col in cols:
            tr[col] = g[col].to_numpy()
        trajs.append(tr)

    return trajs


# =========================================================
# Lane pose pipeline (training-aligned)
# =========================================================

def _norm_class(v):
    """
    Normalize vehicle class strings to {car, truck} when possible.
    """
    s = str(v).strip().lower()
    if "truck" in s or s == "2":
        return "truck"
    if "car" in s or s == "1":
        return "car"
    return s


def build_lane_pose_df(df_all, df_tracks_meta, df_recording_meta):
    """
    Build a lane-pose-ready table aligned to add_lane_position_feature().

    Returns
    -------
    pd.DataFrame
        Includes:
          - y
          - meta_drivingDirection
          - rec_upperLaneMarkings / rec_lowerLaneMarkings
          - meta_class
        Also keeps: id, frame, x, y, recording_id for debug plotting.
    """
    df = df_all.copy()

    if df_tracks_meta is not None and not df_tracks_meta.empty:
        tm = df_tracks_meta.rename(
            columns={"class": "meta_class", "drivingDirection": "meta_drivingDirection"}
        ).copy()
        tm["meta_class"] = tm["meta_class"].astype(str).str.lower()

        df = df.merge(
            tm[["recording_id", "id", "meta_class", "meta_drivingDirection"]],
            on=["recording_id", "id"],
            how="left",
        )

    if df_recording_meta is not None and not df_recording_meta.empty:
        rm = df_recording_meta.rename(
            columns={
                "upperLaneMarkings": "rec_upperLaneMarkings",
                "lowerLaneMarkings": "rec_lowerLaneMarkings",
            }
        ).copy()

        df = df.merge(
            rm[["recording_id", "rec_upperLaneMarkings", "rec_lowerLaneMarkings"]],
            on="recording_id",
            how="left",
        )

    return df


def pick_representative_violators(df_lane, violation_values=(0, 4)):
    """
    Pick up to 4 representative VIOLATION vehicles based on lane_pos.

    Groups
    ------
      (dir=1, car), (dir=1, truck), (dir=2, car), (dir=2, truck)

    Selection rule
    --------------
    Pick the vehicle with the most frames where lane_pos is in violation_values
    (default: (0,4)) within each group.

    Returns
    -------
    pd.DataFrame
        Columns: recording_id, id, meta_drivingDirection, meta_class_norm, n_bad
    """
    viol_frames = df_lane[df_lane["lane_pos"].isin(list(violation_values))].copy()
    if viol_frames.empty:
        return pd.DataFrame()

    viol = (
        viol_frames.groupby(["recording_id", "id"], as_index=False)
        .agg(
            meta_drivingDirection=("meta_drivingDirection", "first"),
            meta_class=("meta_class", "first"),
            n_bad=("lane_pos", "size"),
        )
    )
    viol["meta_class_norm"] = viol["meta_class"].map(_norm_class)

    picks = []
    for direction in [1, 2]:
        for cls in ["car", "truck"]:
            cand = viol[(viol["meta_drivingDirection"] == direction) & (viol["meta_class_norm"] == cls)]
            if not cand.empty:
                picks.append(cand.sort_values("n_bad", ascending=False).iloc[0])

    if not picks:
        # fallback: at least one per direction
        for direction in [1, 2]:
            cand = viol[viol["meta_drivingDirection"] == direction]
            if not cand.empty:
                picks.append(cand.sort_values("n_bad", ascending=False).iloc[0])

    return pd.DataFrame(picks).drop_duplicates(subset=["recording_id", "id"])


def save_lane_pose_debug_plots(df_lane, df_recording_meta, out_dir,
                              violation_values=(0, 4), highlight_unknown=True):
    """
    Save debug plots for representative violation vehicles.

    Plot style matches your aerial road view:
      - Drivable lanes: grey
      - Median: green
      - lane_pos in violation_values (default 0/4): red points
      - lane_pos == -1 (optional): orange points
      - y-axis inverted by default
    """
    out_dir = Path(out_dir)
    debug_dir = out_dir / "debug_plots"
    debug_dir.mkdir(parents=True, exist_ok=True)

    targets = pick_representative_violators(df_lane, violation_values=violation_values) 
    if targets.empty:
        print("[lane_pose] No violation vehicles found (lane_pos in %s)." % (str(tuple(violation_values)),))
        return

    for row in targets.itertuples(index=False):
        rec_id = str(row.recording_id)
        vid = int(row.id)
        dd = int(row.meta_drivingDirection) if pd.notna(row.meta_drivingDirection) else None
        vclass = str(row.meta_class_norm)

        tracks_one = df_lane[
            (df_lane["recording_id"].astype(str) == rec_id) & (df_lane["id"] == vid)
        ][["id", "frame", "x", "y", "recording_id", "lane_pos"]].copy()
        tracks_one.sort_values("frame", inplace=True)

        rec_meta_one = df_recording_meta[df_recording_meta["recording_id"].astype(str) == rec_id].copy()
        if tracks_one.empty or rec_meta_one.empty:
            print(f"[lane_pose] Skip rec={rec_id} vid={vid} (missing tracks/meta).")
            continue

        fname = f"lane_pose_violation_rec{rec_id}_vid{vid}_dir{dd}_class{vclass}.png"
        out_png = debug_dir / fname

        plot_lane_pose_debug(
            tracks_one,
            rec_meta_one,
            vehicle_id=vid,
            recording_id=rec_id,
            out_path=out_png,
            invert_y=True,
            draw_road=True,
            draw_markings=True,
            time_colormap=False,
            dpi=160,
            title=f"Violation example | rec={rec_id} | vid={vid} | dir={dd} | class={vclass}",
            highlight_values=tuple(violation_values),     
            highlight_unknown=bool(highlight_unknown),    
        )

        print(f"[lane_pose] Saved: {out_png}")


def _attach_driving_direction(trajs, df_tracks_meta):
    """
    Attach highD drivingDirection (1/2) to each trajectory dict as tr["drivingDirection"].

    LaneChangeContext uses tr["drivingDirection"] to infer left/right.
    """
    if df_tracks_meta is None or df_tracks_meta.empty:
        return

    if "recording_id" not in df_tracks_meta.columns or "id" not in df_tracks_meta.columns:
        return
    if "drivingDirection" not in df_tracks_meta.columns:
        return

    # build lookup (recording_id, id) -> drivingDirection
    tmp = df_tracks_meta[["recording_id", "id", "drivingDirection"]].copy()
    tmp["recording_id"] = tmp["recording_id"].astype(str)

    dd_map = {}
    for r in tmp.itertuples(index=False):
        try:
            rid = str(r.recording_id)
            vid = int(r.id)
            dd = int(r.drivingDirection) if pd.notna(r.drivingDirection) else None
        except Exception:
            dd = None
        dd_map[(rid, vid)] = dd

    # attach to each trajectory
    for tr in trajs:
        rid = str(tr.get("recording_id"))
        vid = int(tr.get("vehicle_id", tr.get("id")))
        tr["drivingDirection"] = dd_map.get((rid, vid), None)

# =========================================================
# Main report
# =========================================================

def run_report(tracks_dir, out_dir, pattern="*_tracks.csv", max_files=None, show_progress=True):
    """
    Run the dataset analysis and write outputs to disk.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_all, df_tracks_meta, df_recording_meta = load_highd_bundle(
        tracks_dir, pattern=pattern, max_files=max_files, show_progress=show_progress
    )

    # Minimal schema snapshot
    schema = {
        "num_rows": int(len(df_all)),
        "num_cols": int(df_all.shape[1]),
        "num_recordings": int(df_all["recording_id"].nunique()),
        "num_trajectories": int(df_all.groupby(["recording_id", "id"]).ngroups),
        "tracks_cols": list(map(str, df_all.columns)),
        "tracksMeta_cols": list(map(str, df_tracks_meta.columns)) if not df_tracks_meta.empty else [],
        "recordingMeta_cols": list(map(str, df_recording_meta.columns)) if not df_recording_meta.empty else [],
    }
    (out_dir / "schema.json").write_text(json.dumps(schema, indent=2))

    # Kinematics
    cols = ("xVelocity", "yVelocity", "xAcceleration", "yAcceleration")
    plot_global_kinematics(df_all, cols=cols, out_dir=out_dir / "kinematics")
    plot_per_vehicle_variability(df_all, cols=cols, out_dir=out_dir / "kinematics", show_progress=show_progress)
    plot_per_recording_kinematics(
        df_all,
        cols=cols,
        out_dir=out_dir / "kinematics",
        show_progress=show_progress,
        use_minmax_whiskers=False,
    )
    plot_signal_by_direction(df_all, df_tracks_meta, cols=cols, out_dir=out_dir / "kinematics")
    plot_signal_by_class(df_all, df_tracks_meta, cols=cols, out_dir=out_dir / "kinematics")

    # Lane-change analysis
    trajs = build_traj_dicts(
        df_all,
        columns=["laneId", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration"],
        include_frame=True,
        show_progress=show_progress,
    )

    # key aliases expected by your lane-change modules
    for tr in trajs:
        tr["id"] = tr["vehicle_id"]
        lane = tr.get("laneId")
        tr["lane_id"] = pd.to_numeric(lane, errors="coerce") if lane is not None else None
        tr["vx"] = tr.get("xVelocity")
        tr["vy"] = tr.get("yVelocity")
        tr["ax"] = tr.get("xAcceleration")
        tr["ay"] = tr.get("yAcceleration")

    save_lane_change_plots(out_dir=out_dir / "lane_change", trajs=trajs, lane_key="lane_id")
    validate_lane_changes_against_meta(
        out_dir=out_dir / "lane_change",
        trajs=trajs,
        df_tracks_meta=df_tracks_meta,
        lane_key="lane_id",
        meta_col="numLaneChanges",
    )

    lc_cfg = LaneChangeContextConfig(pre_frames=25, post_frames=25, min_gap_frames=10)
    _attach_driving_direction(trajs, df_tracks_meta)
    stats = compute_lane_change_context(trajs, config=lc_cfg, lane_key="lane_id")
    save_lane_change_context_plots(out_dir / "lane_change_context_plots", stats)

    # Lane-pose analysis (training-aligned)
    df_lane = build_lane_pose_df(df_all, df_tracks_meta, df_recording_meta)

    df_lane = add_lane_position_feature(
        df_lane,
        dir_col="meta_drivingDirection",
        y_col="y_center",
        upper_key="rec_upperLaneMarkings",
        lower_key="rec_lowerLaneMarkings",
    )

    # quick sanity counts for new scheme
    vc = df_lane["lane_pos"].value_counts(dropna=False).sort_index()
    print("[lane_pose] lane_pos counts:\n", vc.to_string())

    lane_pose_dir = out_dir / "lane_pose"
    lane_pose_dir.mkdir(parents=True, exist_ok=True)

    # debug plots now show violations (0/4) and optionally unknown (-1)
    save_lane_pose_debug_plots(
        df_lane,
        df_recording_meta,
        lane_pose_dir,
        violation_values=(0, 4),
        highlight_unknown=True,
    )

    cfg = LanePoseReportConfig(
        lane_pos_col="lane_pos",
        class_col="meta_class",
        direction_col="meta_drivingDirection",
        recording_col="recording_id",
        vehicle_col="id",
        include_lane_pos=(-1, 0, 1, 2, 3, 4),  
        count_unique_vehicles=False,  # frame counts
        dpi=160,
    )
    run_lane_pose_report(df_lane, out_dir=lane_pose_dir, cfg=cfg)


if __name__ == "__main__":
    TRACKS_DIR = Path(r"C:\Users\amalj\OneDrive\Desktop\Master's Thesis\Implementation\hdv\data\highd")
    OUT_DIR = Path(__file__).resolve().parent / "data_analysis" / "highd_report"

    print("[report] Starting analysis...")
    print("[report] tracks_dir:", TRACKS_DIR)
    print("[report] out_dir:", OUT_DIR)

    run_report(
        tracks_dir=TRACKS_DIR,
        out_dir=OUT_DIR,
        pattern="*_tracks.csv",
        max_files=10,
        show_progress=True,
    )

    print(f"[report] Analysis completed. Results written to:\n  {OUT_DIR}")
