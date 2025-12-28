"""Standalone analysis entry point for highD *_tracks.csv files."""
import json
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm 

from lane_change_analysis import save_lane_change_plots, validate_lane_changes_against_meta
from kinematic_summary import plot_global_kinematics, plot_per_vehicle_variability, plot_per_recording_kinematics, plot_signal_by_direction, plot_signal_by_class
from lane_change_context import LaneChangeContextConfig, compute_lane_change_context, save_lane_change_context_plots

# -----------------------------
# Loading
# -----------------------------
def load_tracks_and_metadata(tracks_dir, pattern="*_tracks.csv", max_files=None, show_progress=True):
    """
    Load tracks and metadata files together in a single pass.
    
    Parameters
    tracks_dir : str | Path
        Directory containing highD data files.
    pattern : str, optional
        Glob pattern for tracks files. Default "*_tracks.csv".
    max_files : int | None, optional
        Optional limit on number of files loaded.
    show_progress : bool
        If True, show progress bars while reading files.
    
    Returns
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (df_all, df_tracks_meta, df_recording_meta) where:
        - df_all: tracks long-table DataFrame
        - df_tracks_meta: track-level metadata
        - df_recording_meta: recording-level metadata
    """
    TRACKS_META_COLS = [
        "id", "numFrames", "class", "drivingDirection",
        "minXVelocity", "maxXVelocity", "meanXVelocity", "numLaneChanges",
    ]
    RECORDING_META_COLS = ["numVehicles", "numCars", "numTrucks"]
    TRACKS_COLS = [
        "id", "frame", "laneId",
        "xVelocity", "yVelocity", "xAcceleration", "yAcceleration",
    ]
    
    tracks_dir = Path(tracks_dir)
    files = sorted(tracks_dir.glob(pattern))
    if max_files is not None:
        files = files[: int(max_files)]
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {tracks_dir}")
    
    iterator = tqdm(files, desc="Loading tracks and metadata", unit="file") if show_progress else files
    
    dfs_tracks = []
    dfs_tracks_meta = []
    dfs_recording_meta = []
    
    for p in iterator:
        rec_id = p.stem.split("_", 1)[0]
        
        # Load tracks
        df = pd.read_csv(p, usecols=TRACKS_COLS)
        df["id"] = pd.to_numeric(df["id"], downcast="integer")
        df["frame"] = pd.to_numeric(df["frame"], downcast="integer")
        df["laneId"] = pd.to_numeric(df["laneId"], downcast="integer")
        for c in ["xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]:
            df[c] = pd.to_numeric(df[c], downcast="float")
        df["recording_id"] = rec_id
        dfs_tracks.append(df)
        
        # Load tracksMeta
        meta_file = p.parent / f"{rec_id}_tracksMeta.csv"
        if meta_file.exists():
            df_tm = pd.read_csv(meta_file, usecols=TRACKS_META_COLS)
            df_tm["recording_id"] = rec_id
            dfs_tracks_meta.append(df_tm)
        
        # Load recordingMeta
        rec_meta_file = p.parent / f"{rec_id}_recordingMeta.csv"
        if rec_meta_file.exists():
            df_rm = pd.read_csv(rec_meta_file, usecols=RECORDING_META_COLS)
            df_rm["recording_id"] = rec_id
            dfs_recording_meta.append(df_rm)
    
    df_all = pd.concat(dfs_tracks, ignore_index=True, copy=False)
    df_tracks_meta = pd.concat(dfs_tracks_meta, ignore_index=True) if dfs_tracks_meta else pd.DataFrame()
    df_recording_meta = pd.concat(dfs_recording_meta, ignore_index=True) if dfs_recording_meta else pd.DataFrame()
    
    return df_all, df_tracks_meta, df_recording_meta

def compare_meta_vs_tracks_totals(df_tracks, df_tracks_meta, df_recording_meta):
    """
    Compute totals from recording-level metadata and from tracks, and compare.

    Returns a dict with keys: `meta_totals`, `tracks_totals`, `comparison`.
    Fields may be `None` when the corresponding source is missing.
    """
    # recording-meta totals
    if df_recording_meta is None or df_recording_meta.empty:
        meta_totals = {"numVehicles": None, "numCars": None, "numTrucks": None}
    else:
        def _safe_sum(df, col):
            return int(df[col].sum()) if col in df.columns else None
        meta_totals = {
            "numVehicles": _safe_sum(df_recording_meta, "numVehicles"),
            "numCars": _safe_sum(df_recording_meta, "numCars"),
            "numTrucks": _safe_sum(df_recording_meta, "numTrucks"),
        }

    # tracks-derived totals
    if df_tracks is None or df_tracks.empty:
        tracks_totals = {"numVehicles": 0, "numCars": None, "numTrucks": None}
    else:
        # number of unique vehicle trajectories (unique id per recording)
        tracks_totals = {"numVehicles": int(df_tracks.groupby(["recording_id"]) ["id"].nunique().sum())}

        # vehicle class counts from tracksMeta (preferred)
        if df_tracks_meta is None or df_tracks_meta.empty:
            tracks_totals.update({"numCars": None, "numTrucks": None})
        else:
            vc = df_tracks_meta["class"].astype(str).str.lower().value_counts()
            num_cars = None
            num_trucks = None
            car_keys = [k for k in vc.index if "car" in k]
            truck_keys = [k for k in vc.index if "truck" in k]
            if car_keys:
                num_cars = int(vc[car_keys].sum())
            if truck_keys:
                num_trucks = int(vc[truck_keys].sum())
            # fallback for numeric-coded classes (common encoding: 1=car, 2=truck)
            if num_cars is None and "1" in vc.index:
                num_cars = int(vc.get("1", 0))
            if num_trucks is None and "2" in vc.index:
                num_trucks = int(vc.get("2", 0))
            tracks_totals.update({"numCars": num_cars, "numTrucks": num_trucks})

    # comparison (None when either side is missing)
    comparison = {
        "numVehicles_match": (meta_totals["numVehicles"] == tracks_totals.get("numVehicles")) if meta_totals["numVehicles"] is not None else None,
        "numCars_match": (meta_totals["numCars"] == tracks_totals.get("numCars")) if (meta_totals["numCars"] is not None and tracks_totals.get("numCars") is not None) else None,
        "numTrucks_match": (meta_totals["numTrucks"] == tracks_totals.get("numTrucks")) if (meta_totals["numTrucks"] is not None and tracks_totals.get("numTrucks") is not None) else None,
    }

    return {"meta_totals": meta_totals, "tracks_totals": tracks_totals, "comparison": comparison}


# -----------------------------
# Trajectory view (per vehicle)
# -----------------------------
def build_traj_dicts(df_all, columns, id_col="id", frame_col="frame", include_frame=True, show_progress=True):
    """
    Convert the long-table DataFrame into a list of per-vehicle trajectories.
    Each trajectory is represented as a dict with:
      - metadata:
          recording_id (str), vehicle_id (int), T (int)
      - time-series arrays:
          for each selected column: numpy array of length T

    Parameters
    df_all : pd.DataFrame
        Full long-table dataframe containing all columns from all CSV files.
    columns : list[str]
        Columns to extract into each trajectory dict (time series). Only these are converted to numpy arrays, to save memory.
    id_col : str
        Vehicle ID column name in df_all (default: 'id').
    frame_col : str
        Frame/time index column in df_all (default: 'frame').
    include_frame : bool
        Whether to always include the frame column in each trajectory dict.
    show_progress:
        If True, show a progress bar while grouping/constructing trajectories.

    Returns
    trajs:
        List of trajectory dicts.

    Raises
    KeyError:
        If required columns are missing:
          - recording_id
          - id_col
          - frame_col
    """
    required = {"recording_id", id_col, frame_col}
    missing = required - set(df_all.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    # Make a unique, ordered list of columns we will actually pull out
    cols = []
    if include_frame and frame_col not in columns:
        cols.append(frame_col)
    cols.extend([c for c in columns if c != "recording_id"])
    cols = [c for c in cols if c in df_all.columns]  # silently drop missing

    grouped = df_all.groupby(["recording_id", id_col], sort=False)
    total = int(getattr(grouped, "ngroups", 0))

    iterator = grouped
    if show_progress:
        iterator = tqdm(grouped, total=total, desc="Building trajectories", unit="traj")

    trajs = []
    # group by unique trajectory identity
    for (rec_id, vid), g in iterator:
        g = g.sort_values(frame_col) # shape: (T, C+1)
                                     # T = number of frames for that vehicle
                                     # C+1 = all columns including recording_id
        tr = {
            "recording_id": rec_id,   
            "vehicle_id": int(vid),   
            "T": int(len(g))         
        }
        for col in cols:
            tr[col] = g[col].to_numpy()
        trajs.append(tr)
    return trajs

# -----------------------------
# Main report
# -----------------------------
def run_report(tracks_dir, out_dir, pattern="*_tracks.csv", max_files=None, show_progress=True):
    """
    Run a standalone dataset analysis and write a compact report to disk.

    Parameters
    tracks_dir : str | Path
        Directory containing highD `*_tracks.csv` files.
    out_dir : str | Path
        Output directory where summary files and plots will be written.
    pattern : str, optional
        Glob pattern used to select tracks files. Default "*_tracks.csv".
    max_files : int | None, optional
        Optional limit on number of files loaded (debug convenience).
    show_progress:
        If True, show progress bars for long steps (CSV loading, trajectory building).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load full raw table
    df_all, df_tracks_meta, df_recording_meta = load_tracks_and_metadata(Path(tracks_dir), pattern=pattern, max_files=max_files, show_progress=show_progress)
    trajs = build_traj_dicts(df_all, columns=["laneId", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration"], include_frame=True, show_progress=show_progress)
    # Compare recording-level metadata totals with counts derived from tracks
    comp_totals = compare_meta_vs_tracks_totals(df_all, df_tracks_meta, df_recording_meta)
    for tr in trajs:
        tr["id"] = tr["vehicle_id"]
        tr["lane_id"] = tr.get("laneId")
        tr["vx"] = tr.get("xVelocity")
        tr["vy"] = tr.get("yVelocity")
        tr["ax"] = tr.get("xAcceleration")
        tr["ay"] = tr.get("yAcceleration")

    if df_tracks_meta is not None and not df_tracks_meta.empty:
        dd_map = (
            df_tracks_meta[["recording_id", "id", "drivingDirection"]]
            .dropna(subset=["drivingDirection"])
            .copy()
        )
        dd_map["recording_id"] = dd_map["recording_id"].astype(str)
        dd_map["id"] = pd.to_numeric(dd_map["id"], errors="coerce")
        dd_map["drivingDirection"] = pd.to_numeric(dd_map["drivingDirection"], errors="coerce")
        dd_map = dd_map.dropna(subset=["id", "drivingDirection"])
        dd_dict = {(r, int(i)): int(d) for r, i, d in dd_map[["recording_id", "id", "drivingDirection"]].itertuples(index=False, name=None)}

        for tr in trajs:
            key = (str(tr.get("recording_id")), int(tr.get("vehicle_id")))
            tr["drivingDirection"] = dd_dict.get(key, None)

    # Schema overview 
    schema = {
        "num_rows": int(len(df_all)),
        "num_cols": int(df_all.shape[1]),
        "tracks": {"rows": int(len(df_all)), "columns": list(map(str, df_all.columns))},
        "tracksMeta": {"rows": int(len(df_tracks_meta)), "columns": list(map(str, df_tracks_meta.columns)) if not df_tracks_meta.empty else []},
        "recordingMeta": {"rows": int(len(df_recording_meta)), "columns": list(map(str, df_recording_meta.columns)) if not df_recording_meta.empty else []},
        "meta_vs_tracks": comp_totals,
        "num_recordings": int(df_all["recording_id"].nunique()),
        "num_trajectories": int(df_all.groupby(["recording_id", "id"]).ngroups),
    }
    (out_dir / "schema.json").write_text(json.dumps(schema, indent=2))

    # Kinematic summaries and plots
    #cols=("xVelocity", "yVelocity", "xAcceleration", "yAcceleration")
    #plot_global_kinematics(df_all, cols=cols, out_dir=out_dir / "kinematics") # global kinematic distributions that plots for all vehicles together
    #plot_per_vehicle_variability(df_all, cols=cols, out_dir=out_dir / "kinematics", show_progress=show_progress) # per-vehicle kinematic variability plots
    #plot_per_recording_kinematics(df_all, cols=cols, out_dir=out_dir / "kinematics", show_progress=show_progress, use_minmax_whiskers=False) # per-recording kinematic summaries
    #plot_signal_by_direction(df_all, df_tracks_meta, cols=cols, out_dir=out_dir / "kinematics") # kinematic summaries by driving direction
    #plot_signal_by_class(df_all, df_tracks_meta, cols=cols, out_dir=out_dir / "kinematics") # kinematic summaries by vehicle class
    
    # Lane change stats. 
    #save_lane_change_plots(out_dir=out_dir / "lane_change", trajs=trajs, lane_key="lane_id") # lane change count distributions and lane change frequency vs speed
    #validate_lane_changes_against_meta(out_dir=out_dir / "lane_change", trajs=trajs, df_tracks_meta=df_tracks_meta, lane_key="lane_id", meta_col="numLaneChanges", make_plots=True)
    
    # Lane change context stats
    #lc_ctx_cfg = LaneChangeContextConfig(pre_frames=25, post_frames=25, min_gap_frames=10) # 1 second before and after lane change at 25 fps
    #stats = compute_lane_change_context(trajs, config=lc_ctx_cfg, lane_key="lane_id") # computes lane change context statistics
    #save_lane_change_context_plots(out_dir / "lane_change_context_plots", stats)



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
        max_files=None,   # None if all files are to be considered 
        show_progress=True
    )

    print(f"[report] Analysis completed. Results written to:\n  {OUT_DIR}")