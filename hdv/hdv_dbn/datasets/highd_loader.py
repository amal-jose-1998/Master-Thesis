"""
HighD dataset loader (facade).
Orchestrates reading CSVs, merging meta, normalisation, feature derivation,
and returns a per-frame DataFrame.
"""

from pathlib import Path
import numpy as np
import pandas as pd

try:
    # When imported as a package module
    from .dataset import TrajectorySequence
    from ..config import TRAINING_CONFIG, BASELINE_FEATURE_COLS, META_COLS
except ImportError:
    # When run as a standalone script (debug)
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))
    from hdv.hdv_dbn.datasets.dataset import TrajectorySequence
    from hdv.hdv_dbn.config import TRAINING_CONFIG, BASELINE_FEATURE_COLS, META_COLS

from .highd.io import HIGHD_COL_MAP, _read_recording_bundle
from .highd.normalise import normalize_vehicle_centric
from .highd.neighbours import add_direction_aware_context_features
from .highd.lanes import add_lane_position_feature, add_lane_change_feature
from .highd.sequences import df_to_sequences, train_val_test_split, prune_columns
from .highd.scaling import (
    compute_feature_scaler_masked,
    compute_classwise_feature_scalers_masked,
    scale_sequences,
    scale_sequences_classwise,
)

# ---------------------------------------------------------------------
# Main loader: load all recordings with meta and normalization
# ---------------------------------------------------------------------
def load_highd_folder(root, cache_path=None, force_rebuild=False, max_recordings=None, apply_vehicle_centric=True, flip_lateral=True, flip_positions=False):
    """
    Load highD recordings into one unified per-frame DataFrame, including meta.

    Parameters
    root : str or Path
        Directory containing files such as `01_tracks.csv`, `02_tracks.csv`, ...
    cache_path : str or Path, optional
        Path to a cached Feather file. If None, defaults to `root / "highd_all.feather"`.
    force_rebuild : bool, default False
        If True, ignore any existing cache and rebuild from CSV files.
    max_recordings : int, optional
        If provided, only the first `max_recordings` CSV files (after sorting) are loaded. This is useful for debugging and quick experiments.
    apply_vehicle_centric : bool, default True
        If True, apply vehicle-centric normalization to kinematics.
    flip_lateral : bool, default True
        If `apply_vehicle_centric` is True, also flip lateral velocities/accelerations.
    flip_positions : bool, default False
        If `apply_vehicle_centric` is True, also flip x/y positions.

    Returns
    pd.DataFrame
        A single concatenated table with standardized column names: `vehicle_id, frame, x, y, vx, vy, ax, ay, lane_id, recording_id`.
    """
    root = Path(root)
    if cache_path is None:
        cache_path = root / "highd_all_with_meta.feather"
    else:
        cache_path = Path(cache_path)

    # Try to load from cache
    if max_recordings is None and cache_path.exists() and not force_rebuild:
        print(f"[highd_loader] Loading cached DataFrame from: {cache_path}")
        df_cached = pd.read_feather(cache_path)

        # ensure lane_pos exists even for older caches
        if "lane_pos" not in df_cached.columns:
            raise RuntimeError(
                "Cached highD feather is missing lane_pos. Delete the cache or set force_rebuild=True "
                "so lane_pos can be computed from recordingMeta lane markings."
            )
        return df_cached
    
    # Otherwise build from CSVs
    tracks_paths = sorted(root.glob("*_tracks.csv"))
    if not tracks_paths:
        raise RuntimeError(f"No *_tracks.csv files found in {root}")
    
    rec_ids = sorted([int(p.stem.split("_")[0]) for p in tracks_paths])
    if max_recordings is not None:
        rec_ids = rec_ids[:max_recordings]
        print(f"[load_highd_folder] Using only first {len(rec_ids)} recordings.")

    print(f"[load_highd_folder] Building DataFrame from {len(rec_ids)} recordings...")
    dfs = []

    for rec_id in rec_ids:
        df_tracks, df_tracksmeta, df_recmeta = _read_recording_bundle(root, rec_id)

        # Standardize tracks columns
        df_tracks = df_tracks.rename(columns=HIGHD_COL_MAP)
        keep_cols = list(HIGHD_COL_MAP.values())
        df_tracks = df_tracks[keep_cols]
        df_tracks["recording_id"] = rec_id

        # TracksMeta: standardize id column name
        if "id" in df_tracksmeta.columns and "vehicle_id" not in df_tracksmeta.columns:
            df_tracksmeta = df_tracksmeta.rename(columns={"id": "vehicle_id"})

        # Prefix meta columns to avoid name collisions
        tracksmeta_cols = [c for c in df_tracksmeta.columns if c != "vehicle_id"]  
        df_tracksmeta = df_tracksmeta[["vehicle_id"] + tracksmeta_cols].copy()
        df_tracksmeta = df_tracksmeta.rename(columns={c: f"meta_{c}" for c in tracksmeta_cols})  

        # RecordingMeta: usually one row; prefix as rec_
        if len(df_recmeta) >= 1:
            rec_row = df_recmeta.iloc[0].to_dict()
        else:
            rec_row = {}
        rec_row = {f"rec_{k}": v for k, v in rec_row.items()}

        # Merge tracks with tracksMeta (vehicle-wise)
        df = df_tracks.merge(df_tracksmeta, on="vehicle_id", how="left")

        if "width" in df.columns and "height" in df.columns:
            df["x_center"] = df["x"].astype(np.float64) + 0.5 * df["width"].astype(np.float64)
            df["y_center"] = df["y"].astype(np.float64) + 0.5 * df["height"].astype(np.float64)
        else:
            df["x_center"] = df["x"].astype(np.float64)
            df["y_center"] = df["y"].astype(np.float64)

        # Attach recording meta as constants for this recording
        for k, v in rec_row.items():
            df[k] = v
        
        if "meta_class" in df.columns:
            df["meta_class"] = df["meta_class"].astype(str).str.lower()

        # Apply vehicle-centric normalization
        if apply_vehicle_centric:
            dir_col = "meta_drivingDirection" if "meta_drivingDirection" in df.columns else None
            if dir_col is not None:
                df["drivingDirection"] = df[dir_col]
                df = normalize_vehicle_centric(df, dir_col="drivingDirection", flip_longitudinal=True, flip_lateral=flip_lateral, flip_positions=flip_positions)
                df = df.drop(columns=["drivingDirection"], errors="ignore")
            else:
                df["dir_sign"] = 1.0
        
        df = add_direction_aware_context_features(df)
        df = add_lane_position_feature(df, y_col="y_center")
        df = add_lane_change_feature(df, lane_col="lane_id", dir_sign_col="dir_sign")
        
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # Cast neighbor id columns (missing -> 0)
    neighbor_cols = [
        "preceding_id","following_id",
        "left_preceding_id","left_alongside_id","left_following_id",
        "right_preceding_id","right_alongside_id","right_following_id",
    ]
    for c in neighbor_cols:
        if c in df_all.columns:
            df_all[c] = df_all[c].fillna(0).astype(int)

    # cast types explicitly
    df_all["vehicle_id"] = df_all["vehicle_id"].astype(int)
    df_all["frame"] = df_all["frame"].astype(int)
    df_all["lane_pos"] = df_all["lane_pos"].astype(int)
    df_all["recording_id"] = df_all["recording_id"].astype(int)
    if "lc" in df_all.columns:
        df_all["lc"] = df_all["lc"].astype(int)

    float_base = ["x", "y", "x_center", "y_center", "vx", "vy", "ax", "ay"]
    float_rel_suffixes = ("_dx", "_dy", "_dvx", "_dvy")  
    float_cols = []

    for c in float_base:
        if c in df_all.columns:
            float_cols.append(c)
    
    for c in df_all.columns:  
        if any(c.endswith(suf) for suf in float_rel_suffixes):  
            float_cols.append(c) 
    
    float_cols = sorted(set(float_cols))

    for c in float_cols:
        df_all[c] = df_all[c].astype(float)
    
    for c in df_all.columns:  
        if c.endswith("_exists"): 
            df_all[c] = df_all[c].astype(float)  

    df_all = prune_columns(
        df_all,
        feature_cols=BASELINE_FEATURE_COLS,  
        meta_cols=META_COLS,                 
        keep_extra=[],                       
    )

    # Save cache for future calls only for full dataset
    if max_recordings is None:
        print(f"[load_highd_folder] Saving cached DataFrame to: {cache_path}")
        df_all.to_feather(cache_path)

    return df_all
