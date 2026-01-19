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
    from ..config import FRAME_FEATURE_COLS, META_COLS
except ImportError:
    # When run as a standalone script (debug)
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))
    from hdv.hdv_dbn.config import FRAME_FEATURE_COLS, META_COLS

from .highd.io import HIGHD_COL_MAP, _read_recording_bundle
from .highd.normalise import normalize_vehicle_centric
from .highd.neighbours import add_neighbor_exists_flags, add_front_thw_ttc_dhw_from_tracks, add_ego_speed_and_jerk
from .highd.lanes import add_lane_change_feature, add_lane_boundary_distance_features
from .highd.sequences import prune_columns, df_to_sequences, windowize_sequences, train_val_test_split, load_or_build_windowized
from .highd.scaling import compute_feature_scaler, scale_sequences, compute_classwise_feature_scalers, scale_sequences_classwise

def _default_highd_cache_path(root, max_recordings):
    """
    Build a cache filename that depends on max_recordings.

    - max_recordings is None -> "highd_all_with_meta.feather"
    - max_recordings is N    -> "highd_first_{N}_with_meta.feather"
    """
    if max_recordings is None:
        name = "highd_all_with_meta.feather"
    else:
        name = f"highd_first_{int(max_recordings)}_with_meta.feather"
    return root / name


def assert_required_columns_present(df, feature_cols, meta_cols):
    """Hard check that all required columns exist in the dataframe."""
    required = list(feature_cols) + list(meta_cols)
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(
            "[highD loader] Missing required columns in dataframe:\n"
            f"{missing}\n\n"
            f"Available columns ({len(df.columns)}):\n"
            f"{list(df.columns)}"
        )
    
def warn_all_zero_columns(df, feature_cols, tol=1e-12):
    bad = []
    for c in feature_cols:
        if df[c].dtype.kind in "fi" and abs(df[c]).sum() < tol: # all-zero column for numeric types
            bad.append(c)
    if bad:
        print(f"[WARN] All-zero feature columns detected: {bad}")

def enforce_dtypes(df):
    """
    Enforce consistent dtypes based on feature semantics.
    """
    out = df.copy()
    # -----------------------------
    # Identifiers / indices
    # -----------------------------
    for c in ("vehicle_id", "frame", "recording_id"):
        if c in out.columns:
            out[c] = out[c].astype(np.int64, copy=False)
    # Discrete labels
    if "lc" in out.columns:
        out["lc"] = out["lc"].astype(np.int8, copy=False)
    # -----------------------------
    # Other columns
    # -----------------------------
    for c in out.columns:
        if c.endswith("_exists"):
            # binary existence flags
            out[c] = out[c].fillna(0).astype(np.int8)

        elif pd.api.types.is_numeric_dtype(out[c]):
            # continuous numeric features
            out[c] = out[c].astype(np.float64, copy=False)

    return out

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
        cache_path = _default_highd_cache_path(root, max_recordings)
    else:
        cache_path = Path(cache_path)

    # Try to load from cache
    if cache_path.exists() and not force_rebuild:
        print(f"[highd_loader] Loading cached DataFrame from: {cache_path}")
        df_cached = pd.read_feather(cache_path)

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
                df = normalize_vehicle_centric(df, dir_col="meta_drivingDirection", flip_longitudinal=True, flip_lateral=flip_lateral, flip_positions=flip_positions)
            else:
                df["dir_sign"] = 1.0
        
        df = add_neighbor_exists_flags(df)
        df = add_lane_boundary_distance_features(df, y_col="y_center")
        df = add_lane_change_feature(df, lane_col="lane_id", dir_sign_col="dir_sign")
        df = add_front_thw_ttc_dhw_from_tracks(df)
        df = add_ego_speed_and_jerk(df)
        
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True) # concatenate all recordings

    # cast types explicitly
    df_all = enforce_dtypes(df_all)

    df_all = prune_columns(df_all, feature_cols=FRAME_FEATURE_COLS, meta_cols=META_COLS, keep_extra=[])

    assert_required_columns_present(df_all, feature_cols=FRAME_FEATURE_COLS, meta_cols=META_COLS)
    warn_all_zero_columns(df_all, feature_cols=FRAME_FEATURE_COLS)

    # Save cache for future calls (full OR subset)
    print(f"[load_highd_folder] Saving cached DataFrame to: {cache_path}")
    df_all.to_feather(cache_path)

    return df_all
