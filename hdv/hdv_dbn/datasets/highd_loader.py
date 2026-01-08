"""
HighD dataset utilities (loader + sequence builder).

This module:
- loads highD `*_tracks.csv` + `*_tracksMeta.csv` + `*_recordingMeta.csv` files
- merges meta into the per-frame table
- applies vehicle-centric direction normalization (sign flips)
- derives neighbor-relative context features (dx, dy, dvx, dvy) with existence masks
- derives a stable lane-position category (lane_pos) from lane markings (upper/lower) + direction
- converts the table into per-vehicle `TrajectorySequence` objects
- provides scaling helpers (including "masked" scaling that leaves discrete features untouched)
- creates reproducible train/val/test splits (seeded via TRAINING_CONFIG)
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

# Map highD column names to the generic internal names.
HIGHD_COL_MAP = {
    "id": "vehicle_id",
    "frame": "frame",
    "x": "x", 
    "y": "y",
    "width": "width",     
    "height": "height",
    "xVelocity": "vx",
    "yVelocity": "vy",
    "xAcceleration": "ax",
    "yAcceleration": "ay",
    "laneId": "lane_id",

    "frontSightDistance": "front_sight_dist",
    "backSightDistance": "back_sight_dist",
    "dhw": "dhw",
    "thw": "thw",
    "ttc": "ttc",

    "precedingXVelocity": "preceding_vx",
    "precedingId": "preceding_id",
    "followingId": "following_id",

    "leftPrecedingId": "left_preceding_id",
    "leftAlongsideId": "left_alongside_id",
    "leftFollowingId": "left_following_id",

    "rightPrecedingId": "right_preceding_id",
    "rightAlongsideId": "right_alongside_id",
    "rightFollowingId": "right_following_id",
}

_REL_SUFFIXES = ("_dx", "_dy", "_dvx", "_dvy")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _rel_to_exists_pairs(obs_names):
    """
    Build a mapping between neighbor-relative feature columns and their corresponding
    binary existence-mask columns.

    Parameters
    obs_names : Sequence[str]
        Feature names in the observation vector order.

    Returns
    pairs : List[Tuple[int, int]]
        List of (rel_feature_index, exists_feature_index) pairs for all
        relative features that have a matching "<prefix>_exists" feature.
    """
    name_to_idx = {n: i for i, n in enumerate(obs_names)}
    pairs = []
    for rel_idx, name in enumerate(obs_names):
        for suf in _REL_SUFFIXES:
            if name.endswith(suf):
                prefix = name[: -len(suf)]
                ex_name = f"{prefix}_exists"
                if ex_name in name_to_idx:
                    pairs.append((rel_idx, name_to_idx[ex_name]))
                break
    return pairs

def _read_recording_bundle(root, rec_id):
    """
    Read the three CSVs for one recording ID:
        {rec_id}_tracks.csv
        {rec_id}_tracksMeta.csv
        {rec_id}_recordingMeta.csv
    
    Parameters
    root : Path
        Directory containing the highD CSV files.
    rec_id : int
        Recording number (e.g. 1 for `01_tracks.csv`).
    
    Returns
    df_tracks : pd.DataFrame
        DataFrame from `{rec_id}_tracks.csv`.
    df_tracksmeta : pd.DataFrame
        DataFrame from `{rec_id}_tracksMeta.csv`.   
    df_recmeta : pd.DataFrame
        DataFrame from `{rec_id}_recordingMeta.csv`.
    """
    tracks_path = root / f"{rec_id:02d}_tracks.csv"
    tracksmeta_path = root / f"{rec_id:02d}_tracksMeta.csv"
    recmeta_path = root / f"{rec_id:02d}_recordingMeta.csv"

    if not tracks_path.exists():
        raise FileNotFoundError(tracks_path)
    if not tracksmeta_path.exists():
        raise FileNotFoundError(tracksmeta_path)
    if not recmeta_path.exists():
        raise FileNotFoundError(recmeta_path)

    df_tracks = pd.read_csv(tracks_path)
    df_tracksmeta = pd.read_csv(tracksmeta_path)
    df_recmeta = pd.read_csv(recmeta_path)

    return df_tracks, df_tracksmeta, df_recmeta

def _infer_dir_sign(driving_dir):
    """
    Convert driving direction labels into a sign (+1 / -1) such that multiplying
    by sign makes 'forward' consistent across directions.

    Parameters
    driving_dir : pd.Series
        Series of driving direction labels (integers).

    Returns
    np.ndarray  
        Array of shape (N,) with +1.0 or -1.0 values.
    """
    if driving_dir is None:
        return np.ones(0, dtype=np.float64)
    
    vals = pd.unique(driving_dir.dropna())
    vals = np.array(sorted([int(v) for v in vals])) if len(vals) else np.array([])

    if len(vals) == 2 and np.array_equal(vals, np.array([1, 2])):
        sign = np.where(driving_dir.to_numpy() == 2, 1.0, -1.0)
        return sign
    if len(vals) == 2 and np.array_equal(vals, np.array([0, 1])):
        sign = np.where(driving_dir.to_numpy() == 1, 1.0, -1.0)
        return sign

    # Fallback: no flip
    return np.ones(len(driving_dir), dtype=np.float64)

def normalize_vehicle_centric(df, dir_col="drivingDirection", flip_longitudinal=True, flip_lateral=True, flip_positions=False):
    """
    Make kinematics vehicle-centric so that 'forward' and  'right' are consistent across both roadway directions.
    This modifies columns in-place (vx, ax, vy, ay, and optionally x,y) and also stores the applied sign in a new column `dir_sign`.

    Parameters
    df : pd.DataFrame
        Must contain `dir_col` plus vx/ax and optionally vy/ay and x/y.
    dir_col : str
        Column name containing driving direction labels.
    flip_longitudinal : bool
        Flip vx and ax by dir_sign.
    flip_lateral : bool
        Flip vy and ay by dir_sign.
    flip_positions : bool
        If True, also flip x and y. (Not typically needed.)
    
    Returns
    pd.DataFrame
        The modified DataFrame (same as input `df`).
    """
    if dir_col not in df.columns:
        # no direction info => nothing to do
        df["dir_sign"] = 1.0
        return df

    sign = _infer_dir_sign(df[dir_col])
    df["dir_sign"] = sign

    if flip_longitudinal:
        for c in ["vx", "ax"]:
            if c in df.columns:
                df[c] = df[c].astype(np.float64) * sign

    if flip_lateral:
        for c in ["vy", "ay"]:
            if c in df.columns:
                df[c] = df[c].astype(np.float64) * sign

    if flip_positions:
        for c in ["x", "y"]:
            if c in df.columns:
                df[c] = df[c].astype(np.float64) * sign

    return df


def _merge_neighbor_state(df, neighbor_id_col, prefix):
    """
    Attach neighbor state (x,y,vx,vy,lane_id) for the vehicle referenced by neighbor_id_col.
    Adds:
      {prefix}_x_center, {prefix}_y_center, {prefix}_vx, {prefix}_vy, {prefix}_lane_id
    
    Parameters
    df : pd.DataFrame
        Input table containing at least:
          - `recording_id`, `frame`, `vehicle_id`
          - neighbor ID column specified by `neighbor_id_col`
          - x_center, y_center, vx, vy, lane_id columns
    neighbor_id_col : str
        Column name containing neighbor vehicle IDs.
    prefix : str
        Prefix for the new neighbor state columns.
    
    Returns
    pd.DataFrame
        DataFrame with new neighbor state columns added.
    """
    if neighbor_id_col not in df.columns:
        return df

    # lookup: (recording_id, frame, neighbor_id) -> neighbor state
    lookup = df[["recording_id","frame","vehicle_id","x_center","y_center","vx","vy","lane_id"]].copy()
    lookup["vehicle_id"] = lookup["vehicle_id"].astype("Int64")
    lookup = lookup.rename(
        columns={
            "vehicle_id": neighbor_id_col,
            "x_center": f"{prefix}_x_center",
            "y_center": f"{prefix}_y_center",
            "vx": f"{prefix}_vx",
            "vy": f"{prefix}_vy",
            "lane_id": f"{prefix}_lane_id",
        }
    )

    # merge using the neighbor ID for that row
    out = df.merge(
        lookup,
        on=["recording_id", "frame", neighbor_id_col],
        how="left",
    )
    return out

# ---------------------------------------------------------------------
# Lane markings -> lane_pos
# ---------------------------------------------------------------------
def _parse_lane_markings(value):
    """
    Parse recordingMeta lane markings fields into a sorted float array.

    Parameters
    value : str | list | tuple | np.ndarray | None
        Raw lane markings field from recordingMeta.
    
    Returns
    np.ndarray
        Sorted array of lane marking positions (float64).
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.asarray([], dtype=np.float64)

    if isinstance(value, (list, tuple, np.ndarray)):
        return np.sort(np.asarray(value, dtype=np.float64))

    s = str(value).strip()
    if not s:
        return np.asarray([], dtype=np.float64)

    s = s.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    parts = [p.strip() for p in (s.split(";") if ";" in s else s.split(",")) if p.strip()]

    out = []
    for p in parts:
        try:
            out.append(float(p))
        except ValueError:
            continue
    return np.sort(np.asarray(out, dtype=np.float64))


def add_lane_position_feature(df, dir_col="meta_drivingDirection", y_col="y", upper_key="rec_upperLaneMarkings", lower_key="rec_lowerLaneMarkings"):
    """
    Derive stable lane-position categories from lane markings + direction.

    Output:
      df["lane_pos"] in {-1,0,1,2,3,4}
        -1 = unknown / cannot determine (no markings, invalid direction, NaN y)
         0 = left violation  (vehicle is beyond left boundary, vehicle-centric)
         1 = leftmost lane   (vehicle-centric)
         2 = middle lane(s)  (vehicle-centric)
         3 = rightmost lane  (vehicle-centric)
         4 = right violation (vehicle is beyond right boundary, vehicle-centric)

    Direction choice:
      drivingDirection == 1 -> use upperLaneMarkings
      drivingDirection == 2 -> use lowerLaneMarkings

    Vehicle-centric left/right convention:
      - direction == 2 (moving right): smaller world-y is vehicle-left
      - direction == 1 (moving left): smaller world-y is vehicle-right (swap)
    
    Parameters
    df : pd.DataFrame
        Input table containing at least:
          - `y_col` (lateral position)
          - `dir_col` (driving direction)
          - `recording_id` (if available)
          - `upper_key`, `lower_key` columns with lane markings info
    dir_col : str
        Column name containing driving direction labels.
    y_col : str
        Column name containing lateral position.
    upper_key : str
        Column name containing upper lane markings info.
    lower_key : str
        Column name containing lower lane markings info.
    
    Returns
    pd.DataFrame
        DataFrame with new `lane_pos` column added. 
    """
    # If we cannot compute at all, keep a safe default (unknown)
    if y_col not in df.columns or dir_col not in df.columns:
        df["lane_pos"] = -1
        return df

    # Pre-parse markings per recording 
    if "recording_id" in df.columns:
        rec_ids = df["recording_id"].dropna().unique().tolist()
    else:
        rec_ids = [None]

    rec_to_upper = {}
    rec_to_lower = {}

    for rid in rec_ids:
        sub = df[df["recording_id"] == rid] if rid is not None else df

        up_val = sub[upper_key].iloc[0] if (upper_key in sub.columns and len(sub)) else None
        lo_val = sub[lower_key].iloc[0] if (lower_key in sub.columns and len(sub)) else None

        rec_to_upper[rid] = _parse_lane_markings(up_val)
        rec_to_lower[rid] = _parse_lane_markings(lo_val)

    y = df[y_col].to_numpy(dtype=np.float64, copy=False)
    d = df[dir_col].to_numpy(copy=False)
    rid_arr = (
        df["recording_id"].to_numpy(copy=False)
        if "recording_id" in df.columns
        else np.full(len(df), None, dtype=object)
    )

    lane_pos = np.full(len(df), -1, dtype=np.int64)  # unknown by default

    for i in range(len(df)):
        rid = rid_arr[i]
        yi = y[i]

        if np.isnan(yi):
            lane_pos[i] = -1
            continue

        di = int(d[i]) if not pd.isna(d[i]) else -1

        if di == 1:
            marks = rec_to_upper.get(rid, np.asarray([], dtype=np.float64))
        elif di == 2:
            marks = rec_to_lower.get(rid, np.asarray([], dtype=np.float64))
        else:
            marks = np.asarray([], dtype=np.float64)

        # Need at least 2 boundaries to define 1 lane interval
        if marks.size < 2:
            lane_pos[i] = -1
            continue

        left_world = marks[0]
        right_world = marks[-1]

        # Outside boundaries in *world-y*
        outside_left_world = yi < left_world
        outside_right_world = yi >= right_world

        if outside_left_world or outside_right_world:
            # Map boundary violation to *vehicle-centric* left/right depending on direction
            if di == 2:
                # smaller world-y => vehicle-left
                lane_pos[i] = 0 if outside_left_world else 4
            else:
                # di == 1: smaller world-y => vehicle-right (swap)
                lane_pos[i] = 4 if outside_left_world else 0
            continue

        # Inside boundaries -> find lane interval index j in [0, num_lanes-1]
        num_lanes = int(marks.size - 1)
        j = int(np.searchsorted(marks, yi, side="right") - 1)

        if j < 0 or j >= num_lanes:
            # Should be rare because we already handled outside boundaries,
            # but keep it robust.
            lane_pos[i] = -1
            continue

        # Convert to vehicle-centric lane index
        j_vehicle = (num_lanes - 1) - j if di == 1 else j

        if j_vehicle == 0:
            lane_pos[i] = 1  # leftmost lane
        elif j_vehicle == (num_lanes - 1):
            lane_pos[i] = 3  # rightmost lane
        else:
            lane_pos[i] = 2  # middle lanes

    df["lane_pos"] = lane_pos
    return df

def add_lane_change_feature(df, lane_col="lane_id"):
    """
    Add single ternary lane-change feature:
        lc ∈ {-1, 0, +1}
          -1 : left lane change
           0 : no lane change
          +1 : right lane change
    """
    if lane_col not in df.columns:
        df["lc"] = 0
        return df

    df = df.sort_values(["recording_id", "vehicle_id", "frame"], kind="mergesort")

    lane = df[lane_col].to_numpy(dtype=np.float64)
    dlane = np.diff(lane, prepend=lane[0])

    lc = np.sign(dlane).astype(np.int64)
    lc[dlane == 0] = 0  # explicit

    df["lc"] = lc
    return df


# ---------------------------------------------------------------------
# Context features
# ---------------------------------------------------------------------
def add_direction_aware_context_features(df):
    """
    Build direction-consistent relative context features from neighbor IDs.
    Adds, for each neighbor slot:
      - dx = neighbor_x_center - ego_x_center
      - dy = neighbor_y_center - ego_y_center
      - dvx = neighbor_vx - ego_vx
      - dvy = neighbor_vy - ego_vy
      - exists mask (0/1)
    
    Parameters
    df : pd.DataFrame
        Input table containing at least:
          - `recording_id`, `frame`, `vehicle_id`
          - neighbor ID columns (e.g. `preceding_id`, `left_alongside_id`, etc.)
          - x, y, vx, vy columns

    Returns
    pd.DataFrame
        DataFrame with new relative context features added.
    """
    neighbor_specs = [
        ("preceding_id",       "front"),
        ("following_id",       "rear"),
        ("left_preceding_id",  "left_front"),
        ("left_alongside_id",  "left_side"),
        ("left_following_id",  "left_rear"),
        ("right_preceding_id", "right_front"),
        ("right_alongside_id", "right_side"),
        ("right_following_id", "right_rear"),
    ]

    out = df

    # Ensure ego centers exist 
    if "x_center" not in out.columns and "x" in out.columns and "width" in out.columns:
        out["x_center"] = out["x"].astype(np.float64) + 0.5 * out["width"].astype(np.float64)
    if "y_center" not in out.columns and "y" in out.columns and "height" in out.columns:
        out["y_center"] = out["y"].astype(np.float64) + 0.5 * out["height"].astype(np.float64)

    for id_col, prefix in neighbor_specs:
        if id_col not in out.columns:
            continue

        # existence mask before merge
        nid = out[id_col].fillna(0).astype("Int64")
        out[f"{prefix}_exists"] = (nid > 0).astype(np.float64)
        out[id_col] = nid.mask(nid <= 0, pd.NA) # set missing neighbor IDs to NaN for merge

        out = _merge_neighbor_state(out, id_col, prefix)

        nx = f"{prefix}_x_center"
        ny = f"{prefix}_y_center"

        if nx in out.columns:
            out[f"{prefix}_exists"] = out[f"{prefix}_exists"] * (~out[nx].isna()).astype(np.float64)

        # relative features (NaN if neighbor missing)
        out[f"{prefix}_dx"]  = out[nx] - out["x_center"]
        out[f"{prefix}_dy"]  = out[ny] - out["y_center"]
        out[f"{prefix}_dvx"] = out[f"{prefix}_vx"] - out["vx"]
        out[f"{prefix}_dvy"] = out[f"{prefix}_vy"] - out["vy"]

        # direction-aware adjustments
        if "dir_sign" in out.columns:
            out[f"{prefix}_dx"] *= out["dir_sign"]
            out[f"{prefix}_dy"] *= out["dir_sign"]

    return out

# ---------------------------------------------------------------------
# Scaling utilities 
# ---------------------------------------------------------------------
def compute_feature_scaler_masked(sequences, scale_idx):
    """
    Compute per-feature mean and standard deviation from a set of sequences,
    only for the features specified by scale_idx. For neighbor-relative features (*_dx/_dy/_dvx/_dvy),
    compute stats ONLY on frames where the corresponding *_exists == 1.

    Parameters
    sequences : sequence of TrajectorySequence
        Sequences whose `.obs` arrays will be stacked along time.
    scale_idx : list or np.ndarray
        Indices of features to compute mean/std for.
    
    Returns
    mean : np.ndarray
        Feature-wise mean, shape (F,).
    std : np.ndarray
        Feature-wise standard deviation, shape (F,). Very small std values are
        clamped to 1.0 to avoid division by near-zero during scaling.
    """
    if len(sequences) == 0:
        raise ValueError("No sequences provided.")
    
    obs_names = sequences[0].obs_names
    F = int(sequences[0].obs.shape[1])
    scale_idx = np.asarray(list(scale_idx), dtype=int)

    mean = np.zeros((F,), dtype=np.float64)
    std = np.ones((F,), dtype=np.float64)

    X = np.vstack([seq.obs for seq in sequences])  # (sumT, F)

    rel_pairs = dict(_rel_to_exists_pairs(obs_names)) # build rel_idx -> exists_idx mapping

    for j in scale_idx:
        col = X[:, j]
        mask = np.isfinite(col)

        # if this is a relative feature, only use rows where exists==1
        if j in rel_pairs:
            exj = rel_pairs[j]
            mask = mask & (X[:, exj] > 0.5)

        vals = col[mask]
        if vals.size == 0:
            mean[j] = 0.0
            std[j] = 1.0
            continue

        m = float(vals.mean())
        s = float(vals.std())
        mean[j] = m
        std[j] = 1.0 if s < 1e-6 else s

    return mean, std

def compute_classwise_feature_scalers_masked(sequences, scale_idx, class_key="meta_class"):
    """
    Compute one *masked* scaler per class.

    Parameters
    sequences : list[TrajectorySequence]
        Input sequences (usually training split only).
    scale_idx : list or np.ndarray
        Indices of features to compute mean/std for.
    class_key : str
        Key inside seq.meta indicating vehicle class.
    
    Returns
    dict
        Mapping: class_name -> (mean, std)
    """
    buckets = {}
    for seq in sequences:
        if not seq.meta or class_key not in seq.meta:
            continue
        cls = str(seq.meta[class_key])
        buckets.setdefault(cls, []).append(seq)

    scalers = {}
    for cls, seqs in buckets.items():
        scalers[cls] = compute_feature_scaler_masked(seqs, scale_idx=scale_idx)
    return scalers

def scale_sequences(sequences, mean, std):
    """
    Apply z-score scaling and then neutralize neighbor-relative features when the neighbor is absent.
    Scaling is applied as:
        obs_scaled = (obs - mean) / std

    Parameters
    sequences : sequence of TrajectorySequence
        Input sequences to be scaled.
    mean : np.ndarray
        Feature-wise mean, shape (F,).
    std : np.ndarray
        Feature-wise std, shape (F,).

    Returns
    list[TrajectorySequence]
        New list of sequences with scaled observations and neutralized absent-neighbor
        relative features. Metadata (vehicle_id, frames, obs_names, recording_id) is preserved.
    """
    out = []

    if len(sequences) == 0:
        return out
    
    # rel feature ↔ exists feature pairs
    rel_pairs = _rel_to_exists_pairs(sequences[0].obs_names)

    for seq in sequences:
        obs_scaled = (seq.obs - mean) / std
        
        # if neighbor doesn't exist, set its relative features to 0 (neutral in scaled space)
        for rel_idx, ex_idx in rel_pairs:
            absent = seq.obs[:, ex_idx] <= 0.5
            obs_scaled[absent, rel_idx] = 0.0
        
        out.append(
            TrajectorySequence(
                vehicle_id=seq.vehicle_id,
                frames=seq.frames,
                obs=obs_scaled,
                obs_names=seq.obs_names,
                recording_id=seq.recording_id,
                meta=seq.meta
            )
        )
    return out

def scale_sequences_classwise(sequences, scalers, class_key="meta_class"):
    """
    Apply class-specific feature scaling and neutralize absent-neighbor relative features.

    Parameters
    sequences : list[TrajectorySequence]
        Sequences to scale.
    scalers : dict
        Mapping: class_name -> (mean, std)
    class_key : str
        Key inside seq.meta indicating vehicle class.

    Returns
    list[TrajectorySequence]
        Scaled sequences.
    """
    out = []

    if len(sequences) == 0:
        return out
    
    rel_pairs = _rel_to_exists_pairs(sequences[0].obs_names)

    for seq in sequences:
        if seq.meta is None or class_key not in seq.meta:
            raise ValueError("Sequence missing class information for class-wise scaling.")

        cls = seq.meta[class_key]
        if cls not in scalers:
            raise ValueError(f"No scaler found for class '{cls}'.")

        mean, std = scalers[cls]
        obs_scaled = (seq.obs - mean) / std

        for rel_idx, ex_idx in rel_pairs:
            absent = seq.obs[:, ex_idx] <= 0.5
            obs_scaled[absent, rel_idx] = 0.0

        out.append(
            TrajectorySequence(
                vehicle_id=seq.vehicle_id,
                frames=seq.frames,
                obs=obs_scaled,
                obs_names=seq.obs_names,
                recording_id=seq.recording_id,
                meta=seq.meta,
            )
        )

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
        df = add_lane_change_feature(df)

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

# ---------------------------------------------------------------------
# Sequences + split + pruning
# ---------------------------------------------------------------------
def df_to_sequences(df, feature_cols, id_col="vehicle_id", frame_col="frame",  meta_cols=None):
    """
    Convert a per-frame table into per-vehicle sequences. Trajectory is defined by (recording_id, vehicle_id).

    Parameters
    df : pd.DataFrame
        Input table containing at least:
        - `recording_id`, `id_col`, `frame_col`
        - all columns listed in `feature_cols`
    feature_cols : list[str]
        Feature columns to extract into the observation matrix `obs`.
    id_col : str, default "vehicle_id"
        Column identifying vehicles within a recording.
    frame_col : str, default "frame"
        Column representing the time index.
    meta_cols : list[str], optional
        If provided, these columns are extracted from `df` and stored in the `meta` dict of each TrajectorySequence.

    Returns
    list[TrajectorySequence]
        One sequence per (recording_id, vehicle_id). 
    """
    sequences = []

    group_cols = ["recording_id", id_col]

    df = df.sort_values(group_cols + [frame_col], kind="mergesort")

    if meta_cols is None:
        meta_cols = []

    for (rec_id, veh_id), g in df.groupby(group_cols, sort=False): # Groups all rows belonging to the same vehicle in a particular recording
        #g = g.sort_values(frame_col) # DataFrame with only that vehicle’s data. Sorts rows by frame number (time ordering).
        frames = g[frame_col].to_numpy(dtype=np.int64, copy=False) # Extracts time indices
        obs = g[list(feature_cols)].to_numpy(dtype=np.float64, copy=True) # Extracts the observation features into a NumPy array of shape: T × F
        meta = {}
        if meta_cols:
            first = g.iloc[0]
            for c in meta_cols:
                if c in g.columns:
                    meta[c] = first[c]
        # Create one TrajectorySequence object representing this vehicle.
        seq = TrajectorySequence(vehicle_id=veh_id, frames=frames, obs=obs, obs_names=list(feature_cols), recording_id=rec_id,  meta=meta if meta else None)
        sequences.append(seq) # Adds this trajectory to the list.

    return sequences


def train_val_test_split(sequences, train_frac=0.7, val_frac=0.15):
    """
    Randomly partition a list of sequences into train/validation/test sets.

    Parameters
    sequences : list
        Collection of TrajectorySequence objects.
    train_frac : float, default=0.7
        Percentage of sequences assigned to the training set.
    val_frac : float, default=0.15
        Percentage assigned to validation.

    Returns
    (list, list, list)
        The (train, val, test) splits.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be between 0 and 1.")
    if not 0.0 <= val_frac < 1.0:
        raise ValueError("val_frac must be between 0 and 1.")
    if train_frac + val_frac > 1.0:
        raise ValueError("train_frac + val_frac must be <= 1.0.")

    seed = TRAINING_CONFIG.seed
    rng = np.random.default_rng(seed) # Shuffles the order randomly.
    indices = np.arange(len(sequences))
    rng.shuffle(indices)

    # Compute how many sequences go to each set.
    n_train = int(train_frac * len(indices))
    n_val = int(val_frac * len(indices))

    # Slice the shuffled index list.
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    def pick(idxs):
        """Helper function that retrieves sequences by index list."""
        return [sequences[i] for i in idxs]

    return pick(train_idx), pick(val_idx), pick(test_idx)


def prune_columns(df, feature_cols, meta_cols=None, keep_extra=None):
    """
    Keep only essential columns to reduce memory usage.

    Keeps:
      - keys: recording_id, vehicle_id, frame
      - observation features (feature_cols)
      - metadata (meta_cols)
      - optional extras (keep_extra)
    """
    if meta_cols is None:
        meta_cols = []
    if keep_extra is None:
        keep_extra = []

    base_cols = ["recording_id", "vehicle_id", "frame"]
    keep = base_cols + list(feature_cols) + list(meta_cols) + list(keep_extra)

    # remove duplicates, keep only existing columns
    keep = [c for c in dict.fromkeys(keep) if c in df.columns]

    return df[keep].copy()