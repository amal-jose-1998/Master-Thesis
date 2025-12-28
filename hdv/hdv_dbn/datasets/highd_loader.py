"""
HighD dataset utilities.

This module:
- loads highD `*_tracks.csv` + `*_tracksMeta.csv` + `*_recordingMeta.csv` files
- merges meta into the per-frame table
- applies vehicle-centric direction normalization (sign flips)
- converts the table into per-vehicle `TrajectorySequence` objects
- optionally computes and applies feature scaling
- creates reproducible train/val/test splits (seeded via TRAINING_CONFIG)
"""

from pathlib import Path
import numpy as np
import pandas as pd

try:
    # When imported as a package module
    from .dataset import TrajectorySequence
    from ..config import TRAINING_CONFIG
except ImportError:
    # When run as a standalone script (debug)
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))
    from hdv.hdv_dbn.datasets.dataset import TrajectorySequence
    from hdv.hdv_dbn.config import TRAINING_CONFIG

# Map highD column names to the generic internal names.
HIGHD_COL_MAP = {
    "id": "vehicle_id",
    "frame": "frame",
    "x": "x", 
    "y": "y",
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

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
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
      {prefix}_x, {prefix}_y, {prefix}_vx, {prefix}_vy, {prefix}_lane_id
    
    Parameters
    df : pd.DataFrame
        Input table containing at least:
          - `recording_id`, `frame`, `vehicle_id`
          - neighbor ID column specified by `neighbor_id_col`
          - x, y, vx, vy, lane_id columns
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
    lookup = df[["recording_id","frame","vehicle_id","x","y","vx","vy","lane_id"]].copy()
    lookup["vehicle_id"] = lookup["vehicle_id"].astype("Int64")
    lookup = lookup.rename(
        columns={
            "vehicle_id": neighbor_id_col,
            "x": f"{prefix}_x",
            "y": f"{prefix}_y",
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
# Scaling utilities 
# ---------------------------------------------------------------------
def compute_feature_scaler(sequences):
    """
    Compute per-feature mean and standard deviation from a set of sequences. 

    Parameters
    sequences : sequence of TrajectorySequence
        Sequences whose `.obs` arrays will be stacked along time.

    Returns
    mean : np.ndarray
        Feature-wise mean, shape (F,).
    std : np.ndarray
        Feature-wise standard deviation, shape (F,). Very small std values are
        clamped to 1.0 to avoid division by near-zero during scaling.
    """
    X = np.vstack([seq.obs for seq in sequences])  # (N_total, F)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0  # avoid division by ~0
    return mean, std

def compute_classwise_feature_scalers(sequences, class_key="meta_class"):
    """
    Compute one (mean, std) scaler per vehicle class.

    Parameters
    sequences : list[TrajectorySequence]
        Input sequences (usually training split only).
    class_key : str
        Key inside seq.meta indicating vehicle class (e.g. "meta_class").

    Returns
    dict
        Mapping: class_name -> (mean, std)
    """
    buckets = {}

    for seq in sequences:
        if seq.meta is None or class_key not in seq.meta:
            continue
        cls = seq.meta[class_key]
        buckets.setdefault(cls, []).append(seq)

    scalers = {}
    for cls, seqs in buckets.items():
        mean, std = compute_feature_scaler(seqs)
        scalers[cls] = (mean, std)

    return scalers

def scale_sequences_classwise(sequences, scalers, class_key="meta_class"):
    """
    Apply class-specific feature scaling to sequences.

    Parameters
    sequences : list[TrajectorySequence]
        Sequences to scale.
    scalers : dict
        Output of compute_classwise_feature_scalers().
    class_key : str
        Key inside seq.meta indicating vehicle class.

    Returns
    list[TrajectorySequence]
        Scaled sequences.
    """
    out = []

    for seq in sequences:
        if seq.meta is None or class_key not in seq.meta:
            raise ValueError("Sequence missing class information for class-wise scaling.")

        cls = seq.meta[class_key]
        if cls not in scalers:
            raise ValueError(f"No scaler found for class '{cls}'.")

        mean, std = scalers[cls]
        obs_scaled = (seq.obs - mean) / std

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

def scale_sequences(sequences, mean, std):
    """
    Apply z-score feature scaling to a list of sequences.
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
        New sequence objects containing scaled observations. Metadata (vehicle_id,
        frames, obs_names, recording_id) is preserved.
    """
    out = []
    for seq in sequences:
        obs_scaled = (seq.obs - mean) / std
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

# ---------------------------------------------------------------------
# Context features
# ---------------------------------------------------------------------
def add_direction_aware_context_features(df):
    """
    Build direction-consistent relative context features from neighbor IDs.
    Adds, for each neighbor slot:
      - dx  = neighbor_x - ego_x
      - dy  = neighbor_y - ego_y
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
    for id_col, prefix in neighbor_specs:
        if id_col not in out.columns:
            continue

        # existence mask before merge
        nid = out[id_col].fillna(0).astype("Int64")
        out[f"{prefix}_exists"] = (nid > 0).astype(np.float64)
        out[id_col] = nid.mask(nid <= 0, pd.NA) # set missing neighbor IDs to NaN

        out = _merge_neighbor_state(out, id_col, prefix)

        # relative features (NaN if neighbor missing)
        out[f"{prefix}_dx"]  = out[f"{prefix}_x"]  - out["x"]
        out[f"{prefix}_dy"]  = out[f"{prefix}_y"]  - out["y"]
        out[f"{prefix}_dvx"] = out[f"{prefix}_vx"] - out["vx"]
        out[f"{prefix}_dvy"] = out[f"{prefix}_vy"] - out["vy"]

        rel_cols = [f"{prefix}_dx", f"{prefix}_dy", f"{prefix}_dvx", f"{prefix}_dvy"]
        miss = out[f"{prefix}_exists"] == 0.0
        out.loc[miss, rel_cols] = 0.0

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
        return pd.read_feather(cache_path)
    
    # Otherwise build from CSVs
    tracks_paths = sorted(root.glob("*_tracks.csv"))
    if not tracks_paths:
        raise RuntimeError(f"No *_tracks.csv files found in {root}")
    
    rec_ids = [int(p.stem.split("_")[0]) for p in tracks_paths]
    rec_ids = sorted(rec_ids)
    
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
        meta_cols = [c for c in df_tracksmeta.columns if c != "vehicle_id"]
        df_tracksmeta = df_tracksmeta[["vehicle_id"] + meta_cols].copy()
        df_tracksmeta = df_tracksmeta.rename(columns={c: f"meta_{c}" for c in meta_cols})

        # RecordingMeta: usually one row; prefix as rec_
        if len(df_recmeta) >= 1:
            rec_row = df_recmeta.iloc[0].to_dict()
        else:
            rec_row = {}
        rec_row = {f"rec_{k}": v for k, v in rec_row.items()}

        # Merge tracks with tracksMeta (vehicle-wise)
        df = df_tracks.merge(df_tracksmeta, on="vehicle_id", how="left")

        # Attach recording meta as constants for this recording
        for k, v in rec_row.items():
            df[k] = v
        
        if "meta_class" in df.columns:
            df["meta_class"] = df["meta_class"].astype(str).str.lower()

        # Apply vehicle-centric normalization
        if apply_vehicle_centric:
            # highD tracksMeta commonly provides drivingDirection there
            # after prefix it becomes meta_drivingDirection
            dir_col = "meta_drivingDirection" if "meta_drivingDirection" in df.columns else None
            if dir_col is not None:
                # create a temporary un-prefixed column name expected by normalize function
                df["drivingDirection"] = df[dir_col]
                df = normalize_vehicle_centric(df, dir_col="drivingDirection", flip_longitudinal=True, flip_lateral=flip_lateral, flip_positions=flip_positions)
                df = df.drop(columns=["drivingDirection"], errors="ignore")
            else:
                df["dir_sign"] = 1.0
        
        df = add_direction_aware_context_features(df)

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
    df_all["lane_id"] = df_all["lane_id"].astype(int)
    df_all["recording_id"] = df_all["recording_id"].astype(int)
    for col in ["x", "y", "vx", "vy", "ax", "ay"]:
        df_all[col] = df_all[col].astype(float)

    # Save cache for future calls only for full dataset
    if max_recordings is None:
        print(f"[load_highd_folder] Saving cached DataFrame to: {cache_path}")
        df_all.to_feather(cache_path)

    return df_all

# ---------------------------------------------------------------------
# Sequences: keep obs clean, store meta in seq.meta
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

# ---------------------------------------------------------------------
# Main for quick sanity check
# ---------------------------------------------------------------------
def main():
    """Quick sanity check for the highD loader."""
    HIGH_D_ROOT = Path(
        r"C:\Users\amalj\OneDrive\Desktop\Master's Thesis\Implementation\hdv\data\highd"
    )

    print("Loading highD data...")
    df = load_highd_folder(
        root=HIGH_D_ROOT,
        max_recordings=2,          # limit for quick testing
        apply_vehicle_centric=True,
        flip_lateral=True,
        flip_positions=False,
    )

    print("DataFrame loaded.")

    bad = df[(df["front_exists"] == 1) & (df["front_x"].isna())]
    print("front_exists=1 but front_x is NaN:", len(bad))

    bad2 = df[(df["preceding_id"] == 0) & (df["front_exists"] == 1)]
    print("preceding_id=0 but front_exists=1:", len(bad2))


    print("Columns:", list(df.columns))
    print("Number of rows:", len(df))

    # -----------------------------
    # Feature + meta selection
    # -----------------------------
    # baseline feature set
    feature_cols = [
        "y","vx","vy","ax","ay","lane_id",
        "front_exists","front_dx","front_dvx",
        "rear_exists","rear_dx","rear_dvx",
    ]
    # extended feature set with more context
    #feature_cols = [
    #    "y","vx","vy","ax","ay","lane_id",
    #    "front_exists","front_dx","front_dvx",
    #    "rear_exists","rear_dx","rear_dvx",
    #
    #    "left_front_exists","left_front_dx","left_front_dvx",
    #    "right_front_exists","right_front_dx","right_front_dvx",
    #    "left_side_exists","left_side_dy",
    #    "right_side_exists","right_side_dy",
    #]

    meta_cols = [
        "meta_class",
        "meta_drivingDirection"
    ]

    print("\nBuilding sequences...")
    seqs = df_to_sequences(
        df,
        feature_cols=feature_cols,
        meta_cols=meta_cols,
    )

    print(f"Total sequences: {len(seqs)}")

    # -----------------------------
    # Inspect a few sequences
    # -----------------------------
    for i, seq in enumerate(seqs[:3]):
        print(f"\nSequence {i}")
        print("  vehicle_id:", seq.vehicle_id)
        print("  recording_id:", seq.recording_id)
        print("  T:", seq.T)
        print("  F:", seq.F)
        print("  meta:", seq.meta)
        print("  obs mean (first 3 features):", seq.obs.mean(axis=0)[:3])

    # -----------------------------
    # Train / val / test split
    # -----------------------------
    train_seqs, val_seqs, test_seqs = train_val_test_split(seqs)

    print("\nSplit sizes:")
    print("  train:", len(train_seqs))
    print("  val  :", len(val_seqs))
    print("  test :", len(test_seqs))

    # -----------------------------
    # Class-wise scaling
    # -----------------------------
    print("\nComputing class-wise scalers...")
    scalers = compute_classwise_feature_scalers(train_seqs)

    for cls, (mean, std) in scalers.items():
        print(f"  class '{cls}': mean[0]={mean[0]:.3f}, std[0]={std[0]:.3f}")

    train_scaled = scale_sequences_classwise(train_seqs, scalers)
    val_scaled   = scale_sequences_classwise(val_seqs, scalers)
    test_scaled  = scale_sequences_classwise(test_seqs, scalers)

    print("\nScaling successful.")
    print("Example scaled obs mean (train, first seq):",
          train_scaled[0].obs.mean(axis=0)[:3])


if __name__ == "__main__":
    main()
