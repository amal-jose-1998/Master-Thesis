"""
HighD dataset utilities.

This module:
- loads one or more highD `*_tracks.csv` files into a single unified DataFrame
- converts the table into per-vehicle `TrajectorySequence` objects
- optionally computes and applies feature scaling
- creates reproducible train/val/test splits (seeded via TRAINING_CONFIG)
"""

from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

from .dataset import TrajectorySequence
from ..config import TRAINING_CONFIG

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
}

def compute_feature_scaler(sequences):
    """
    Compute per-feature mean and standard deviation from a set of sequences.
    The scaler is typically computed on the training split only, then applied to
    train/val/test using the same statistics.

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
            )
        )
    return out

def load_highd_folder(root, cache_path=None, force_rebuild=False, max_recordings=None):
    """
    Load highD `*_tracks.csv` files and merge them into one unified DataFrame.
    If a cached Feather file exists and `max_recordings` is None, the cached
    table is loaded (unless `force_rebuild=True`).

    Parameters
    root : str or Path
        Directory containing files such as `01_tracks.csv`, `02_tracks.csv`, ...
    cache_path : str or Path, optional
        Path to a cached Feather file. If None, defaults to `root / "highd_all.feather"`.
    force_rebuild : bool, default False
        If True, ignore any existing cache and rebuild from CSV files.
    max_recordings : int, optional
        If provided, only the first `max_recordings` CSV files (after sorting)
        are loaded. This is useful for debugging and quick experiments.

    Returns
    pd.DataFrame
        A single concatenated table with standardized column names:
        `vehicle_id, frame, x, y, vx, vy, ax, ay, lane_id, recording_id`.
    """
    root = Path(root)
    if cache_path is None:
        cache_path = root / "highd_all.feather"
    else:
        cache_path = Path(cache_path)

    # Try to load from cache
    if max_recordings is None and cache_path.exists() and not force_rebuild:
        print(f"[highd_loader] Loading cached DataFrame from: {cache_path}")
        return pd.read_feather(cache_path)
    
    # Otherwise build from CSVs
    csv_paths = sorted(root.glob("*_tracks.csv"))
    if not csv_paths:
        raise RuntimeError(f"No *_tracks.csv files found in {root}")
    
    if max_recordings is not None:
        csv_paths = csv_paths[:max_recordings]
        print(f"[load_highd_folder] Using only first {len(csv_paths)} recordings.")

    print(f"[load_highd_folder] Building DataFrame from {len(csv_paths)} CSV files...")
    dfs = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path) # Loads the highD CSV file into a pandas DataFrame.
        df = df.rename(columns=HIGHD_COL_MAP) # Renames highD columns to our defined names.
        keep_cols = list(HIGHD_COL_MAP.values())
        df = df[keep_cols] # Keeps only these essential columns.
        rec_id = int(csv_path.stem.split("_")[0]) # Extracts the recording number from the filename (e.g. "01" from "01_tracks.csv")
        df["recording_id"] = rec_id
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True) # Concatenates all _tracks.csv DataFrames into one big table.

    # cast types explicitly
    df_all["vehicle_id"] = df_all["vehicle_id"].astype(int)
    df_all["frame"] = df_all["frame"].astype(int)
    df_all["lane_id"] = df_all["lane_id"].astype(int)
    for col in ["x", "y", "vx", "vy", "ax", "ay"]:
        df_all[col] = df_all[col].astype(float)

    # Save cache for future calls only for full dataset
    if max_recordings is None:
        print(f"[load_highd_folder] Saving cached DataFrame to: {cache_path}")
        df_all.to_feather(cache_path)
    
    return df_all


def df_to_sequences(df, feature_cols, id_col="vehicle_id", frame_col="frame"):
    """
    Convert a flat per-frame table into per-vehicle trajectories.
    A trajectory is defined by the unique pair (recording_id, vehicle_id).
    Rows are sorted by time (`frame_col`) within each group.

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

    Returns
    list[TrajectorySequence]
        One sequence per (recording_id, vehicle_id). Each sequence contains:
        - `frames`: shape (T,)
        - `obs`: shape (T, F)
        - `obs_names`: the provided `feature_cols`
        - `recording_id`: the group recording identifier
    """
    sequences: List[TrajectorySequence] = []

    group_cols = ["recording_id", id_col]

    df = df.sort_values(group_cols + [frame_col], kind="mergesort")

    for (rec_id, veh_id), g in df.groupby(group_cols, sort=False): # Groups all rows belonging to the same vehicle in a particular recording
        #g = g.sort_values(frame_col) # DataFrame with only that vehicle’s data. Sorts rows by frame number (time ordering).
        frames = g[frame_col].to_numpy(dtype=np.int64, copy=False) # Extracts time indices
        obs = g[list(feature_cols)].to_numpy(dtype=np.float64, copy=True) # Extracts the observation features into a NumPy array of shape: T × F
        # Create one TrajectorySequence object representing this vehicle.
        seq = TrajectorySequence(vehicle_id=veh_id, frames=frames, obs=obs, obs_names=list(feature_cols), recording_id=rec_id)
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
