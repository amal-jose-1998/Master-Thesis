"""
Utilities to load the highD dataset and convert it into TrajectorySequence objects for DBN training.
"""
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .dataset import TrajectorySequence

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

def load_highd_folder(root):
    """
    Load all *_tracks.csv files in the highD folder and merge them into one DataFrame.

    Parameters
    root : str or Path
        Directory containing files such as `01_tracks.csv`, `02_tracks.csv`, ...

    Returns
    DataFrame
        A single table with unified column names:
        `vehicle_id, frame, x, y, vx, vy, ax, ay, lane_id, recording_id`.

    Notes
    - Only columns in ``HIGHD_COL_MAP`` are retained.
    - ``recording_id`` is inferred from the filename prefix.
    """
    root = Path(root)
    csv_paths = sorted(root.glob("*_tracks.csv"))
    if not csv_paths:
        raise RuntimeError(f"No *_tracks.csv files found in {root}")

    dfs = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path) # Loads the highD CSV file into a pandas DataFrame.
        df = df.rename(columns=HIGHD_COL_MAP) # Renames highD columns to our defined names.
        keep_cols = list(HIGHD_COL_MAP.values())
        df = df[keep_cols] # Keeps only these essential columns.
        rec_id = csv_path.stem.split("_")[0] # Extracts the recording number from the filename (e.g. "01" from "01_tracks.csv")
        df["recording_id"] = rec_id
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True) # Concatenates all _tracks.csv DataFrames into one big table.

    # cast types explicitly
    df_all["vehicle_id"] = df_all["vehicle_id"].astype(int)
    df_all["frame"] = df_all["frame"].astype(int)
    df_all["lane_id"] = df_all["lane_id"].astype(int)
    for col in ["x", "y", "vx", "vy", "ax", "ay"]:
        df_all[col] = df_all[col].astype(float)
    return df_all


def df_to_sequences(df, feature_cols, id_col="vehicle_id", frame_col="frame"):
    """
    Group a flat highD table into per-vehicle TrajectorySequence objects.

    Parameters
    df : DataFrame
        Input table containing at least ``id_col``, ``frame_col``, and the feature columns.
    feature_cols : list of str
        Names of the observation features to extract.
    id_col : str, default="vehicle_id"
        Column used to identify individual trajectories.
    frame_col : str, default="frame"
        Column containing the time index.

    Returns
    list of TrajectorySequence
        One sequence per unique vehicle ID. Each sequence contains:
        - sorted frame indices,
        - a T×F observation array,
        - the feature name list,
        - the associated recording ID if present.
    """
    sequences: List[TrajectorySequence] = []

    for veh_id, g in df.groupby(id_col): # Groups all rows belonging to the same vehicle
        g = g.sort_values(frame_col) # DataFrame with only that vehicle’s data. Sorts rows by frame number (time ordering).
        frames = g[frame_col].to_numpy(dtype=np.int64) # Extracts time indices
        obs = g[list(feature_cols)].to_numpy(dtype=np.float64) # Extracts the observation features into a NumPy array of shape: T × F
        # Create one TrajectorySequence object representing this vehicle.
        seq = TrajectorySequence(vehicle_id=veh_id, frames=frames, obs=obs, obs_names=list(feature_cols), 
                                 recording_id=g["recording_id"].iloc[0] if "recording_id" in g else None
                                 )
        sequences.append(seq) # Adds this trajectory to the list.

    return sequences


def train_val_test_split(sequences, train_frac=0.7, val_frac=0.15, seed=123):
    """
    Randomly partition a list of sequences into train/validation/test sets.

    Parameters
    sequences : list
        Collection of TrajectorySequence objects.
    train_frac : float, default=0.7
        Percentage of sequences assigned to the training set.
    val_frac : float, default=0.15
        Percentage assigned to validation.
    seed : int, default=123
        RNG seed for reproducible shuffling.

    Returns
    (list, list, list)
        The (train, val, test) splits.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be between 0 and 1.")
    if not 0.0 <= val_frac < 1.0:
        raise ValueError("val_frac must be between 0 and 1.")

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
