import numpy as np
import pandas as pd
from ..dataset import TrajectorySequence
from ...config import TRAINING_CONFIG, WINDOW_FEATURE_COLS

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

def _nanmean(x):
    return float(np.nanmean(x)) if np.any(np.isfinite(x)) else np.nan

def _nanstd(x):
    return float(np.nanstd(x)) if np.any(np.isfinite(x)) else np.nan

def _nanmin(x):
    x = x[np.isfinite(x)]
    return float(x.min()) if x.size else np.nan

def windowize_sequences(sequences, W=150, stride=10):
    """
    Convert per-frame sequences into per-window sequences by computing summary statistics over sliding windows.

    Parameters
    sequences : list[TrajectorySequence]
        Input per-frame sequences.
    W : int, default=150
        Window size in frames.
    stride : int, default=10
        Stride between window starts in frames.
    
    Returns
    list[TrajectorySequence]
        Output per-window sequences.
    """
    out = []
    if not sequences:
        return out
    
    # Get feature names and build index maps
    per_frame_feature_names = list(sequences[0].obs_names)
    in_idx = {n: i for i, n in enumerate(per_frame_feature_names)}

    win_names = list(WINDOW_FEATURE_COLS) # schema of window-level features in the output.
    out_idx = {n: i for i, n in enumerate(win_names)}

    def _set(Y, t, name, value):
        """Write value into Y[t, idx] if feature name exists in output."""
        idx = out_idx.get(name)
        if idx is not None:
            Y[t, idx] = value

    #strict input validation (fail fast) 
    required_frame = {"vx", "ax", "vy", "ay", "lc"}
    # jerk outputs require jerk_x
    if "jerk_mean" in out_idx or "jerk_std" in out_idx:
        required_frame.add("jerk_x")
    # lane outputs require lane distances
    if "d_left_lane_mean" in out_idx or "d_left_lane_min" in out_idx:
        required_frame.add("d_left_lane")
    if "d_right_lane_mean" in out_idx or "d_right_lane_min" in out_idx:
        required_frame.add("d_right_lane")
    # risk outputs require corresponding risk columns
    for r in ["front_thw", "front_ttc", "front_dhw"]:
        if f"{r}_mean" in out_idx or f"{r}_min" in out_idx or f"{r}_vfrac" in out_idx:
            required_frame.add(r)

    # exists outputs require exists inputs
    exists_cols = [
        "front_exists", "rear_exists",
        "left_front_exists", "left_side_exists", "left_rear_exists",
        "right_front_exists", "right_side_exists", "right_rear_exists",
    ]
    for c in exists_cols:
        if f"{c}_frac" in out_idx:
            required_frame.add(c)

    missing = [c for c in sorted(required_frame) if c not in in_idx]
    if missing:
        raise RuntimeError(f"[windowize_sequences] Missing required frame cols: {missing}")

    has_front_exists = "front_exists" in in_idx  

    def risk_valid_mask(r):
        return np.isfinite(r) & (r > 0.0)

    # Process each sequence individually
    for seq in sequences:
        X = seq.obs    # per-frame observation matrix for one vehicle: shape (T, F_in)
        T = X.shape[0] # number of frames 
        if T < W:      # vehicle’s trajectory length T is shorter than window size W; skip it
            continue

        starts = np.arange(0, T - W + 1, stride, dtype=np.int64)        # Starting indices of each window
        Tw = int(starts.size)                                           # number of windows
        Y = np.full((Tw, len(win_names)), np.nan, dtype=np.float64)     # Output observation matrix; shape: Tw × F_out; initialized to NaN (missing stays missing))

        for t, s0 in enumerate(starts):     # For each window: slice the per-frame data
            s1 = s0 + W                     # Ending index (exclusive)
            win = X[s0:s1, :]               # per-frame data inside this window; shape: W, F_in 

            for c in ["vx", "ax", "vy", "ay"]:
                col = win[:, in_idx[c]] # extract the W values in that window
                # compute mean/std ignoring NaNs
                _set(Y, t, f"{c}_mean", _nanmean(col))
                _set(Y, t, f"{c}_std",  _nanstd(col))

            # jerk_x -> jerk_mean/std
            if "jerk_mean" in out_idx or "jerk_std" in out_idx:
                col = win[:, in_idx["jerk_x"]]
                _set(Y, t, "jerk_mean", _nanmean(col))
                _set(Y, t, "jerk_std",  _nanstd(col))

            # LC presence flags
            lc = win[:, in_idx["lc"]]  
            lc_finite = lc[np.isfinite(lc)] 
            # set presence flags
            _set(Y, t, "lc_left_present",  float(np.any(lc_finite == -1))) 
            _set(Y, t, "lc_right_present", float(np.any(lc_finite == +1)))

            # lane boundary distances; NaNs are ignored.
            if "d_left_lane_mean" in out_idx or "d_left_lane_min" in out_idx:
                dl = win[:, in_idx["d_left_lane"]]
                _set(Y, t, "d_left_lane_mean", _nanmean(dl))
                _set(Y, t, "d_left_lane_min",  _nanmin(dl))

            if "d_right_lane_mean" in out_idx or "d_right_lane_min" in out_idx:
                dr = win[:, in_idx["d_right_lane"]]
                _set(Y, t, "d_right_lane_mean", _nanmean(dr))
                _set(Y, t, "d_right_lane_min",  _nanmin(dr))

            # risk summaries
            def fill_risk(colname): 
                # Compute mean, min, vfrac for a risk feature
                r = win[:, in_idx[colname]].astype(np.float64, copy=False) # extracts the W values for the risk feature.
                valid = risk_valid_mask(r)
                if has_front_exists:
                    ex = win[:, in_idx["front_exists"]] 
                    valid = valid & (ex > 0.5) # only consider frames where the front vehicle exists.
                vfrac = float(np.mean(valid)) if valid.size else 0.0 # fraction of valid frames
                rv = r[valid] # list of valid risk values.
                _set(Y, t, f"{colname}_mean", (float(rv.mean()) if rv.size else np.nan))
                _set(Y, t, f"{colname}_min",  (float(rv.min())  if rv.size else np.nan))
                _set(Y, t, f"{colname}_vfrac", vfrac)

            for r in ["front_thw", "front_ttc", "front_dhw"]: 
                if f"{r}_mean" in out_idx or f"{r}_min" in out_idx or f"{r}_vfrac" in out_idx:
                    fill_risk(r) 

            # existence fractions
            for c in exists_cols:
                out_name = f"{c}_frac"
                if out_name in out_idx:
                    ex = win[:, in_idx[c]]
                    _set(Y, t, out_name, float(np.mean(np.isfinite(ex) & (ex > 0.5)))) # fraction of frames where neighbor exists

        out.append(TrajectorySequence(
            vehicle_id=seq.vehicle_id,
            frames=starts, # now equals starts (window start indices), not original frame numbers
            obs=Y,         # now window-level matrix Y of shape (Tw, F_out)
            obs_names=win_names,
            recording_id=seq.recording_id,
            meta=seq.meta,
        ))

    return out