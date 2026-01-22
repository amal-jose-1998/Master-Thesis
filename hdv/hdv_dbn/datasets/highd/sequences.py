import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
import joblib
import json

from ..dataset import TrajectorySequence
from ...config import (FRAME_FEATURE_COLS, META_COLS, WINDOW_FEATURE_COLS)

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
    obs_names = list(feature_cols)

    df = df.sort_values(group_cols + [frame_col], kind="mergesort")

    if meta_cols is None:
        meta_cols = []

    grouped = df.groupby(group_cols, sort=False)
    pbar = tqdm(grouped, total=grouped.ngroups, desc="Building sequences", unit="seq")

    for (rec_id, veh_id), g in pbar: # Groups all rows belonging to the same vehicle in a particular recording
        frames = g[frame_col].to_numpy(dtype=np.int64, copy=False) # Extracts time indices
        obs = g[obs_names].to_numpy(dtype=np.float64, copy=False) # Extracts the observation features into a NumPy array of shape: T × F
        obs.setflags(write=False) # Make read-only to prevent accidental modification
       
        meta = {}
        if meta_cols:
            first = g.iloc[0]
            for c in meta_cols:
                if c in g.columns:
                    meta[c] = first[c] # Extracts metadata from the first row of this vehicle's data. (Eg: meta_class, meta_drivingDirection)
       
        # Create one TrajectorySequence object representing this vehicle.
        seq = TrajectorySequence(vehicle_id=veh_id, frames=frames, obs=obs, obs_names=obs_names, recording_id=rec_id,  meta=meta if meta else None)
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
        Random seed for reproducibility.

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

def _nanmax(x):
    x = x[np.isfinite(x)]
    return float(x.max()) if x.size else np.nan

def _nanp95(x: np.ndarray):
    x = x[np.isfinite(x)]
    return float(np.percentile(x, 95)) if x.size else np.nan

def _nanrms(x: np.ndarray):
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x * x))) if x.size else np.nan

def _seti(Y, t, j, v):
    """
    Set Y[t, j] if j is not None.
    
    Parameters
    Y : np.ndarray
        Output observation matrix.
    t : int
        Current time index.
    j : int or None
        Column index to set, or None to skip.
    v : float
        Value to set.
    """
    if j is not None:
        Y[t, j] = v

def _risk_valid_mask(r):
    return np.isfinite(r) & (r > 0.0)

def _sign_fractions(x):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan

    n = float(x.size)

    neg = float((x < 0.0).sum()) / n
    pos = float((x > 0.0).sum()) / n
    zer = float((x == 0.0).sum()) / n

    return neg, pos, zer

def _compute_minmax_and_sign_fracs(win, Y, t, in_idx, out, c):
    """
    For c in {"ax","ay"} compute (if requested in out):
      {c}_min, {c}_max
      {c}_neg_frac: frac(c < 0)
      {c}_pos_frac: frac(c > 0)
      {c}_zero_frac: frac(c == 0) 
    """
    jmin = out.get(f"{c}_min")
    jmax = out.get(f"{c}_max")
    jneg = out.get(f"{c}_neg_frac")
    jpos = out.get(f"{c}_pos_frac")
    jzer = out.get(f"{c}_zero_frac")

    # nothing requested → do nothing
    if jmin is None and jmax is None and jneg is None and jpos is None and jzer is None:
        return

    col = win[:, in_idx[c]].astype(np.float64, copy=False)
    finite = np.isfinite(col)
    if not np.any(finite):
        return

    v = col[finite]  # finite only

    # min/max (use your nan-style helpers)
    _seti(Y, t, jmin, _nanmin(v))
    _seti(Y, t, jmax, _nanmax(v))

    # fractions (compute once, set only those requested)
    neg, pos, zer = _sign_fractions(v)
    _seti(Y, t, jneg, neg)
    _seti(Y, t, jpos, pos)
    _seti(Y, t, jzer, zer)

def _compute_kinematics(win, Y, t, in_idx, out):
    """
    Compute kinematic summaries: mean/std of vx, ax, vy, ay.
    
    Parameters
    win : np.ndarray
        Input window of shape (W, F_in).
    Y : np.ndarray
        Output observation matrix of shape (Tw, F_out).
    t : int
        Current window index in Y.
    in_idx : dict
        Mapping from input feature names to their column indices in `win`.
    out : dict
        Mapping from output feature names to their column indices in `Y`.
    """
    for c in ("vx", "ax", "vy", "ay"):
        col = win[:, in_idx[c]]
        _seti(Y, t, out.get(f"{c}_mean"), _nanmean(col))
        _seti(Y, t, out.get(f"{c}_std"),  _nanstd(col))
        if c == "ax" or c == "ay":
            _compute_minmax_and_sign_fracs(win, Y, t, in_idx, out, c)

def _compute_jerk(win, Y, t, in_idx, out):
    """
    Compute jerk summaries: mean/std/rms/p95 of jerk_x and jerk_y.

    Parameters
    win : np.ndarray
        Input window of shape (W, F_in).
    Y : np.ndarray
        Output observation matrix of shape (Tw, F_out).
    t : int
        Current window index in Y.
    in_idx : dict
        Mapping from input feature names to their column indices in `win`.
    out : dict
        Mapping from output feature names to their column indices in `Y`.
    """
    # jerk_x stats
    if "jerk_x" in in_idx:
        _seti(Y, t, out.get("jerk_x_mean"), _nanmean(win[:, in_idx["jerk_x"]]))
        _seti(Y, t, out.get("jerk_x_std"),  _nanstd(win[:, in_idx["jerk_x"]]))
        _seti(Y, t, out.get("jerk_x_rms"),  _nanrms(win[:, in_idx["jerk_x"]]))
        _seti(Y, t, out.get("jerk_x_p95"),  _nanp95(win[:, in_idx["jerk_x"]]))
    # jerk_y stats
    if "jerk_y" in in_idx:
        _seti(Y, t, out.get("jerk_y_mean"), _nanmean(win[:, in_idx["jerk_y"]]))
        _seti(Y, t, out.get("jerk_y_std"),  _nanstd(win[:, in_idx["jerk_y"]]))
        _seti(Y, t, out.get("jerk_y_rms"),  _nanrms(win[:, in_idx["jerk_y"]]))
        _seti(Y, t, out.get("jerk_y_p95"),  _nanp95(win[:, in_idx["jerk_y"]]))

def _compute_lane_change_flags(win, Y, t, in_idx, out):
    # lc: -1 left, +1 right, 0 none, NaNs ignored
    lc = win[:, in_idx["lc"]]
    lc_f = lc[np.isfinite(lc)]
    _seti(Y, t, out.get("lc_left_present"),  float(np.any(lc_f == -1))) # 1 if any left lane changes in window; else 0
    _seti(Y, t, out.get("lc_right_present"), float(np.any(lc_f == +1))) # 1 if any right lane changes in window; else 0

def _compute_lane_boundaries(win, Y, t, in_idx, out):
    j = out.get("d_left_lane_mean")
    k = out.get("d_left_lane_min")
    if j is not None or k is not None:
        dl = win[:, in_idx["d_left_lane"]]
        _seti(Y, t, j, _nanmean(dl))
        _seti(Y, t, k, _nanmin(dl))

    j = out.get("d_right_lane_mean")
    k = out.get("d_right_lane_min")
    if j is not None or k is not None:
        dr = win[:, in_idx["d_right_lane"]]
        _seti(Y, t, j, _nanmean(dr))
        _seti(Y, t, k, _nanmin(dr))

def _compute_risk(win, Y, t, in_idx, out, has_front_exists):
    # Only compute if any risk outputs exist
    for colname in ("front_thw", "front_ttc", "front_dhw"):
        j_mean = out.get(f"{colname}_mean")
        j_min  = out.get(f"{colname}_min")
        j_vfr  = out.get(f"{colname}_vfrac")
        if j_mean is None and j_min is None and j_vfr is None:
            continue

        r = win[:, in_idx[colname]].astype(np.float64, copy=False)
        valid = _risk_valid_mask(r)
        if has_front_exists:
            ex = win[:, in_idx["front_exists"]]
            valid = valid & (ex > 0.5)

        vfrac = float(np.mean(valid)) if valid.size else 0.0
        rv = r[valid]
        _seti(Y, t, j_mean, (float(rv.mean()) if rv.size else np.nan))
        _seti(Y, t, j_min,  (float(rv.min())  if rv.size else np.nan))
        _seti(Y, t, j_vfr,  vfrac)

def _compute_existence_fracs(win, Y, t, in_idx, out, exists_cols):
    for c in exists_cols:
        j = out.get(f"{c}_frac")
        if j is None:
            continue
        ex = win[:, in_idx[c]]
        _seti(Y, t, j, float(np.mean(np.isfinite(ex) & (ex > 0.5))))

def _validate_required_columns(in_idx, out_idx):
    required_frame = {"vx", "ax", "vy", "ay", "lc"}

    jerk_out_keys = {
        "jerk_x_mean", "jerk_x_std", "jerk_x_rms", "jerk_x_p95",
        "jerk_y_mean", "jerk_y_std", "jerk_y_rms", "jerk_y_p95",
    }
    if any(k in out_idx for k in jerk_out_keys):
        # require per-frame jerk columns only if you actually want jerk outputs
        if any(k.startswith("jerk_x_") for k in out_idx.keys() if k in jerk_out_keys):
            required_frame.add("jerk_x")
        if any(k.startswith("jerk_y_") for k in out_idx.keys() if k in jerk_out_keys):
            required_frame.add("jerk_y")

    if "d_left_lane_mean" in out_idx or "d_left_lane_min" in out_idx:
        required_frame.add("d_left_lane")
    if "d_right_lane_mean" in out_idx or "d_right_lane_min" in out_idx:
        required_frame.add("d_right_lane")

    for r in ("front_thw", "front_ttc", "front_dhw"):
        if (f"{r}_mean" in out_idx) or (f"{r}_min" in out_idx) or (f"{r}_vfrac" in out_idx):
            required_frame.add(r)

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

    return exists_cols

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

    #strict input validation (fail fast) 
    exists_cols = _validate_required_columns(in_idx, out_idx)

    has_front_exists = "front_exists" in in_idx  

    total_windows = 0
    skipped = 0
    pbar = tqdm(sequences, desc="Windowizing sequences", unit="seq")

    # Process each sequence individually
    for seq in pbar:
        X = seq.obs    # per-frame observation matrix for one vehicle: shape (T, F_in)
        T = X.shape[0] # number of frames 
        if T < W:      # vehicle’s trajectory length T is shorter than window size W; skip it
            skipped += 1
            pbar.set_postfix(skipped=skipped, out=len(out), windows=total_windows)
            continue

        starts = np.arange(0, T - W + 1, stride, dtype=np.int64)        # Starting indices of each window
        Tw = int(starts.size)                                           # number of windows
        Y = np.full((Tw, len(win_names)), np.nan, dtype=np.float64)     # Output observation matrix; shape: Tw × F_out; initialized to NaN (missing stays missing))
        total_windows += Tw

        for t, s0 in enumerate(starts):     # For each window: slice the per-frame data
            s1 = s0 + W                     # Ending index (exclusive)
            win = X[s0:s1, :]               # per-frame data inside this window; shape: W, F_in 

            _compute_kinematics(win, Y, t, in_idx, out_idx)        # compute mean/std of vx, ax, vy, ay
            _compute_jerk(win, Y, t, in_idx, out_idx)              # compute mean/std of jerk_x
            _compute_lane_change_flags(win, Y, t, in_idx, out_idx) # lane change presence flags
            _compute_lane_boundaries(win, Y, t, in_idx, out_idx)   # lane boundary distances
            _compute_risk(win, Y, t, in_idx, out_idx, has_front_exists)  # risk summaries
            _compute_existence_fracs(win, Y, t, in_idx, out_idx, exists_cols) # existence fractions
        
        frames = seq.frames
        out_frames = frames[starts]  # window start frame numbers

        out.append(TrajectorySequence(
            vehicle_id=seq.vehicle_id,
            frames=out_frames, # window start frames
            obs=Y,         # now window-level matrix Y of shape (Tw, F_out)
            obs_names=win_names,
            recording_id=seq.recording_id,
            meta=seq.meta,
        ))
        pbar.set_postfix(skipped=skipped, out=len(out), windows=total_windows,)

    return out


def window_cache_paths(cache_dir, W, stride):
    """
    Get cache file paths for windowized sequences.

    Parameters
    cache_dir : Path
        Directory where cache files are stored.
    exp_name : str
        Experiment name.
    W : int
        Window size.
    stride : int
        Window stride.
    S : int
        Number of hidden styles.
    A : int
        Number of action types.

    Returns
    (Path, Path)
        Paths to the joblib and json cache files.
    """
    base = cache_dir / f"W{W}_Stride{stride}"
    return base.with_suffix(".joblib"), base.with_suffix(".json")

def load_or_build_windowized(df, cache_dir, W, stride, force_rebuild=False):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True) # ensure cache directory exists else create it

    p_joblib, p_meta = window_cache_paths(cache_dir, W, stride)

    if p_joblib.exists() and (not force_rebuild):
        print(f"[window-cache] Loading: {p_joblib}")
        return joblib.load(p_joblib)
    
    frame_feature_cols = list(FRAME_FEATURE_COLS)
    meta_cols = list(META_COLS)
    
    # Build per-vehicle frame sequences 
    frame_sequences = df_to_sequences(df, feature_cols=frame_feature_cols, meta_cols=meta_cols)
    print(f"[load_or_build_windowized] Total FRAME sequences (vehicles) loaded: {len(frame_sequences)}")

    print(f"[load_or_build_windowized] Building windowized sequences (W={W}, stride={stride})")
    win_seqs = windowize_sequences(frame_sequences, W=W, stride=stride)

    joblib.dump(win_seqs, p_joblib, compress=3) # save to cache
    meta = {
        "W": int(W),
        "stride": int(stride),
        "num_sequences": int(len(win_seqs)),
        "win_names": list(WINDOW_FEATURE_COLS),
    }
    p_meta.write_text(json.dumps(meta, indent=2))
    print(f"[load_or_build_windowized] Saved: {p_joblib}")
    return win_seqs