import numpy as np
import pandas as pd

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
