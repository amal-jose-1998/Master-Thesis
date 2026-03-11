import numpy as np
import pandas as pd

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


def add_lane_boundary_distance_features(df, y_col="y_center", dir_col="meta_drivingDirection", upper_key="rec_upperLaneMarkings", lower_key="rec_lowerLaneMarkings"):
    """
    Add vehicle-centric distances to left/right lane boundaries.
    Outputs:
      - d_left_lane  >= 0
      - d_right_lane >= 0

    If boundaries cannot be determined, values are NaN.
    """
    if y_col not in df.columns or dir_col not in df.columns:
        df["d_left_lane"] = np.nan
        df["d_right_lane"] = np.nan
        return df

    # Pre-parse markings per recording 
    if "recording_id" in df.columns:
        rec_ids = df["recording_id"].dropna().unique().tolist()
    else:
        rec_ids = [None]

    rec_to_upper, rec_to_lower = {}, {}
    for rid in rec_ids:
        sub = df[df["recording_id"] == rid] if rid is not None else df
        rec_to_upper[rid] = _parse_lane_markings(sub[upper_key].iloc[0]) if upper_key in sub else np.array([])
        rec_to_lower[rid] = _parse_lane_markings(sub[lower_key].iloc[0]) if lower_key in sub else np.array([])

    y = df[y_col].to_numpy(dtype=np.float64, copy=False)
    d = df[dir_col].to_numpy(copy=False)
    rid_arr = df["recording_id"].to_numpy(copy=False) if "recording_id" in df.columns else np.full(len(df), None)

    d_left = np.full(len(df), np.nan, dtype=np.float64)
    d_right = np.full(len(df), np.nan, dtype=np.float64)

    for i in range(len(df)):    # iterate rows
        yi = y[i] # lateral position
        if np.isnan(yi):
            continue

        di = int(d[i]) if not pd.isna(d[i]) else -1 # driving direction
        rid = rid_arr[i] # recording id

        if di == 1:
            marks = rec_to_upper.get(rid, np.array([])) # use upper lane markings
        elif di == 2:
            marks = rec_to_lower.get(rid, np.array([])) # use lower lane markings
        else:
            continue

        if marks.size < 2: # need at least 2 boundaries to define lane intervals
            continue

        # find lane interval
        j = int(np.searchsorted(marks, yi, side="right") - 1) # find left boundary index
        if j < 0 or j >= marks.size - 1:
            continue

        left_world = marks[j]
        right_world = marks[j + 1]

        # convert to vehicle-centric left/right
        if di == 2:
            # smaller world-y = vehicle-left
            d_left[i] = yi - left_world
            d_right[i] = right_world - yi
        else:
            # di == 1: smaller world-y = vehicle-right (swap)
            d_left[i] = right_world - yi
            d_right[i] = yi - left_world

    df["d_left_lane"] = d_left
    df["d_right_lane"] = d_right
    return df


def add_lane_change_feature(df, lane_col="lane_id", dir_sign_col="dir_sign", meta_dir_col="meta_drivingDirection", upper_direction_value=1,):
    """
    Add single ternary lane-change feature:
        lc ∈ {-1, 0, +1}
          -1 : left lane change (vehicle-centric)
           0 : no lane change
          +1 : right lane change (vehicle-centric)
    """
    if lane_col not in df.columns:
        df["lc"] = 0
        return df

    # Per-trajectory stable sort
    df = df.sort_values(["recording_id", "vehicle_id", "frame"], kind="mergesort").copy()

    lane = df[lane_col].to_numpy(dtype=np.float64, copy=False) # lane ids
    lane_f = np.where(np.isfinite(lane), lane, np.nan)         # treat non-finite as NaN

   # reset diff at each new trajectory boundary
    rec = df["recording_id"].to_numpy()
    vid = df["vehicle_id"].to_numpy()
    new_traj = np.r_[True, (rec[1:] != rec[:-1]) | (vid[1:] != vid[:-1])] # new trajectory flags which reset diffs

    prev_lane = np.r_[lane_f[0], lane_f[:-1]] # shifted lane ids
    dlane = lane_f - prev_lane       # lane id differences
    dlane[new_traj] = 0.0            # reset at new trajs
    dlane[~np.isfinite(dlane)] = 0.0 # treat NaN diffs as 0

    lc = np.sign(dlane).astype(np.int64) # {-1,0,+1} initial lane change without vehicle-centric adjustment

    if dir_sign_col in df.columns:
        ds = df[dir_sign_col].to_numpy()
        ds = np.where(np.isfinite(ds) & (ds != 0), np.sign(ds), 1).astype(np.int64)
        lc = lc * ds
    elif meta_dir_col in df.columns:
        # Treat "upper" direction as needing sign flip; everything else -> no flip
        # If md is NaN/invalid, default to no flip.
        md = df[meta_dir_col].to_numpy()
        ds = np.where(np.isfinite(md) & (md == upper_direction_value), -1, 1).astype(np.int64) # upper -> -1, else +1
        lc = lc * ds

    df["lc"] = np.clip(lc, -1, 1).astype(np.int8)
    return df