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

def add_lane_change_feature(df, lane_col="lane_id", dir_sign_col="dir_sign"):
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

    lane = df[lane_col].to_numpy(dtype=np.float64)

   # reset diff at each new trajectory boundary
    rec = df["recording_id"].to_numpy()
    vid = df["vehicle_id"].to_numpy()
    new_traj = np.r_[True, (rec[1:] != rec[:-1]) | (vid[1:] != vid[:-1])]

    prev_lane = np.r_[lane[0], lane[:-1]]
    dlane = lane - prev_lane
    dlane[new_traj] = 0.0   

    lc = np.sign(dlane).astype(np.int64)

    # Flip sign for upper-lane direction (vehicle heading opposite)
    if dir_sign_col in df.columns:
        ds = df[dir_sign_col].to_numpy()
        ds = np.where(np.isfinite(ds) & (ds != 0), np.sign(ds), 1).astype(np.int64)
        lc = lc * ds

    df["lc"] = np.clip(lc, -1, 1).astype(np.int8)
    return df