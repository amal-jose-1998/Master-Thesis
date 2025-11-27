"""
Rule-based preprocessing to derive discrete maneuver labels from continuous trajectory data.

Expected input columns per row:
    - vehicle_id : vehicle identifier
    - frame    : time/frame index (or something sortable in time)
    - ax       : longitudinal acceleration [m/s^2]
    - lane_id  : integer lane index (1, 2, 3, ...)
    - y        : lateral position [m] (for approximate lane center offset)
    - vy       : lateral velocity [m/s] (for lateral motion magnitude)

Output:
    - DataFrame with two extra columns:
        - maneuver_long : longitudinal maneuver label
        - maneuver_lat  : lateral / lane-change maneuver label
"""
import numpy as np
import pandas as pd

from .config import OBS_CONFIG, DBN_STATES


def _lane_center(lane_id, lane_width):
    """
    Approximate lane center position given lane_id (1,2,3,...) and lane width.
    Model: center_y = (lane_id - 0.5) * lane_width

    parameters
    lane_id : np.ndarray
        Array of lane indices (1,2,3,...)
    lane_width : float
        Width of each lane in meters.
    
    returns
    np.ndarray
        Array of lane center y-positions.
    """
    return (lane_id.astype(float) - 0.5) * lane_width


def classify_long_lat_maneuvers_for_vehicle(df_vehicle, lane_col="lane_id", accel_col="ax", lat_pos_col="y", lat_vel_col="vy"):
    """
    Classify maneuvers for a single vehicle.

    Parameters
    df_vehicle : pd.DataFrame
        Trajectory for one vehicle, already sorted by time/frame. Index preserved.
    lane_col : str
        Column name for lane index.
    accel_col : str
        Column name for longitudinal acceleration.
    lat_pos_col : str or None
        Column name for lateral position (if available).
    lat_vel_col : str or None
        Column name for lateral velocity (if available).

    Returns
    (pd.Series, pd.Series)
        - maneuver_long : longitudinal maneuver labels
        - maneuver_lat  : lateral / lane-change maneuver labels
    """
    n = len(df_vehicle) # number of time steps for this vehicle
    # default maneuver is assigned as "maintain_speed" and "keep_lane"
    long_m = np.full(n, "maintain_speed", dtype=object)
    lat_m = np.full(n, "keep_lane", dtype=object)
    # Require acceleration for longitudinal classification
    if accel_col is None or accel_col not in df_vehicle.columns:
        raise ValueError(
            f"Acceleration column '{accel_col}' is missing in df_vehicle; "
            "cannot derive longitudinal maneuvers."
        )
    has_lane = lane_col is not None and lane_col in df_vehicle.columns # check if lane column is available
    has_y = lat_pos_col is not None and lat_pos_col in df_vehicle.columns # check if lateral position column is available
    has_vy = lat_vel_col is not None and lat_vel_col in df_vehicle.columns # check if lateral velocity column is available
    y = df_vehicle[lat_pos_col].to_numpy() if has_y else None # lateral position array or None
    vy = df_vehicle[lat_vel_col].to_numpy() if has_vy else None # lateral velocity array or None
    ax = df_vehicle[accel_col].to_numpy() # longitudinal acceleration
    lane = df_vehicle[lane_col].to_numpy() if has_lane else None # lane indices
    # ------------------------------------------------------------------
    # Longitudinal maneuvers: hard_brake, decelerate, accelerate
    # ------------------------------------------------------------------
    # Hard brake (dominates decelerate)
    hard_brake_mask = ax <= OBS_CONFIG.hard_brake_threshold
    long_m[hard_brake_mask] = "hard_brake"
    # Decelerate (but not already hard_brake)
    decel_mask = (ax < -OBS_CONFIG.accel_threshold) & ~hard_brake_mask
    long_m[decel_mask] = "decelerate"
    # Accelerate
    accel_mask = ax > OBS_CONFIG.accel_threshold
    long_m[accel_mask] = "accelerate"
    # Everything else stays "maintain_speed" for now
    # ------------------------------------------------------------------
    # Lateral maneuvers via lane_id transitions
    # ------------------------------------------------------------------
    if n > 1 and lane is not None:
        lane_change_bool = lane[1:] != lane[:-1] # detect lane changes by comparing consecutive lane IDs
        lane_change_indices = np.where(lane_change_bool)[0] + 1  # get indices of lane changes (shift by 1 due to diff)
        for idx in lane_change_indices:
            old_lane = lane[idx - 1]
            new_lane = lane[idx]
            # Determine direction of lane change
            if new_lane > old_lane: # assume higher lane ID means left #TODO: check this assumption
                direction = "left"
            elif new_lane < old_lane: # lower lane ID means right
                direction = "right"
            else:
                continue  # should not happen
            # New lane center
            new_center = _lane_center(np.array([new_lane]), OBS_CONFIG.lane_width)[0]
            t = idx
            # Extend t forward until the vehicle is "settled" in new lane (near lane center and low lateral velocity).
            while t < n: # look ahead from lane change index
                # Position close to new lane center?
                if has_y:
                    close_to_center = abs(y[t] - new_center) < OBS_CONFIG.prepare_lc_offset # close enough to center of new lane such that the lc_prepare offset is not exceeded
                else:
                    close_to_center = True  # can't check, so don't stop on this criterion
                # Lateral velocity small?
                if has_vy:
                    small_vy = abs(vy[t]) < OBS_CONFIG.lateral_vel_threshold
                else:
                    small_vy = True
                # Once both are true after we have entered the new lane, we consider LC done
                if (t > idx) and close_to_center and small_vy:
                    break
                t += 1
            # t is now the first "settled" frame, or n if never settled
            perform_start = idx
            perform_end = t # exclusive. 
            # Mark the perform_lc_* maneuver over this interval
            lat_m[perform_start:perform_end] = f"perform_lc_{direction}"
            # Preparation window before the lane change
            prep_start = max(0, perform_start - OBS_CONFIG.lc_prep_frames)
            prep_end = perform_start  # up to but not including
            prep_label = "prepare_lc_left" if direction == "left" else "prepare_lc_right"
            for j in range(prep_start, prep_end):
                use_prepare = True
                if has_y:
                    # Distance from old lane center at frame j
                    old_center_j = _lane_center(np.array([old_lane]), OBS_CONFIG.lane_width)[0] 
                    offset = abs(y[j] - old_center_j)
                else: # no y available
                    offset = None
                speed = abs(vy[j]) if has_vy else None
                if has_y and has_vy:
                    # lc_prepare if either clearly off center OR has noticeable lateral speed
                    use_prepare = (offset >= OBS_CONFIG.prepare_lc_offset) or (speed >= OBS_CONFIG.lateral_vel_threshold)
                elif has_y:
                    use_prepare = offset >= OBS_CONFIG.prepare_lc_offset
                elif has_vy:
                    use_prepare = speed >= OBS_CONFIG.lateral_vel_threshold
                else:
                    # no lateral info at all -> we can't reliably say "prepare"
                    use_prepare = False
                # Only overwrite if not already in a perform_lc_* state
                if use_prepare and not str(lat_m[j]).startswith("perform_lc_"):
                    lat_m[j] = prep_label
    else:
        # No lane data or single time step: cannot detect lane changes
        #print("Warning: insufficient lane data to classify lane-change maneuvers.")
        pass
    # ------------------------------------------------------------------
    # Sanity: ensure all maneuvers are valid
    # ------------------------------------------------------------------
    # valid maneuver labels
    long_valid = set(DBN_STATES.long_maneuver_states)
    lat_valid = set(DBN_STATES.lat_maneuver_states) 
    # detect any invalid labels
    long_invalid = ~np.isin(long_m, list(long_valid))
    lat_invalid = ~np.isin(lat_m, list(lat_valid))
    if long_invalid.any():
        print("Warning: invalid longitudinal maneuver labels found; resetting to 'maintain_speed'.")
        long_m[long_invalid] = "maintain_speed"

    if lat_invalid.any():
        print("Warning: invalid lateral maneuver labels found; resetting to 'keep_lane'.")
        lat_m[lat_invalid] = "keep_lane"

    long_series = pd.Series(long_m, index=df_vehicle.index, name="maneuver_long")
    lat_series = pd.Series(lat_m, index=df_vehicle.index, name="maneuver_lat")

    return long_series, lat_series


def add_maneuver_labels(df, vehicle_id_col="vehicle_id", time_col="frame", lane_col="lane_id", accel_col="ax", lat_pos_col="y", lat_vel_col="vy"):
    """
    Add maneuver labels to the full dataframe by classifying each track

    Parameters
    df : pd.DataFrame
        Full dataset containing many vehicles and time steps.
    vehicle_id_col : str
        Name of the column identifying each vehicle.
    time_col : str
        Name of the column used to sort time within each vehicle.
    lane_col : str
        Name of the column indicating the lane.
    accel_col : str
        Name of the column indicating longitudinal acceleration.
    lat_pos_col : str or None
        Name of the column for lateral position (if available).
    lat_vel_col : str or None
        Name of the column for lateral velocity (if available).

    Returns
    pd.DataFrame
        A copy of df with new columns:
            - maneuver_long
            - maneuver_lat
    """
    df = df.copy()
    df.sort_values([vehicle_id_col, time_col], inplace=True) # sort based on vehicle and time
    # to collect per-vehicle maneuver Series
    long_list = []
    lat_list = []
    for _, df_vehicle in df.groupby(vehicle_id_col): # process each vehicle
        df_vehicle = df_vehicle.sort_values(time_col) # ensure time-sorted
        m_long, m_lat = classify_long_lat_maneuvers_for_vehicle(df_vehicle=df_vehicle, lane_col=lane_col, accel_col=accel_col, lat_pos_col=lat_pos_col, lat_vel_col=lat_vel_col) # get maneuver labels for this vehicle
        # collect the result
        long_list.append(m_long)
        lat_list.append(m_lat)
    # combine and sort back to original order
    maneuver_long_all = pd.concat(long_list).sort_index()
    maneuver_lat_all = pd.concat(lat_list).sort_index()
    df["maneuver_long"] = maneuver_long_all
    df["maneuver_lat"] = maneuver_lat_all
    return df
