"""
Rule-based preprocessing to derive discrete maneuver labels from continuous trajectory data.

Expected input columns per row:
    - track_id : vehicle/track identifier
    - frame    : time/frame index (or something sortable in time)
    - ax       : longitudinal acceleration [m/s^2]
    - lane_id  : integer lane index (1, 2, 3, ...)
    - y        : lateral position [m] (for approximate lane center offset)
    - vy       : lateral velocity [m/s] (for lateral motion magnitude)

Output:
    - DataFrame with one extra column 'maneuver'
"""

from dataclasses import dataclass
from typing import List, Optional
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


def classify_maneuvers_for_vehicle(df_vehicle, lane_col="lane_id", accel_col="ax", lat_pos_col="y", lat_vel_col="vy"):
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
    pd.Series
        A Series of maneuver labels indexed like df_vehicle.index.
        Values are in DBN_STATES.maneuver_states.
    """
    n = len(df_vehicle) # number of time steps for this vehicle
    maneuvers = np.full(n, "maintain_speed", dtype=object) # default maneuver is assigned as "maintain_speed"

    has_ax = accel_col is not None and accel_col in df_vehicle.columns # check if acceleration column is available
    has_lane = lane_col is not None and lane_col in df_vehicle.columns # check if lane column is available
    has_y = lat_pos_col is not None and lat_pos_col in df_vehicle.columns # check if lateral position column is available
    has_vy = lat_vel_col is not None and lat_vel_col in df_vehicle.columns # check if lateral velocity column is available

    y = df_vehicle[lat_pos_col].to_numpy() if has_y else None # lateral position array or None
    vy = df_vehicle[lat_vel_col].to_numpy() if has_vy else None # lateral velocity array or None
    ax = df_vehicle[accel_col].to_numpy() if has_ax else None # longitudinal acceleration
    lane = df_vehicle[lane_col].to_numpy() if has_lane else None # lane indices

    # ------------------------------------------------------------------
    # Longitudinal maneuvers: hard_brake, decelerate, accelerate
    # ------------------------------------------------------------------
    # Hard brake (dominates decelerate)
    hard_brake_mask = ax <= OBS_CONFIG.hard_brake_threshold
    maneuvers[hard_brake_mask] = "hard_brake"

    # Decelerate (but not already hard_brake)
    decel_mask = (ax < -OBS_CONFIG.accel_threshold) & ~hard_brake_mask
    maneuvers[decel_mask] = "decelerate"

    # Accelerate
    accel_mask = ax > OBS_CONFIG.accel_threshold
    maneuvers[accel_mask] = "accelerate"

    # Everything else stays "maintain_speed" for now

    # ------------------------------------------------------------------
    # Lane-change maneuvers via lane_id transitions
    # ------------------------------------------------------------------
    if n > 1:
        lane_change_bool = lane[1:] != lane[:-1]
        lane_change_indices = np.where(lane_change_bool)[0] + 1  # shift by 1

        for idx in lane_change_indices:
            # Determine direction of lane change
            if lane[idx] > lane[idx - 1]:
                direction = "left"
            elif lane[idx] < lane[idx - 1]:
                direction = "right"
            else:
                continue  # degenerate case

            # Mark perform_lc_* for a short window starting at the change index
            start = idx
            end = min(idx + OBS_CONFIG.lc_frames, n)

            if direction == "left":
                maneuvers[start:end] = "perform_lc_left"
            else:
                maneuvers[start:end] = "perform_lc_right"

            # Preparation window before the lane change
            prep_start = max(0, idx - OBS_CONFIG.prep_frames)
            prep_end = idx  # up to but not including the perform index

            prep_label = "prepare_lc_left" if direction == "left" else "prepare_lc_right"

            for j in range(prep_start, prep_end):
                use_prepare = True

                if has_y:
                    center_j = _lane_center(lane[j:j+1], OBS_CONFIG.lane_width)[0]
                    # If still very close to lane center, maybe not preparing yet
                    if abs(y[j] - center_j) < OBS_CONFIG.prepare_lane_offset:
                        use_prepare = False

                if has_vy:
                    # If lateral velocity is very small, also less likely preparing
                    if abs(vy[j]) < OBS_CONFIG.lateral_vel_threshold:
                        # If we also have y, require at least offset OR lateral speed
                        if has_y:
                            center_j = _lane_center(lane[j:j+1], OBS_CONFIG.lane_width)[0]
                            if abs(y[j] - center_j) < OBS_CONFIG.prepare_lane_offset:
                                use_prepare = False
                        else:
                            use_prepare = False

                # Only overwrite if not already in a perform_lc_* state
                if use_prepare and not maneuvers[j].startswith("perform_lc_"):
                    maneuvers[j] = prep_label

    # ------------------------------------------------------------------
    # 3) Sanity: ensure all maneuvers are valid
    # ------------------------------------------------------------------
    valid = set(DBN_STATES.maneuver_states)
    invalid_mask = ~np.isin(maneuvers, list(valid))
    if invalid_mask.any():
        maneuvers[invalid_mask] = "maintain_speed"

    return pd.Series(maneuvers, index=df_vehicle.index, name="maneuver")


def add_maneuver_labels(df, vehicle_id_col="vehicle_id", time_col="frame", lane_col="lane_id", accel_col="ax", lat_pos_col="y", lat_vel_col="vy"):
    """
    Add a 'maneuver' column to the full dataframe by classifying each track.

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
        A copy of df with a new 'maneuver' column.
    """

    df = df.copy()
    df.sort_values([vehicle_id_col, time_col], inplace=True) # sort based on vehicle and time
    maneuver_series_list = [] # to collect per-vehicle maneuver Series
    for _, df_vehicle in df.groupby(vehicle_id_col): # process each vehicle
        df_vehicle = df_vehicle.sort_values(time_col) # ensure time-sorted
        m_labels = classify_maneuvers_for_vehicle(df_vehicle=df_vehicle, lane_col=lane_col, accel_col=accel_col, lat_pos_col=lat_pos_col, lat_vel_col=lat_vel_col) # get maneuver labels for this vehicle
        maneuver_series_list.append(m_labels) # collect the result
    maneuver_all = pd.concat(maneuver_series_list).sort_index() # combine and sort back to original order
    df["maneuver"] = maneuver_all # add maneuver column
    return df
