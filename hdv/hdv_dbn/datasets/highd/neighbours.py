import numpy as np
import pandas as pd

_REL_SUFFIXES = ("_dx", "_dy", "_dvx", "_dvy")

def _rel_to_exists_pairs(obs_names):
    """
    Build a mapping between neighbor-relative feature columns and their corresponding
    binary existence-mask columns.

    Parameters
    obs_names : Sequence[str]
        Feature names in the observation vector order.

    Returns
    pairs : List[Tuple[int, int]]
        List of (rel_feature_index, exists_feature_index) pairs for all
        relative features that have a matching "<prefix>_exists" feature.
    """
    name_to_idx = {n: i for i, n in enumerate(obs_names)}
    pairs = []
    for rel_idx, name in enumerate(obs_names):
        for suf in _REL_SUFFIXES:
            if name.endswith(suf):
                prefix = name[: -len(suf)]
                ex_name = f"{prefix}_exists"
                if ex_name in name_to_idx:
                    pairs.append((rel_idx, name_to_idx[ex_name]))
                break
    return pairs

def _merge_neighbor_state(df, neighbor_id_col, prefix):
    """
    Attach neighbor state (x,y,vx,vy,lane_id) for the vehicle referenced by neighbor_id_col.
    Adds:
      {prefix}_x_center, {prefix}_y_center, {prefix}_vx, {prefix}_vy, {prefix}_lane_id
    
    Parameters
    df : pd.DataFrame
        Input table containing at least:
          - `recording_id`, `frame`, `vehicle_id`
          - neighbor ID column specified by `neighbor_id_col`
          - x_center, y_center, vx, vy, lane_id columns
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
    lookup = df[["recording_id","frame","vehicle_id","x_center","y_center","vx","vy","lane_id"]].copy()
    lookup["vehicle_id"] = lookup["vehicle_id"].astype("Int64")
    lookup = lookup.rename(
        columns={
            "vehicle_id": neighbor_id_col,
            "x_center": f"{prefix}_x_center",
            "y_center": f"{prefix}_y_center",
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

def add_direction_aware_context_features(df):
    """
    Build direction-consistent relative context features from neighbor IDs.
    Adds, for each neighbor slot:
      - dx = neighbor_x_center - ego_x_center
      - dy = neighbor_y_center - ego_y_center
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

    # Ensure ego centers exist 
    if "x_center" not in out.columns and "x" in out.columns and "width" in out.columns:
        out["x_center"] = out["x"].astype(np.float64) + 0.5 * out["width"].astype(np.float64)
    if "y_center" not in out.columns and "y" in out.columns and "height" in out.columns:
        out["y_center"] = out["y"].astype(np.float64) + 0.5 * out["height"].astype(np.float64)

    for id_col, prefix in neighbor_specs:
        if id_col not in out.columns:
            continue

        # existence mask before merge
        nid = out[id_col].fillna(0).astype("Int64")
        out[f"{prefix}_exists"] = (nid > 0).astype(np.float64)
        out[id_col] = nid.mask(nid <= 0, pd.NA) # set missing neighbor IDs to NaN for merge

        out = _merge_neighbor_state(out, id_col, prefix)

        nx = f"{prefix}_x_center"
        ny = f"{prefix}_y_center"

        if nx in out.columns:
            out[f"{prefix}_exists"] = out[f"{prefix}_exists"] * (~out[nx].isna()).astype(np.float64)

        # relative features (NaN if neighbor missing)
        out[f"{prefix}_dx"]  = out[nx] - out["x_center"]
        out[f"{prefix}_dy"]  = out[ny] - out["y_center"]
        out[f"{prefix}_dvx"] = out[f"{prefix}_vx"] - out["vx"]
        out[f"{prefix}_dvy"] = out[f"{prefix}_vy"] - out["vy"]

        # direction-aware adjustments
        if "dir_sign" in out.columns:
            out[f"{prefix}_dx"] *= out["dir_sign"]
            out[f"{prefix}_dy"] *= out["dir_sign"]

    return out


def add_front_thw_ttc_dhw_from_tracks(df):
    """
    Rename highD tracks.csv front-interaction metrics to standardized names.

    Renames (if present):
        thw -> front_thw
        ttc -> front_ttc
        dhw -> front_dhw

    Does not create new data or modify values.
    """
    out = df.copy()

    rename_map = {
        "thw": "front_thw",
        "ttc": "front_ttc",
        "dhw": "front_dhw",
    }

    existing = {k: v for k, v in rename_map.items() if k in out.columns} 
    if existing:
        out = out.rename(columns=existing) # rename only existing columns

    return out

def add_ego_speed_and_jerk(df, dt=1.0/25.0):
    """
    Add:
        - speed  = sqrt(vx^2 + vy^2)
        - jerk_x = (ax[t] - ax[t-1]) /dt computed groupwise per (recording_id, vehicle_id)
        - jerk_y = d(ay)/dt

    Conventions:
        - speed is NaN if vx or vy is missing
        - jerk_x = 0.0 for the first frame of each vehicle
        - jerk_x = NaN if ax[t] or ax[t-1] is NaN
    """
    out = df.copy()

    # speed
    if ("vx" in out.columns) and ("vy" in out.columns):
        vx = out["vx"].to_numpy(dtype=np.float64, copy=False)
        vy = out["vy"].to_numpy(dtype=np.float64, copy=False)
        out["speed"] = np.sqrt(vx * vx + vy * vy)
    else:
        out["speed"] = np.nan

    out = out.sort_values(["recording_id", "vehicle_id", "frame"], kind="mergesort") # sorted so that all frames of a vehicle in a recording are contiguous and in time order
    rec = out["recording_id"].to_numpy(copy=False)
    vid = out["vehicle_id"].to_numpy(copy=False)

    same = (rec == np.roll(rec, 1)) & (vid == np.roll(vid, 1))
    same[0] = False  # prevent wrap-around artifact

    if dt is None or dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")

    # jerk (groupwise)
    for a_col, j_col in (("ax", "jerk_x"), ("ay", "jerk_y")):
        if a_col in out.columns:
            a = out[a_col].to_numpy(dtype=np.float64, copy=False)

            jerk = np.full_like(a, np.nan, dtype=np.float64)

            idx = np.where(same)[0] # valid indices where we can compute jerk
            a_now = a[idx]
            a_prev = a[idx - 1]

            ok = np.isfinite(a_now) & np.isfinite(a_prev)
            jerk[idx[ok]] = (a_now[ok] - a_prev[ok]) / dt

            # first frame per trajectory
            jerk[~same] = 0.0

            out[j_col] = jerk
        else:
            out[j_col] = 0.0 

    return out


def add_neighbor_exists_flags(df):
    """
    Add only neighbor existence flags from neighbor ID columns.
    Produces:
      front_exists, rear_exists,
      left_front_exists, left_side_exists, left_rear_exists,
      right_front_exists, right_side_exists, right_rear_exists
    """
    specs = [
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
    for id_col, prefix in specs:
        if id_col in out.columns:
            nid = out[id_col].fillna(0)
            out[f"{prefix}_exists"] = (nid.astype("Int64") > 0).astype(np.float64)
        else:
            out[f"{prefix}_exists"] = 0.0
    return out
