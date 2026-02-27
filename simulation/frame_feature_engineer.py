import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from hdv.hdv_dbn.config import FRAME_FEATURE_COLS, META_COLS
from hdv.hdv_dbn.datasets.highd.io import HIGHD_COL_MAP

# Neighbor slots used by training pipeline (same names as in HIGHD_COL_MAP/neighbours.py)
_NEIGHBOR_SPECS = [
    ("preceding_id",       "front"),
    ("following_id",       "rear"),
    ("left_preceding_id",  "left_front"),
    ("left_alongside_id",  "left_side"),
    ("left_following_id",  "left_rear"),
    ("right_preceding_id", "right_front"),
    ("right_alongside_id", "right_side"),
    ("right_following_id", "right_rear"),
]

def _dir_sign_from_driving_direction(driving_dir):
    """
    Match normalize_vehicle_centric() sign convention:
      - if drivingDirection is {1,2}: sign = +1 for 2, -1 for 1
      - fallback: +1
    """
    try:
        d = int(driving_dir)
    except Exception:
        return 1.0
    if d == 2:
        return 1.0
    if d == 1:
        return -1.0
    return 1.0

def _parse_lane_markings(x):
    if x is None:
        return np.array([], dtype=np.float64)

    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x, dtype=np.float64).ravel()
        return arr[np.isfinite(arr)]

    return np.array([], dtype=np.float64)

def _lane_boundary_distances(y, driving_dir, upper_marks, lower_marks):
    """
    Ego-only version of add_lane_boundary_distance_features() logic.
    Returns (d_left_lane, d_right_lane) in vehicle-centric sense.
    """
    if not np.isfinite(y):
        return (np.nan, np.nan)

    if driving_dir == 1:
        marks = upper_marks
    elif driving_dir == 2:
        marks = lower_marks
    else:
        return (np.nan, np.nan)

    if marks.size < 2:
        return (np.nan, np.nan)

    # index of the lane marking just to the left of the vehicle's y position in world coordinates, if driving_dir==2; 
    # or just to the right if driving_dir==1 (since lane markings are flipped in that case)
    j = int(np.searchsorted(marks, y, side="right") - 1) 
    if j < 0 or j >= marks.size - 1:
        return (np.nan, np.nan)

    left_world = float(marks[j])
    right_world = float(marks[j + 1])

    # Match lanes.py convention:
    # - for direction==2: smaller world-y = vehicle-left
    # - for direction==1: smaller world-y = vehicle-right (swap)
    if driving_dir == 2:
        d_left = y - left_world
        d_right = right_world - y
    else:
        d_left = right_world - y
        d_right = y - left_world

    return (float(d_left), float(d_right))

@dataclass
class OnlineFrameFeatureEngineer:
    """
    Online frame feature engineering for a single vehicle's trajectory.
    - Maintains internal state to compute features that require temporal context (e.g., jerk, lane change flags).
    - Designed to be called sequentially for each new frame of the vehicle, updating internal state as needed.
    - Output of each call is a single frame vector with engineered features, ordered as FRAME_FEATURE_COLS.
    - Requires tracks_meta_df and recording_meta_df for the vehicle to compute certain features (e.g., lane boundaries).
    - Can apply vehicle-centric normalization and lane-relative features on the fly.    
    """
    apply_vehicle_centric: bool = True
    flip_lateral: bool = True
    flip_positions: bool = False
    dt: float = 1.0 / 25.0 # time step between frames, used for computing jerk

    def __post_init__(self):
        self._prev_state: Dict[Tuple[int, int], Dict[str, float]] = {} # (recording_id, vehicle_id) -> {"ax":..., "ay":..., "lane_id":..., "dir_sign":...}
        # cache derived from tracks_meta_df (recomputed if df object changes)
        self._tracks_meta_token: Optional[int] = None
        self._veh_to_dir: Dict[int, int] = {}
        self._veh_to_class: Dict[int, str] = {}

        # lane markings cache (recomputed if recording_meta dict identity changes)
        self._rec_meta_token: Optional[int] = None
        self._upper_marks: np.ndarray = np.array([], dtype=np.float64)
        self._lower_marks: np.ndarray = np.array([], dtype=np.float64)

    def reset(self):
        self._prev_state.clear()

    # ---------------------------------------------------------------------
    # Caching helpers
    # ---------------------------------------------------------------------
    def _build_tracks_meta_cache(self, tracks_meta_df: pd.DataFrame):
        token = id(tracks_meta_df) # id-based token to detect if the DataFrame object has changed since last cache build
        if self._tracks_meta_token == token: # cache is still the same
            return

        df = tracks_meta_df
        if "id" in df.columns and "vehicle_id" not in df.columns:
            df = df.rename(columns={"id": "vehicle_id"})

        veh = df["vehicle_id"].to_numpy(dtype=np.int64, copy=False)

        if "drivingDirection" in df.columns:
            dd = df["drivingDirection"].to_numpy(copy=False)
            self._veh_to_dir = {int(v): int(d) for v, d in zip(veh, dd)} # dict mapping vehicle_id to drivingDirection for all vehicles in the tracks_meta_df
        else:
            self._veh_to_dir = {}

        if "class" in df.columns:
            cls = df["class"].astype(str).str.lower().to_numpy(copy=False)
            self._veh_to_class = {int(v): str(c) for v, c in zip(veh, cls)} # dict mapping vehicle_id to class for all vehicles in the tracks_meta_df
        else:
            self._veh_to_class = {}

        self._tracks_meta_token = token # update token to current DataFrame object


    def _build_lane_markings_cache(self, recording_meta: dict):
        """
        Builds lane markings cache from recording_meta dict, which should contain 'upperLaneMarkings' and 'lowerLaneMarkings' as lists of floats.
        Uses the identity of the recording_meta dict to determine if the cache needs to be updated.
        """
        if recording_meta is None:
            self._upper_marks = np.array([], dtype=np.float64)
            self._lower_marks = np.array([], dtype=np.float64)
            self._rec_meta_token = None
            return

        token = id(recording_meta)
        if self._rec_meta_token == token:
            return
        
        up = recording_meta.get("upperLaneMarkings", None)
        lo = recording_meta.get("lowerLaneMarkings", None)
        
        self._upper_marks = _parse_lane_markings(up)
        self._lower_marks = _parse_lane_markings(lo)
        self._rec_meta_token = token

     # ---------------------------------------------------------------------
    # Core per-frame extraction
    # ---------------------------------------------------------------------
    def _standardize_frame_df(self, raw_frame_df: pd.DataFrame):
        """
        Accept either:
          - already-standardized columns (vehicle_id/vx/ax/...)
          - raw highD columns (id/xVelocity/xAcceleration/...)
        """
        if "vehicle_id" in raw_frame_df.columns:
            return raw_frame_df
        if "id" in raw_frame_df.columns:
            # very small rename; avoids full pipeline
            return raw_frame_df.rename(columns=HIGHD_COL_MAP)
        return raw_frame_df  # if columns don't match expected formats, just pass through and let downstream code handle missing columns as needed

    def _get_row_by_vehicle_id(self, df: pd.DataFrame, ego_vehicle_id):
        col = "vehicle_id" if "vehicle_id" in df.columns else "id"
        m = (df[col].to_numpy() == ego_vehicle_id)
        if not np.any(m):
            raise ValueError(f"ego_vehicle_id={ego_vehicle_id} not present in frame")
        return df.loc[df.index[np.argmax(m)]] # row of the ego vehicle in the current frame, with standardized column names


    def add_frame(self, raw_frame_df: pd.DataFrame, ego_vehicle_id, tracks_meta_df: pd.DataFrame, recording_meta: dict, recording_id):
        self._build_tracks_meta_cache(tracks_meta_df) # ensure tracks_meta cache is up to date for this frame's DataFrame object
        self._build_lane_markings_cache(recording_meta) # ensure lane markings cache is up to date for this frame's recording_meta dict

        df = self._standardize_frame_df(raw_frame_df) # standardize expected column names for downstream feature computations, if possible; otherwise downstream code will handle missing columns as needed

        # Build numpy views for fast lookup
        veh_col = "vehicle_id" if "vehicle_id" in df.columns else "id"
        vid = df[veh_col].to_numpy(dtype=np.int64, copy=False) # vehicle_id column as NumPy array for fast lookup of neighbor rows by vehicle_id

        # ego
        ego_row = self._get_row_by_vehicle_id(df, int(ego_vehicle_id))
        x = float(ego_row.get("x", np.nan))
        y = float(ego_row.get("y", np.nan))
        w = float(ego_row.get("width", np.nan))
        h = float(ego_row.get("height", np.nan))

        x_center = x + 0.5 * w if np.isfinite(x) and np.isfinite(w) else x
        y_center = y + 0.5 * h if np.isfinite(y) and np.isfinite(h) else y

        vx = float(ego_row.get("vx", np.nan))
        vy = float(ego_row.get("vy", np.nan))
        ax = float(ego_row.get("ax", np.nan))
        ay = float(ego_row.get("ay", np.nan))
        lane_id = float(ego_row.get("lane_id", np.nan))

        driving_dir = int(self._veh_to_dir.get(int(ego_vehicle_id), 0))  # 1/2 expected
        dir_sign = _dir_sign_from_driving_direction(driving_dir)

        # Apply vehicle-centric normalization (match normalize_vehicle_centric() behavior)
        if self.apply_vehicle_centric:
            if np.isfinite(vx): vx = vx * dir_sign
            if np.isfinite(ax): ax = ax * dir_sign
            if self.flip_lateral:
                if np.isfinite(vy): vy = vy * dir_sign
                if np.isfinite(ay): ay = ay * dir_sign
            if self.flip_positions:
                if np.isfinite(x): x = x * dir_sign
                if np.isfinite(y): y = y * dir_sign
                if np.isfinite(x_center): x_center = x_center * dir_sign
                if np.isfinite(y_center): y_center = y_center * dir_sign

        # Lane boundary distances (ego-only)
        d_left_lane, d_right_lane = _lane_boundary_distances(
            y_center, driving_dir, self._upper_marks, self._lower_marks
        )

        # Rename risk metrics if still raw: thw/ttc/dhw -> front_thw/front_ttc/front_dhw
        front_thw = float(ego_row.get("front_thw", ego_row.get("thw", np.nan)))
        front_ttc = float(ego_row.get("front_ttc", ego_row.get("ttc", np.nan)))
        front_dhw = float(ego_row.get("front_dhw", ego_row.get("dhw", np.nan)))

        # Jerk + lane-change from prev ego state (cheap, online)
        key = (int(recording_id), int(ego_vehicle_id))
        prev = self._prev_state.get(key)

        if prev is None:
            jerk_x = 0.0
            jerk_y = 0.0
            lc = 0
        else:
            prev_ax = prev.get("ax", np.nan)
            prev_ay = prev.get("ay", np.nan)
            prev_lane = prev.get("lane_id", np.nan)
            jerk_x = (ax - prev_ax) / self.dt if np.isfinite(ax) and np.isfinite(prev_ax) else np.nan
            jerk_y = (ay - prev_ay) / self.dt if np.isfinite(ay) and np.isfinite(prev_ay) else np.nan
            if np.isfinite(lane_id) and np.isfinite(prev_lane):
                # match your earlier online convention: multiply by sign(dir_sign)
                lc = int(np.sign(lane_id - prev_lane) * np.sign(dir_sign))
            else:
                lc = 0

        lc = int(np.clip(lc, -1, 1))
        self._prev_state[key] = {"ax": ax, "ay": ay, "lane_id": lane_id, "dir_sign": dir_sign}

        # -----------------------------------------------------------------
        # Neighbor-relative features for the 8 slots (ego-only)
        # -----------------------------------------------------------------
        # Default: exists=0, rel features NaN
        feat = {name: np.nan for name in FRAME_FEATURE_COLS}

        # Fill core ego + lane/risk + online temporal
        feat.update({
            "vx": vx, "vy": vy, "ax": ax, "ay": ay,
            "jerk_x": jerk_x, "jerk_y": jerk_y,
            "lc": float(lc),
            "d_left_lane": d_left_lane,
            "d_right_lane": d_right_lane,
            "front_thw": front_thw,
            "front_ttc": front_ttc,
            "front_dhw": front_dhw,
        })

        # Build id->index for this frame once
        id_to_idx = {int(v): i for i, v in enumerate(vid)}

        # For neighbor computations we need neighbor centers + velocities (raw columns may exist)
        xs = df.get("x", pd.Series(np.nan, index=df.index)).to_numpy(dtype=np.float64, copy=False) 
        ys = df.get("y", pd.Series(np.nan, index=df.index)).to_numpy(dtype=np.float64, copy=False)
        ws = df.get("width", pd.Series(np.nan, index=df.index)).to_numpy(dtype=np.float64, copy=False)
        hs = df.get("height", pd.Series(np.nan, index=df.index)).to_numpy(dtype=np.float64, copy=False)
        vxs = df.get("vx", pd.Series(np.nan, index=df.index)).to_numpy(dtype=np.float64, copy=False)
        vys = df.get("vy", pd.Series(np.nan, index=df.index)).to_numpy(dtype=np.float64, copy=False)

        xcs = xs + 0.5 * ws 
        ycs = ys + 0.5 * hs

        # Apply same vehicle-centric flip to neighbor velocities (positions not flipped in training by default)
        if self.apply_vehicle_centric:
            vxs = vxs * dir_sign # match normalize_vehicle_centric() convention
            if self.flip_lateral:
                vys = vys * dir_sign # match normalize_vehicle_centric() convention

        # Extract ego index (for robustness, not reusing ego_row index)
        ego_i = id_to_idx.get(int(ego_vehicle_id), None)
        if ego_i is None:
            # should not happen due to earlier check
            raise ValueError(f"ego_vehicle_id={ego_vehicle_id} not found in id_to_idx")

        ex0 = float(xcs[ego_i]) if np.isfinite(xcs[ego_i]) else float(x_center) # ego center x
        ey0 = float(ycs[ego_i]) if np.isfinite(ycs[ego_i]) else float(y_center) # ego center y

        for id_col, prefix in _NEIGHBOR_SPECS:
            # existence feature name in FRAME_FEATURE_COLS: f"{prefix}_exists"
            exists_name = f"{prefix}_exists"
            dx_name = f"{prefix}_dx"
            dy_name = f"{prefix}_dy"
            dvx_name = f"{prefix}_dvx"
            dvy_name = f"{prefix}_dvy"

            nid_raw = ego_row.get(id_col, 0) # neighbor vehicle_id from ego row for this slot; defaults to 0 (non-existing) if column/value is missing/invalid
            try:
                nid = int(nid_raw)
            except Exception:
                nid = 0

            if nid <= 0:
                feat[exists_name] = 0.0
                continue

            j = id_to_idx.get(nid, None)
            if j is None:
                feat[exists_name] = 0.0
                continue

            # neighbor must have a finite center to be considered existing (matches neighbours.py check)
            if not np.isfinite(xcs[j]) or not np.isfinite(ycs[j]):
                feat[exists_name] = 0.0
                continue

            feat[exists_name] = 1.0

            # relative positions in ego-centered frame
            dx = float(xcs[j] - ex0)
            dy = float(ycs[j] - ey0)

            # direction-aware adjustment (matches add_direction_aware_context_features: dx,dy *= dir_sign)
            dx *= float(dir_sign)
            dy *= float(dir_sign)

            # relative velocities in (already) vehicle-centric kinematics
            dvx = float(vxs[j] - vx) if np.isfinite(vxs[j]) and np.isfinite(vx) else np.nan
            dvy = float(vys[j] - vy) if np.isfinite(vys[j]) and np.isfinite(vy) else np.nan

            feat[dx_name] = dx
            feat[dy_name] = dy
            feat[dvx_name] = dvx
            feat[dvy_name] = dvy

        # Ensure all *_exists that are in FRAME_FEATURE_COLS but not set default to 0.0 (not NaN)
        for _, prefix in _NEIGHBOR_SPECS:
            en = f"{prefix}_exists"
            if en in feat and not np.isfinite(feat[en]):
                feat[en] = 0.0

        # Return in canonical order
        return np.asarray([feat[c] for c in FRAME_FEATURE_COLS], dtype=np.float64)