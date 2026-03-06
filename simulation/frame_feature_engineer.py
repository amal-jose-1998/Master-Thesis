import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from hdv.hdv_dbn.config import FRAME_FEATURE_COLS
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


@dataclass(frozen=True)
class FrameContext:
    """
    Cached per-frame numpy views to avoid rebuilding arrays/dicts for each ego vehicle.
    """
    df: pd.DataFrame
    veh_col: str
    vid: np.ndarray                 # (N,)
    id_to_idx: Dict[int, int]       # vehicle_id -> row index
    xs: np.ndarray
    ys: np.ndarray
    ws: np.ndarray
    hs: np.ndarray
    xcs: np.ndarray
    ycs: np.ndarray
    vxs: np.ndarray
    vys: np.ndarray


@dataclass
class OnlineFrameFeatureEngineer:
    """
    Online frame feature engineering for a single vehicle's trajectory.
    - Maintains internal state to compute features that require temporal context (e.g., jerk, lane change flags).
    - Output is a single frame vector ordered as FRAME_FEATURE_COLS.
    - Can apply vehicle-centric normalization on the fly.
    """
    apply_vehicle_centric: bool = True
    flip_lateral: bool = True
    flip_positions: bool = False
    dt: float = 1.0 / 25.0

    def __post_init__(self):
        self._prev_state: Dict[Tuple[int, int], Dict[str, float]] = {}  # (recording_id, vehicle_id) -> state
        self._tracks_meta_token: Optional[int] = None
        self._veh_to_dir: Dict[int, int] = {}

        self._rec_meta_token: Optional[int] = None
        self._upper_marks: np.ndarray = np.array([], dtype=np.float64)
        self._lower_marks: np.ndarray = np.array([], dtype=np.float64)

    def reset(self):
        self._prev_state.clear()

    # ---------------------------------------------------------------------
    # Caching helpers
    # ---------------------------------------------------------------------
    def _build_tracks_meta_cache(self, tracks_meta_df: pd.DataFrame):
        token = id(tracks_meta_df)
        if self._tracks_meta_token == token:
            return

        df = tracks_meta_df
        if "id" in df.columns and "vehicle_id" not in df.columns:
            df = df.rename(columns={"id": "vehicle_id"})

        veh = df["vehicle_id"].to_numpy(dtype=np.int64, copy=False)

        if "drivingDirection" in df.columns:
            dd = df["drivingDirection"].to_numpy(copy=False)
            self._veh_to_dir = {int(v): int(d) for v, d in zip(veh, dd)}
        else:
            self._veh_to_dir = {}

        self._tracks_meta_token = token

    def _build_lane_markings_cache(self, recording_meta: dict):
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
    # Standardization
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
            return raw_frame_df.rename(columns=HIGHD_COL_MAP)
        return raw_frame_df

    # ---------------------------------------------------------------------
    # New two-step API for multi-ego efficiency
    # ---------------------------------------------------------------------
    def prepare_frame(self, raw_frame_df, tracks_meta_df, recording_meta: dict):
        """
        Build cached numpy views for this frame (do once per frame).
        """
        self._build_tracks_meta_cache(tracks_meta_df)
        self._build_lane_markings_cache(recording_meta)

        df = self._standardize_frame_df(raw_frame_df)

        veh_col = "vehicle_id" if "vehicle_id" in df.columns else "id"
        vid = df[veh_col].to_numpy(dtype=np.int64, copy=False)
        id_to_idx = {int(v): i for i, v in enumerate(vid)}

        xs = df.get("x", pd.Series(np.nan, index=df.index)).to_numpy(dtype=np.float64, copy=False)
        ys = df.get("y", pd.Series(np.nan, index=df.index)).to_numpy(dtype=np.float64, copy=False)
        ws = df.get("width", pd.Series(np.nan, index=df.index)).to_numpy(dtype=np.float64, copy=False)
        hs = df.get("height", pd.Series(np.nan, index=df.index)).to_numpy(dtype=np.float64, copy=False)
        vxs = df.get("vx", pd.Series(np.nan, index=df.index)).to_numpy(dtype=np.float64, copy=False)
        vys = df.get("vy", pd.Series(np.nan, index=df.index)).to_numpy(dtype=np.float64, copy=False)

        xcs = xs + 0.5 * ws
        ycs = ys + 0.5 * hs

        return FrameContext(
            df=df,
            veh_col=veh_col,
            vid=vid,
            id_to_idx=id_to_idx,
            xs=xs, ys=ys, ws=ws, hs=hs,
            xcs=xcs, ycs=ycs,
            vxs=vxs, vys=vys
        )

    def compute_ego(self, ctx: FrameContext, ego_vehicle_id, *, recording_id):
        """
        Compute one ego vehicle's engineered frame vector using cached ctx.
        Returns np.float64 vector ordered as FRAME_FEATURE_COLS.
        """
        ego_vehicle_id = int(ego_vehicle_id)
        ego_i = ctx.id_to_idx.get(ego_vehicle_id, None)
        if ego_i is None:
            raise ValueError(f"ego_vehicle_id={ego_vehicle_id} not present in frame")

        ego_row = ctx.df.iloc[int(ego_i)]

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

        driving_dir = int(self._veh_to_dir.get(ego_vehicle_id, 0))
        dir_sign = _dir_sign_from_driving_direction(driving_dir)

        # vehicle-centric normalization (keeps behavior consistent with your current add_frame)
        if self.apply_vehicle_centric:
            if np.isfinite(vx):
                vx *= dir_sign
            if np.isfinite(ax):
                ax *= dir_sign
            if self.flip_lateral:
                if np.isfinite(vy):
                    vy *= dir_sign
                if np.isfinite(ay):
                    ay *= dir_sign
            if self.flip_positions:
                if np.isfinite(x):
                    x *= dir_sign
                if np.isfinite(y):
                    y *= dir_sign
                if np.isfinite(x_center):
                    x_center *= dir_sign
                if np.isfinite(y_center):
                    y_center *= dir_sign

        d_left_lane, d_right_lane = _lane_boundary_distances(
            y_center, driving_dir, self._upper_marks, self._lower_marks
        )

        front_thw = float(ego_row.get("front_thw", ego_row.get("thw", np.nan)))
        front_ttc = float(ego_row.get("front_ttc", ego_row.get("ttc", np.nan)))
        front_dhw = float(ego_row.get("front_dhw", ego_row.get("dhw", np.nan)))

        # jerk + lane-change (online state)
        key = (int(recording_id), ego_vehicle_id)
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
                lc = int(np.sign(lane_id - prev_lane) * np.sign(dir_sign))
            else:
                lc = 0

        lc = int(np.clip(lc, -1, 1))
        self._prev_state[key] = {"ax": ax, "ay": ay, "lane_id": lane_id, "dir_sign": dir_sign}

        # initialize feature dict
        feat = {name: np.nan for name in FRAME_FEATURE_COLS}
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

        # neighbor arrays (apply ego’s dir_sign like before)
        vxs = ctx.vxs * dir_sign if self.apply_vehicle_centric else ctx.vxs
        if self.apply_vehicle_centric and self.flip_lateral:
            vys = ctx.vys * dir_sign
        else:
            vys = ctx.vys

        ex0 = float(ctx.xcs[ego_i]) if np.isfinite(ctx.xcs[ego_i]) else float(x_center)
        ey0 = float(ctx.ycs[ego_i]) if np.isfinite(ctx.ycs[ego_i]) else float(y_center)

        for id_col, prefix in _NEIGHBOR_SPECS:
            exists_name = f"{prefix}_exists"
            dx_name = f"{prefix}_dx"
            dy_name = f"{prefix}_dy"
            dvx_name = f"{prefix}_dvx"
            dvy_name = f"{prefix}_dvy"

            nid_raw = ego_row.get(id_col, 0)
            try:
                nid = int(nid_raw)
            except Exception:
                nid = 0

            if nid <= 0:
                feat[exists_name] = 0.0
                continue

            j = ctx.id_to_idx.get(nid, None)
            if j is None or (not np.isfinite(ctx.xcs[j])) or (not np.isfinite(ctx.ycs[j])):
                feat[exists_name] = 0.0
                continue

            feat[exists_name] = 1.0

            dx = float(ctx.xcs[j] - ex0) * float(dir_sign)
            dy = float(ctx.ycs[j] - ey0) * float(dir_sign)

            dvx = float(vxs[j] - vx) if np.isfinite(vxs[j]) and np.isfinite(vx) else np.nan
            dvy = float(vys[j] - vy) if np.isfinite(vys[j]) and np.isfinite(vy) else np.nan

            feat[dx_name] = dx
            feat[dy_name] = dy
            feat[dvx_name] = dvx
            feat[dvy_name] = dvy

        # ensure exists defaults are 0.0
        for _, prefix in _NEIGHBOR_SPECS:
            en = f"{prefix}_exists"
            if en in feat and not np.isfinite(feat[en]):
                feat[en] = 0.0

        return np.asarray([feat[c] for c in FRAME_FEATURE_COLS], dtype=np.float64)
