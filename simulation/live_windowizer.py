import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass
from typing import Any

from hdv.hdv_dbn.config import FRAME_FEATURE_COLS, WINDOW_FEATURE_COLS, META_COLS, WINDOW_CONFIG
from hdv.hdv_dbn.datasets.highd.io import HIGHD_COL_MAP
from hdv.hdv_dbn.datasets.highd.normalise import normalize_vehicle_centric
from hdv.hdv_dbn.datasets.highd.lanes import add_lane_boundary_distance_features, add_lane_change_feature
from hdv.hdv_dbn.datasets.highd.neighbours import (
    add_direction_aware_context_features,
    add_front_thw_ttc_dhw_from_tracks,
    add_ego_speed_and_jerk,
)
from hdv.hdv_dbn.datasets.highd.sequences import (
    _compute_kinematics,
    _compute_jerk,
    _compute_lane_change_flags,
    _compute_lane_boundaries,
    _compute_risk,
    _compute_existence_fracs,
    _compute_neighbor_rel,
    _validate_required_columns,
    prune_columns,
)


def _serialize_lane_markings(value):
    """Serialize lane markings list/dict into a string for easier merging into frame features. Returns None if input is None."""
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return ";".join(str(v) for v in value)
    return str(value)


def ensure_columns(df: pd.DataFrame, cols, *, fill_value=np.nan):
    """Ensure `df` contains each column in `cols`. Missing columns are created with `fill_value`."""
    out = df
    for c in cols:
        if c not in out.columns:
            out[c] = fill_value
    return out


def engineer_frame_features(df: pd.DataFrame, *, tracks_meta_df=None, recording_meta=None, recording_id=None, apply_vehicle_centric=True, 
                            flip_lateral=True, flip_positions=False, strict=True, verbose=True):
    """
    Engineer *frame-level* features needed by LiveWindowizer from raw highD tracks.
    This function is designed to be usable both:
      - batch mode (entire vehicle trajectory), and
      - streaming mode (rolling buffer of last K frames).
      - Output contains at least all `FRAME_FEATURE_COLS` (added as NaN if missing unless strict=True).
      - Output is pruned to `FRAME_FEATURE_COLS + META_COLS` to keep it lightweight.
    """
    out = df.rename(columns=HIGHD_COL_MAP).copy()

    if recording_id is not None:
        out["recording_id"] = int(recording_id)

    # Merge per-vehicle metadata (driving direction, class)
    if tracks_meta_df is not None:
        meta = tracks_meta_df.copy()
        if "id" in meta.columns and "vehicle_id" not in meta.columns:
            meta = meta.rename(columns={"id": "vehicle_id"})

        tracksmeta_cols = [c for c in meta.columns if c != "vehicle_id"]
        meta = meta[["vehicle_id"] + tracksmeta_cols].copy()
        meta = meta.rename(columns={c: f"meta_{c}" for c in tracksmeta_cols})
        out = out.merge(meta, on="vehicle_id", how="left")

        if "meta_class" in out.columns:
            out["meta_class"] = out["meta_class"].astype(str).str.lower()

    # Add per-recording metadata (lane markings)
    if recording_meta is not None:
        upper = _serialize_lane_markings(
            recording_meta.get("upperLaneMarkings", recording_meta.get("lane_markings_upper"))
        )
        lower = _serialize_lane_markings(
            recording_meta.get("lowerLaneMarkings", recording_meta.get("lane_markings_lower"))
        )
        if upper is not None:
            out["rec_upperLaneMarkings"] = upper
        if lower is not None:
            out["rec_lowerLaneMarkings"] = lower

    # Centers
    if "width" in out.columns and "height" in out.columns:
        out["x_center"] = out["x"].astype(np.float64) + 0.5 * out["width"].astype(np.float64)
        out["y_center"] = out["y"].astype(np.float64) + 0.5 * out["height"].astype(np.float64)
    else:
        out["x_center"] = out["x"].astype(np.float64)
        out["y_center"] = out["y"].astype(np.float64)

    # Vehicle-centric normalization
    if apply_vehicle_centric:
        dir_col = "meta_drivingDirection" if "meta_drivingDirection" in out.columns else None
        if dir_col is not None:
            out = normalize_vehicle_centric(
                out,
                dir_col=dir_col,
                flip_longitudinal=True,
                flip_lateral=flip_lateral,
                flip_positions=flip_positions,
            )
        else:
            # fall back: treat as "forward"
            out["dir_sign"] = 1.0

    # ---- Engineer derived features (mirrors dataset pipeline)
    out = add_direction_aware_context_features(out)
    out = add_lane_boundary_distance_features(out, y_col="y_center")
    out = add_lane_change_feature(out, lane_col="lane_id", dir_sign_col="dir_sign")
    out = add_front_thw_ttc_dhw_from_tracks(out)
    out = add_ego_speed_and_jerk(out)

    # ---- Prune to required frame features (+ meta)
    out = prune_columns(out, feature_cols=FRAME_FEATURE_COLS, meta_cols=META_COLS)

    # ---- Ensure schema completeness
    missing = [c for c in FRAME_FEATURE_COLS if c not in out.columns]
    if missing:
        msg = f"[engineer_frame_features] Missing required frame features: {missing}"
        if strict:
            raise KeyError(msg)
        out = ensure_columns(out, missing, fill_value=np.nan)
        if verbose:
            print(msg + " (added as NaN)")
    else:
        if verbose:
            print("[engineer_frame_features] All required frame features present.")

    return out


@dataclass
class OnlineFrameFeatureEngineer:
    """
    True online feature engineer for a single vehicle stream.

    Stores a rolling buffer of raw frames and runs `engineer_frame_features`
    on that buffer. Returns the engineered `FRAME_FEATURE_COLS` vector for the newest frame.
    """
    tracks_meta_df: Any = None
    recording_meta: Any = None
    recording_id: Any = None
    buffer_len: int = 200
    apply_vehicle_centric: bool = True
    flip_lateral: bool = True
    flip_positions: bool = False
    strict: bool = False

    def __post_init__(self):
        self._raw_buf = None

    def reset(self):
        self._raw_buf = None

    def push_row(self, raw_row: dict) -> np.ndarray:
        """
        Push one raw highD row (dict-like) and get engineered frame vector.

        Returns:
          np.ndarray shape (len(FRAME_FEATURE_COLS),) dtype float64
        """
        new_row = pd.DataFrame([raw_row])

        if self._raw_buf is None:
            self._raw_buf = new_row
        else:
            self._raw_buf = pd.concat([self._raw_buf, new_row], ignore_index=True)

        if len(self._raw_buf) > self.buffer_len:
            self._raw_buf = self._raw_buf.iloc[-self.buffer_len:].reset_index(drop=True)

        feat_df = engineer_frame_features(
            self._raw_buf,
            tracks_meta_df=self.tracks_meta_df,
            recording_meta=self.recording_meta,
            recording_id=self.recording_id,
            apply_vehicle_centric=self.apply_vehicle_centric,
            flip_lateral=self.flip_lateral,
            flip_positions=self.flip_positions,
            strict=self.strict,
            verbose=False,
        )

        last = feat_df.iloc[-1]
        return last[list(FRAME_FEATURE_COLS)].to_numpy(dtype=np.float64)


class LiveWindowizer:
    """
    Online window feature computation from streaming *frame-level* features.

    - Feed engineered frame vectors (ordered as `obs_names`) using `add_frame`.
    - When enough frames exist and stride condition is met, outputs one window vector
      ordered as `WINDOW_FEATURE_COLS`.
    """
    def __init__(self, obs_names):
        self.W = int(WINDOW_CONFIG.W)
        self.stride = int(WINDOW_CONFIG.stride)
        self.obs_names = list(obs_names)

        self.buffer = deque(maxlen=self.W)
        self.frame_count = 0

        self.win_names = list(WINDOW_FEATURE_COLS)
        self.in_idx = {n: i for i, n in enumerate(self.obs_names)}
        self.out_idx = {n: i for i, n in enumerate(self.win_names)}

        # Validate required inputs for existence/relational features
        self.exists_cols, self.rel_tasks = _validate_required_columns(self.in_idx, self.out_idx)
        self.has_front_exists = "front_exists" in self.in_idx

    def add_frame(self, frame: np.ndarray):
        """
        Add one engineered frame vector.

        Args:
          frame: np.ndarray shape (len(obs_names),) dtype float64

        Returns:
          list[np.ndarray]: usually empty or one element with shape (len(WINDOW_FEATURE_COLS),)
        """
        if not isinstance(frame, np.ndarray) or frame.ndim != 1 or frame.dtype != np.float64:
            raise TypeError(
                "LiveWindowizer.add_frame expects a 1D NumPy array of dtype float64, ordered as obs_names."
            )
        if frame.shape[0] != len(self.obs_names):
            raise ValueError(
                f"Frame length {frame.shape[0]} does not match obs_names length {len(self.obs_names)}."
            )

        self.buffer.append(frame)
        self.frame_count += 1

        windows = []
        ready = (len(self.buffer) == self.W) and ((self.frame_count - self.W) % self.stride == 0)
        if not ready:
            return windows

        win = np.stack(self.buffer)  # (W, D_in)
        Y = np.full((1, len(self.win_names)), np.nan, dtype=np.float64)
        t = 0

        _compute_kinematics(win, Y, t, self.in_idx, self.out_idx)
        _compute_jerk(win, Y, t, self.in_idx, self.out_idx)
        _compute_lane_change_flags(win, Y, t, self.in_idx, self.out_idx)
        _compute_lane_boundaries(win, Y, t, self.in_idx, self.out_idx)
        _compute_risk(win, Y, t, self.in_idx, self.out_idx, self.has_front_exists)
        _compute_existence_fracs(win, Y, t, self.in_idx, self.out_idx, self.exists_cols)
        _compute_neighbor_rel(win, Y, t, self.rel_tasks)

        windows.append(Y[0])
        return windows

    def reset(self):
        self.buffer.clear()
        self.frame_count = 0