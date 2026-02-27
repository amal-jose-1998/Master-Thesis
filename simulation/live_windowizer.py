import numpy as np
from collections import deque

from hdv.hdv_dbn.config import WINDOW_FEATURE_COLS, WINDOW_CONFIG, FRAME_FEATURE_COLS
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


class LiveWindowizer:
    """
    Online window feature computation from streaming *frame-level* features.

    - Feed engineered frame vectors (ordered as `obs_names`) using `add_frame`.
    - When enough frames exist and stride condition is met, outputs one window vector
      ordered as `WINDOW_FEATURE_COLS`.
    """
    def __init__(self):
        self.W = int(WINDOW_CONFIG.W)
        self.stride = int(WINDOW_CONFIG.stride)

        self.obs_names = list(FRAME_FEATURE_COLS)
        self.win_names = list(WINDOW_FEATURE_COLS)

        self.buffer = deque(maxlen=self.W) # holds up to W frames, each frame is a vector of length len(obs_names)
        self.frame_count = 0

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