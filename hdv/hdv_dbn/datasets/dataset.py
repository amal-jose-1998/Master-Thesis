"""Generic utilities for turning trajectory tables into sequences that can be used to train the HDV DBN."""
from dataclasses import dataclass
import numpy as np
from typing import List, Optional

@dataclass
class TrajectorySequence:
    """
    One vehicle trajectory represented as a time series.

    Attributes
    vehicle_id :
        Identifier of the vehicle (e.g. highD 'id').
    frames :
        Array of shape (T,) with frame indices, sorted in time.
    obs :
        Array of shape (T, F) with observation features used by the DBN.
    obs_names :
        List of length F with feature names corresponding to columns of `obs`.

    Optional attributes (for later use)
    recording_id :
        For datasets like highD, which contain multiple recordings (01, 02,..),
        this can store from which recording this trajectory was taken.
    mask :
        Optional boolean array of shape (T,). True can mean "valid"; this
        lets you mask out invalid steps without changing the length.
    """

    vehicle_id: object
    frames: np.ndarray
    obs: np.ndarray
    obs_names: List[str]

    recording_id: Optional[object] = None
    mask: Optional[np.ndarray] = None

    @property
    def T(self):
        """Number of time steps in the sequence."""
        return int(len(self.frames))

    @property
    def F(self):
        """Number of observation features per time step."""
        return int(self.obs.shape[1])