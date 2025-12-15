"""Generic utilities for turning trajectory tables into sequences that can be used to train the HDV DBN."""
from dataclasses import dataclass
import numpy as np
from typing import List, Optional

@dataclass
class TrajectorySequence:
    """
    One vehicle trajectory represented as a time series.

    Attributes
    vehicle_id : object
        Identifier of the vehicle (e.g. highD `id` column).
    frames : np.ndarray
        Array of shape (T,) containing frame indices, sorted in increasing time order.
    obs : np.ndarray
        Observation matrix of shape (T, F), where:
            T = number of time steps,
            F = number of observation features.
    obs_names : list[str]
        Names of the observation features, length F, matching columns of `obs`.
    recording_id : object, optional
        Identifier of the recording (e.g. highD recording number). Useful when
        datasets contain multiple independent recordings.
    """

    vehicle_id: object
    frames: np.ndarray
    obs: np.ndarray
    obs_names: List[str]

    recording_id: Optional[object] = None

    @property
    def T(self):
        """
        Length of the trajectory.

        Returns
        int
            Number of time steps T in the sequence.
        """
        return int(len(self.frames))

    @property
    def F(self):
        """
        Observation dimensionality.

        Returns
        int
            Number of observation features per time step.
        """
        return int(self.obs.shape[1])