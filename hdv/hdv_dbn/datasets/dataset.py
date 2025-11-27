"""
Generic utilities for turning trajectory tables into sequences that can be used to train the HDV DBN.

Expected input format:
Any DataFrame with at least:

    vehicle_id : identifier of each vehicle trajectory.
    frame      : sortable time index
    x          : longitudinal position  [m]
    y          : lateral position       [m]
    vx         : longitudinal velocity  [m/s]
    vy         : lateral velocity       [m/s]
    ax         : longitudinal accel.    [m/s^2]
    lane_id    : lane index (1,2,3)
"""

from dataclasses import dataclass
import numpy as np
from typing import Dict, List

@dataclass
class TrajectorySequence:
    """One vehicle trajectory represented as a time series."""
    vehicle_id: object            # of the vehicle whose trajectory this sequence represents
    frames: np.ndarray            # frame numbers (time indices) for this vehicle’s trajectory, sorted in order.
    features: np.ndarray          # size=(T, F); actual numerical data used by the DBN; T = number of time steps (frames), F = number of features per frame
    feature_names: List[str]      # names of the columns inside features
    labels: Dict[str, np.ndarray] # maneuver annotation