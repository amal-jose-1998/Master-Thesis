from dataclasses import dataclass
from typing import List, Tuple

@dataclass(frozen=True)
class DBNStates:
    """Dataclass to hold the different state categories for the DBN model."""
    driving_style_states: Tuple[str, ...]
    intent_states: Tuple[str, ...]
    maneuver_states: Tuple[str, ...]

# Define the DBN states
DBN_STATES = DBNStates(
    driving_style_states=("cautious", "normal", "aggressive"),
    intent_states=("keep_lane", "lane_change_left", "lane_change_right", "speed_up", "slow_down"),
    maneuver_states=("maintain_speed", "accelerate", "decelerate", "hard_brake", "prepare_lc_left", "prepare_lc_right", "perform_lc_left", "perform_lc_right")
)

@dataclass(frozen=True)
class ObsConfig:
    """Dataclass to hold the observation configuration for the DBN model."""
    lane_ids: List[int]
    # Longitudinal thresholds (m/s²)
    accel_threshold: float                      # above → accelerate
    hard_brake_threshold: float                 # below → hard brake
    # Lateral thresholds
    lateral_vel_threshold: float                # |vy| above → lateral movement for prepare
    prepare_lane_offset: float                  # |y - lane_center| above → preparing LC
    # Lane-change episode parameters
    prep_frames_lc: int                         # frames before LC index to consider for prepare_lc_* 
    lc_frames: int                              # minimum frames for perform_lc_*
    lane_width: float                           # approximate lane width [m]
    
    
OBS_CONFIG = ObsConfig(
    lane_ids=[1, 2, 3],
    accel_threshold=0.2,         # m/s²
    hard_brake_threshold=-2.0,   # m/s²
    lateral_vel_threshold=0.2,   # m/s
    prepare_lane_offset=0.8,     # meters toward boundary
    prep_frames_lc=25,           # frames before LC labeled as "prepare"
    lc_frames=15,                # frames marked as "perform"
    lane_width=3.5               # approximate lane width for lateral offset
    )