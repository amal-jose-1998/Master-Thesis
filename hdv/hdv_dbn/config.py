from dataclasses import dataclass
from typing import Tuple, Optional, Literal, List

# =============================================================================
# Latent state space configuration
# =============================================================================
@dataclass(frozen=True)
class DBNStates:
    """
    Container for the discrete latent state spaces used by the model.

    Attributes
    driving_style : Tuple[str, ...]
        Names of discrete driving style states (e.g., conservative/normal/aggressive).
    action : Tuple[str, ...]
        Names of discrete maneuver/action states (e.g., keep_lane, lane_change_left, ...).
    """
    driving_style: Tuple[str, ...]
    action: Tuple[str, ...]
   

# Define the DBN states
DBN_STATES = DBNStates(
    driving_style=("style_0", "style_1", "style_2"),
    action=("action_0", "action_1", "action_2",)
    )

# =============================================================================
# Observation feature configuration for highD dataset
# =============================================================================

# meta columns stored per sequence (not part of obs vector)
META_COLS: List[str] = [
    "meta_class",           # vehicle class (e.g., car/truck)
    "meta_drivingDirection",# driving direction (+1 or -1)
]

# -----------------------------
# Ego features
# -----------------------------
EGO_FEATURES: List[str] = [
    "vx", "vy",
    "ax", "ay",
    "lane_pos", "lc",
    "speed", "jerk_x",
]

# -----------------------------
# Neighbor feature blocks
# -----------------------------
FRONT_FEATURES: List[str] = [
    "front_exists",
    "front_dx", "front_dy",
    "front_dvx", "front_dvy",
    "front_thw", "front_ttc",
]

REAR_FEATURES: List[str] = [
    "rear_exists",
    "rear_dx", "rear_dy",
    "rear_dvx", "rear_dvy",
]

LEFT_FRONT_FEATURES: List[str] = [
    "left_front_exists",
    "left_front_dx", "left_front_dy",
    "left_front_dvx", "left_front_dvy",
]

LEFT_REAR_FEATURES: List[str] = [
    "left_rear_exists",
    "left_rear_dx", "left_rear_dy",
    "left_rear_dvx", "left_rear_dvy",
]

RIGHT_FRONT_FEATURES: List[str] = [
    "right_front_exists",
    "right_front_dx", "right_front_dy",
    "right_front_dvx", "right_front_dvy",
]

RIGHT_REAR_FEATURES: List[str] = [
    "right_rear_exists",
    "right_rear_dx", "right_rear_dy",
    "right_rear_dvx", "right_rear_dvy",
]

LEFT_SIDE_FEATURES: List[str] = [  
    "left_side_exists",            
    "left_side_dx", "left_side_dy",
    "left_side_dvx", "left_side_dvy",
]                                 

RIGHT_SIDE_FEATURES: List[str] = [
    "right_side_exists",           
    "right_side_dx", "right_side_dy",
    "right_side_dvx", "right_side_dvy",
]     

# Full baseline observation vector (feature order matters!)
BASELINE_FEATURE_COLS: List[str] = (
     EGO_FEATURES
    + FRONT_FEATURES
    + LEFT_FRONT_FEATURES
    + LEFT_SIDE_FEATURES
    + LEFT_REAR_FEATURES
    + RIGHT_FRONT_FEATURES
    + RIGHT_SIDE_FEATURES
    + RIGHT_REAR_FEATURES
)

# -----------------------------
# Discrete vs continuous split
# -----------------------------
"""
Discrete features are handled by non-Gaussian emission components:
  - lane_pos: categorical distribution (K=5)
      coding: {0,1,2,3,4} (ignore/mask -1 if present)
  - lc: categorical distribution (K=3; left/none/right)
      coding: {-1, 0, +1} (we will map internally to {0,1,2})
  - *_exists: independent Bernoulli distributions

Continuous features are modeled by a multivariate Gaussian per latent state.
Only continuous features should be z-score normalized.
"""
# Categorical (multinomial) discrete features
CATEGORICAL_FEATURES: List[str] = [
    "lane_pos",
    "lc",
]

# Number of categories for each categorical feature
CATEGORICAL_FEATURE_SIZES = {
    "lane_pos": 5,
    "lc": 3,
}

# Binary discrete features (modeled with independent Bernoulli emissions)
BERNOULLI_FEATURES: List[str] = [
    "front_exists", "rear_exists",
    "left_front_exists", "left_side_exists", "left_rear_exists",
    "right_front_exists", "right_side_exists", "right_rear_exists",
]

# All discrete features (categorical + Bernoulli)
DISCRETE_FEATURES: List[str] = CATEGORICAL_FEATURES + BERNOULLI_FEATURES

# Everything else is continuous
CONTINUOUS_FEATURES: List[str] = [f for f in BASELINE_FEATURE_COLS if f not in DISCRETE_FEATURES]

# =============================================================================
# Training configuration
# =============================================================================
@dataclass(frozen=True)
class TrainingConfig:
    """
    Global configuration for EM training and numerical settings.

    EM / early stopping
    seed : int
        Random seed for reproducibility (splits, init, sampling).
    em_num_iters : int
        Maximum number of EM iterations.
    early_stop_patience : int
        Stop if no meaningful improvement is observed for this many iterations.
    early_stop_min_delta_per_obs : float
        Minimum increase in average log-likelihood per observation required
        to be considered an improvement.
    early_stop_delta_A_thresh : float
        Minimum change in transition matrix A required; if A stabilizes below
        this threshold, training may stop early.

    Emission numerical stability
    emission_jitter : float
        Small diagonal term added to covariances for numerical stability.
    min_cov_diag : float
        Lower bound for covariance diagonal entries.
    gauss_min_state_mass : float
        Skip Gaussian parameter update for states with total responsibility mass
        below this threshold.
    gauss_min_eig : float
        Minimum eigenvalue used when projecting covariances to SPD (Symmetric Positive Definite).

    Data / initialization
    use_classwise_scaling : bool
        If True, compute separate scalers for different vehicle classes (e.g., car vs truck).
    max_kmeans_samples : int
        Max number of observations used for k-means emission initialization.
    max_highd_recordings : Optional[int]
        If set, limit the number of recordings for debugging.

    Runtime / logging
    backend : Literal["torch"]
        Numerical backend (currently Torch).
    device : Literal["cuda", "cpu"]
        Device selection.
    dtype : Literal["float32", "float64"]
        Floating point precision for Torch computations.
    use_wandb : bool
        Enable Weights & Biases logging.
    wandb_project : str
        W&B project name.
    wandb_run_name : Optional[str]
        W&B run name.

    Transition initialization
    cpd_init : Literal["uniform", "random", "sticky"]
        How to initialize transition probabilities.
    cpd_alpha : float
        Strength of Dirichlet smoothing / pseudocounts for transitions.
    cpd_stay_style : float
        If using "sticky" init, prior probability of staying in the same style state.
    cpd_seed : int
        Seed used for random CPD initialization (kept separate for clarity).
    """
    seed: int = 123
    em_num_iters: int = 100

    exists_as_bernoulli: bool = True          # E1,E3=True, E2=False
    bern_weight: float = 1.0                   # E1=1.0, E3=0.5 or 0.25
    lane_weight: float = 0.2  
    lc_weight: float = 1.0

    early_stop_patience: int = 3
    early_stop_min_delta_per_obs: float = 5e-3
    early_stop_delta_A_thresh: float = 1e-3

    verbose: int = 1
    use_progress: bool = True
    use_classwise_scaling: bool = True

    emission_jitter: float = 1e-6
    cat_alpha: float = 1.0
    min_cov_diag: float = 1e-5
    gauss_min_state_mass: float = 50.0
    gauss_min_eig: float = 1e-4

    max_kmeans_samples: int = 100000
    max_highd_recordings: Optional[int] = 5

    use_wandb: bool = True
    wandb_project: str = "hdv_dbn_highd"
    wandb_run_name: Optional[str] = "Experiment 1: uniform"

    backend: Literal["torch"] = "torch"
    device: Literal["cuda", "cpu"] = "cuda"
    dtype: Literal["float32", "float64"] = "float64"

    cpd_init: Literal["uniform", "random", "sticky"] = "uniform"
    cpd_alpha: float = 1.0
    cpd_stay_style: float = 0.8
    cpd_seed: int = 123

    
    

TRAINING_CONFIG = TrainingConfig()