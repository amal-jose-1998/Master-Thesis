from dataclasses import dataclass, field
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
    driving_style=("style_0", "style_1",),
    #driving_style=("dummy",),
    action=("action_0", "action_1", "action_2", "action_3",)
    #action=("dummy")
    )


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
    em_num_iters: int = 200

    # -------------------------------------------------------------
    # Emission model selection
    # -------------------------------------------------------------
    # "poe"    : Product-of-Experts emissions (your PoE module)
    # "linear" : additive log-likelihoods (no PoE logZ coupling)
    emission_model: Literal["poe", "hierarchical"] = "poe"
    # Only used for "poe" if that implementation uses gradient M-step
    poe_em_lr: float = 3e-3
    poe_em_steps: int = 20
    # -------------------------------------------------------------
    # PoE emission M-step stabilization / regularization
    # -------------------------------------------------------------
    # Weak MAP-style priors to stabilize gradient-based PoE M-step
    poe_em_lam_mu: float = 0.0        # L2 penalty on Gaussian means
    poe_em_lam_logvar: float = 0.0    # L2 penalty on log-variances
    poe_em_lam_logit: float = 0.0     # L2 penalty on Bernoulli logits
    # Optional optimizer-level damping (usually keep 0.0 if explicit priors are used)
    poe_em_weight_decay: float = 0.0
    # Inner-loop early stopping for PoE M-step
    poe_em_inner_patience: int = 3
    # Optional chunking over time steps (0 = disabled)
    poe_em_chunk_size: int = 0

    learn_pi0: bool = False  
    pi0_alpha: float = 0.0
    disable_discrete_obs: bool = False
    bern_weight: float = 1     #(dont change)                
    lc_weight: float = 25   # to clip
    # Lane-change imbalance handling in EM
    #   - 'none': no special weighting
    #   - 'A'   : likelihood tempering (logB[t] *= w_t) before forward-backward. This can distort inference more strongly.
    #   - 'B'   : weighted sufficient statistics (gamma and xi) after forward-backward. This is usually safer and more interpretable. 
    lc_weight_mode: Literal["none", "A", "B"] = "B"       
    # How to weight xi_t (transition counts) in mode 'B'
    #   - 'next': use w_{t+1}
    #   - 'avg' : use 0.5*(w_t + w_{t+1})
    lc_xi_weight: Literal['next', 'avg'] = 'avg'  

    early_stop_patience: int = 3
    early_stop_min_delta_per_obs: float = 5e-3
    early_stop_delta_A_thresh: float = 1e-3
    early_stop_delta_pi_thresh: float = 1e-3

    # Transition MAP priors (Dirichlet + stickiness)
    alpha_A_s: float = 0.01   # smoothing for style rows
    kappa_A_s: float = 0.0    # extra self-transition mass for style (stickier)
    alpha_A_a: float = 0.01   # smoothing for action rows
    kappa_A_a: float = 0.0     # extra self-transition mass for action (less sticky than style)

    verbose: int = 1
    use_progress: bool = True
    use_classwise_scaling: bool = True

    emission_jitter: float = 1e-6
    cat_alpha: float = 1.0
    min_cov_diag: float = 1e-5
    gauss_min_state_mass: float = 50.0
    gauss_min_eig: float = 1e-4

    max_kmeans_samples: int = 100000
    max_highd_recordings: Optional[int] = 25

    use_wandb: bool = True
    wandb_project: str = "hdv_dbn_highd"
    wandb_run_name: Optional[str] = "26.poe-uni_cpd-uni_pi-lc_b-bern_on"

    backend: Literal["torch"] = "torch"
    device: Literal["cuda", "cpu"] = "cuda"
    dtype: Literal["float32", "float64"] = "float32"

    cpd_init: Literal["uniform", "random", "sticky"] = "uniform"
    cpd_alpha: float = 1.0
    cpd_stay_style: float = 0.8
    cpd_seed: int = 123

    EPSILON: float = 1e-6 # small constant for numerical stability
    debug_asserts: bool = True

TRAINING_CONFIG = TrainingConfig()


# =============================================================================
# Observation feature configuration for highD dataset
# =============================================================================
# meta columns stored per sequence (not part of obs vector)
META_COLS: List[str] = [
    "meta_class",           # vehicle class (e.g., car/truck)
    "meta_drivingDirection",# driving direction (+1 or -1)
]

# Per-frame columns we keep in the dataframe (to build windows) 
FRAME_FEATURE_COLS: List[str] = [
    # ego kinematics
    "vx", "vy", "ax", "ay", "jerk_x", "jerk_y",

    # lane change evidence  ∈ { -1, 0, +1 }
    "lc",       

    # lane boundary distances (from add_lane_boundary_distance_features)
    # These may be NaN if lane boundaries are undefined.
    "d_left_lane", "d_right_lane",

    # front risk from highD tracks (kept as NaN when undefined)
    "front_thw", "front_ttc", "front_dhw",

    # existence flags (frame-level, later windowised into fractions)
   "left_front_exists",  "front_exists",  "right_front_exists", 
   "left_side_exists", "right_side_exists", 
   "left_rear_exists", "rear_exists", "right_rear_exists",

    # neighbor-relative interaction signals (frame-level) 
    "front_dx", "front_dy", "front_dvx", "front_dvy",
    "left_front_dx", "left_front_dy", "left_front_dvx", "left_front_dvy",
    "right_front_dx", "right_front_dy", "right_front_dvx", "right_front_dvy",
    "left_side_dx", "left_side_dy", "left_side_dvx", "left_side_dvy",
    "right_side_dx", "right_side_dy", "right_side_dvx", "right_side_dvy",
    "left_rear_dx", "left_rear_dy", "left_rear_dvx", "left_rear_dvy",
    "right_rear_dx", "right_rear_dy", "right_rear_dvx", "right_rear_dvy",
    "rear_dx", "rear_dy", "rear_dvx", "rear_dvy",
]


# =============================================================================
# Windowing configuration (window = timestep)
# =============================================================================
@dataclass(frozen=True)
class WindowConfig:
    """
    Controls conversion from per-frame trajectories to per-window timesteps.
    W and stride are in frames. 
    """
    W: int = 50  # highD is 25 Hz (so 50 frames = 2s).
    stride: int = 10 # 10 frames ~= 0.4s between window starts

WINDOW_CONFIG = WindowConfig()

# Window-level columns produced by windowize_sequences() 
WINDOW_EGO_FEATURES: List[str] = [
    "vx_last", "vx_slope",
    "ax_last",
    "vy_last", "vy_slope", 
    "ay_last", 
    "ax_neg_frac", "ax_pos_frac", "ax_zero_frac",
    "ay_neg_frac", "ay_pos_frac", "ay_zero_frac",
    "jerk_x_p95",
    "jerk_y_p95",
]

WINDOW_LC_FEATURES: List[str] = [
    "lc_left_present",
    "lc_right_present", 
]

WINDOW_LANE_GEOM_FEATURES: List[str] = [
    "d_left_lane_last",  "d_left_lane_min", 
    "d_right_lane_last", "d_right_lane_min", 
]

# *_vfrac = fraction of frames where the signal was valid
# if vfrac == 0 → mean and min are NaN (masked later in emissions / E-step)
WINDOW_FRONT_RISK_FEATURES: List[str] = [
    #"front_thw_slope", 
    "front_thw_last", "front_thw_vfrac", 
    #"front_ttc_slope", #"front_ttc_last", 
    "front_ttc_min", "front_ttc_vfrac", 
    #"front_dhw_mean", "front_dhw_min", "front_dhw_vfrac",
]

WINDOW_NEIGHBOR_REL_FEATURES: List[str] = [
    "front_dx_min", #"front_dx_slope",
    #"front_dvx_min", 
    "front_dvx_slope",
    "front_exists_frac",

    #"left_side_dx_last", "left_side_dx_slope",
    "left_side_dvx_last", 
    "left_side_dvx_slope",
    "left_side_dy_last", "left_side_exists_frac",

    #"right_side_dx_last", "right_side_dx_slope",
    "right_side_dvx_last", 
    "right_side_dvx_slope",
    "right_side_dy_last", "right_side_exists_frac",

    "left_front_dx_last", "left_front_dvx_slope", "left_front_dy_last", "left_front_exists_frac", 
    #"left_rear_dx_last", "left_rear_exists_frac",

    "right_front_dx_last", "right_front_dvx_slope", "right_front_dy_last", "right_front_exists_frac", 
    #"right_rear_dx_last", "right_rear_exists_frac",
]


WINDOW_FEATURE_COLS: List[str] = (
    WINDOW_EGO_FEATURES
    + WINDOW_LC_FEATURES
    + WINDOW_LANE_GEOM_FEATURES
    + WINDOW_FRONT_RISK_FEATURES
    + WINDOW_NEIGHBOR_REL_FEATURES
)

BERNOULLI_FEATURES: List[str] = [
    "lc_left_present",
    "lc_right_present",
]

CONTINUOUS_FEATURES: List[str] = [
    n for n in WINDOW_FEATURE_COLS
    if n not in set(BERNOULLI_FEATURES)
]


@dataclass(frozen=True)
class SemanticAnalysisConfig:
    # Paths (edit these)
    model_path: str = r"C:\\Users\\amalj\\OneDrive\\Desktop\\Master's Thesis\\Implementation\\hdv\\models\\final.npz"
    data_root: str = r"C:\\Users\\amalj\\OneDrive\\Desktop\\Master's Thesis\\Implementation\\hdv\\data\\highd"

    # Speed/debug controls
    max_sequences: int | None = None   # e.g. 200 for quick run; None = use all
    split_name : str = "train"
    print_joint_table: bool = True     # print (s,a)
    print_style_table: bool = True     # derived marginal over a
    print_action_table: bool = True    # derived marginal over s

    
    semantic_feature_cols: List[str] = field(
        default_factory=lambda: list(WINDOW_FEATURE_COLS)
    )

SEMANTIC_CONFIG = SemanticAnalysisConfig()