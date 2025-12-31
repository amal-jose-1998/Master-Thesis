from dataclasses import dataclass
from typing import Tuple, Optional, Literal, List

@dataclass(frozen=True)
class DBNStates:
    """Dataclass to hold the different state categories for the DBN model."""
    driving_style: Tuple[str, ...]
    action: Tuple[str, ...]
   

# Define the DBN states
DBN_STATES = DBNStates(
    driving_style=("style_0", "style_1", "style_2"),
    action=("dummy",)
    )

# =============================================================================
# Observation feature configuration for highD dataset
# =============================================================================

# meta columns stored per sequence (not part of obs vector)
META_COLS: List[str] = [
    "meta_class",
    "meta_drivingDirection",
]

EGO_FEATURES: List[str] = [
    "y",
    "vx", "vy",
    "ax", "ay",
    "lane_pos",
]

FRONT_FEATURES: List[str] = [
    "front_exists",
    "front_dx", "front_dy",
    "front_dvx", "front_dvy",
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

BASELINE_FEATURE_COLS: List[str] = (
    EGO_FEATURES
    + FRONT_FEATURES
    + REAR_FEATURES
    + LEFT_FRONT_FEATURES
    + LEFT_REAR_FEATURES
    + LEFT_SIDE_FEATURES
    + RIGHT_SIDE_FEATURES 
    + RIGHT_FRONT_FEATURES
    + RIGHT_REAR_FEATURES
)

# discrete features (do NOT z-score scale; do NOT model with a plain Gaussian)
# lane_pos is categorical; *_exists are binary masks.
DISCRETE_FEATURES: List[str] = [
    "lane_pos",
    "front_exists", "rear_exists",
    "left_front_exists", "left_side_exists", "left_rear_exists",      
    "right_front_exists", "right_side_exists", "right_rear_exists",   
]

# continuous features = all baseline minus discrete
CONTINUOUS_FEATURES: List[str] = [f for f in BASELINE_FEATURE_COLS if f not in DISCRETE_FEATURES]
# =============================================================================

@dataclass(frozen=True)
class TrainingConfig:
    """
    Global training-related configuration.

    seed : int
        Base random seed used for data splits and initialisation.
    em_num_iters : int
        Maximum number of EM iterations.
    em_tol : float
        Convergence threshold for EM based on improvement in criterion.
    verbose : int
        Default verbosity level for training.
            0 = no prints,
            1 = per-iteration summary,
            2 = detailed (more debug prints).
    use_progress : bool
        Whether to show tqdm progress bars during training.
    max_kmeans_samples : int
        Max number of samples for k-means initialisation.
    max_highd_recordings : int | None
        If not None, limit number of highD CSV recordings used (for debugging).
    backend : str
        Numerical backend identifier. Currently only "torch" is supported,
        but this field allows future extension (e.g., "cupy").
    device : str
        Torch device string:
            "cuda" → use NVIDIA GPU (if available)
            "cpu"  → force CPU execution
    dtype : str
        Floating-point precision used for Torch tensors.
            "float32" → faster, lower memory, usually sufficient
            "float64" → higher precision, slower, safer for numerical stability
    use_batched_padding : bool
        If True, variable-length trajectories can be grouped into padded batches
        with masking. This enables higher GPU utilization but requires additional
        logic in the trainer.
    batch_size_seqs : int
        Number of trajectories per batch when padded batching is enabled.
    """
    seed: int = 123
    em_num_iters: int = 2
    early_stop_patience: int = 3
    early_stop_min_delta_per_obs: float = 5e-3
    early_stop_delta_A_thresh: float = 1e-5
    verbose: int = 1
    use_progress: bool = True
    use_classwise_scaling: bool = True
    emission_jitter: float = 1e-6
    cat_alpha: float = 1.0
    min_cov_diag: float = 1e-5

    max_kmeans_samples: int = 100000
    max_highd_recordings: Optional[int] = None

    use_wandb: bool = True
    wandb_project: str = "hdv_dbn_highd"
    wandb_run_name: Optional[str] = "Training with the latest observation set with baseline (latent style only)"

    backend: Literal["torch"] = "torch"
    device: Literal["cuda", "cpu"] = "cuda"
    dtype: Literal["float32", "float64"] = "float64"

    use_batched_padding: bool = False
    batch_size_seqs: int = 64

    cpd_init: Literal["uniform", "random", "sticky"] = "sticky"
    cpd_alpha: float = 1.0
    cpd_stay_style: float = 0.8
    cpd_seed: Optional[int] = 123

TRAINING_CONFIG = TrainingConfig()