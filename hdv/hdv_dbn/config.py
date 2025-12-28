from dataclasses import dataclass
from typing import Tuple, Optional, Literal

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
    em_num_iters: int = 100
    early_stop_patience: int = 3
    early_stop_min_delta_per_obs: float = 5e-3
    early_stop_delta_A_thresh: float = 1e-5
    verbose: int = 1
    use_progress: bool = True
    use_classwise_scaling: bool = True
    emission_jitter: float = 1e-6
    min_cov_diag: float = 1e-5

    max_kmeans_samples: int = 100000
    max_highd_recordings: Optional[int] = None

    use_wandb: bool = True
    wandb_project: str = "hdv_dbn_highd"
    wandb_run_name: Optional[str] = "Training with all track.csv files"

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