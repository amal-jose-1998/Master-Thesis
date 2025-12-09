from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class DBNStates:
    """Dataclass to hold the different state categories for the DBN model."""
    driving_style: Tuple[str, ...]
    action: Tuple[str, ...]
   

# Define the DBN states
DBN_STATES = DBNStates(
    driving_style=("style_0", "style_1", "style_2"), #("cautious", "normal", "aggressive")
    action=("action_0", "action_1", "action_2", "action_3", "action_4", "action_5") #("maintain_speed", "accelerate", "decelerate", "hard_brake" "lane_change_left", "lane_change_right") 
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
    """
    seed: int = 123
    em_num_iters: int = 100
    em_tol: float = 1e-3
    verbose: int = 1
    use_progress: bool = True
    max_kmeans_samples: int = 100000
    max_highd_recordings: int | None = 5


TRAINING_CONFIG = TrainingConfig()