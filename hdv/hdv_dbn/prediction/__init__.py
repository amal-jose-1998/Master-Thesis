from .filtering import StructuredDBNFilter, BeliefState
from .model_interface import HDVDbnModel
from .online_predictor import OnlinePredictor, PredictionOutput
from .metrics import (
    JointStateMetrics,
    HitAtHorizon,
    TimeToEvent,
    MetricsAccumulator,
)
from .validate import ValidationStep, TrajectoryData, ValidationConfig
from .visualize_metrics import (
    visualize_all_metrics,
    plot_time_to_hit_histogram,
    plot_cumulative_hit_rate,
    plot_hit_count_summary,
    plot_confusion_matrix
)

__all__ = [
    # Filtering (core algorithm)
    "StructuredDBNFilter",
    "BeliefState",
    
    # Model interface (model-agnostic)
    "HDVDbnModel",
    
    # Online prediction (streaming)
    "OnlinePredictor",
    "PredictionOutput",
    
    # Metrics (evaluation)
    "JointStateMetrics",
    "HitAtHorizon",
    "TimeToEvent",
    "MetricsAccumulator",
    
    # Validation (orchestration)
    "ValidationStep",
    "TrajectoryData",
    "ValidationConfig",
    
    # Visualization
    "visualize_all_metrics",
    "plot_time_to_hit_histogram",
    "plot_cumulative_hit_rate",
    "plot_hit_count_summary",
    "plot_confusion_matrix",
]
