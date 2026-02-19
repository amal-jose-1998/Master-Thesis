"""
This module prepares the input in the right form for both:
    a. rule-based GT labeling (needs raw units),
    b. model inference (needs scaled input).

1. Reads split.json in the experiment folder to know which vehicle trajectories belong to the test split.
2. Loads the windowized test sequences (each sequence is a list of window-feature vectors).
3. Loads the trained model checkpoint (final.npz) into an HDVTrainer.
4. Applies the saved scaler (global or classwise) to produce:
    raw_obs: unscaled features (physical units)
    scaled_obs: scaled features (model input)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, cast
import numpy as np

from ..evaluate_highd_dbn import load_test_sequences_from_experiment_split
from ..utils.eval_common import scale_obs_masked
from ..trainer import HDVTrainer
from ..config import WINDOW_FEATURE_COLS, CONTINUOUS_FEATURES
from ..datasets.dataset import TrajectorySequence


@dataclass(frozen=True)
class TestSequences:
    """
    Test-split sequences prepared for prediction-time evaluation.
    Provides RAW and SCALED window features:
        - RAW: used for rule-based GT labeling (thresholds in physical units)
        - SCALED: used for model inference (must match training-time scaling)

    Attributes
    raw_obs:
        List of RAW observation matrices (T,F).
    scaled_obs:
        List of SCALED observation matrices (T,F) in training feature space.
    feature_cols:
        The feature name list aligned with columns of raw_obs/scaled_obs.
    trajectory_ids:
        Trajectory identifier for each sequence (for logging/tracking).
    scaler_mode:
        Either "global" or "classwise" depending on checkpoint scalers.
    """
    raw_obs: List[np.ndarray]
    scaled_obs: List[np.ndarray]
    feature_cols: List[str]
    trajectory_ids: List[str]
    scaler_mode: str

def _get_meta_class(seq):
    """Retrieve meta_class from TrajectorySequence definition."""
    meta = getattr(seq, "meta", None)
    if isinstance(meta, dict):
        cls = meta.get("meta_class", None)
        return None if cls is None else str(cls)
    return None
 
def load_test_data_for_prediction(*, exp_dir, data_root, checkpoint_name="final.npz"):
    """
    Load the test split and prepare RAW + SCALED window features for prediction.
    Single source of truth:
      - exp_dir/split.json determines which vehicles belong to test, and fixes W/stride used to windowize sequences.
      - highD window sequences are loaded/built via the existing dataset utilities.
      - scaling is reproduced from the checkpoint's saved scalers using `scale_obs_masked`, scaling only CONTINUOUS_FEATURES.

    Parameters
    exp_dir:
        Experiment directory containing split.json and the checkpoint file.
    data_root:
        Path to highD dataset root.
    checkpoint_name:
        Filename of checkpoint inside exp_dir (default: "final.npz").
    
    Returns
    trainer:
        Loaded HDVTrainer instance (model + scalers).
    test:
        TestSequences containing aligned RAW and SCALED observations for each test vehicle.
    """
    exp_dir = Path(exp_dir)
    data_root = Path(data_root)

    ckpt_path = exp_dir / checkpoint_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # 1) Load sequences (split.json + window cache)
    raw_seqs, feature_cols, _, _ = load_test_sequences_from_experiment_split(exp_dir=exp_dir, data_root=data_root)  # feature_cols aligns with seq.obs columns
    # Note: split_payload and test_keys discarded (not needed for validation)
    
    feature_cols = list(feature_cols) # Verify feature column ordering matches config
    if list(feature_cols) != list(WINDOW_FEATURE_COLS):
        raise RuntimeError(
            "Feature column mismatch vs config.WINDOW_FEATURE_COLS.\n"
            f"From sequences ({len(feature_cols)}): {feature_cols}\n"
            f"From config    ({len(WINDOW_FEATURE_COLS)}): {list(WINDOW_FEATURE_COLS)}"
        )
    
    sn = getattr(raw_seqs[0], "obs_names", None) # Validate sequence obs_names if present
    if sn is not None and list(sn) != list(feature_cols):
        raise RuntimeError(
            "Sequence obs_names does not match feature_cols returned by loader.\n"
            f"seq.obs_names ({len(sn)}): {list(sn)}\n"
            f"feature_cols  ({len(feature_cols)}): {list(feature_cols)}"
        )

    # 2) Load checkpoint to get scaler + model params
    trainer: HDVTrainer = HDVTrainer.load(ckpt_path)

    # 3) Build scale indices 
    cont_set = set(CONTINUOUS_FEATURES)
    scale_idx = np.array([i for i, name in enumerate(feature_cols) if name in cont_set], dtype=np.int64) # indices of only CONTINUOUS_FEATURES. (Binary flags etc. remain unscaled.)

    # 4) Decide scaler mode from checkpoint contents.
    is_classwise = isinstance(trainer.scaler_mean, dict) and isinstance(trainer.scaler_std, dict)
    scaler_mode = "classwise" if is_classwise else "global"

    raw_obs = []
    scaled_obs = []
    trajectory_ids = []

    # For each test sequence in raw_seqs: store RAW and compute SCALED
    raw_seqs_typed: List[TrajectorySequence] = cast(List[TrajectorySequence], raw_seqs)
    for seq in raw_seqs_typed: 
        # Create trajectory ID from recording_id and vehicle_id
        traj_id = f"{seq.recording_id}:{seq.vehicle_id}"
        trajectory_ids.append(str(traj_id))
        
        obs_raw = np.asarray(seq.obs, dtype=np.float64) # shape (T,F)
        if obs_raw.ndim != 2:
            raise ValueError(f"Expected seq.obs to be (T,F), got {obs_raw.shape}")
        if obs_raw.shape[1] != len(feature_cols):
            raise ValueError(
                f"Feature dimension mismatch: seq.obs has F={obs_raw.shape[1]}, "
                f"but feature_cols has {len(feature_cols)}"
            )
        raw_obs.append(obs_raw)
        
        # scales with either global or classwise scalers
        if is_classwise:
            cls = _get_meta_class(seq)
            if cls is None:
                raise AttributeError(
                    "Classwise scaling enabled, but seq.meta['meta_class'] is missing."
                )
            scaler_mean_dict = cast(dict, trainer.scaler_mean)
            scaler_std_dict = cast(dict, trainer.scaler_std)
            if cls not in scaler_mean_dict or cls not in scaler_std_dict:
                raise KeyError(
                    f"meta_class='{cls}' missing in checkpoint scalers. "
                    f"Available: {sorted(str(k) for k in scaler_mean_dict.keys())}"
                )
            obs_scaled = scale_obs_masked(obs_raw, scaler_mean_dict[cls], scaler_std_dict[cls], scale_idx)
        else:
            if trainer.scaler_mean is None or trainer.scaler_std is None:
                raise RuntimeError("Checkpoint missing global scaler_mean/std.")
            obs_scaled = scale_obs_masked(obs_raw, trainer.scaler_mean, trainer.scaler_std, scale_idx)

        scaled_obs.append(obs_scaled.astype(np.float32, copy=False))

    return trainer, TestSequences(
        raw_obs=raw_obs,
        scaled_obs=scaled_obs,
        feature_cols=feature_cols,
        trajectory_ids=trajectory_ids,
        scaler_mode=scaler_mode,
    )
