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
from typing import List
import numpy as np

from ..evaluate_highd_dbn import load_test_sequences_from_experiment_split
from ..utils.eval_common import scale_obs_masked
from ..trainer import HDVTrainer
from ..config import WINDOW_FEATURE_COLS, CONTINUOUS_FEATURES


@dataclass(frozen=True)
class TestSequences:
    """
    Test-split sequences prepared for prediction-time evaluation.
    This container intentionally provides both RAW and SCALED window features:
        - RAW: used for rule-based GT labeling (thresholds in physical units)
        - SCALED: used for model inference (must match training-time scaling)

    Attributes
    raw_seqs:
        List of per-vehicle window sequences from the test split. Each sequence
        is expected to follow `TrajectorySequence`, with fields:
            - seq.obs: (T,F) RAW window features
            - seq.obs_names: list[str] of length F
            - seq.meta: optional dict with keys like "meta_class"
    raw_obs:
        List of RAW observation matrices (T,F).
    scaled_obs:
        List of SCALED observation matrices (T,F) in training feature space.
    feature_cols:
        The feature name list aligned with columns of raw_obs/scaled_obs.
        This is taken from the sequences (and should match seq.obs_names).
    split_payload:
        Parsed split.json dictionary (contains W, stride, keys, etc.).
    test_keys:
        Set of vehicle keys ("recording_id:vehicle_id") for the test split.
    scaler_mode:
        Either "global" or "classwise" depending on checkpoint scalers.
    """
    raw_seqs: List[object]
    raw_obs: List[np.ndarray] # for rule labels
    scaled_obs: List[np.ndarray] # for model inference
    feature_cols: List[str]
    split_payload: dict # contents of split.json
    test_keys: set # set of "recording:vehicle" strings.
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
    raw_seqs, feature_cols, split_payload, test_keys = load_test_sequences_from_experiment_split(exp_dir=exp_dir, data_root=data_root)  # feature_cols aligns with seq.obs columns
    
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
    trainer = HDVTrainer.load(ckpt_path)

    # 3) Build scale indices 
    cont_set = set(CONTINUOUS_FEATURES)
    scale_idx = np.array([i for i, name in enumerate(feature_cols) if name in cont_set], dtype=np.int64)

    # 4) Decide scaler mode from checkpoint contents.
    is_classwise = isinstance(trainer.scaler_mean, dict) and isinstance(trainer.scaler_std, dict)
    scaler_mode = "classwise" if is_classwise else "global"

    raw_obs = []
    scaled_obs = []

    for seq in raw_seqs: # For each test sequence: store RAW and compute SCALED
        obs_raw = np.asarray(seq.obs, dtype=np.float64)
        if obs_raw.ndim != 2:
            raise ValueError(f"Expected seq.obs to be (T,F), got {obs_raw.shape}")
        if obs_raw.shape[1] != len(feature_cols):
            raise ValueError(
                f"Feature dimension mismatch: seq.obs has F={obs_raw.shape[1]}, "
                f"but feature_cols has {len(feature_cols)}"
            )

        raw_obs.append(obs_raw)

        if is_classwise:
            cls = _get_meta_class(seq)
            if cls is None:
                raise AttributeError(
                    "Classwise scaling enabled, but seq.meta['meta_class'] is missing."
                )
            if cls not in trainer.scaler_mean or cls not in trainer.scaler_std:
                raise KeyError(
                    f"meta_class='{cls}' missing in checkpoint scalers. "
                    f"Available: {sorted(trainer.scaler_mean.keys())}"
                )
            obs_scaled = scale_obs_masked(obs_raw, trainer.scaler_mean[cls], trainer.scaler_std[cls], scale_idx)
        else:
            if trainer.scaler_mean is None or trainer.scaler_std is None:
                raise RuntimeError("Checkpoint missing global scaler_mean/std.")
            obs_scaled = scale_obs_masked(obs_raw, trainer.scaler_mean, trainer.scaler_std, scale_idx)

        scaled_obs.append(obs_scaled.astype(np.float32, copy=False))

    return trainer, TestSequences(
        raw_seqs=raw_seqs,
        raw_obs=raw_obs,
        scaled_obs=scaled_obs,
        feature_cols=feature_cols,
        split_payload=split_payload,
        test_keys=test_keys,
        scaler_mode=scaler_mode,
    )
