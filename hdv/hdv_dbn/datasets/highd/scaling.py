import numpy as np
from ..dataset import TrajectorySequence
from .neighbours import _rel_to_exists_pairs

def compute_feature_scaler_masked(sequences, scale_idx):
    """
    Compute per-feature mean and standard deviation from a set of sequences,
    only for the features specified by scale_idx. For neighbor-relative features (*_dx/_dy/_dvx/_dvy),
    compute stats ONLY on frames where the corresponding *_exists == 1.

    Parameters
    sequences : sequence of TrajectorySequence
        Sequences whose `.obs` arrays will be stacked along time.
    scale_idx : list or np.ndarray
        Indices of features to compute mean/std for.
    
    Returns
    mean : np.ndarray
        Feature-wise mean, shape (F,).
    std : np.ndarray
        Feature-wise standard deviation, shape (F,). Very small std values are
        clamped to 1.0 to avoid division by near-zero during scaling.
    """
    if len(sequences) == 0:
        raise ValueError("No sequences provided.")
    
    obs_names = sequences[0].obs_names
    F = int(sequences[0].obs.shape[1])
    scale_idx = np.asarray(list(scale_idx), dtype=int)

    mean = np.zeros((F,), dtype=np.float64)
    std = np.ones((F,), dtype=np.float64)

    X = np.vstack([seq.obs for seq in sequences])  # (sumT, F)

    rel_pairs = dict(_rel_to_exists_pairs(obs_names)) # build rel_idx -> exists_idx mapping

    for j in scale_idx:
        col = X[:, j]
        mask = np.isfinite(col)

        # if this is a relative feature, only use rows where exists==1
        if j in rel_pairs:
            exj = rel_pairs[j]
            mask = mask & (X[:, exj] > 0.5)

        vals = col[mask]
        if vals.size == 0:
            mean[j] = 0.0
            std[j] = 1.0
            continue

        m = float(vals.mean())
        s = float(vals.std())
        mean[j] = m
        std[j] = 1.0 if s < 1e-6 else s

    return mean, std

def compute_classwise_feature_scalers_masked(sequences, scale_idx, class_key="meta_class"):
    """
    Compute one *masked* scaler per class.

    Parameters
    sequences : list[TrajectorySequence]
        Input sequences (usually training split only).
    scale_idx : list or np.ndarray
        Indices of features to compute mean/std for.
    class_key : str
        Key inside seq.meta indicating vehicle class.
    
    Returns
    dict
        Mapping: class_name -> (mean, std)
    """
    buckets = {}
    for seq in sequences:
        if not seq.meta or class_key not in seq.meta:
            continue
        cls = str(seq.meta[class_key])
        buckets.setdefault(cls, []).append(seq)

    scalers = {}
    for cls, seqs in buckets.items():
        scalers[cls] = compute_feature_scaler_masked(seqs, scale_idx=scale_idx)
    return scalers

def scale_sequences(sequences, mean, std):
    """
    Apply z-score scaling and then neutralize neighbor-relative features when the neighbor is absent.
    Scaling is applied as:
        obs_scaled = (obs - mean) / std

    Parameters
    sequences : sequence of TrajectorySequence
        Input sequences to be scaled.
    mean : np.ndarray
        Feature-wise mean, shape (F,).
    std : np.ndarray
        Feature-wise std, shape (F,).

    Returns
    list[TrajectorySequence]
        New list of sequences with scaled observations and neutralized absent-neighbor
        relative features. Metadata (vehicle_id, frames, obs_names, recording_id) is preserved.
    """
    out = []

    if len(sequences) == 0:
        return out
    
    # rel feature ↔ exists feature pairs
    rel_pairs = _rel_to_exists_pairs(sequences[0].obs_names)

    for seq in sequences:
        obs_scaled = (seq.obs - mean) / std
        
        # if neighbor doesn't exist, set its relative features to 0 (neutral in scaled space)
        for rel_idx, ex_idx in rel_pairs:
            absent = seq.obs[:, ex_idx] <= 0.5
            obs_scaled[absent, rel_idx] = 0.0
        
        out.append(
            TrajectorySequence(
                vehicle_id=seq.vehicle_id,
                frames=seq.frames,
                obs=obs_scaled,
                obs_names=seq.obs_names,
                recording_id=seq.recording_id,
                meta=seq.meta
            )
        )
    return out

def scale_sequences_classwise(sequences, scalers, class_key="meta_class"):
    """
    Apply class-specific feature scaling and neutralize absent-neighbor relative features.

    Parameters
    sequences : list[TrajectorySequence]
        Sequences to scale.
    scalers : dict
        Mapping: class_name -> (mean, std)
    class_key : str
        Key inside seq.meta indicating vehicle class.

    Returns
    list[TrajectorySequence]
        Scaled sequences.
    """
    out = []

    if len(sequences) == 0:
        return out
    
    rel_pairs = _rel_to_exists_pairs(sequences[0].obs_names)

    for seq in sequences:
        if seq.meta is None or class_key not in seq.meta:
            raise ValueError("Sequence missing class information for class-wise scaling.")

        cls = seq.meta[class_key]
        if cls not in scalers:
            raise ValueError(f"No scaler found for class '{cls}'.")

        mean, std = scalers[cls]
        obs_scaled = (seq.obs - mean) / std

        for rel_idx, ex_idx in rel_pairs:
            absent = seq.obs[:, ex_idx] <= 0.5
            obs_scaled[absent, rel_idx] = 0.0

        out.append(
            TrajectorySequence(
                vehicle_id=seq.vehicle_id,
                frames=seq.frames,
                obs=obs_scaled,
                obs_names=seq.obs_names,
                recording_id=seq.recording_id,
                meta=seq.meta,
            )
        )

    return out