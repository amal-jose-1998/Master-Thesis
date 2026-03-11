import numpy as np
from ..dataset import TrajectorySequence

def compute_feature_scaler(sequences, scale_idx):
    """
    Compute per-feature mean/std for selected features (scale_idx), ignoring NaNs/Infs.
    If a feature has a companion "<prefix>_vfrac", statistics are computed only on
    rows where vfrac > 0 (feature defined at least once in the window).

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
        raise ValueError("[compute_feature_scaler] No sequences provided.")
    
    obs_names = list(sequences[0].obs_names) # list of feature names from the first sequence.
    name_to_idx = {n: i for i, n in enumerate(obs_names)} # dict mapping feature name → column index.

    F = int(sequences[0].obs.shape[1]) # number of features
    scale_idx = np.asarray(scale_idx, dtype=int)

    mean = np.zeros(F, dtype=np.float64)
    std = np.ones(F, dtype=np.float64)

    # Prepare vfrac gating info
    gated = []                                              # list of (j, vf_idx_or_None)
    for j in scale_idx:
        fname = obs_names[j]                                # name of this feature
        vf_idx = None
        if fname.endswith(("_mean", "_min", "_std")):
            vfrac_name = fname.rsplit("_", 1)[0] + "_vfrac" # corresponding vfrac feature name
            vf_idx = name_to_idx.get(vfrac_name)            # index of vfrac feature, or None if not found
        gated.append((int(j), vf_idx))                      # store index and vfrac index (or None)

    # Pass 1: mean 
    sum_j = np.zeros(F, dtype=np.float64) # sum accumulator per feature
    cnt_j = np.zeros(F, dtype=np.int64)   # count of valid entries per feature

    for seq in sequences:
        X = np.asarray(seq.obs, dtype=np.float64)  # (T, F)
        for j, vf_idx in gated:     # Loop over features to scale with vfrac info
            col = X[:, j]           # Extracts the entire feature column across all stacked rows.
            mask = np.isfinite(col) # True where col is not NaN and not ±Inf.

            if vf_idx is not None: 
                vf = X[:, vf_idx]                    # corresponding vfrac column for those features that have it
                mask &= np.isfinite(vf) & (vf > 0.0) # Update mask to include only rows where vfrac > 0.

            if np.any(mask): 
                vals = col[mask]              # Select only valid entries for this feature.
                sum_j[j] += float(vals.sum()) # accumulate sum
                cnt_j[j] += int(vals.size)    # accumulate count

    # finalize mean only for requested features
    for j in scale_idx:
        j = int(j)
        if cnt_j[j] > 0:
            mean[j] = sum_j[j] / cnt_j[j]
        else:
            mean[j] = 0.0  # keep identity scaling


    # Pass 2: std (uses computed mean)
    ssq_j = np.zeros(F, dtype=np.float64) # sum of squares accumulator per feature

    for seq in sequences:
        X = np.asarray(seq.obs, dtype=np.float64)
        for j, vf_idx in gated:
            col = X[:, j]
            mask = np.isfinite(col)

            if vf_idx is not None:
                vf = X[:, vf_idx]
                mask &= np.isfinite(vf) & (vf > 0.0)

            if np.any(mask):
                vals = col[mask] - mean[j]
                ssq_j[j] += float(np.dot(vals, vals))  # sum of squares

    for j in scale_idx:
        j = int(j)
        if cnt_j[j] > 0:
            var = ssq_j[j] / cnt_j[j]  # matches numpy std with ddof=0
            s = float(np.sqrt(var))
            std[j] = 1.0 if s < 1e-6 else s
        else:
            std[j] = 1.0

    return mean, std

def compute_classwise_feature_scalers(sequences, scale_idx, class_key="meta_class"):
    """
    Compute one *masked* scaler per class: {class_name: (mean, std)}.

    Parameters
    sequences : list[TrajectorySequence]
        Input sequences (usually training split only).
    scale_idx : list or np.ndarray
        Indices of features to compute mean/std for.
    class_key : str
        Key inside seq.meta indicating vehicle class.
    
    Returns
    dict
        Mapping: class_name -> (mean_vec, std_vec)
    """
    buckets = {} # to hold sequences per class
    for seq in sequences:
        if seq.meta and class_key in seq.meta:
            buckets.setdefault(str(seq.meta[class_key]), []).append(seq) # vehicle sequence assigned to its class bucket

    # Returns: { "car": (mean_vec,std_vec), "truck": (mean_vec,std_vec), ... }
    return {
        cls: compute_feature_scaler(seqs, scale_idx)
        for cls, seqs in buckets.items()
    }

def _scale_obs(x, mean, std):
    """
    Scale only finite entries in x. NaNs/Infs are preserved exactly.
    """
    x = np.asarray(x, dtype=np.float64) # shape (T, F)
    # Convert mean/std to 1D float arrays of shape (F,).
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    std = np.asarray(std, dtype=np.float64).reshape(-1)

    if x.shape[1] != mean.shape[0]:
        raise ValueError(f"mean/std shape mismatch: mean {mean.shape}, obs {x.shape}")

    finite = np.isfinite(x) # True where x[t,f] is a real number.
    z = (x - mean[None, :]) / std[None, :]

    out = x.copy()
    out[finite] = z[finite] # Only update finite entries.
    return out

def scale_sequences(sequences, mean, std):
    """
    Apply z-score scaling to finite entries only (masked NaNs design).
    NaNs/Infs remain NaN/Inf. No imputation is performed.

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
    if len(sequences) == 0:
        return []

    out = []
    for seq in sequences:
        obs_scaled = _scale_obs(seq.obs, mean, std)
        out.append(TrajectorySequence(
            vehicle_id=seq.vehicle_id,
            frames=seq.frames,
            obs=obs_scaled,
            obs_names=seq.obs_names,
            recording_id=seq.recording_id,
            meta=seq.meta
        ))
    return out

def scale_sequences_classwise(sequences, scalers, class_key="meta_class"):
    """
    Apply class-specific masked scaling (finite entries only).

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
    if len(sequences) == 0:
        return []

    out = []
    for seq in sequences:
        if not seq.meta or class_key not in seq.meta:
            raise ValueError("[scale_sequences_classwise] Missing class information.")

        cls = seq.meta[class_key] # Read class label.
        if cls not in scalers:
            raise ValueError(f"[scale_sequences_classwise] No scaler for class '{cls}'.")

        mean, std = scalers[cls]
        obs_scaled = _scale_obs(seq.obs, mean, std)

        out.append(TrajectorySequence(
            vehicle_id=seq.vehicle_id,
            frames=seq.frames,
            obs=obs_scaled,
            obs_names=seq.obs_names,
            recording_id=seq.recording_id,
            meta=seq.meta
        ))
    return out