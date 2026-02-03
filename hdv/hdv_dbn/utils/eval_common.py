import numpy as np


def scale_obs_masked(obs, mean, std, scale_idx):
    """
    Z-score scale only selected columns, but only where entries are finite.
    NaN/Inf are preserved exactly.
    """
    x = np.asarray(obs, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    std = np.asarray(std, dtype=np.float64).reshape(-1)

    if x.ndim != 2:
        raise ValueError(f"obs must be 2D (T,F), got {x.shape}")
    if x.shape[1] != mean.shape[0] or x.shape[1] != std.shape[0]:
        raise ValueError(f"mean/std mismatch: mean {mean.shape}, std {std.shape}, obs {x.shape}")

    out = x.copy()
    scale_idx = np.asarray(scale_idx, dtype=np.int64).reshape(-1)
    if scale_idx.size == 0:
        return out

    cols = out[:, scale_idx]
    finite = np.isfinite(cols)

    denom = std[scale_idx] + 1e-12
    z = (cols - mean[scale_idx][None, :]) / denom[None, :]

    cols_out = cols.copy()
    cols_out[finite] = z[finite]
    out[:, scale_idx] = cols_out
    return out


def seq_key(seq):
    """
    Canonical stable key for a (recording_id, vehicle_id) sequence.
    Returns "rid:vid" as strings.
    """
    return f"{seq.recording_id}:{seq.vehicle_id}"