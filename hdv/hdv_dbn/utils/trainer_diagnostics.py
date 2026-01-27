import numpy as np
import torch
import math


import numpy as np
import torch

def _run_lengths_from_hard_labels(idx_1d: np.ndarray):
    """Helper: contiguous run lengths for a 1D integer label sequence."""
    T = int(idx_1d.shape[0])
    if T <= 0:
        return np.asarray([], dtype=np.int64)

    runs = []
    cur = 1
    for t in range(1, T):
        if idx_1d[t] == idx_1d[t - 1]:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    return np.asarray(runs, dtype=np.int64)

def run_lengths_from_gamma_sa(gamma_sa_seqs):
    """
    Compute run-length diagnostics from gamma[t,s,a] (T,S,A):

      - joint (s,a): argmax over (s,a)
      - style only  : argmax over s of sum_a gamma[t,s,a]
      - action only : argmax over a of sum_s gamma[t,s,a]

    Returns
    -------
    joint_runs, joint_median_per_traj
    style_runs, style_median_per_traj
    action_runs, action_median_per_traj
    """
    all_joint, med_joint = [], []
    all_s, med_s = [], []
    all_a, med_a = [], []

    for gamma in gamma_sa_seqs:
        if gamma is None:
            med_joint.append(np.nan); med_s.append(np.nan); med_a.append(np.nan)
            continue

        T, S, A = map(int, gamma.shape)
        if T <= 0:
            med_joint.append(np.nan); med_s.append(np.nan); med_a.append(np.nan)
            continue

        g = gamma.reshape(T, S, A)

        # --- joint hard labels (flatten only for argmax convenience) ---
        flat = g.reshape(T, S * A)
        idx_joint = torch.argmax(flat, dim=1).detach().cpu().numpy().astype(np.int64)

        # --- style hard labels: marginalize over action ---
        g_s = g.sum(dim=2)  # (T,S)
        idx_s = torch.argmax(g_s, dim=1).detach().cpu().numpy().astype(np.int64)

        # --- action hard labels: marginalize over style ---
        g_a = g.sum(dim=1)  # (T,A)
        idx_a = torch.argmax(g_a, dim=1).detach().cpu().numpy().astype(np.int64)

        # run lengths
        rj = _run_lengths_from_hard_labels(idx_joint)
        rs = _run_lengths_from_hard_labels(idx_s)
        ra = _run_lengths_from_hard_labels(idx_a)

        all_joint.append(rj); med_joint.append(float(np.median(rj)) if rj.size else np.nan)
        all_s.append(rs);     med_s.append(float(np.median(rs)) if rs.size else np.nan)
        all_a.append(ra);     med_a.append(float(np.median(ra)) if ra.size else np.nan)

    joint_runs = np.concatenate(all_joint) if all_joint else np.asarray([], dtype=np.int64)
    style_runs = np.concatenate(all_s) if all_s else np.asarray([], dtype=np.int64)
    action_runs = np.concatenate(all_a) if all_a else np.asarray([], dtype=np.int64)

    return (
        joint_runs,  np.asarray(med_joint, dtype=np.float64),
        style_runs,  np.asarray(med_s, dtype=np.float64),
        action_runs, np.asarray(med_a, dtype=np.float64),
    )

def _entropy_normalized(p: np.ndarray, axis: int = -1, eps: float = 1e-15) -> np.ndarray:
    """
    Normalized Shannon entropy along `axis`:
      H(p)/log(K), where K = size along axis.
    Uses 0*log(0)=0 (no clipping distortion).
    """
    K = int(p.shape[axis])
    K = max(K, 2)
    logK = float(np.log(K))

    den = np.maximum(p.sum(axis=axis, keepdims=True), eps)
    p = p / den

    # 0*log(0)=0
    mask = (p > 0.0)
    H = -np.sum(np.where(mask, p * np.log(p), 0.0), axis=axis)
    return H / logK


def posterior_entropy_from_gamma_sa(gamma_sa_seqs, eps: float = 1e-15):
    """
    Compute normalized entropies from gamma[t,s,a] (T,S,A):

      - joint entropy: H(S,A) normalized by log(S*A)
      - style entropy: H(S)   normalized by log(S)   using gamma_s[t,s] = sum_a gamma[t,s,a]
      - action entropy:H(A)   normalized by log(A)   using gamma_a[t,a] = sum_s gamma[t,s,a]

    Returns
    -------
    ent_joint_all : (sum_T,)
    ent_joint_mean_per_traj : (num_traj,)
    ent_style_all : (sum_T,)
    ent_style_mean_per_traj : (num_traj,)
    ent_action_all : (sum_T,)
    ent_action_mean_per_traj : (num_traj,)
    """
    joint_list, style_list, action_list = [], [], []
    joint_mean, style_mean, action_mean = [], [], []

    for gamma in gamma_sa_seqs:
        if gamma is None:
            joint_mean.append(np.nan); style_mean.append(np.nan); action_mean.append(np.nan)
            continue

        T, S, A = map(int, gamma.shape)
        if T <= 0:
            joint_mean.append(np.nan); style_mean.append(np.nan); action_mean.append(np.nan)
            continue

        g = gamma.detach().cpu().numpy().astype(np.float64)  # (T,S,A)

        # --- joint entropy over (s,a): flatten last two dims ---
        g_joint = g.reshape(T, S * A)                         # (T, SA)
        Hj = _entropy_normalized(g_joint, axis=1, eps=eps)     # (T,)

        # --- style marginal entropy ---
        g_s = g.sum(axis=2)                                   # (T,S)
        Hs = _entropy_normalized(g_s, axis=1, eps=eps)         # (T,)

        # --- action marginal entropy ---
        g_a = g.sum(axis=1)                                   # (T,A)
        Ha = _entropy_normalized(g_a, axis=1, eps=eps)         # (T,)

        joint_list.append(Hj); style_list.append(Hs); action_list.append(Ha)
        joint_mean.append(float(np.mean(Hj))); style_mean.append(float(np.mean(Hs))); action_mean.append(float(np.mean(Ha)))

    ent_joint_all = np.concatenate(joint_list) if joint_list else np.asarray([], dtype=np.float64)
    ent_style_all = np.concatenate(style_list) if style_list else np.asarray([], dtype=np.float64)
    ent_action_all = np.concatenate(action_list) if action_list else np.asarray([], dtype=np.float64)

    return (
        ent_joint_all,  np.asarray(joint_mean, dtype=np.float64),
        ent_style_all,  np.asarray(style_mean, dtype=np.float64),
        ent_action_all, np.asarray(action_mean, dtype=np.float64),
    )

def posterior_weighted_feature_stats(obs_names, obs_seqs, gamma_sa_seqs, semantic_feature_names=None, include_derived=True, S=None, A=None):
    """
    Posterior-weighted mean ± std per (s,a) state for semantics features.

    - Config-driven (no hard-coded feature lists).
    - Computes posterior-weighted moments:
        mean = Σ_t γ_t(s,a) x_t / Σ_t γ_t(s,a)
        var  = Σ_t γ_t(s,a) x_t^2 / Σ_t γ_t(s,a) - mean^2
    - Finite masking per feature.

    Returns
    -------
    feat_names : list[str]
    means : np.ndarray, shape (S,A,F)
    stds  : np.ndarray, shape (S,A,F)
    """
    _assert_obs_gamma_aligned(obs_seqs, gamma_sa_seqs)

    # ------------------------------------------------------------------
    # Resolve semantics feature list from config (single source of truth)
    # ------------------------------------------------------------------
    if semantic_feature_names is None:
        try:
            from ..config import WINDOW_FEATURE_COLS  # type: ignore
            semantic_feature_names = list(WINDOW_FEATURE_COLS)
        except Exception:
            # Fallback if called outside package context
            semantic_feature_names = list(obs_names)

    # indices in obs vector
    name_to_idx = {n: i for i, n in enumerate(obs_names)}
    def idx(name):
        return name_to_idx.get(name, None)

    # ------------------------------------------------------------------
    # Optional derived features (only if base components exist)
    # ------------------------------------------------------------------
    i_vx, i_vy = idx("vx_mean"), idx("vy_mean")
    i_ax, i_ay = idx("ax_mean"), idx("ay_mean")
    compute_speed_mag = (i_vx is not None and i_vy is not None)
    compute_acc_mag = (i_ax is not None and i_ay is not None)

    feat_specs = []
    if include_derived and compute_speed_mag:
        feat_specs.append(("speed_mag_mean", None, "ego_speed_mag"))
    if include_derived and compute_acc_mag:
        feat_specs.append(("acc_mag_mean", None, "ego_acc_mag"))

    # Direct features from config list (in order), only those present in obs_names
    for n in semantic_feature_names:
        j = idx(str(n))
        if j is not None:
            feat_specs.append((str(n), j, "direct"))

    feat_names = [fs[0] for fs in feat_specs]
    F = len(feat_names)

    # infer S,A from first non-empty gamma
    infer_S = infer_A = None
    for g in gamma_sa_seqs:
        if g is not None and g.numel() > 0:
            _, infer_S, infer_A = map(int, g.shape)
            break

    if infer_S is None:
        return feat_names, np.zeros((0, 0, F)), np.zeros((0, 0, F))

    if S is None:
        S = infer_S
    if A is None:
        A = infer_A

    if int(S) != int(infer_S) or int(A) != int(infer_A):
        raise ValueError(f"Provided S,A=({S},{A}) but gamma has ({infer_S},{infer_A}).")

    S = int(S)
    A = int(A)

    sum_w   = np.zeros((S, A, F), dtype=np.float64)
    sum_wx  = np.zeros((S, A, F), dtype=np.float64)
    sum_wx2 = np.zeros((S, A, F), dtype=np.float64)

    for obs, gamma in zip(obs_seqs, gamma_sa_seqs):
        if obs is None or len(obs) == 0:
            continue

        x = np.asarray(obs, dtype=np.float64)                 # (T,D)
        g = gamma.detach().cpu().numpy().astype(np.float64)   # (T,S,A)

        if g.shape[0] != x.shape[0]:
            raise ValueError(f"obs T={x.shape[0]} != gamma T={g.shape[0]}")
        if g.shape[1] != S or g.shape[2] != A:
            raise ValueError(f"gamma has shape {g.shape}, expected (T,{S},{A})")

        # Normalize per timestep WITHOUT element clipping (avoids artificial mass injection)
        den = np.maximum(g.sum(axis=(1, 2), keepdims=True), 1e-15)
        g = g / den

        # Precompute derived vectors once per trajectory
        v_mag = None
        a_mag = None
        if include_derived and compute_speed_mag:
            v_mag = np.sqrt(x[:, i_vx] ** 2 + x[:, i_vy] ** 2)
        if include_derived and compute_acc_mag:
            a_mag = np.sqrt(x[:, i_ax] ** 2 + x[:, i_ay] ** 2)

        for col, (label, j, special) in enumerate(feat_specs):
            if special == "ego_speed_mag":
                val = v_mag
            elif special == "ego_acc_mag":
                val = a_mag
            else:
                val = x[:, j]

            finite = np.isfinite(val)
            if not np.any(finite):
                continue

            v = np.where(finite, val, 0.0).astype(np.float64)   # (T,)
            fm = finite.astype(np.float64)                      # (T,)

            W = g * fm[:, None, None]                           # (T,S,A)
            sum_w[:, :, col]  += W.sum(axis=0)                  # (S,A)
            sum_wx[:, :, col] += np.einsum("tsa,t->sa", W, v)   # (S,A)
            sum_wx2[:, :, col]+= np.einsum("tsa,t->sa", W, v*v) # (S,A)

    den = np.maximum(sum_w, 1e-15)
    means = sum_wx / den
    var = np.maximum(sum_wx2 / den - means * means, 0.0)
    stds = np.sqrt(var)

    means[sum_w <= 0.0] = np.nan
    stds[sum_w <= 0.0] = np.nan

    return feat_names, means, stds


def _assert_obs_gamma_aligned(obs_seqs, gamma_sa_seqs):
    if len(obs_seqs) != len(gamma_sa_seqs):
        raise ValueError(f"len(obs_used)={len(obs_seqs)} != len(gamma_all)={len(gamma_sa_seqs)}")

    for i, (obs, gamma) in enumerate(zip(obs_seqs, gamma_sa_seqs)):
        if obs is None:
            raise ValueError(f"obs_used[{i}] is None (filter earlier or handle explicitly)")
        if gamma is None:
            raise ValueError(f"gamma_all[{i}] is None")
        if len(obs) != int(gamma.shape[0]):
            raise ValueError(f"Trajectory {i} length mismatch: obs T={len(obs)} vs gamma T={int(gamma.shape[0])}")