import numpy as np

def _entropy_normalized(p, axis=-1, eps=1e-15):
    """
    Normalized Shannon entropy along `axis`:
      H(p)/log(K), where K = size along axis.
    """
    K = int(p.shape[axis])
    K = max(K, 2)
    logK = float(np.log(K)) # normalization constant

    den = np.maximum(p.sum(axis=axis, keepdims=True), eps) # the total mass along axis
    p = p / den # turns p into a proper probability distribution.

    # 0*log(0)=0
    mask = (p > 0.0)
    logp = np.zeros_like(p)
    logp[mask] = np.log(p[mask])   # log computed ONLY where safe

    H = -np.sum(p * logp, axis=axis) # Shannon entropy
    return H / logK

def _finite_1d(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    return x[np.isfinite(x)]

def _summarize_entropy(prefix, per_traj):
    x = _finite_1d(per_traj)
    if x.size == 0:
        return {
            f"{prefix}_vehicle_n": 0,
            f"{prefix}_vehicle_p10": float("nan"),
            f"{prefix}_vehicle_median": float("nan"),
            f"{prefix}_vehicle_p90": float("nan"),
        }

    return {
        f"{prefix}_vehicle_n": int(x.size),
        f"{prefix}_vehicle_p10": float(np.percentile(x, 10)),
        f"{prefix}_vehicle_median": float(np.median(x)),
        f"{prefix}_vehicle_p90": float(np.percentile(x, 90)),
    }

def posterior_entropy_from_gamma_sa(gamma_sa_seqs, eps=1e-15):
    """
    Compute normalized posterior entropies for an EM-trained DBN with
    joint latent states (style, action). 
    For each trajectory and each timestep t, the function computes:
        - Joint entropy H(S_t, A_t) from the joint posterior γ_t(s, a), normalized by log(S · A).
        - Style entropy H(S_t) from the marginal posterior γ_t(s) = Σ_a γ_t(s, a), normalized by log(S).
        - Action entropy H(A_t) from the marginal posterior γ_t(a) = Σ_s γ_t(s, a), normalized by log(A).
    Entropies are normalized to lie in [0, 1], allowing comparison across models with different numbers of latent states.

    Parameters
    gamma_sa_seqs
        List of posterior tensors γ_t(s, a), one per trajectory, each of shape (T, S, A).
    eps
        Small constant used to prevent numerical instability when normalizing probability distributions.

    Returns
    H_joint_mat, H_style_mat, H_action_mat
        Arrays of shape (N, Tmax) with NaN padding.
    lengths
        Array of shape (N,) containing each trajectory length.
    summary
        Dict of robust scalars for cross-experiment comparison.
    """
    # Per-trajectory entropy series
    Hj_list, Hs_list, Ha_list = [], [], []
    lengths = []

    for gamma in gamma_sa_seqs:
        if gamma is None:
            Hj_list.append(np.asarray([], dtype=np.float64))
            Hs_list.append(np.asarray([], dtype=np.float64))
            Ha_list.append(np.asarray([], dtype=np.float64))
            lengths.append(0)
            continue

        T, S, A = map(int, gamma.shape)
        if T <= 0:
            Hj_list.append(np.asarray([], dtype=np.float64))
            Hs_list.append(np.asarray([], dtype=np.float64))
            Ha_list.append(np.asarray([], dtype=np.float64))
            lengths.append(0)
            continue

        g = gamma.detach().cpu().numpy().astype(np.float64)  # (T,S,A)

        # joint entropy over SA
        g_joint = g.reshape(T, S * A)                         # (T, SA)
        Hj = _entropy_normalized(g_joint, axis=1, eps=eps)     # (T,)

        # style marginal entropy
        g_s = g.sum(axis=2)                                   # (T,S)
        Hs = _entropy_normalized(g_s, axis=1, eps=eps)         # (T,)

        # action marginal entropy
        g_a = g.sum(axis=1)                                   # (T,A)
        Ha = _entropy_normalized(g_a, axis=1, eps=eps)         # (T,)

        Hj_list.append(Hj)
        Hs_list.append(Hs)
        Ha_list.append(Ha)
        lengths.append(T)

    lengths = np.asarray(lengths, dtype=np.int64)
    N = int(len(lengths))
    Tmax = int(lengths.max()) if N > 0 else 0

    # Padded heatmap matrices (vehicles x time)
    H_joint_mat = np.full((N, Tmax), np.nan, dtype=np.float64)
    H_style_mat = np.full((N, Tmax), np.nan, dtype=np.float64)
    H_action_mat = np.full((N, Tmax), np.nan, dtype=np.float64)
    for i in range(N):
        Ti = int(lengths[i])
        if Ti <= 0:
            continue
        H_joint_mat[i, :Ti] = Hj_list[i]
        H_style_mat[i, :Ti] = Hs_list[i]
        H_action_mat[i, :Ti] = Ha_list[i]
    
    # Per-trajectory means (vehicle-level)
    #ent_joint_per_traj = np.asarray(
    #    [np.nanmean(H_joint_mat[i, :lengths[i]]) if lengths[i] > 0 else np.nan for i in range(N)],
    #    dtype=np.float64
    #)
    #ent_style_per_traj = np.asarray(
    #    [np.nanmean(H_style_mat[i, :lengths[i]]) if lengths[i] > 0 else np.nan for i in range(N)],
    #    dtype=np.float64
    #)
    #ent_action_per_traj = np.asarray(
    #    [np.nanmean(H_action_mat[i, :lengths[i]]) if lengths[i] > 0 else np.nan for i in range(N)],
    #    dtype=np.float64
    #)

    # Per-trajectory medians (vehicle-level, robust)
    ent_joint_per_traj = np.asarray(
        [np.nanmedian(H_joint_mat[i, :lengths[i]]) if lengths[i] > 0 else np.nan for i in range(N)],
        dtype=np.float64
    )
    ent_style_per_traj = np.asarray(
        [np.nanmedian(H_style_mat[i, :lengths[i]]) if lengths[i] > 0 else np.nan for i in range(N)],
        dtype=np.float64
    )
    ent_action_per_traj = np.asarray(
        [np.nanmedian(H_action_mat[i, :lengths[i]]) if lengths[i] > 0 else np.nan for i in range(N)],
        dtype=np.float64
    )

    summary = {}
    summary.update(_summarize_entropy("ent_joint",  ent_joint_per_traj))
    summary.update(_summarize_entropy("ent_style",  ent_style_per_traj))
    summary.update(_summarize_entropy("ent_action", ent_action_per_traj))

    return H_joint_mat, H_style_mat, H_action_mat, lengths, summary



def posterior_weighted_feature_stats(obs_names, obs_seqs, gamma_sa_seqs, semantic_feature_names=None, include_derived=True, S=None, A=None):
    """
    Posterior-weighted mean ± std per (s,a) state for semantics features.
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

        # Normalize per timestep 
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