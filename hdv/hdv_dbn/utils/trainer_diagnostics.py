import numpy as np
import torch
import math


def _run_lengths_from_gamma_argmax(gamma_all):
    """
    Compute run-lengths (segment durations) from hard labels:
        z_hat[t] = argmax_k gamma[t,k]
    It is a per-timestep MAP assignment from the marginals

    Returns
    run_lengths : np.ndarray, shape (num_segments,)
        Lengths of contiguous segments across all trajectories.
    per_traj_median : np.ndarray, shape (num_traj,)
        Median run length within each trajectory (NaN if T==0).
    """
    all_runs = []
    traj_medians = []

    for gamma in gamma_all:
        T = int(gamma.shape[0])
        if T <= 0:
            traj_medians.append(np.nan)
            continue

        # hard path from marginals
        z_hat = torch.argmax(gamma, dim=1).detach().cpu().numpy().astype(np.int64)

        # compute contiguous run lengths
        runs = []
        cur = 1
        for t in range(1, T):
            if z_hat[t] == z_hat[t - 1]:
                cur += 1
            else:
                runs.append(cur)
                cur = 1
        runs.append(cur)

        runs_np = np.asarray(runs, dtype=np.int64)
        all_runs.append(runs_np)
        traj_medians.append(float(np.median(runs_np)) if runs_np.size > 0 else np.nan)

    if len(all_runs) == 0:
        run_lengths = np.asarray([], dtype=np.int64)
    else:
        run_lengths = np.concatenate(all_runs, axis=0)

    return run_lengths.astype(np.int64), np.asarray(traj_medians, dtype=np.float64)

def _posterior_entropy_from_gamma(num_states, gamma_all):
    """
    Compute normalized Shannon entropy of the posterior from gamma per timestep.
        0 → totally confident (posterior is ~one-hot)
        1 → totally uncertain (posterior ~uniform)

    Returns
    -------
    ent_all : np.ndarray, shape (sum_T,)
        Normalized entropies pooled over all timesteps in all trajectories. In [0,1].
    ent_mean_per_traj : np.ndarray, shape (num_traj,)
        Mean normalized entropy per trajectory (NaN if T==0).
    """
    K = int(num_states)
    logK = float(np.log(max(K, 2)))  # avoid divide-by-zero; K>=2 in practice

    ent_list = []
    ent_mean_traj = []

    for gamma in gamma_all:
        T = int(gamma.shape[0])
        if T <= 0:
            ent_mean_traj.append(np.nan)
            continue

        # gamma: (T,K). Ensure numerical stability.
        g = gamma.detach().cpu().numpy().astype(np.float64)
        g = np.clip(g, 1e-15, 1.0)
        g = g / g.sum(axis=1, keepdims=True)

        # H_t = -sum_k g*log(g)  -> (T,)
        H = -np.sum(g * np.log(g), axis=1)

        # normalized entropy in [0,1]
        Hn = H / logK
        ent_list.append(Hn)
        ent_mean_traj.append(float(np.mean(Hn)))

    ent_all = np.concatenate(ent_list, axis=0) if len(ent_list) else np.asarray([], dtype=np.float64)
    return ent_all.astype(np.float64), np.asarray(ent_mean_traj, dtype=np.float64)

def _posterior_weighted_key_feature_stats(num_states, obs_names, obs_used, gamma_all):
    """
    Compute posterior-weighted mean ± std per state for a small set of derived key features (semantics).

    Returns
    feat_names : list[str]
    means : np.ndarray, shape (K, F)
    stds  : np.ndarray, shape (K, F)
    """
    _assert_obs_gamma_aligned(obs_used, gamma_all)

    K = int(num_states)
    # indices in obs vector
    name_to_idx = {n: i for i, n in enumerate(obs_names)}

    def idx(name: str):
        return name_to_idx.get(name, None)

    # ego indices
    i_vx, i_vy = idx("vx_mean"), idx("vy_mean")
    i_ax, i_ay = idx("ax_mean"), idx("ay_mean")
    compute_speed_mag = (i_vx is not None and i_vy is not None)
    compute_acc_mag = (i_ax is not None and i_ay is not None)

    feat_specs = []  # (label, j, special)

    if compute_speed_mag:
        feat_specs.append(("speed_mag_mean", None, "ego_speed_mag"))
    if compute_acc_mag:
        feat_specs.append(("acc_mag_mean", None, "ego_acc_mag"))
    
    candidate_names = [
        "vx_mean","vx_std","vy_mean","vy_std",
        "ax_mean","ax_std","ay_mean","ay_std",
        "jerk_mean","jerk_std",
        "d_left_lane_mean","d_left_lane_min",
        "d_right_lane_mean","d_right_lane_min",
        "front_thw_mean","front_thw_min","front_thw_vfrac",
        "front_ttc_mean","front_ttc_min","front_ttc_vfrac",
        "front_dhw_mean","front_dhw_min","front_dhw_vfrac",
        "lc_left_present","lc_right_present",
        "front_exists_frac","rear_exists_frac",
        "left_front_exists_frac","left_side_exists_frac","left_rear_exists_frac",
        "right_front_exists_frac","right_side_exists_frac","right_rear_exists_frac",
    ]
    for n in candidate_names:
        j = idx(n)
        if j is not None:
            feat_specs.append((n, j, "direct")) 

    feat_names = [fs[0] for fs in feat_specs]
    F = len(feat_names)

    means = np.full((K, F), np.nan, dtype=np.float64)
    stds  = np.full((K, F), np.nan, dtype=np.float64)

    sum_w   = np.zeros((K, F), dtype=np.float64)
    sum_wx  = np.zeros((K, F), dtype=np.float64)
    sum_wx2 = np.zeros((K, F), dtype=np.float64)

    for obs, gamma in zip(obs_used, gamma_all):
        x = np.asarray(obs, dtype=np.float64)  # (T, D)
        g = gamma.detach().cpu().numpy().astype(np.float64)  # (T, K)

        # defensive normalization
        g = np.clip(g, 1e-15, 1.0)
        g = g / g.sum(axis=1, keepdims=True)

        # Precompute derived ego vectors once per trajectory
        v_mag = None
        a_mag = None
        if compute_speed_mag:
            v_mag = np.sqrt(x[:, i_vx] ** 2 + x[:, i_vy] ** 2)
        if compute_acc_mag:
            a_mag = np.sqrt(x[:, i_ax] ** 2 + x[:, i_ay] ** 2)

    # Accumulate posterior-weighted moments with finite masking
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

            v = np.where(finite, val, 0.0).astype(np.float64)
            fm = finite.astype(np.float64)

            W = g * fm[:, None]                # (T,K)
            sum_w[:, col]  += W.sum(axis=0)
            sum_wx[:, col] += (W * v[:, None]).sum(axis=0)
            sum_wx2[:, col]+= (W * (v*v)[:, None]).sum(axis=0)

    # finalize mean/std
    for k in range(K):
        for col in range(F):
            sw = sum_w[k, col]
            if sw <= 0.0:
                continue
            m = sum_wx[k, col] / sw
            ex2 = sum_wx2[k, col] / sw
            var = max(ex2 - m*m, 0.0)
            means[k, col] = m
            stds[k, col] = math.sqrt(var)

    return feat_names, means, stds

def _assert_obs_gamma_aligned(obs_used, gamma_all):
    if len(obs_used) != len(gamma_all):
        raise ValueError(f"len(obs_used)={len(obs_used)} != len(gamma_all)={len(gamma_all)}")

    for i, (obs, gamma) in enumerate(zip(obs_used, gamma_all)):
        if obs is None:
            raise ValueError(f"obs_used[{i}] is None (filter earlier or handle explicitly)")
        if gamma is None:
            raise ValueError(f"gamma_all[{i}] is None")
        if (len(obs) != int(gamma.shape[0])):
            raise ValueError(
                f"Trajectory {i} length mismatch: obs T={len(obs)} vs gamma T={int(gamma.shape[0])}"
            )
