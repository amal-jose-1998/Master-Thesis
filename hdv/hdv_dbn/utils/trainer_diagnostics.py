import numpy as np
import torch
import math


def run_lengths_from_gamma_sa_argmax(gamma_sa_seqs):
    """
    Compute run-lengths (segment durations) from hard labels:
        (s_hat[t], a_hat[t]) = argmax_{s,a} gamma[t,s,a]

    Returns
    run_lengths : np.ndarray, shape (num_segments,)
        Lengths of contiguous segments across all trajectories.
    per_traj_median : np.ndarray, shape (num_traj,)
        Median run length within each trajectory (NaN if T==0).
    """
    all_runs = []
    traj_medians = []

    for gamma in gamma_sa_seqs:
        if gamma is None:
            traj_medians.append(np.nan)
            continue

        T = int(gamma.shape[0])
        if T <= 0:
            traj_medians.append(np.nan)
            continue

        # argmax over (s,a) jointly
        flat = gamma.reshape(T, -1)                          # (T, S*A)
        idx = torch.argmax(flat, dim=1).detach().cpu().numpy().astype(np.int64)  # (T,)

        # compute contiguous run lengths
        runs = []
        cur = 1
        for t in range(1, T):
            if idx[t] == idx[t - 1]:
                cur += 1
            else:
                runs.append(cur)
                cur = 1
        runs.append(cur)

        runs_np = np.asarray(runs, dtype=np.int64)
        all_runs.append(runs_np)
        traj_medians.append(float(np.median(runs_np)) if runs_np.size > 0 else np.nan)

    run_lengths = np.concatenate(all_runs, axis=0) if len(all_runs) else np.asarray([], dtype=np.int64)
    return run_lengths.astype(np.int64), np.asarray(traj_medians, dtype=np.float64)

def posterior_entropy_from_gamma(gamma_sa_seqs, eps=1e-15):
    """
    Compute normalized Shannon entropy of the posterior from gamma per timestep.
        H_t = -Σ_{s,a} p_t(s,a) log p_t(s,a)
        Hn_t = H_t / log(S*A)  in [0,1] (approximately)

        0 → totally confident (posterior is ~one-hot)
        1 → totally uncertain (posterior ~uniform)

    Returns
    ent_all : np.ndarray, shape (sum_T,)
        Normalized entropies pooled over all timesteps in all trajectories. In [0,1].
    ent_mean_per_traj : np.ndarray, shape (num_traj,)
        Mean normalized entropy per trajectory (NaN if T==0).
    """
    ent_list = []
    ent_mean_traj = []

    for gamma in gamma_sa_seqs:
        if gamma is None:
            ent_mean_traj.append(np.nan)
            continue

        T, S, A = map(int, gamma.shape)
        if T <= 0:
            ent_mean_traj.append(np.nan)
            continue

        K = max(S * A, 2)
        logK = float(np.log(K))

        # gamma: (T,S,A). Ensure numerical stability.
        g = gamma.detach().cpu().numpy().astype(np.float64)  # (T,S,A)
        g = np.clip(g, eps, 1.0)
        g = g / g.sum(axis=(1, 2), keepdims=True)

        # H_t = -sum_k g*log(g)  -> (T,)
        H = -np.sum(g * np.log(g), axis=(1, 2))     # (T,)

        # normalized entropy in [0,1]
        Hn = H / logK                               # (T,)
        
        ent_list.append(Hn)
        ent_mean_traj.append(float(np.mean(Hn)) if Hn.size else np.nan)

    ent_all = np.concatenate(ent_list, axis=0) if len(ent_list) else np.asarray([], dtype=np.float64)
    return ent_all.astype(np.float64), np.asarray(ent_mean_traj, dtype=np.float64)

def posterior_weighted_feature_stats(obs_names, obs_seqs, gamma_sa_seqs, S=None, A=None):
    """
    Compute posterior-weighted mean ± std per state for a small set of derived key features (semantics).

    Returns
    feat_names : list[str]
    means : np.ndarray, shape (S,A,F)
    stds  : np.ndarray, shape (S,A,F)
    """
    _assert_obs_gamma_aligned(obs_seqs, gamma_sa_seqs)

    # indices in obs vector
    name_to_idx = {n: i for i, n in enumerate(obs_names)}
    
    def idx(name: str):
        return name_to_idx.get(name, None)

    # ego indices
    i_vx, i_vy = idx("vx_mean"), idx("vy_mean")
    i_ax, i_ay = idx("ax_mean"), idx("ay_mean")
    compute_speed_mag = (i_vx is not None and i_vy is not None)
    compute_acc_mag = (i_ax is not None and i_ay is not None)

    feat_specs = []
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

        x = np.asarray(obs, dtype=np.float64)                # (T,D)
        g = gamma.detach().cpu().numpy().astype(np.float64)  # (T,S,A)

        if g.shape[0] != x.shape[0]:
            raise ValueError(f"obs T={x.shape[0]} != gamma T={g.shape[0]}")

        if g.shape[1] != S or g.shape[2] != A:
            raise ValueError(f"gamma has shape {g.shape}, expected (T,{S},{A})")

        # defensive normalization
        g = np.clip(g, 1e-15, 1.0)
        g = g / g.sum(axis=(1, 2), keepdims=True)

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

            v = np.where(finite, val, 0.0).astype(np.float64)   # (T,)
            fm = finite.astype(np.float64)                      # (T,)

            # weight mask: g[t,s,a] * fm[t]
            W = g * fm[:, None, None]                           # (T,S,A)

            sum_w[:, :, col]  += W.sum(axis=0)                                  # (S,A)
            sum_wx[:, :, col] += np.einsum("tsa,t->sa", W, v)                   # (S,A)
            sum_wx2[:, :, col]+= np.einsum("tsa,t->sa", W, v * v)               # (S,A)

    den = np.maximum(sum_w, 1e-15)
    means = sum_wx / den
    var = np.maximum(sum_wx2 / den - means * means, 0.0)
    stds = np.sqrt(var)

    # keep NaNs where no mass
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
        if (len(obs) != int(gamma.shape[0])):
            raise ValueError(
                f"Trajectory {i} length mismatch: obs T={len(obs)} vs gamma T={int(gamma.shape[0])}"
            )
