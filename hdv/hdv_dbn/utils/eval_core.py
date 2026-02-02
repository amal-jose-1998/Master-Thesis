import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass
import torch

from ..trainer import HDVTrainer
from ..config import CONTINUOUS_FEATURES, DBN_STATES, TRAINING_CONFIG
from ..inference import infer_posterior  
from .trainer_diagnostics import posterior_entropy_from_gamma_sa


# -----------------------------
# small utilities
# -----------------------------

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

# -----------------------------
# metrics helpers
# -----------------------------
def param_count(trainer):
    """
    Compute the exact number of free parameters k for the *implemented* model.

    Counts:
    - CPD parameters (with simplex constraints: K-1 free params per categorical)
    - Emission parameters based on the configured emission model class:
        * hierarchical: (S,A,Dc) Gaussian mean/var + (S,A,B) Bernoulli p
        * poe: (S,Dc) and (A,Dc) Gaussian mean/var + (S,B) and (A,B) Bernoulli p

    Notes:
    - If TRAINING_CONFIG.learn_pi0 is False, initial distributions are treated as fixed
      and are not counted.
    - If TRAINING_CONFIG.disable_discrete_obs is True, Bernoulli parameters are not counted.

    Returns:
    - k_total: int
    """
    S=int(len(DBN_STATES.driving_style))
    A=int(len(DBN_STATES.action))
    
    learn_pi0 = bool(getattr(TRAINING_CONFIG, "learn_pi0", False))
    if learn_pi0:
        k_pi = (S - 1) + S * (A - 1)
    else:
        k_pi = 0

    k_As = S * (S - 1)
    k_Aa = S * A * (A - 1)

    k_em = 0
    try:
        em = trainer.emissions
        disable_discrete = bool(getattr(TRAINING_CONFIG, "disable_discrete_obs", False))
        
        Dc = int(getattr(em, "cont_dim", 0))
        B = int(getattr(em, "bin_dim", 0))
        if disable_discrete:
            B = 0
        
        em_name = str(getattr(TRAINING_CONFIG, "emission_model", "")).lower().strip()
        if em_name == "hierarchical":
            # gauss_mean (S,A,Dc), gauss_var (S,A,Dc), bern_p (S,A,B)
            k_em = 2 * S * A * Dc + S * A * B
        elif em_name == "poe":
            # style mean/var (S,Dc), action mean/var (A,Dc),
            # style p (S,B), action p (A,B)
            k_em = 2 * (S + A) * Dc + (S + A) * B
        else:
            raise ValueError(f"Unknown emission_model='{em_name}'. Expected 'hierarchical' or 'poe'.")

    except Exception as e:
        raise RuntimeError("Emission parameter counting failed") from e
    
    k_total = int(k_pi + k_As + k_Aa + k_em)
    breakdown = dict(k_pi=int(k_pi), k_As=int(k_As), k_Aa=int(k_Aa), k_em=int(k_em))

    return k_total, breakdown


def occupancy_and_keff(gamma_sa_seqs):
    joint_sum = None
    style_sum = None
    action_sum = None
    total_T = 0

    for g in gamma_sa_seqs:
        if g is None or g.numel() == 0:
            continue
        gg = g.detach()
        T, S, A = map(int, gg.shape)
        total_T += T

        j = gg.reshape(T, S * A).sum(dim=0)  # (SA,)
        s = gg.sum(dim=2).sum(dim=0)         # (S,)
        a = gg.sum(dim=1).sum(dim=0)         # (A,)

        joint_sum = j if joint_sum is None else (joint_sum + j)
        style_sum = s if style_sum is None else (style_sum + s)
        action_sum = a if action_sum is None else (action_sum + a)

    if total_T <= 0 or joint_sum is None:
        return dict(
            occ_joint_max=np.nan, occ_style_max=np.nan, occ_action_max=np.nan,
            keff_joint=np.nan, keff_style=np.nan, keff_action=np.nan,
        )

    def norm_np(x):
        z = x / (x.sum() + 1e-12)
        return z.cpu().numpy().astype(np.float64)

    pj = norm_np(joint_sum)
    ps = norm_np(style_sum)
    pa = norm_np(action_sum)

    def keff(p):
        p = np.maximum(p, 1e-15)
        H = -np.sum(p * np.log(p))
        return float(np.exp(H))

    return dict(
        occ_joint_max=float(np.max(pj)),
        occ_style_max=float(np.max(ps)),
        occ_action_max=float(np.max(pa)),
        keff_joint=keff(pj),
        keff_style=keff(ps),
        keff_action=keff(pa),
    )

def plot_entropy_heatmaps(H_joint, H_style, H_action, lengths, out_dir, prefix="entropy", 
                          sort_by="joint_median",   # "joint_median", "length", or None 
                          vmax=1.0,               # entropy is normalized in [0,1]
                          ):
    """
    Save three entropy heatmaps (joint/style/action) as PNG files.

    Parameters
    H_joint, H_style, H_action
        Arrays shaped (N, Tmax) with NaN padding.
    lengths
        Array shaped (N,) holding valid lengths for each row.
    out_dir
        Folder to save PNGs.
    prefix
        File prefix for outputs.
    sort_by
        Row ordering strategy:
          - "joint_median": sort by median joint entropy per vehicle (ascending)
          - "length": sort by sequence length (descending)
          - None: keep original order
    vmax
        Upper limit for color scaling (use 1.0 for normalized entropy).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lengths = np.asarray(lengths, dtype=np.int64)
    N = int(H_joint.shape[0])
    if N == 0:
        return {}

    # Sorting (stable across runs if you keep same input order)
    if sort_by == "joint_median":
        row_score = np.nanmedian(H_joint, axis=1)
        order = np.argsort(row_score)  # low -> high uncertainty
    elif sort_by == "length":
        order = np.argsort(-lengths)   # long -> short
    else:
        order = np.arange(N)

    Hj = H_joint[order]
    Hs = H_style[order]
    Ha = H_action[order]

    def _save_one(H, title, fname):
        plt.figure(figsize=(10, 6))
        plt.imshow(H, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0.0, vmax=vmax)
        plt.colorbar(label="Normalized posterior entropy")
        plt.xlabel("Time step (window index)")
        plt.ylabel("Vehicle (sorted)" if sort_by else "Vehicle")
        plt.title(title)
        path = out_dir / fname
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        return str(path)

    paths = {}
    paths["entropy_joint_png"]  = _save_one(Hj, f"{prefix}: Joint entropy H(S,A)",  f"{prefix}_joint.png")
    paths["entropy_style_png"]  = _save_one(Hs, f"{prefix}: Style entropy H(S)",    f"{prefix}_style.png")
    paths["entropy_action_png"] = _save_one(Ha, f"{prefix}: Action entropy H(A)",   f"{prefix}_action.png")

    return paths


def plot_T_vs_avg_nll(per_seq, out_dir, fname="T_vs_avg_nll.png", title="Online predictive: T vs avg NLL"):
    """
    Scatter plot of trajectory length T (windows) vs avg NLL per window.

    Parameters
    per_seq : dict
        key -> {"T": int, "avg_nll": float}
    out_dir : str or Path
        Where to save the PNG.
    fname : str
        File name for saved figure.
    title : str
        Plot title.

    Returns
    str : path to saved PNG
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    T_list = []
    nll_list = []

    for _, rec in per_seq.items():
        T = rec.get("T", None)
        avg_nll = rec.get("avg_nll", None)
        if T is None or avg_nll is None:
            continue
        if np.isfinite(T) and np.isfinite(avg_nll) and int(T) > 0:
            T_list.append(int(T))
            nll_list.append(float(avg_nll))

    if len(T_list) == 0:
        # nothing to plot
        return ""

    T_arr = np.asarray(T_list, dtype=np.int64)
    nll_arr = np.asarray(nll_list, dtype=np.float64)

    plt.figure(figsize=(8, 5))
    plt.scatter(T_arr, nll_arr, s=10, alpha=0.6)
    plt.xlabel("Trajectory length T (windows)")
    plt.ylabel("Average NLL per window")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()

    path = out_dir / fname
    plt.savefig(path, dpi=200)
    plt.close()
    return str(path)


# -----------------------------
# core reusable evaluator
# -----------------------------
def evaluate_checkpoint(model_path, test_seqs, feature_cols, out_dir=None, save_heatmaps=True):
    """Load checkpoint -> scale test obs -> infer -> compute metrics."""
    t0 = time.perf_counter()
    print(f"[eval_core] Loading model: {model_path}", flush=True)
    trainer = HDVTrainer.load(model_path)  
    emissions = trainer.emissions
    print(f"[eval_core] Model loaded in {time.perf_counter()-t0:.2f}s", flush=True)

    scaler_mean = trainer.scaler_mean
    scaler_std = trainer.scaler_std
    if scaler_mean is None or scaler_std is None:
        raise RuntimeError(f"Model '{model_path}' missing scaler_mean/std.")

    scale_idx = np.array([i for i, n in enumerate(feature_cols) if n in CONTINUOUS_FEATURES], dtype=np.int64)

    print(f"[eval_core] Scaling {len(test_seqs)} test sequences", flush=True)
    # scale test obs
    test_obs = []
    for i, seq in enumerate(test_seqs):
        if isinstance(scaler_mean, dict) and isinstance(scaler_std, dict):
            meta = getattr(seq, "meta", None) or {}
            cls = str(meta.get("meta_class", "NA"))
            mean = scaler_mean[cls]
            std = scaler_std[cls]
        else:
            mean, std = scaler_mean, scaler_std
        test_obs.append(scale_obs_masked(seq.obs, mean, std, scale_idx))
        if (i + 1) % 50 == 0 or (i + 1) == len(test_seqs):
            print(f"[eval_core] scaled {i+1}/{len(test_seqs)}", flush=True)

    total_ll = 0.0 # total_ll = Σ_i log p(O^{(i)} | model), mainly an internal quantity used for normalization and BIC.

    total_T = 0 # Total number of timesteps across all test trajectories.
    per_traj_ll_per_window = []
    gamma_sa_all = []

    print("[eval_core] Starting inference per trajectory", flush=True)
    t_inf0 = time.perf_counter()
    # infer per trajectory 
    for i, obs in enumerate(test_obs):
        gamma_s_a, _, _, loglik = infer_posterior(obs=obs, pi_s0=trainer.pi_s0, pi_a0_given_s0=trainer.pi_a0_given_s0, A_s=trainer.A_s, A_a=trainer.A_a, emissions=emissions)

        gamma_sa_all.append(gamma_s_a)
        total_ll += float(loglik)
        per_traj_ll_per_window.append(float(loglik) / max(obs.shape[0], 1))
        total_T += int(obs.shape[0])

        if (i + 1) % 20 == 0 or (i + 1) == len(test_obs):
            dt = time.perf_counter() - t_inf0
            avg = dt / (i + 1)
            print(f"[eval_core] infer {i+1}/{len(test_obs)}  avg={avg:.3f}s/seq  totalT={total_T}", flush=True)


    print("[eval_core] Computing entropy/occupancy/BIC", flush=True)

    ll_per_t = total_ll / max(total_T, 1) # Average log-likelihood per timestep; How good is the model, independent of dataset size

    H_joint, H_style, H_action, lengths, ent_summary = posterior_entropy_from_gamma_sa(gamma_sa_all)

    occ = occupancy_and_keff(gamma_sa_all)

    # BIC
    k, k_parts = param_count(trainer)
    N = max(total_T, 1)
    bic = (k * np.log(N)) - 2.0 * total_ll

    p = np.percentile(per_traj_ll_per_window, [0, 5, 25, 50, 75, 95, 100]) if per_traj_ll_per_window else np.full(7, np.nan)

    metrics = dict(
        total_ll=float(total_ll),
        total_T=int(total_T),
        ll_per_timestep=float(ll_per_t),

        # log-likelihood scores, aggregated by rank.
        traj_ll_pw_p0=float(p[0]),
        traj_ll_pw_p5=float(p[1]), # very poorly explained vehicles
        traj_ll_pw_p25=float(p[2]),
        traj_ll_pw_p50=float(p[3]), # median vehicle performance
        traj_ll_pw_p75=float(p[4]),
        traj_ll_pw_p95=float(p[5]), # very well explained vehicles
        traj_ll_pw_p100=float(p[6]),

        k_params=int(k),
        BIC=float(bic),
    )

    metrics.update({k: float(v) for k, v in occ.items()})
    metrics.update(ent_summary)
    
    if save_heatmaps:
        if out_dir is None:
            raise ValueError("save_heatmaps=True requires out_dir to be set.")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        heatmap_npz = out_dir / "entropy_heatmaps.npz"
        np.savez_compressed(
            heatmap_npz,
            H_joint=H_joint,
            H_style=H_style,
            H_action=H_action,
            lengths=lengths,
        )

        # Save 3 PNGs
        plot_paths = plot_entropy_heatmaps(
            H_joint=H_joint,
            H_style=H_style,
            H_action=H_action,
            lengths=lengths,
            out_dir=out_dir,
            prefix="entropy",
            sort_by="joint_mean",  # stable, readable
            vmax=1.0,
        )

        # Record artifact paths in JSON
        metrics.setdefault("artifacts", {})
        metrics["artifacts"].update({"entropy_heatmaps_npz": str(heatmap_npz), **plot_paths})

    print("[eval_core] Done metrics", flush=True)
    return metrics

# =============================================================================
# Online  predictive log-likelihood via filtering only
# =============================================================================
@torch.no_grad()
def _online_ll_terms_single(obs_scaled, pi_s0, pi_a0_given_s0, A_s, A_a, emissions):
    """
    Compute ll_terms[t] = log p(o_t | o_{0:t-1}) using forward filtering only.

    Shapes:
      obs_scaled: (T,F)
      emissions.loglikelihood(obs_scaled) -> logB_sa: (T,S,A)

    Returns:
      ll_terms: list length T
    """
    if obs_scaled is None or obs_scaled.shape[0] == 0:
        return []

    logB_sa = emissions.loglikelihood(obs_scaled)  # (T,S,A); logB_sa[t, s, a] is the emission log-likelihood for the observed window at time t
    if not torch.is_tensor(logB_sa):
        logB_sa = torch.as_tensor(logB_sa)

    device = getattr(emissions, "_device", pi_s0.device)
    dtype = getattr(emissions, "_dtype", pi_s0.dtype)

    logB_sa = logB_sa.to(device=device, dtype=dtype)
    pi_s0 = pi_s0.to(device=device, dtype=dtype)
    pi_a0_given_s0 = pi_a0_given_s0.to(device=device, dtype=dtype)
    A_s = A_s.to(device=device, dtype=dtype)
    A_a = A_a.to(device=device, dtype=dtype)

    T, S, A = map(int, logB_sa.shape)
    if pi_a0_given_s0.shape != (S, A):
        raise ValueError(f"pi_a0_given_s0 shape {tuple(pi_a0_given_s0.shape)} != ({S},{A})")
    if A_s.shape != (S, S):
        raise ValueError(f"A_s shape {tuple(A_s.shape)} != ({S},{S})")
    if A_a.shape != (S, A, A):
        raise ValueError(f"A_a shape {tuple(A_a.shape)} != ({S},{A},{A})")

    eps = 1e-6

    # log predicted belief before seeing o_0: log p(s0,a0)
    log_b_pred = torch.log(pi_s0 + eps)[:, None] + torch.log(pi_a0_given_s0 + eps)  # (S,A); p(s0​,a0​)=p(s0​)p(a0​∣s0​)

    logAs = torch.log(A_s + eps)  # (S,S)
    logAa = torch.log(A_a + eps)  # (S,A,A)
    # convenience: (A_prev, S_next, A_next)
    logAa_ap_s_an = logAa.permute(1, 0, 2).contiguous()
    assert logAa_ap_s_an.shape == (A, S, A), (
        f"logAa_ap_s_a has shape {logAa_ap_s_an.shape}, expected ({A},{S},{A})"
    )

    ll_terms = []

    for t in range(T):
        # predictive likelihood for o_t given past
        # log_b_pred = what the model currently believe about (s, a) before seeing this window
        # logB_sa[t] = If the driver were in (s, a), how likely is this observation?
        log_joint = log_b_pred + logB_sa[t]  # (S,A); log[p(s_t​,a_t ​∣ o_{0:t−1}​) p(o_t ​∣ s_t​,a_t​)]; How plausible is it that the driver was in this state and produced this window
        logZ_t = torch.logsumexp(log_joint.reshape(-1), dim=0)  # log[p(o_t ​∣ o_{0:t−1​})]; scalar normalization constant, adds everything up across all possible states.
        ll_terms.append(float(logZ_t.detach().cpu().item())) # save it because this is the prediction quality at time t.

        # filtering posterior (After seeing this window, what the model believe about the driver’s style and action)
        log_b_post = log_joint - logZ_t  # normalized log p(s_t,a_t | o_{0:t}); becomes a proper probability distribution

        if t < T - 1:
            # predict next belief: If the driver is currently in (s, a), how likely are they to move to (s′, a′)
            # log b_{t+1|t}(s',a') = logsumexp_{s,a} [ log b_post(s,a) + logA_s[s,s'] + logA_a[s',a,a'] ]
            tmp = (
                log_b_post[:, :, None, None] +
                logAs[:, None, :, None] +
                logAa_ap_s_an[None, :, :, :]
            )  # (S,A,S',A')
            log_b_pred = torch.logsumexp(tmp, dim=(0, 1))  # (S',A'); belief before seeing the next window
            log_b_pred = log_b_pred - torch.logsumexp(log_b_pred.reshape(-1), dim=0)

    return ll_terms # [logp(o_0​), logp(o_1 ​∣ o_0​), …, logp(o_{T−1​} ∣ o_{0:T−2}​)]


def evaluate_online_predictive_ll(model_path, test_seqs, feature_cols, out_dir=None, save_plot=True):
    """
    Online (strictly-causal) evaluation using filtering-only predictive log-likelihood.

    Output:
      summary 
    """
    print(f"[eval_core.online_ll] Loading model: {model_path}", flush=True)
    trainer = HDVTrainer.load(model_path)
    emissions = trainer.emissions

    scaler_mean = trainer.scaler_mean
    scaler_std = trainer.scaler_std
    if scaler_mean is None or scaler_std is None:
        raise RuntimeError(f"Model '{model_path}' missing scaler_mean/std.")

    scale_idx = np.array([i for i, n in enumerate(feature_cols) if n in CONTINUOUS_FEATURES], dtype=np.int64)

    # model params (torch)
    pi_s0 = trainer.pi_s0
    pi_a0_given_s0 = trainer.pi_a0_given_s0
    A_s = trainer.A_s
    A_a = trainer.A_a

    per_seq = {}
    total_ll = 0.0
    total_T = 0
    avg_nll_list = []

    for i, seq in enumerate(test_seqs):
        key = seq_key(seq)

        # scaling (classwise vs global)
        if isinstance(scaler_mean, dict) and isinstance(scaler_std, dict):
            meta = getattr(seq, "meta", None) or {}
            cls = str(meta.get("meta_class", "NA"))
            mean = scaler_mean[cls]
            std = scaler_std[cls]
        else:
            mean, std = scaler_mean, scaler_std

        obs_scaled = scale_obs_masked(seq.obs, mean, std, scale_idx)

        ll_terms = _online_ll_terms_single(
            obs_scaled=obs_scaled,
            pi_s0=pi_s0,
            pi_a0_given_s0=pi_a0_given_s0,
            A_s=A_s,
            A_a=A_a,
            emissions=emissions
        )

        ll_total = float(np.sum(ll_terms))
        T = int(len(ll_terms))
        nll_total = float(-ll_total)
        avg_nll = float(nll_total / max(T, 1))

        total_ll += ll_total
        total_T += T
        avg_nll_list.append(avg_nll)

        rec = {
            "T": T,
            "avg_nll": avg_nll,
        }

        per_seq[key] = rec

        if (i + 1) % 50 == 0 or (i + 1) == len(test_seqs):
            print(f"[eval_core.online_ll] processed {i+1}/{len(test_seqs)} totalT={total_T}", flush=True)

    x = np.asarray(avg_nll_list, dtype=np.float64)
    x = x[np.isfinite(x)]

    def _summ(z):
        if z.size == 0:
            return {"mean": np.nan, "median": np.nan, "p10": np.nan, "p90": np.nan}
        return {
            "mean": float(np.mean(z)),
            "median": float(np.median(z)),
            "p10": float(np.percentile(z, 10)),
            "p90": float(np.percentile(z, 90)),
        }

    summary = {
        "model_path": str(model_path),
        "n_sequences": int(len(test_seqs)),
        "total_T": int(total_T), # Total number of windows across all test vehicles
        "total_ll": float(total_ll), # Raw log-likelihood, summed over all timesteps of all vehicles
        "total_nll": float(-total_ll),
        "avg_nll_per_timestep_weighted": float((-total_ll) / max(int(total_T), 1)), # global average NLL per window; long trajectories contribute more weight than short ones
        "per_sequence_avg_nll": _summ(x), # Each vehicle counts equally, regardless of length.
    }

    artifacts = {}
    if save_plot:
        if out_dir is None:
            raise ValueError("save_plot=True requires out_dir to be set.")
        png_path = plot_T_vs_avg_nll(
            per_seq=per_seq,
            out_dir=out_dir,
            fname="T_vs_avg_nll.png",
            title="Online predictive: T vs avg NLL"
        )
        if png_path:
            artifacts["T_vs_avg_nll_png"] = png_path

    out = {"summary": summary}
    if artifacts:
        out["artifacts"] = artifacts

    return out