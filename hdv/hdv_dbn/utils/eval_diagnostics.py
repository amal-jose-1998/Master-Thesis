import numpy as np
import time
from pathlib import Path

from .eval_plots import plot_entropy_heatmaps
from .eval_common import scale_obs_masked
from ..trainer import HDVTrainer
from ..config import CONTINUOUS_FEATURES, DBN_STATES, TRAINING_CONFIG
from ..inference import infer_posterior  
from .trainer_diagnostics import posterior_entropy_from_gamma_sa



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
        total_T=int(total_T),
        ll_per_timestep=float(ll_per_t),

        # log-likelihood scores, aggregated by rank.
        traj_ll_pw_p5=float(p[1]), # very poorly explained vehicles
        traj_ll_pw_p50=float(p[3]), # median vehicle performance
        traj_ll_pw_p95=float(p[5]), # very well explained vehicles

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