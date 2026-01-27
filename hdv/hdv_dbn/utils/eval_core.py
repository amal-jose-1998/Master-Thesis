import json
import re

import numpy as np
import torch

from ..trainer import HDVTrainer
from ..config import CONTINUOUS_FEATURES
from ..inference import infer_posterior  
from .trainer_diagnostics import posterior_entropy_from_gamma_sa, run_lengths_from_gamma_sa


# -----------------------------
# small utilities
# -----------------------------

def scale_obs_masked(obs, mean, std, scale_idx):
    """
    Scale only selected feature indices (continuous dims) using mean/std.
    Discrete dims remain unchanged.
    """
    x = np.asarray(obs, dtype=np.float64).copy()
    m = np.asarray(mean, dtype=np.float64)
    s = np.asarray(std, dtype=np.float64)
    denom = s[scale_idx] + 1e-12
    x[:, scale_idx] = (x[:, scale_idx] - m[scale_idx]) / denom
    return x


def traj_key(seq):
    meta = getattr(seq, "meta", None) or {}
    rid = meta.get("recording_id", meta.get("recordingId", meta.get("rec_id", "NA")))
    vid = meta.get("vehicle_id", meta.get("vehicleId", meta.get("veh_id", "NA")))
    return f"{rid}:{vid}"


def save_split_json(path, train_keys, val_keys, test_keys, seed, note=""):
    payload = {
        "seed": int(seed),
        "note": str(note),
        "train": list(train_keys),
        "val": list(val_keys),
        "test": list(test_keys),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_split_json(path):
    payload = json.loads(path.read_text())
    return payload["train"], payload["val"], payload["test"], payload.get("seed", None)


def apply_split_by_keys(seqs, train_keys, val_keys, test_keys):
    k2seq = {traj_key(s): s for s in seqs}
    train = [k2seq[k] for k in train_keys if k in k2seq]
    val   = [k2seq[k] for k in val_keys if k in k2seq]
    test  = [k2seq[k] for k in test_keys if k in k2seq]
    return train, val, test


def list_checkpoints(exp_dir):
    ckpts = sorted(exp_dir.glob("*.npz"))

    def key(p):
        name = p.stem.lower()
        if name == "final":
            return (10**9, name)
        m = re.search(r"iter(\d+)", name)
        if m:
            return (int(m.group(1)), name)
        m = re.search(r"(\d+)", name)
        if m:
            return (int(m.group(1)), name)
        return (0, name)

    return sorted(ckpts, key=key)


# -----------------------------
# metrics helpers
# -----------------------------

def approx_param_count(trainer):
    """
    Same approximate counting logic you already use in evaluate_experiments.py. :contentReference[oaicite:6]{index=6}
    """
    S = int(trainer.pi_s0.numel())
    A = int(trainer.pi_a0_given_s0.shape[1])

    k_trans = (S - 1) + S * (A - 1) + S * (S - 1) + S * A * (A - 1)

    k_em = 0
    try:
        em = trainer.emissions.to_arrays()
        exclude = {"obs_names", "bernoulli_names", "cont_idx", "bin_idx"}
        for kk, vv in em.items():
            if kk in exclude:
                continue
            arr = np.asarray(vv)
            if arr.dtype.kind in "iuf" and arr.size > 0:
                k_em += int(arr.size)
    except Exception:
        k_em = 0

    return int(k_trans + k_em)


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

    def norm_np(x: torch.Tensor):
        z = x / (x.sum() + 1e-12)
        return z.cpu().numpy().astype(np.float64)

    pj = norm_np(joint_sum)
    ps = norm_np(style_sum)
    pa = norm_np(action_sum)

    def keff(p) -> float:
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


# -----------------------------
# core reusable evaluator
# -----------------------------

def evaluate_checkpoint(model_path, test_seqs, feature_cols):
    """Load checkpoint -> scale test obs -> infer -> compute metrics."""
    trainer = HDVTrainer.load(model_path)  
    emissions = trainer.emissions

    scaler_mean = trainer.scaler_mean
    scaler_std = trainer.scaler_std
    if scaler_mean is None or scaler_std is None:
        raise RuntimeError(f"Model '{model_path}' missing scaler_mean/std.")

    scale_idx = np.array([i for i, n in enumerate(feature_cols) if n in CONTINUOUS_FEATURES], dtype=np.int64)

    # scale test obs
    test_obs = []
    for seq in test_seqs:
        if isinstance(scaler_mean, dict) and isinstance(scaler_std, dict):
            meta = getattr(seq, "meta", None) or {}
            cls = str(meta.get("meta_class", "NA"))
            mean = scaler_mean[cls]
            std = scaler_std[cls]
        else:
            mean, std = scaler_mean, scaler_std
        test_obs.append(scale_obs_masked(seq.obs, mean, std, scale_idx))

    total_ll = 0.0
    total_T = 0
    per_traj_ll = []
    gamma_sa_all = []

    # infer per trajectory 
    for obs in test_obs:
        gamma_s_a, gamma_s, gamma_a, loglik = infer_posterior(
            obs=obs,
            pi_s0=trainer.pi_s0,
            pi_a0_given_s0=trainer.pi_a0_given_s0,
            A_s=trainer.A_s,
            A_a=trainer.A_a,
            emissions=emissions,
        )

        gamma_sa_all.append(gamma_s_a)
        total_ll += float(loglik)
        per_traj_ll.append(float(loglik))
        total_T += int(obs.shape[0])

    ll_per_t = total_ll / max(total_T, 1)
    ll_per_traj = total_ll / max(len(test_obs), 1)

    ent_joint_mean = ent_style_mean = ent_action_mean = np.nan
    runlen_joint_median = runlen_style_median = runlen_action_median = np.nan

    _, ent_joint_mean, _, ent_style_mean, _, ent_action_mean = posterior_entropy_from_gamma_sa(gamma_sa_all)
    _, runlen_joint_med,  _, runlen_style_med, _, runlen_action_med = run_lengths_from_gamma_sa(gamma_sa_all)

    ent_joint_mean = float(np.nanmean(ent_joint_mean))
    ent_style_mean = float(np.nanmean(ent_style_mean))
    ent_action_mean = float(np.nanmean(ent_action_mean))

    runlen_joint_median = float(np.nanmedian(runlen_joint_med))
    runlen_style_median = float(np.nanmedian(runlen_style_med))
    runlen_action_median = float(np.nanmedian(runlen_action_med))

    occ = occupancy_and_keff(gamma_sa_all)

    # BIC
    k = approx_param_count(trainer)
    N = max(total_T, 1)
    bic = (k * np.log(N)) - 2.0 * total_ll

    p = np.percentile(per_traj_ll, [0, 5, 25, 50, 75, 95, 100]) if per_traj_ll else np.full(7, np.nan)

    return dict(
        total_ll=float(total_ll),
        total_T=int(total_T),
        ll_per_timestep=float(ll_per_t),
        ll_per_traj=float(ll_per_traj),

        traj_ll_p0=float(p[0]),
        traj_ll_p5=float(p[1]),
        traj_ll_p25=float(p[2]),
        traj_ll_p50=float(p[3]),
        traj_ll_p75=float(p[4]),
        traj_ll_p95=float(p[5]),
        traj_ll_p100=float(p[6]),

        ent_joint_mean=float(ent_joint_mean),
        ent_style_mean=float(ent_style_mean),
        ent_action_mean=float(ent_action_mean),

        runlen_joint_median=float(runlen_joint_median),
        runlen_style_median=float(runlen_style_median),
        runlen_action_median=float(runlen_action_median),

        k_params_approx=int(k),
        BIC_approx=float(bic),

        occ=occ,
    )
