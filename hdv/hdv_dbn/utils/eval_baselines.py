import numpy as np
import torch

from .eval_plots import plot_T_vs_avg_nll
from .eval_common import scale_obs_masked, seq_key
from ..trainer import HDVTrainer
from ..config import CONTINUOUS_FEATURES

_EPS = 1e-12


def _scale_idx_from_feature_cols(feature_cols):
    return np.asarray([i for i, n in enumerate(feature_cols) if n in CONTINUOUS_FEATURES], dtype=np.int64)


def _infer_cont_bin_idx(feature_cols):
    cont_idx = np.asarray([i for i, n in enumerate(feature_cols) if n in CONTINUOUS_FEATURES], dtype=np.int64)
    bin_idx = np.asarray([i for i, n in enumerate(feature_cols) if n not in CONTINUOUS_FEATURES], dtype=np.int64)
    return cont_idx, bin_idx


def _fit_iid_params_from_scaled_obs(train_seqs, feature_cols, scaler_mean, scaler_std):
    """
    Fit IID params on TRAIN ONLY after scaling with checkpoint mean/std.
    Continuous dims: Gaussian diag
    Non-cont dims: Bernoulli (assumed 0/1; clamped)
    """
    scale_idx = _scale_idx_from_feature_cols(feature_cols)
    cont_idx, bin_idx = _infer_cont_bin_idx(feature_cols)

    X_list = []
    for seq in train_seqs:
        obs_scaled = scale_obs_masked(seq.obs, scaler_mean, scaler_std, scale_idx)
        if obs_scaled is not None and obs_scaled.shape[0] > 0:
            X_list.append(np.asarray(obs_scaled, dtype=np.float64))

    if len(X_list) == 0:
        raise ValueError("IID fit: no usable training observations.")

    X = np.concatenate(X_list, axis=0)  # (N,F)
    F = int(X.shape[1])

    mu = np.zeros(F, dtype=np.float64)
    var = np.ones(F, dtype=np.float64)
    p = np.full(F, 0.5, dtype=np.float64)

    for j in cont_idx.tolist():
        x = X[:, j]
        m = np.isfinite(x)
        if np.any(m):
            mu[j] = float(np.mean(x[m]))
            var[j] = max(float(np.var(x[m])), 1e-6)

    for j in bin_idx.tolist():
        x = X[:, j]
        m = np.isfinite(x)
        if np.any(m):
            x01 = np.clip(x[m], 0.0, 1.0)
            pj = float(np.mean(x01))
            p[j] = float(np.clip(pj, 1e-4, 1.0 - 1e-4))

    return {
        "mu": mu,
        "var": var,
        "p": p,
        "scale_idx": scale_idx,
        "cont_idx": cont_idx,
        "bin_idx": bin_idx,
    }


def _iid_ll_terms_single(obs_scaled, params):
    """Per-timestep ll: log p(o_t) under IID factorized model."""
    X = np.asarray(obs_scaled, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        return []

    mu = params["mu"]; var = params["var"]; p = params["p"]
    cont_idx = params["cont_idx"]; bin_idx = params["bin_idx"]

    T = int(X.shape[0])
    ll_t = np.zeros(T, dtype=np.float64)

    if cont_idx.size > 0:
        x = X[:, cont_idx]
        m = np.isfinite(x)
        mu_c = mu[cont_idx][None, :]
        var_c = var[cont_idx][None, :]
        ll = -(0.5 * np.log(2.0 * np.pi * var_c) + 0.5 * ((x - mu_c) ** 2) / var_c)
        ll = np.where(m, ll, 0.0)
        ll_t += np.sum(ll, axis=1)

    if bin_idx.size > 0:
        x = X[:, bin_idx]
        m = np.isfinite(x)
        x01 = np.clip(x, 0.0, 1.0)
        p_b = p[bin_idx][None, :]
        ll = x01 * np.log(p_b + _EPS) + (1.0 - x01) * np.log(1.0 - p_b + _EPS)
        ll = np.where(m, ll, 0.0)
        ll_t += np.sum(ll, axis=1)

    return [float(v) for v in ll_t]


@torch.no_grad()
def _frozen_belief_ll_terms_single(obs_scaled, pi_s0, pi_a0_given_s0, emissions):
    """
    No-dynamics ablation (frozen belief):
      - do filtering update to get log p(s_t,a_t | o_{0:t})
      - for the next prediction step, set predictive belief = current posterior (skip transitions)
    """
    if obs_scaled is None or obs_scaled.shape[0] == 0:
        return []

    logB_sa = emissions.loglikelihood(obs_scaled)  # (T,S,A)
    if not torch.is_tensor(logB_sa):
        logB_sa = torch.as_tensor(logB_sa)

    device = getattr(emissions, "_device", pi_s0.device)
    dtype = getattr(emissions, "_dtype", pi_s0.dtype)

    logB_sa = logB_sa.to(device=device, dtype=dtype)
    pi_s0 = pi_s0.to(device=device, dtype=dtype)
    pi_a0_given_s0 = pi_a0_given_s0.to(device=device, dtype=dtype)

    eps = 1e-6
    log_b_pred = torch.log(pi_s0 + eps)[:, None] + torch.log(pi_a0_given_s0 + eps)  # (S,A)

    T = int(logB_sa.shape[0])
    ll_terms = []

    for t in range(T):
        log_joint = log_b_pred + logB_sa[t]
        logZ_t = torch.logsumexp(log_joint.reshape(-1), dim=0)
        ll_terms.append(float(logZ_t.detach().cpu().item()))
        log_b_post = log_joint - logZ_t  # (S,A)

        # freeze: predictive belief for next step = current posterior
        log_b_pred = log_b_post
        log_b_pred = log_b_pred - torch.logsumexp(log_b_pred.reshape(-1), dim=0)

    return ll_terms


def _summarize_avg_nll(avg_nll_list):
    x = np.asarray(avg_nll_list, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": np.nan, "median": np.nan, "p10": np.nan, "p90": np.nan}
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
    }


def evaluate_iid_baseline(model_path, train_seqs, test_seqs, feature_cols, out_dir=None, save_plot=True):
    """
    IID baseline:
      - fit (mu,var,p) on TRAIN ONLY (after applying checkpoint scaler)
      - evaluate NLL on TEST
      - return summary compatible with evaluate_online_predictive_ll
    """
    print(f"[eval_baselines.iid] Loading model: {model_path}", flush=True)
    trainer = HDVTrainer.load(model_path)

    scaler_mean = trainer.scaler_mean
    scaler_std = trainer.scaler_std
    if scaler_mean is None or scaler_std is None:
        raise RuntimeError(f"Model '{model_path}' missing scaler_mean/std.")

    if isinstance(scaler_mean, dict) or isinstance(scaler_std, dict):
        raise NotImplementedError("IID baseline currently assumes global scaler_mean/std (not classwise dict).")

    params = _fit_iid_params_from_scaled_obs(train_seqs, feature_cols, scaler_mean, scaler_std)

    per_seq = {}
    total_ll = 0.0
    total_T = 0
    avg_nll_list = []

    for i, seq in enumerate(test_seqs):
        key = seq_key(seq)
        obs_scaled = scale_obs_masked(seq.obs, scaler_mean, scaler_std, params["scale_idx"])
        ll_terms = _iid_ll_terms_single(obs_scaled, params)

        ll_total = float(np.sum(ll_terms))
        T = int(len(ll_terms))
        avg_nll = float((-ll_total) / max(T, 1))

        total_ll += ll_total
        total_T += T
        avg_nll_list.append(avg_nll)
        per_seq[key] = {"T": T, "avg_nll": avg_nll}

        if (i + 1) % 50 == 0 or (i + 1) == len(test_seqs):
            print(f"[eval_baselines.iid] processed {i+1}/{len(test_seqs)} totalT={total_T}", flush=True)

    summary = {
        "n_sequences": int(len(test_seqs)),
        "total_T": int(total_T),
        "avg_nll_per_timestep_weighted": float((-total_ll) / max(int(total_T), 1)),
        "per_sequence_avg_nll": _summarize_avg_nll(avg_nll_list),
    }

    artifacts = {}
    if save_plot:
        if out_dir is None:
            raise ValueError("save_plot=True requires out_dir to be set.")
        png_path = plot_T_vs_avg_nll(
            per_seq=per_seq,
            out_dir=out_dir,
            fname="T_vs_avg_nll.png",
            title="IID baseline: T vs avg NLL"
        )
        if png_path:
            artifacts["T_vs_avg_nll_png"] = png_path

    out = {"summary": summary}
    if artifacts:
        out["artifacts"] = artifacts
    return out


def evaluate_frozen_belief_online_ll(model_path, test_seqs, feature_cols, out_dir=None, save_plot=True):
    """
    No-dynamics baseline:
      - keep the belief frozen between steps (no transition propagation)
      - still scores the real observations (strictly causal)
    """
    print(f"[eval_baselines.frozen] Loading model: {model_path}", flush=True)
    trainer = HDVTrainer.load(model_path)
    emissions = trainer.emissions

    scaler_mean = trainer.scaler_mean
    scaler_std = trainer.scaler_std
    if scaler_mean is None or scaler_std is None:
        raise RuntimeError(f"Model '{model_path}' missing scaler_mean/std.")

    if isinstance(scaler_mean, dict) or isinstance(scaler_std, dict):
        raise NotImplementedError("Frozen-belief baseline currently assumes global scaler_mean/std (not classwise dict).")

    scale_idx = _scale_idx_from_feature_cols(feature_cols)

    pi_s0 = trainer.pi_s0
    pi_a0_given_s0 = trainer.pi_a0_given_s0

    per_seq = {}
    total_ll = 0.0
    total_T = 0
    avg_nll_list = []

    for i, seq in enumerate(test_seqs):
        key = seq_key(seq)
        obs_scaled = scale_obs_masked(seq.obs, scaler_mean, scaler_std, scale_idx)

        ll_terms = _frozen_belief_ll_terms_single(
            obs_scaled=obs_scaled,
            pi_s0=pi_s0,
            pi_a0_given_s0=pi_a0_given_s0,
            emissions=emissions,
        )

        ll_total = float(np.sum(ll_terms))
        T = int(len(ll_terms))
        avg_nll = float((-ll_total) / max(T, 1))

        total_ll += ll_total
        total_T += T
        avg_nll_list.append(avg_nll)
        per_seq[key] = {"T": T, "avg_nll": avg_nll}

        if (i + 1) % 50 == 0 or (i + 1) == len(test_seqs):
            print(f"[eval_baselines.frozen] processed {i+1}/{len(test_seqs)} totalT={total_T}", flush=True)

    summary = {
        "n_sequences": int(len(test_seqs)),
        "total_T": int(total_T),
        "avg_nll_per_timestep_weighted": float((-total_ll) / max(int(total_T), 1)),
        "per_sequence_avg_nll": _summarize_avg_nll(avg_nll_list),
    }

    artifacts = {}
    if save_plot:
        if out_dir is None:
            raise ValueError("save_plot=True requires out_dir to be set.")
        png_path = plot_T_vs_avg_nll(
            per_seq=per_seq,
            out_dir=out_dir,
            fname="T_vs_avg_nll.png",
            title="Frozen-belief baseline: T vs avg NLL"
        )
        if png_path:
            artifacts["T_vs_avg_nll_png"] = png_path

    out = {"summary": summary}
    if artifacts:
        out["artifacts"] = artifacts
    return out
