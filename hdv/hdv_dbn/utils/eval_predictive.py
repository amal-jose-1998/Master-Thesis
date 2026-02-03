import numpy as np
import torch

from .eval_plots import plot_T_vs_avg_nll, plot_ap_nll_vs_horizon
from .eval_common import scale_obs_masked, seq_key
from .eval_baselines import evaluate_iid_baseline, evaluate_frozen_belief_online_ll
from ..trainer import HDVTrainer
from ..config import CONTINUOUS_FEATURES

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


@torch.no_grad()
def _predict_latent_belief(log_b_init, A_s, A_a, H):
    """
    Roll a joint latent belief forward H steps without observations.

    log_b_init : (S,A)  log p(s_t,a_t | o_{0:t})
    Returns:
        log_b_pred[h] : list of (S,A) log-beliefs for h=1..H
    """
    eps = 1e-6
    logAs = torch.log(A_s + eps)              # (S,S)
    logAa = torch.log(A_a + eps)              # (S,A,A)
    logAa_ap_s_an = logAa.permute(1, 0, 2)    # (A,S,A)

    log_b = log_b_init
    out = []

    for _ in range(H): # Repeat H times: compute one-step prediction, then repeat from that prediction.
        tmp = (
            log_b[:, :, None, None] +     # logp(s_t​,a_t ​∣ past)
            logAs[:, None, :, None] +     # logp(s_{t+1} ​∣ s_t​)
            logAa_ap_s_an[None, :, :, :]  # logp(a_{t+1} ​∣ a_t, s_{t+1}​)
        )  # (S,A,S',A'); log[p(s_t​, a_t ​∣ o_{0:t}​) p(s_{t+1} ​∣ s_t​) p(a_{t+1} ​∣ a_t​, s_{t+1}​)]

        log_b = torch.logsumexp(tmp, dim=(0, 1))  # (S',A'); logp(s_{t+1}​,a_{t+1} ​∣ o_{0:t}​)
        log_b = log_b - torch.logsumexp(log_b.reshape(-1), dim=0)
        out.append(log_b)

    return out


@torch.no_grad()
def _anticipatory_ll_terms_single(obs_scaled, pi_s0, pi_a0_given_s0, A_s, A_a, emissions, H, t_warmup):
    """
    Compute anticipatory log-likelihood terms:
      log p(o_{t+h} | o_{0:t}) for h=1..H
    """
    # This is the “emission score table” used in filtering and prediction scoring.
    # This is where the actual observation enters.
    logB_sa = emissions.loglikelihood(obs_scaled)  # (T,S,A); computes log[p(o_t ​∣ s_t​,a_t​)] for every time and every latent pair.
    if not torch.is_tensor(logB_sa):
        logB_sa = torch.as_tensor(logB_sa)

    eps = 1e-6

    # initial predictive belief over joint state (s,a) at time 0:
    log_b_pred = (
        torch.log(pi_s0 + eps)[:, None] +
        torch.log(pi_a0_given_s0 + eps)
    )

    T = logB_sa.shape[0]
    scores = []  # list of lists (per t); each element corresponds to one time t and contains H floats

    for t in range(T):
        # filtering update
        # log_b_pred is the predicted prior belief at time t given past observations.
        # logB_sa[t] is log[p(o_t ​∣ s_t​,a_t​)] for the actual observation at time t
        log_joint = log_b_pred + logB_sa[t] # unnormalized posterior (in log).
        logZ = torch.logsumexp(log_joint.reshape(-1), dim=0) # normalization constant so posterior sums to 1.
        log_b_post = log_joint - logZ # log[p(s_t​,a_t ​∣ o_{0:t}​)] filtering posterior at time t

        if t >= t_warmup and (t + H) < T: # Skip early times (warmup) and avoid running past sequence end.
            preds = _predict_latent_belief(log_b_post, A_s, A_a, H) # predict future latent beliefs

            ll_th = []
            for h, log_b_h in enumerate(preds, start=1): # Loop over horizons
                # log_b_h is predicted belief over latent at time t+h.
                log_joint_h = log_b_h + logB_sa[t + h]
                # log p(o_{t+h} | o_{0:t}) = log ∑_{s,a} p(o_{t+h} | s_{t+h}=s, a_{t+h}=a) · p(s_{t+h}=s, a_{t+h}=a | o_{0:t})
                ll = torch.logsumexp(log_joint_h.reshape(-1), dim=0) # predictive log-likelihood at horizon h
                ll_th.append(float(ll.detach().cpu().item()))

            scores.append(ll_th)

        # propagate belief to next time
        if t < T - 1:
            tmp = (
                log_b_post[:, :, None, None] +
                torch.log(A_s + eps)[:, None, :, None] +
                torch.log(A_a + eps).permute(1, 0, 2)[None, :, :, :]
            )
            log_b_pred = torch.logsumexp(tmp, dim=(0, 1)) # logp(s_{t+1}​,a_{t+1} ​∣ o_{0:t}​)
            log_b_pred = log_b_pred - torch.logsumexp(log_b_pred.reshape(-1), dim=0) # Normalize it so it’s a proper log-prob distribution

    return scores


def evaluate_anticipatory_predictive_ll(model_path, test_seqs, feature_cols, H=10, t_warmup=5, out_dir=None, save_plot=True):
    """
    Anticipatory (strictly-causal) evaluation:
      For each pause time t >= t_warmup:
        score future observations o_{t+h} under p(o_{t+h} | o_{0:t}), h=1..H

    Returns:
      summary: includes ap_nll_by_h (curve) + scalar summaries
      artifacts: optional plot path
    """
    print(f"[eval_core.ap_ll] Loading model: {model_path}", flush=True)
    trainer = HDVTrainer.load(model_path)
    emissions = trainer.emissions

    pi_s0 = trainer.pi_s0
    pi_a0_given_s0 = trainer.pi_a0_given_s0
    A_s = trainer.A_s
    A_a = trainer.A_a

    scaler_mean = trainer.scaler_mean
    scaler_std = trainer.scaler_std
    if scaler_mean is None or scaler_std is None:
        raise RuntimeError(f"Model '{model_path}' missing scaler_mean/std.")

    scale_idx = np.array(
        [i for i, n in enumerate(feature_cols) if n in CONTINUOUS_FEATURES],
        dtype=np.int64
    )

    # Aggregate across all vehicles:
    ll_rows_all = []  # list of [ll(h=1..H)] rows (one row per pause-time)

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

        # list of rows (number_of_pause_times_for_this_vehicle, H), each row is ll for horizons 1..H
        ll_rows_i = _anticipatory_ll_terms_single(obs_scaled, pi_s0, pi_a0_given_s0, A_s, A_a, emissions, H=H, t_warmup=t_warmup)

        if ll_rows_i:
            arr = np.asarray(ll_rows_i, dtype=np.float64)  # (n_pause_i, H)
            ll_rows_all.append(arr)

        if (i + 1) % 50 == 0 or (i + 1) == len(test_seqs):
            print(f"[eval_core.ap_ll] processed {i+1}/{len(test_seqs)}", flush=True)

    if len(ll_rows_all) == 0:
        summary = {
            "H": int(H),
            "t_warmup": int(t_warmup),
            "n_pause_total": 0,
            "ap_nll_by_h": [float("nan")] * int(H),
            "ap_nll_mean_over_h": float("nan"),
        }
        return {"summary": summary}

    LL = np.concatenate(ll_rows_all, axis=0)  # (N_pause_total, H)
    mean_ll_h = np.nanmean(LL, axis=0)        # (H,) average log-likelihood at each horizon
    ap_nll_by_h = -mean_ll_h                  # (H,) convert to negative log-likelihood
    ap_nll_mean_over_h = float(np.nanmean(ap_nll_by_h)) # average performance across horizons

    summary = {
        "H": int(H),
        "t_warmup": int(t_warmup),
        "n_pause_total": int(LL.shape[0]),
        "ap_nll_by_h": ap_nll_by_h.astype(np.float64).tolist(),
        "ap_nll_mean_over_h": ap_nll_mean_over_h, 
    }

    artifacts = {}
    if save_plot:
        if out_dir is None:
            raise ValueError("save_plot=True requires out_dir to be set.")
        png = plot_ap_nll_vs_horizon(
            ap_nll_by_h=ap_nll_by_h,
            out_dir=out_dir,
            fname="ap_nll_vs_horizon.png",
            title=f"Anticipatory predictive NLL vs horizon (H={H})"
        )
        if png:
            artifacts["ap_nll_vs_horizon_png"] = png

    out = {"summary": summary}
    if artifacts:
        out["artifacts"] = artifacts
    return out


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
        "n_sequences": int(len(test_seqs)),
        "total_T": int(total_T), # Total number of windows across all test vehicles
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