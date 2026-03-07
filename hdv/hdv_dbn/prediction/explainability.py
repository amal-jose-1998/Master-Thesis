from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import numpy as np
import torch

from ..config import (
    DBN_STATES,
    TRAINING_CONFIG,
    WINDOW_EGO_FEATURES,
    WINDOW_FRONT_RISK_FEATURES,
    WINDOW_LANE_GEOM_FEATURES,
    WINDOW_LC_FEATURES,
    WINDOW_NEIGHBOR_REL_FEATURES,
)
from .filtering import StructuredDBNFilter, BeliefState


_EPS = float(getattr(TRAINING_CONFIG, "EPSILON", 1e-6))


@dataclass(frozen=True)
class StateScore:
    s: int
    a: int
    style_name: str
    action_name: str
    prob: float
    logprob: float


def _state_name_s(s: int) -> str:
    names = tuple(getattr(DBN_STATES, "driving_style", ()))
    return str(names[s]) if 0 <= s < len(names) else f"style_{s}"


def _state_name_a(a: int) -> str:
    names = tuple(getattr(DBN_STATES, "action", ()))
    return str(names[a]) if 0 <= a < len(names) else f"action_{a}"


def _normalize_logprob(logp: torch.Tensor) -> torch.Tensor:
    z = torch.logsumexp(logp.reshape(-1), dim=0)
    return logp - z


def _to_prob(logp: torch.Tensor) -> torch.Tensor:
    return torch.exp(_normalize_logprob(logp))


def _topk_states(logp_sa: torch.Tensor, k: int) -> list[StateScore]:
    lp = _normalize_logprob(logp_sa)
    p = torch.exp(lp)
    flat_p = p.reshape(-1)
    flat_lp = lp.reshape(-1)
    k = max(1, min(int(k), int(flat_p.numel())))
    vals, idxs = torch.topk(flat_p, k=k, dim=0)
    A = int(logp_sa.shape[1])
    out: list[StateScore] = []
    for rank in range(k):
        idx = int(idxs[rank].item())
        s = int(idx // A)
        a = int(idx % A)
        out.append(
            StateScore(
                s=s,
                a=a,
                style_name=_state_name_s(s),
                action_name=_state_name_a(a),
                prob=float(vals[rank].item()),
                logprob=float(flat_lp[idx].item()),
            )
        )
    return out


def _default_feature_groups() -> dict[str, list[str]]:
    return {
        "ego_kinematics": list(WINDOW_EGO_FEATURES),
        "lane_change_flags": list(WINDOW_LC_FEATURES),
        "lane_geometry": list(WINDOW_LANE_GEOM_FEATURES),
        "front_risk": list(WINDOW_FRONT_RISK_FEATURES),
        "neighbor_interaction": list(WINDOW_NEIGHBOR_REL_FEATURES),
    }


def _tv_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    return float(0.5 * torch.abs(p - q).sum().item())


def _decode_idx(idx: int, A: int) -> tuple[int, int]:
    return int(idx // A), int(idx % A)


def _state_summary_from_logp(logp_sa: torch.Tensor) -> dict[str, Any]:
    p = _to_prob(logp_sa)
    flat = p.reshape(-1)
    idx = int(torch.argmax(flat).item())
    S, A = map(int, p.shape)
    s, a = _decode_idx(idx, A)
    return {
        "s": s,
        "a": a,
        "style_name": _state_name_s(s),
        "action_name": _state_name_a(a),
        "prob": float(flat[idx].item()),
        "entropy": float((-flat * torch.log(flat.clamp_min(1e-30))).sum().item()),
        "num_states": int(S * A),
    }


def _group_contrib_template(group_names: list[str]) -> dict[str, float]:
    return {g: 0.0 for g in group_names}


def _build_group_index(feature_cols: list[str], feature_groups: dict[str, list[str]]) -> dict[str, list[int]]:
    idx_map = {name: i for i, name in enumerate(feature_cols)}
    out: dict[str, list[int]] = {}
    for group, names in feature_groups.items():
        out[group] = [idx_map[n] for n in names if n in idx_map]
    return out


def _hierarchical_feature_group_contrib(
    *,
    emissions,
    x_t: np.ndarray,
    s: int,
    a: int,
    feature_cols: list[str],
    feature_groups: dict[str, list[str]],
) -> dict[str, Any]:
    group_idx = _build_group_index(feature_cols, feature_groups)
    group_scores = _group_contrib_template(list(group_idx.keys()))

    cont_idx = list(getattr(emissions, "cont_idx", []))
    bin_idx = list(getattr(emissions, "bin_idx", []))
    cont_set = set(cont_idx)
    bin_set = set(bin_idx)

    gauss_mean = np.asarray(emissions.gauss.mean, dtype=np.float64)
    gauss_var = np.asarray(emissions.gauss.var, dtype=np.float64)

    bern_p = None
    if (not bool(getattr(emissions, "disable_discrete_obs", False))) and len(bin_idx) > 0:
        bern_p = np.asarray(emissions.bern.p, dtype=np.float64)

    cont_pos = {orig: j for j, orig in enumerate(cont_idx)}
    bin_pos = {orig: j for j, orig in enumerate(bin_idx)}
    w_bern = float(getattr(TRAINING_CONFIG, "bern_weight", 1.0))

    for group, idxs in group_idx.items():
        total = 0.0
        for i in idxs:
            xi = float(x_t[i])
            if (i in cont_set) and np.isfinite(xi):
                j = cont_pos[i]
                mu = float(gauss_mean[s, a, j])
                var = max(float(gauss_var[s, a, j]), _EPS)
                total += -0.5 * (math.log(2.0 * math.pi) + math.log(var) + ((xi - mu) * (xi - mu) / var))
            elif (i in bin_set) and (bern_p is not None) and np.isfinite(xi):
                j = bin_pos[i]
                b = 1.0 if xi > 0.5 else 0.0
                p = float(np.clip(bern_p[s, a, j], _EPS, 1.0 - _EPS))
                total += w_bern * (b * math.log(p) + (1.0 - b) * math.log(1.0 - p))
        group_scores[group] = float(total)

    sorted_groups = sorted(group_scores.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "group_scores_loglik": {k: float(v) for k, v in group_scores.items()},
        "ranked_groups": [{"group": k, "loglik": float(v)} for k, v in sorted_groups],
    }


def _poe_feature_group_contrib(
    *,
    emissions,
    x_t: np.ndarray,
    s: int,
    a: int,
    feature_cols: list[str],
    feature_groups: dict[str, list[str]],
) -> dict[str, Any]:
    group_idx = _build_group_index(feature_cols, feature_groups)
    group_scores = _group_contrib_template(list(group_idx.keys()))

    cont_idx = list(getattr(emissions, "cont_idx", []))
    bin_idx = list(getattr(emissions, "bin_idx", []))
    cont_set = set(cont_idx)
    bin_set = set(bin_idx)

    mu_s = np.asarray(emissions.style_gauss.mean, dtype=np.float64)
    var_s = np.asarray(emissions.style_gauss.var, dtype=np.float64)
    mu_a = np.asarray(emissions.action_gauss.mean, dtype=np.float64)
    var_a = np.asarray(emissions.action_gauss.var, dtype=np.float64)

    ps = None
    pa = None
    if (not bool(getattr(emissions, "disable_discrete_obs", False))) and len(bin_idx) > 0:
        ps = np.asarray(emissions.style_bern.p, dtype=np.float64)
        pa = np.asarray(emissions.action_bern.p, dtype=np.float64)

    cont_pos = {orig: j for j, orig in enumerate(cont_idx)}
    bin_pos = {orig: j for j, orig in enumerate(bin_idx)}
    w_bern = float(getattr(TRAINING_CONFIG, "bern_weight", 1.0))

    for group, idxs in group_idx.items():
        total = 0.0
        for i in idxs:
            xi = float(x_t[i])
            if (i in cont_set) and np.isfinite(xi):
                j = cont_pos[i]
                mus = float(mu_s[s, j])
                vas = max(float(var_s[s, j]), _EPS)
                mua = float(mu_a[a, j])
                vaa = max(float(var_a[a, j]), _EPS)
                style_term = -0.5 * (math.log(2.0 * math.pi) + math.log(vas) + ((xi - mus) * (xi - mus) / vas))
                action_term = -0.5 * (math.log(2.0 * math.pi) + math.log(vaa) + ((xi - mua) * (xi - mua) / vaa))
                v = max(vas + vaa, _EPS)
                z_term = -0.5 * (math.log(2.0 * math.pi) + math.log(v) + (((mus - mua) * (mus - mua)) / v))
                total += (style_term + action_term - z_term)
            elif (i in bin_set) and (ps is not None) and (pa is not None) and np.isfinite(xi):
                j = bin_pos[i]
                b = 1.0 if xi > 0.5 else 0.0
                p_s = float(np.clip(ps[s, j], _EPS, 1.0 - _EPS))
                p_a = float(np.clip(pa[a, j], _EPS, 1.0 - _EPS))
                z = float(np.clip(p_s * p_a + (1.0 - p_s) * (1.0 - p_a), _EPS, 1.0))
                total += w_bern * (
                    b * (math.log(p_s) + math.log(p_a) - math.log(z))
                    + (1.0 - b) * (math.log(1.0 - p_s) + math.log(1.0 - p_a) - math.log(z))
                )
        group_scores[group] = float(total)

    sorted_groups = sorted(group_scores.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "group_scores_loglik": {k: float(v) for k, v in group_scores.items()},
        "ranked_groups": [{"group": k, "loglik": float(v)} for k, v in sorted_groups],
    }


def feature_group_contributions(
    *,
    emissions,
    x_t: np.ndarray,
    s: int,
    a: int,
    feature_cols: list[str],
    feature_groups: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    groups = feature_groups or _default_feature_groups()
    if hasattr(emissions, "gauss") and hasattr(emissions, "bern"):
        return _hierarchical_feature_group_contrib(
            emissions=emissions,
            x_t=x_t,
            s=s,
            a=a,
            feature_cols=feature_cols,
            feature_groups=groups,
        )
    if hasattr(emissions, "style_gauss") and hasattr(emissions, "action_gauss"):
        return _poe_feature_group_contrib(
            emissions=emissions,
            x_t=x_t,
            s=s,
            a=a,
            feature_cols=feature_cols,
            feature_groups=groups,
        )
    return {
        "group_scores_loglik": {},
        "ranked_groups": [],
    }


@torch.no_grad()
def explain_prediction_at_t(
    *,
    obs_scaled: np.ndarray | torch.Tensor,
    emissions,
    pi_s0: np.ndarray | torch.Tensor,
    pi_a0_given_s0: np.ndarray | torch.Tensor,
    A_s: np.ndarray | torch.Tensor,
    A_a: np.ndarray | torch.Tensor,
    t: int,
    feature_cols: list[str],
    obs_raw: np.ndarray | torch.Tensor | None = None,
    frames: np.ndarray | list[int] | None = None,
    top_k: int = 3,
    feature_groups: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    x = np.asarray(obs_scaled, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"obs_scaled must be shape (T,F), got {x.shape}")

    T, F = map(int, x.shape)
    if F != len(feature_cols):
        raise ValueError(f"feature_cols length {len(feature_cols)} does not match obs feature dim {F}")
    if T == 0:
        raise ValueError("obs_scaled is empty")

    ti = int(t)
    if ti < 0 or ti >= T:
        raise IndexError(f"t={ti} out of bounds for T={T}")

    device = getattr(emissions, "_device", torch.device("cpu"))
    dtype = getattr(emissions, "_dtype", torch.float32)

    pi_s0_t = torch.as_tensor(pi_s0, device=device, dtype=dtype)
    pi_a0_s0_t = torch.as_tensor(pi_a0_given_s0, device=device, dtype=dtype)
    A_s_t = torch.as_tensor(A_s, device=device, dtype=dtype)
    A_a_t = torch.as_tensor(A_a, device=device, dtype=dtype)
    obs_torch = torch.as_tensor(x, device=device, dtype=dtype)

    fb = StructuredDBNFilter(
        pi_s0=pi_s0_t,
        pi_a0_given_s0=pi_a0_s0_t,
        A_s=A_s_t,
        A_a=A_a_t,
        device=device,
        dtype=dtype,
    )

    log_emit_all = emissions.loglikelihood(obs_torch, device=device, dtype=dtype)
    if not torch.is_tensor(log_emit_all):
        log_emit_all = torch.as_tensor(log_emit_all, device=device, dtype=dtype)

    if log_emit_all.ndim != 3:
        raise ValueError(f"Expected emission loglikelihood with shape (T,S,A), got {tuple(log_emit_all.shape)}")

    belief_prev = fb.initial_belief().log_prob
    prior_t = None
    post_t = None
    emit_t = None

    for i in range(T):
        if i == 0:
            pred_i = belief_prev
        else:
            pred_i = fb.predict(BeliefState(belief_prev, t=i - 1)).log_prob

        post_i = fb.update(BeliefState(pred_i, t=i), log_emit_all[i]).log_prob
        if i == ti:
            prior_t = pred_i
            post_t = post_i
            emit_t = log_emit_all[i]
            break
        belief_prev = post_i

    if prior_t is None or post_t is None or emit_t is None:
        raise RuntimeError("Failed to compute explanation state at requested timestep")

    forecast_t1 = fb.predict(BeliefState(post_t, t=ti)).log_prob

    prior_prob = _to_prob(prior_t)
    post_prob = _to_prob(post_t)
    emit_prob = _to_prob(emit_t)
    forecast_prob = _to_prob(forecast_t1)

    top_prior = _topk_states(prior_t, top_k)
    top_emit = _topk_states(emit_t, top_k)
    top_post = _topk_states(post_t, top_k)
    top_forecast = _topk_states(forecast_t1, top_k)

    top_prior_state = top_prior[0]
    top_post_state = top_post[0]

    tv = _tv_distance(prior_prob, post_prob)
    prior_post_map_changed = (top_prior_state.s != top_post_state.s) or (top_prior_state.a != top_post_state.a)

    if tv >= 0.20 or prior_post_map_changed:
        driver = "observation_driven"
    elif tv <= 0.08:
        driver = "persistence_driven"
    else:
        driver = "mixed"

    raw_vec = None
    if obs_raw is not None:
        raw_arr = np.asarray(obs_raw, dtype=np.float64)
        if raw_arr.shape == x.shape:
            raw_vec = raw_arr[ti]

    frame_val = None
    if frames is not None:
        fr = np.asarray(frames)
        if fr.shape[0] == T:
            frame_val = int(fr[ti])

    winning_s = int(top_post_state.s)
    winning_a = int(top_post_state.a)
    group_contrib = feature_group_contributions(
        emissions=emissions,
        x_t=x[ti],
        s=winning_s,
        a=winning_a,
        feature_cols=feature_cols,
        feature_groups=feature_groups,
    )

    feature_values_scaled = {
        feature_cols[j]: (float(x[ti, j]) if np.isfinite(x[ti, j]) else None)
        for j in range(F)
    }
    feature_values_raw = None
    if raw_vec is not None:
        feature_values_raw = {
            feature_cols[j]: (float(raw_vec[j]) if np.isfinite(raw_vec[j]) else None)
            for j in range(F)
        }

    return {
        "t": ti,
        "frame": frame_val,
        "sequence_length": T,
        "feature_dim": F,
        "topk": int(top_k),
        "observation": {
            "feature_values_scaled": feature_values_scaled,
            "feature_values_raw": feature_values_raw,
        },
        "emission": {
            "map": _state_summary_from_logp(emit_t),
            "topk": [s.__dict__ for s in top_emit],
            "prob_matrix": emit_prob.detach().cpu().numpy().astype(np.float64).tolist(),
            "loglik_matrix": emit_t.detach().cpu().numpy().astype(np.float64).tolist(),
        },
        "prior": {
            "map": _state_summary_from_logp(prior_t),
            "topk": [s.__dict__ for s in top_prior],
            "prob_matrix": prior_prob.detach().cpu().numpy().astype(np.float64).tolist(),
            "logprob_matrix": _normalize_logprob(prior_t).detach().cpu().numpy().astype(np.float64).tolist(),
        },
        "posterior": {
            "map": _state_summary_from_logp(post_t),
            "topk": [s.__dict__ for s in top_post],
            "prob_matrix": post_prob.detach().cpu().numpy().astype(np.float64).tolist(),
            "logprob_matrix": _normalize_logprob(post_t).detach().cpu().numpy().astype(np.float64).tolist(),
        },
        "forecast_t_plus_1": {
            "map": _state_summary_from_logp(forecast_t1),
            "topk": [s.__dict__ for s in top_forecast],
            "prob_matrix": forecast_prob.detach().cpu().numpy().astype(np.float64).tolist(),
            "logprob_matrix": _normalize_logprob(forecast_t1).detach().cpu().numpy().astype(np.float64).tolist(),
        },
        "diagnostics": {
            "tv_distance_prior_to_posterior": float(tv),
            "prior_to_posterior_map_changed": bool(prior_post_map_changed),
            "driver": driver,
            "winning_state": {
                "s": winning_s,
                "a": winning_a,
                "style_name": _state_name_s(winning_s),
                "action_name": _state_name_a(winning_a),
            },
        },
        "feature_group_contributions": group_contrib,
    }