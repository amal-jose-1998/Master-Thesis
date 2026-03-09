from dataclasses import dataclass
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
from ..hierarchical_emissions import MixedEmissionModel as HierarchicalMixedEmissionModel


_EPS = float(getattr(TRAINING_CONFIG, "EPSILON", 1e-6))


@dataclass(frozen=True)
class StateScore:
    s: int
    a: int
    style_name: str
    action_name: str
    prob: float


def _semantic_names_for_sa(semantic_map, s, a, default_style, default_action):
    if semantic_map is None:
        return default_style, default_action
    try:
        s_key = f"s{int(s)}"
        a_key = f"a{int(a)}"
        style_name = semantic_map.get("styles", {}).get(s_key, {}).get("name")
        action_name = semantic_map.get("actions_by_style", {}).get(s_key, {}).get(a_key, {}).get("name")
        return str(style_name or default_style), str(action_name or default_action)
    except Exception:
        return default_style, default_action


def _state_name_s(s, semantic_map=None):
    names = tuple(getattr(DBN_STATES, "driving_style", ()))
    default_style = str(names[s]) if 0 <= s < len(names) else f"style_{s}"
    style_name, _ = _semantic_names_for_sa(semantic_map, int(s), -1, default_style, "")
    return style_name


def _state_name_a(a, *, s=None, semantic_map=None):
    names = tuple(getattr(DBN_STATES, "action", ()))
    default_action = str(names[a]) if 0 <= a < len(names) else f"action_{a}"
    if s is None:
        return default_action
    _, action_name = _semantic_names_for_sa(semantic_map, int(s), int(a), "", default_action)
    return action_name


def _normalize_logprob(logp: torch.Tensor):
    z = torch.logsumexp(logp.reshape(-1), dim=0)
    norm = logp - z
    return torch.exp(norm) 


def _topk_states(logp_sa: torch.Tensor, k, semantic_map=None):
    p = _normalize_logprob(logp_sa) # convert log probabilities to actual probabilities; shape: (S, A) 
    flat_p = p.reshape(-1) # flatten to 1D for topk selection
    k = max(1, min(int(k), int(flat_p.numel())))
    vals, idxs = torch.topk(flat_p, k=k, dim=0)
    A = int(logp_sa.shape[1])
    out: list[StateScore] = []
    for rank in range(k):
        idx = int(idxs[rank].item())
        # decodes each flat index back into (s,a)
        s = int(idx // A)
        a = int(idx % A)
        out.append(
            StateScore(
                s=s,
                a=a,
                style_name=_state_name_s(s, semantic_map=semantic_map),
                action_name=_state_name_a(a, s=s, semantic_map=semantic_map),
                prob=float(vals[rank].item()),
            )
        )
    return out


def _default_feature_groups():
    return {
        "ego_kinematics": list(WINDOW_EGO_FEATURES),
        "lane_change_flags": list(WINDOW_LC_FEATURES),
        "lane_geometry": list(WINDOW_LANE_GEOM_FEATURES),
        "front_risk": list(WINDOW_FRONT_RISK_FEATURES),
        "neighbor_interaction": list(WINDOW_NEIGHBOR_REL_FEATURES),
    }


def _tv_distance(p: torch.Tensor, q: torch.Tensor):
    """Computes total variation distance 0.5 * sum(|p-q|) between two probability distributions p and q."""
    return float(0.5 * torch.abs(p - q).sum().item())


def _state_summary_from_logp(logp_sa, semantic_map=None):
    p = _normalize_logprob(logp_sa)
    flat = p.reshape(-1)
    idx = int(torch.argmax(flat).item())
    S, A = map(int, p.shape)
    s, a = int(idx // A), int(idx % A)
    return {
        "s": s,
        "a": a,
        "style_name": _state_name_s(s, semantic_map=semantic_map),
        "action_name": _state_name_a(a, s=s, semantic_map=semantic_map),
        "prob": float(flat[idx].item()), # the probability of the MAP state
        "num_states": int(S * A),
    }


def _build_group_index(feature_cols, feature_groups):
    idx_map = {name: i for i, name in enumerate(feature_cols)}
    out = {}
    for group, names in feature_groups.items():
        out[group] = [idx_map[n] for n in names if n in idx_map]
    return out


def feature_group_contributions(*, emissions: HierarchicalMixedEmissionModel, x_t: np.ndarray, s, a, feature_cols, feature_groups=None):
    """Computes the contribution of each feature group to the emission loglikelihood for a given observation x_t and state (s,a)."""
    groups = feature_groups or _default_feature_groups()
    group_idx = _build_group_index(feature_cols, groups) # maps group names to lists of feature indices
    group_scores = {g: 0.0 for g in group_idx.keys()} # initialize all group scores to zero

    # put continuous and binary feature indices into sets for fast lookup
    cont_idx = list(getattr(emissions, "cont_idx", []))
    bin_idx = list(getattr(emissions, "bin_idx", []))
    cont_set = set(cont_idx)
    bin_set = set(bin_idx)

    # reads Gaussian means/variances and Bernoulli probabilities
    gauss_mean = np.asarray(emissions.gauss.mean, dtype=np.float64)
    gauss_var = np.asarray(emissions.gauss.var, dtype=np.float64)
    bern_p = None
    if (not bool(getattr(emissions, "disable_discrete_obs", False))) and len(bin_idx) > 0:
        bern_p = np.asarray(emissions.bern.p, dtype=np.float64)

    cont_pos = {orig: j for j, orig in enumerate(cont_idx)} # maps original feature indices to their position in the Gaussian parameters
    bin_pos = {orig: j for j, orig in enumerate(bin_idx)} # maps original feature indices to their position in the Bernoulli parameters
    w_bern = float(getattr(TRAINING_CONFIG, "bern_weight", 1.0))

    # for the winning state (s,a), sums per-feature log-likelihood contributions group by group
    for group, idxs in group_idx.items():# iterate through each feature group 
        total = 0.0
        for i in idxs: # iterate through the feature indices for this group
            xi = float(x_t[i])
            if (i in cont_set) and np.isfinite(xi):
                j = cont_pos[i] # get the position of this feature in the Gaussian parameters
                mu = float(gauss_mean[s, a, j])
                var = max(float(gauss_var[s, a, j]), _EPS)
                total += -0.5 * (math.log(2.0 * math.pi) + math.log(var) + ((xi - mu) * (xi - mu) / var))
            elif (i in bin_set) and (bern_p is not None) and np.isfinite(xi):
                j = bin_pos[i] # get the position of this feature in the Bernoulli parameters
                b = 1.0 if xi > 0.5 else 0.0
                p = float(np.clip(bern_p[s, a, j], _EPS, 1.0 - _EPS))
                total += w_bern * (b * math.log(p) + (1.0 - b) * math.log(1.0 - p))
        group_scores[group] = float(total) # store the total loglikelihood contribution of this group for state (s,a)

    return {
        "group_scores_loglik": {k: float(v) for k, v in group_scores.items()}, # the raw loglikelihood contribution of each group
    }



@torch.no_grad()
def explain_prediction_at_t(*, obs_scaled, emissions: HierarchicalMixedEmissionModel, pi_s0, pi_a0_given_s0, A_s, A_a, t, feature_cols, obs_raw=None, frames=None, top_k=3, feature_groups=None, semantic_map=None):
    x = np.asarray(obs_scaled, dtype=np.float64) # (T_seq, F)
    if x.ndim != 2:
        raise ValueError(f"obs_scaled must be shape (T,F), got {x.shape}")

    T, F = map(int, x.shape)
    if F != len(feature_cols):
        raise ValueError(f"feature_cols length {len(feature_cols)} does not match obs feature dim {F}")
    if T == 0:
        raise ValueError("obs_scaled is empty")

    ti = int(t) # local index of the timestep to explain
    if ti < 0 or ti >= T: # out of bounds
        raise IndexError(f"t={ti} out of bounds for T={T}")

    device = getattr(emissions, "_device", torch.device("cpu"))
    dtype = getattr(emissions, "_dtype", torch.float32)

    # convert all model parameters to torch tensors on the same device/dtype as emissions for efficient computation
    pi_s0_t = torch.as_tensor(pi_s0, device=device, dtype=dtype)
    pi_a0_s0_t = torch.as_tensor(pi_a0_given_s0, device=device, dtype=dtype)
    A_s_t = torch.as_tensor(A_s, device=device, dtype=dtype)
    A_a_t = torch.as_tensor(A_a, device=device, dtype=dtype)
    obs_torch = torch.as_tensor(x, device=device, dtype=dtype)

    # initialize the DBN filter with the model parameters
    fb = StructuredDBNFilter(pi_s0=pi_s0_t, pi_a0_given_s0=pi_a0_s0_t, A_s=A_s_t, A_a=A_a_t, device=device, dtype=dtype)

    # compute the loglikelihood of all observations under the emission model for each (s,a) state, resulting in a (T,S,A) tensor
    log_emit_all = emissions.loglikelihood(obs_torch, device=device, dtype=dtype)
    if not torch.is_tensor(log_emit_all):
        log_emit_all = torch.as_tensor(log_emit_all, device=device, dtype=dtype)

    if log_emit_all.ndim != 3:
        raise ValueError(f"Expected emission loglikelihood with shape (T,S,A), got {tuple(log_emit_all.shape)}")

    # initial belief over latent states before processing the sequence.
    belief_prev = fb.initial_belief().log_prob # log probabilities for numerical stability; shape: (S, A)
    
    # placeholders that will be filled once the loop reaches the requested timestep ti
    prior_t = None
    post_t = None
    emit_t = None

    for i in range(T): # iterate through the sequence, updating beliefs at each step
        if i == 0: # At the first timestep, the prior is just the initial belief.
            pred_i = belief_prev
        else: # For subsequent timesteps, the prior is the one-step prediction from the previous posterior.
            pred_i = fb.predict(BeliefState(belief_prev, t=i - 1)).log_prob

        # Bayesian update step for time i
        post_i = fb.update(BeliefState(pred_i, t=i), log_emit_all[i]).log_prob
        if i == ti: # at the requested timestep, save the prior, posterior, and emission log probabilities for explanation
            prior_t = pred_i # what the DBN expected before seeing the current observation
            emit_t = log_emit_all[i] # how compatible the current window features are with each state 
            post_t = post_i # combines them into the filtered belief after observing the current timestep 
            break

        # If the target has not yet been reached, the posterior becomes the belief carried forward into the next iteration.
        belief_prev = post_i

    if prior_t is None or post_t is None or emit_t is None:
        raise RuntimeError("Failed to compute explanation state at requested timestep")

    # One-step forecast after the explained timestep
    forecast_t1 = fb.predict(BeliefState(post_t, t=ti)).log_prob # This takes the posterior at the explained time and propagates it forward once.

    # Convert log probabilities to actual probabilities 
    prior_prob = _normalize_logprob(prior_t)
    post_prob = _normalize_logprob(post_t)
    emit_prob = _normalize_logprob(emit_t)
    forecast_prob = _normalize_logprob(forecast_t1)

    # Get the top-k most likely states for each distribution 
    top_prior = _topk_states(prior_t, top_k, semantic_map=semantic_map)
    top_emit = _topk_states(emit_t, top_k, semantic_map=semantic_map)
    top_post = _topk_states(post_t, top_k, semantic_map=semantic_map)
    top_forecast = _topk_states(forecast_t1, top_k, semantic_map=semantic_map)

    # pick the MAP states.
    top_prior_state = top_prior[0]
    top_post_state = top_post[0]

    tv = _tv_distance(prior_prob, post_prob) # computes 0.5 * sum(|p-q|) between prior and posterior.
    prior_post_map_changed = (top_prior_state.s != top_post_state.s) or (top_prior_state.a != top_post_state.a) # checks whether the MAP state changed from prior to posterior.
        
    raw_vec = None
    # extracts the raw, unscaled feature vector at the explained timestep, but only if obs_raw was supplied and has matching shape.
    if obs_raw is not None:
        raw_arr = np.asarray(obs_raw, dtype=np.float64)
        if raw_arr.shape == x.shape:
            raw_vec = raw_arr[ti] # (F,)

    frame_val = None
    if frames is not None:
        fr = np.asarray(frames)
        if fr.shape[0] == T:
            frame_val = int(fr[ti]) # the absolute frame number corresponding to the explained timestep, if frames were supplied and have matching length.

    # Choosing the winning posterior state
    winning_s = int(top_post_state.s)
    winning_a = int(top_post_state.a)
    # Feature-group contribution analysis
    group_contrib = feature_group_contributions(emissions=emissions, x_t=x[ti], s=winning_s, a=winning_a, feature_cols=feature_cols, feature_groups=feature_groups)

    feature_values_scaled = { # maps feature names to their scaled values at the explained timestep
        feature_cols[j]: (float(x[ti, j]) if np.isfinite(x[ti, j]) else None)
        for j in range(F)
    }
    feature_values_raw = None
    if raw_vec is not None:
        feature_values_raw = { # maps feature names to their raw values at the explained timestep
            feature_cols[j]: (float(raw_vec[j]) if np.isfinite(raw_vec[j]) else None)
            for j in range(F)
        }

    return {
        "window_index": ti, # the window-local index being explained
        "frame": frame_val, # the absolute frame number corresponding to the explained timestep
        "sequence_length": T,
        "feature_dim": F,
        "topk": int(top_k),
        "observation": {
            "feature_values_scaled": feature_values_scaled,
            "feature_values_raw": feature_values_raw,
        },
        "emission": {
            "map": _state_summary_from_logp(emit_t, semantic_map=semantic_map),
            "topk": [s.__dict__ for s in top_emit],
            "prob_matrix": emit_prob.detach().cpu().numpy().astype(np.float64).tolist(),
            "loglik_matrix": emit_t.detach().cpu().numpy().astype(np.float64).tolist(),
        },
        "prior": {
            "map": _state_summary_from_logp(prior_t, semantic_map=semantic_map),
            "topk": [s.__dict__ for s in top_prior],
            "prob_matrix": prior_prob.detach().cpu().numpy().astype(np.float64).tolist(),
            "logprob_matrix": _normalize_logprob(prior_t).detach().cpu().numpy().astype(np.float64).tolist(),
        },
        "posterior": {
            "map": _state_summary_from_logp(post_t, semantic_map=semantic_map),
            "topk": [s.__dict__ for s in top_post],
            "prob_matrix": post_prob.detach().cpu().numpy().astype(np.float64).tolist(),
            "logprob_matrix": _normalize_logprob(post_t).detach().cpu().numpy().astype(np.float64).tolist(),
        },
        "forecast_t_plus_1": {
            "map": _state_summary_from_logp(forecast_t1, semantic_map=semantic_map),
            "topk": [s.__dict__ for s in top_forecast],
            "prob_matrix": forecast_prob.detach().cpu().numpy().astype(np.float64).tolist(),
            "logprob_matrix": _normalize_logprob(forecast_t1).detach().cpu().numpy().astype(np.float64).tolist(),
        },
        "diagnostics": {
            "tv_distance_prior_to_posterior": float(tv),
            "prior_to_posterior_map_changed": bool(prior_post_map_changed),
            "winning_state": {
                "s": winning_s,
                "a": winning_a,
                "style_name": _state_name_s(winning_s, semantic_map=semantic_map),
                "action_name": _state_name_a(winning_a, s=winning_s, semantic_map=semantic_map),
            },
        },
        "feature_group_contributions": group_contrib,
    }