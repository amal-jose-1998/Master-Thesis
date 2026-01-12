import torch
import numpy as np

from .trainer import forward_backward_torch

def infer_posterior_structured(obs, pi_s0, pi_a0_given_s0, A_s, A_a, emissions):
    """
    Structured posterior inference for a trained HDV DBN (test-time).

    Latent state is z_t = (s_t, a_t) with factorized transitions:
      P(s_t | s_{t-1})         = A_s[s_prev, s]
      P(a_t | a_{t-1}, s_t)    = A_a[s, a_prev, a]

    Emissions are evaluated on the JOINT state z via:
      logB[t, z] = log p(o_t | s_t, a_t)
    where MixedEmissionModel already handles:
      - Gaussian(diag) on continuous dims
      - Bernoulli on *_exists
      - Categorical on lane_pos and lc (weighted)

    Parameters
    pi_s0 : torch.Tensor
        Initial distribution over styles. Shape (S,).
    pi_a0_given_s0 : torch.Tensor
        Initial action distribution given style. Shape (S, A).
    A_s : torch.Tensor
        Style transition matrix. Shape (S, S).
    A_a : torch.Tensor
        Action transition tensor. Shape (S, A, A) as A_a[s_cur, a_prev, a_cur].
    emissions : MixedEmissionModel
        Trained emission model instance with loglik_all_states(obs)->(T, N).
 
    Returns
    gamma_flat : torch.Tensor
        Joint posterior marginals γ_t(z), shape (T, S*A).
    gamma_style : torch.Tensor
        Marginal posterior γ_t(s), shape (T, S).
    gamma_action : torch.Tensor
        Marginal posterior γ_t(a), shape (T, A).
    loglik : float
        Sequence log-likelihood log p(O_1:T).
    """
    # accept NumPy or Torch inputs for transitions
    if not torch.is_tensor(pi_s0):
        pi_s0 = torch.as_tensor(pi_s0)
    if not torch.is_tensor(pi_a0_given_s0):
        pi_a0_given_s0 = torch.as_tensor(pi_a0_given_s0)
    if not torch.is_tensor(A_s):
        A_s = torch.as_tensor(A_s)
    if not torch.is_tensor(A_a):
        A_a = torch.as_tensor(A_a)

    # Prefer the emission model's configured device/dtype
    device = getattr(emissions, "_device", pi_s0.device)
    dtype = getattr(emissions, "_dtype", pi_s0.dtype)
    pi_s0 = pi_s0.to(device=device, dtype=dtype)
    pi_a0_given_s0 = pi_a0_given_s0.to(device=device, dtype=dtype)
    A_s = A_s.to(device=device, dtype=dtype)
    A_a = A_a.to(device=device, dtype=dtype)
 
    # ------------------------------------------------------------------
    # Emission likelihoods: log p(o_t | s_t, a_t)
    # ------------------------------------------------------------------
    if obs is None or getattr(obs, "shape", (0,))[0] == 0:
        S = int(getattr(emissions, "num_style", 1))
        A = int(getattr(emissions, "num_action", 1))
        gamma = torch.empty((0, S * A), device=device, dtype=dtype)
        return (
            gamma,
            torch.empty((0, S), device=device, dtype=dtype),
            torch.empty((0, A), device=device, dtype=dtype),
            0.0,
        )

    logB_flat = emissions.loglik_all_states(obs)  # (T, S*A)
    T = int(logB_flat.shape[0])
    S = int(getattr(emissions, "num_style"))
    A = int(getattr(emissions, "num_action"))
    logB_s_a = logB_flat.view(T, S, A)            # (T, S, A)
 
    # ------------------------------------------------------------------
    # Forward–backward
    # ------------------------------------------------------------------
    with torch.no_grad():
        gamma_s_a, _, _, loglik_t = forward_backward_torch(
            pi_s0=pi_s0,
            pi_a0_given_s0=pi_a0_given_s0,
            A_s=A_s,
            A_a=A_a,
            logB_s_a=logB_s_a,
        )
    # ------------------------------------------------------------------
    # Marginalisation
    # ------------------------------------------------------------------
    gamma_style = gamma_s_a.sum(dim=2)      # (T, S)
    gamma_action = gamma_s_a.sum(dim=1)     # (T, A)
    gamma_flat = gamma_s_a.reshape(T, S * A)

    return gamma_flat, gamma_style, gamma_action, float(loglik_t.item())

