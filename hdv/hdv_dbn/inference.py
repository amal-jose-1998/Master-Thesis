import torch
import numpy as np

from .trainer import forward_backward_torch

def infer_posterior(obs, pi_z, A_zz, emissions):
    """
    Compute posteriors for one trajectory under the joint-state HMM (Style, Action).

    For an observation sequence o_0:T-1, this function:
      1) Computes emission log-likelihoods for all joint states in one vectorised call:
           logB[t, z] = log p(o_t | Z_t = z)
         via `emissions.loglik_all_states(obs)`.
      2) Runs log-domain forward–backward in Torch to obtain:
           gamma[t, z] = p(Z_t = z | o_0:T-1)
           loglik      = log p(o_0:T-1)
      3) Marginalises gamma into separate posteriors over Style_t and Action_t.

    Parameters
    obs : np.ndarray
        Observation sequence of shape (T, obs_dim).
    pi_z : torch.Tensor
        Initial distribution over joint states.
        Shape: (N,), where N = S * A.
    A_zz : torch.Tensor
        Joint-state transition matrix.
        Shape: (N, N).
    emissions : GaussianEmissionModel
        Trained emission model.
        Must provide:
          - num_style 
          - num_action 
          - loglik_all_states(obs) 

    Returns
    gamma : torch.Tensor
        Joint posterior marginals.
        Shape: (T, N).
    gamma_style : torch.Tensor
        Marginal posterior over styles.
        Shape: (T, S).
    gamma_action : torch.Tensor
        Marginal posterior over actions.
        Shape: (T, A).
    loglik : float
        Log-likelihood of the full observation sequence.
    """
    device = pi_z.device
    dtype = pi_z.dtype

    # ------------------------------------------------------------------
    # Emission likelihoods 
    # ------------------------------------------------------------------
    logB = emissions.loglik_all_states(obs) # logB shape: (T, N)

    # ------------------------------------------------------------------
    # Forward–backward
    # ------------------------------------------------------------------
    with torch.no_grad():
        gamma, _, loglik_t = forward_backward_torch(pi_z, A_zz, logB) # gamma: (T, N)

    # ------------------------------------------------------------------
    # Marginalisation
    # ------------------------------------------------------------------
    S = emissions.num_style
    A = emissions.num_action
    T, N = gamma.shape

    gamma_style = torch.zeros((T, S), device=device, dtype=dtype)
    gamma_action = torch.zeros((T, A), device=device, dtype=dtype)

    gamma_sa = gamma.view(T, S, A)          # (T, S, A)
    gamma_style = gamma_sa.sum(dim=2)       # (T, S)
    gamma_action = gamma_sa.sum(dim=1)      # (T, A)

    return gamma, gamma_style, gamma_action, float(loglik_t.item())

