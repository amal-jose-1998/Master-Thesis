import torch
import numpy as np

from .forward_backward import forward_backward_torch  

def infer_posterior(obs, pi_s0, pi_a0_given_s0, A_s, A_a, emissions):
    """
    Posterior inference for the structured DBN (test-time).

    Latents:
      s_t (style), a_t (action)

    Transitions:
      P(s_t | s_{t-1})      = A_s[s_prev, s]
      P(a_t | a_{t-1}, s_t) = A_a[s_t, a_prev, a_t]

    Emissions:
      emissions.loglikelihood(obs) -> logB_sa with shape (T, S, A),
      where logB_sa[t,s,a] = log p(o_t | s_t=s, a_t=a).

    Returns
    gamma_s_a : torch.Tensor
        Joint posterior over (s,a), shape (T, S, A).
    gamma_style : torch.Tensor
        Marginal posterior over style, shape (T, S).
    gamma_action : torch.Tensor
        Marginal posterior over action, shape (T, A).
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

    # prefer the emission model's configured device/dtype
    device = getattr(emissions, "_device", pi_s0.device)
    dtype = getattr(emissions, "_dtype", pi_s0.dtype)

    pi_s0 = pi_s0.to(device=device, dtype=dtype)
    pi_a0_given_s0 = pi_a0_given_s0.to(device=device, dtype=dtype)
    A_s = A_s.to(device=device, dtype=dtype)
    A_a = A_a.to(device=device, dtype=dtype)

    # infer S,A from transitions
    S = int(pi_s0.numel())
    A = int(pi_a0_given_s0.shape[1])

    # empty obs handling
    if obs is None or getattr(obs, "shape", (0,))[0] == 0:
        gamma_s_a = torch.empty((0, S, A), device=device, dtype=dtype)
        return (
            gamma_s_a,
            torch.empty((0, S), device=device, dtype=dtype),
            torch.empty((0, A), device=device, dtype=dtype),
            0.0,
        )

    # emission log-likelihoods: (T,S,A)
    logB_sa = emissions.loglikelihood(obs)

    if not torch.is_tensor(logB_sa):
        logB_sa = torch.as_tensor(logB_sa)

    logB_sa = logB_sa.to(device=device, dtype=dtype)

    if logB_sa.ndim != 3:
        raise ValueError(f"Expected logB_sa to be 3D (T,S,A). Got shape {tuple(logB_sa.shape)}")

    T = int(logB_sa.shape[0])
    if logB_sa.shape[1] != S or logB_sa.shape[2] != A:
        raise ValueError(
            f"logB_sa has shape {tuple(logB_sa.shape)} but transitions imply S={S}, A={A}."
        )

    # forward–backward
    with torch.no_grad():
        gamma_s_a, _, _, loglik_t = forward_backward_torch(
            pi_s0=pi_s0,
            pi_a0_given_s0=pi_a0_given_s0,
            A_s=A_s,
            A_a=A_a,
            logB_s_a=logB_sa,
            return_xi_t=False,
        )

    # marginals
    gamma_style = gamma_s_a.sum(dim=2)   # (T,S)
    gamma_action = gamma_s_a.sum(dim=1)  # (T,A)

    return gamma_s_a, gamma_style, gamma_action, float(loglik_t.item())