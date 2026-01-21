import torch

from .config import TRAINING_CONFIG

# -----------------------------------------------------------------------------
# Numerical stability constants
# -----------------------------------------------------------------------------
EPSILON = TRAINING_CONFIG.EPSILON if hasattr(TRAINING_CONFIG, "EPSILON") else 1e-6
# =============================================================================
# Torch forward-backward 
# =============================================================================
def forward_backward_torch(pi_s0, pi_a0_given_s0, A_s, A_a, logB_s_a, return_xi_t=False):
    """
    Structured forward–backward in log-domain for z_t = (s_t, a_t), using factorized transitions:
        P(s_t | s_{t-1}) = A_s[s_prev, s]
        P(a_t | a_{t-1}, s_t) = A_a[s, a_prev, a]
    Emissions are provided over the joint state:
        logB_s_a[t, s, a] = log p(o_t | s_t=s, a_t=a)

    Parameters
    pi_s0 : torch.Tensor
        Initial style distribution, shape (S,). P(s_0 = s).
    pi_a0_given_s0 : torch.Tensor
        Initial action distribution given style, shape (S, A). P(a_0 = a | s_0 = s).
    A_s : torch.Tensor
        Style transition matrix, shape (S, S). Row-stochastic over s given s_prev.
        A_s[s_prev, s] = P(s_t = s | s_{t-1} = s_prev).
    A_a : torch.Tensor
        Action transition matrix, shape (S, A, A). Row-stochastic over a given (s, a_prev).
        A_a[s, a_prev, a] = P(a_t = a | a_{t-1} = a_prev, s_t = s).
    logB_s_a : torch.Tensor
        Emission log-likelihoods, shape (T, S, A).
        logB_s_a[t, s, a] = log p(o_t | s_t=s, a_t=a).
    return_xi_t : bool
        If True, also return per-timestep expected transition marginals:
          xi_s_t: (T-1, S, S)
          xi_a_t: (T-1, S, A, A)

    Returns
    gamma_s_a : torch.Tensor
        Posterior marginal over joint states, shape (T, S, A).
        gamma_s_a[t, s, a] = P(s_t=s, a_t=a | O).
    xi_s_sum : torch.Tensor
        Expected style transition counts summed over time, shape (S, S).
        xi_s_sum[s_prev, s] = sum_t P(s_t=s_prev, s_{t+1}=s | O).
    xi_a_sum : torch.Tensor
        Expected action transition counts summed over time, shape (S, A, A).
        xi_a_sum[s, a_prev, a] = sum_t P(s_{t+1}=s, a_t=a_prev, a_{t+1}=a | O).
    loglik : torch.Tensor
        Log-likelihood of the observation sequence: log P(O).
    xi_s_t : torch.Tensor, optional
        If return_xi_t is True, per-timestep expected style transition marginals, shape (T-1, S, S).
        xi_s_t[t, s_prev, s] = P(s_t=s_prev, s_{t+1}=s | O).
    xi_a_t : torch.Tensor, optional 
        If return_xi_t is True, per-timestep expected action transition marginals, shape (T-1, S, A, A).
        xi_a_t[t, s, a_prev, a] = P(s_{t+1}=s, a_t=a_prev, a_{t+1}=a | O).
    """
    device = logB_s_a.device
    dtype = logB_s_a.dtype

    T, S, A = logB_s_a.shape

    assert A_a.shape == (S, A, A), f"A_a shape {A_a.shape} != ({S},{A},{A})"
    assert pi_a0_given_s0.shape == (S, A)
    assert A_s.shape == (S, S)
    assert pi_s0.shape == (S,)

    # Smoothing with epsilon: additional numerical stability to avoid log(0)
    log_pi_s0 = torch.log(pi_s0 + EPSILON)                           # (S,)
    log_pi_a0_given_s0 = torch.log(pi_a0_given_s0 + EPSILON)         # (S,A)
    logAs = torch.log(A_s + EPSILON)                                 # (S,S) : s_prev -> s_next
    logAa = torch.log(A_a + EPSILON)                                 # (S,A,A): a_t depends on (s_t,a_{t-1})

    # For easier indexing: (s_next, a_prev, a_next) -> (a_prev, s_next, a_next)
    logAa_ap_s_a = logAa.permute(1, 0, 2).contiguous()  # (A, S, A); we want to sum over a_prev easily

    assert logAa_ap_s_a.shape == (A, S, A), (
        f"logAa_ap_s_a has shape {logAa_ap_s_a.shape}, expected ({A},{S},{A})"
    )

    #------------------------------------------
    # forward pass 
    #------------------------------------------
    alpha = torch.empty((T, S, A), device=device, dtype=dtype) # store log forward messages but scaled (normalized) at each timestep; shape (T,S,A)
    c = torch.empty((T,), device=device, dtype=dtype) # scaling factors per timestep; shape (T,)

    # t=0
    alpha0 = log_pi_s0[:, None] + log_pi_a0_given_s0 + logB_s_a[0] # unscaled log joint at t=0: α0​(s, a) = log p(o_0, s_0, a_0); shape (S, A)
    c0 = torch.logsumexp(alpha0.reshape(-1), dim=0)                # log normalizer (log-likelihood of the first observation) at t=0: log p(o_0)
    alpha[0] = alpha0 - c0                                         # scaled log-forward message: log p(s_0, a_0 | o_0)
    c[0] = c0                                                      # store for computing total log-likelihood later and for consistent backward scaling.

    # t>=1
    for t in range(1, T): # iterates over 1,...,T-1
        prev = alpha[t - 1]  # scaled log-forward message at time t-1: log p(s_{t-1}, a_{t-1} | o_{0:t-1}); shape (S_prev, A_prev)

        # m[s_prev, s_next, a_next] = logsumexp over a_prev of prev[s_prev,a_prev] + logAa[s_next,a_prev,a_next]
        m = torch.logsumexp(
            prev[:, :, None, None] + logAa_ap_s_a[None, :, :, :], # # broadcast -> (S_prev, A_prev, S_t, A_t)
            dim=1                                           # sum over a_prev
            )                                               # (S_prev, S_t, A_t)

        # next[s_next, a_next] = logsumexp over s_prev of m + logAs[s_prev,s_next]
        nxt = torch.logsumexp(
            m + logAs[:, :, None], # (S_prev, S_t, A_t)
            dim=0                  # sum over s_prev
            )                      # (S_t, A_t)

        alpha_t = nxt + logB_s_a[t]                         # αt​(st​, at​) = log p(o_{0:t}, s_t, a_t); shape (S, A)
        ct = torch.logsumexp(alpha_t.reshape(-1), dim=0)    # log normalizer at time t: log p(o_t | o_{0:t-1})
        alpha[t] = alpha_t - ct                             # scaled log-forward message: log p(s_t, a_t | o_{0:t})
        c[t] = ct                                           # store per-timestep log normalizer (used to recover total log-likelihood)

    loglik = c.sum() # total sequence log-likelihood: log P(O) = log p(o_{0:T-1})

    #------------------------------------------
    # backward pass
    #------------------------------------------
    # This corresponds to the boundary condition at the end.
    beta = torch.zeros((T, S, A), device=device, dtype=dtype) # scaled log backward messages initialised with zeros (log1); shape (T,S,A)

    for t in range(T - 2, -1, -1): # Iterates backward from T-2 down to 0
        nb = beta[t + 1] + logB_s_a[t + 1]  # “future message + next emission”. shape (S_next, A_next)

        # h[s_prev, s_next, a_prev] = logsumexp over a_next of logAa[s_next,a_prev,a_next] + nb[s_next,a_next]
        h = torch.logsumexp(
            logAa + nb[:, None, :],   # (S_next, A_t, A_next)
            dim=2                     # sum over a_next
            )                         # (S_next, A_t)

        # beta[t,s_prev,a_prev] = logsumexp over s_next of logAs[s_prev,s_next] + h[s_next,a_prev]
        beta_t = torch.logsumexp(
            logAs[:, :, None] + h[None, :, :],  # (S_t, S_next, A_t)  
            dim=1                               # sum over s_next
            )                                   # (S_t, A_t)

        beta[t] = beta_t - c[t + 1]  # scaling consistency

    #------------------------------------------
    # gamma 
    #----------------------------------------
    log_gamma = alpha + beta               # unnormalized log gamma: log p(s_t, a_t | O); shape (T,S,A)
    log_gamma = log_gamma - torch.logsumexp(log_gamma.reshape(T, -1), dim=1).view(T, 1, 1) # normalize per t by dividing by sum_{s,a} gamma[t,s,a]
    gamma_s_a = torch.exp(log_gamma)       # P(s_t​=s, a_t​=a ∣ o_{0:T−1}​); shape (T,S,A)

    #------------------------------------------
    # xi sums (and optional per-t)
    #------------------------------------------
    xi_s_sum = torch.zeros((S, S), device=device, dtype=dtype)
    xi_a_sum = torch.zeros((S, A, A), device=device, dtype=dtype)

    if return_xi_t: # if requested, store per-timestep xi
        xi_s_t = torch.zeros((max(T - 1, 0), S, S), device=device, dtype=dtype)    # (T-1,S,S)
        xi_a_t = torch.zeros((max(T - 1, 0), S, A, A), device=device, dtype=dtype) # (T-1,S,A,A)
    else:
        xi_s_t = None
        xi_a_t = None

    for t in range(T - 1): # iterates over 0,...,T-2
        nb = beta[t + 1] + logB_s_a[t + 1]  # nb(s_{t+1}​, a_{t+1}​) = log p(o_{t+1:T−1} ​∣ s_{t+1​}, a_{t+1}​); shape (S_next,A_next)

        # log_xi = alpha[t][:,:,None,None] + logAs[:,None,:,None] + logAa_ap_s_a[None,:,:,:] + nb[None,None,:,:]
        log_xi = (
            alpha[t][:, :, None, None]        # (S_prev,A_prev,1,1); anchors the transition at time t.
            + logAs[:, None, :, None]         # (S_prev,1,S_next,1); style transition
            + logAa_ap_s_a[None, :, :, :]     # (1,A_prev,S_next,A_next); action transition
            + nb[None, None, :, :]            # (1,1,S_next,A_next); future message + emission = future evidence
        )   # (S_prev,A_prev,S_next,A_next)

        # normalize to get xi
        Z = torch.logsumexp(log_xi.reshape(-1), dim=0) # log normalizer for xi at time t (so that xi sums to 1 over all (s,a,s',a'))
        xi = torch.exp(log_xi - Z)  # (S_prev,A_prev,S_next,A_next)

        xi_s = xi.sum(dim=(1, 3))                 # (S_prev,S_next); Expected style transition counts at time t.
        xi_a = xi.sum(dim=0).permute(1, 0, 2)     # (S_next,A_prev,A_next); Expected action transition counts (conditioned on s_{t+1}) at time t.

        # acumulate sums over time
        xi_s_sum += xi_s # shape (S,S)
        xi_a_sum += xi_a # shape (S,A,A)

        if return_xi_t: # store per-timestep xi
            xi_s_t[t] = xi_s # shape (S,S)
            xi_a_t[t] = xi_a # shape (S,A,A)

    if return_xi_t:
        return gamma_s_a, xi_s_sum, xi_a_sum, loglik, xi_s_t, xi_a_t
    else:
        return gamma_s_a, xi_s_sum, xi_a_sum, loglik