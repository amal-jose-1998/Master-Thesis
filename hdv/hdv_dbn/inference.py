import numpy as np
from typing import Tuple

from .trainer import forward_backward  
from .emissions import GaussianEmissionModel

def viterbi(pi_z, A_zz, logB):
    """
    Viterbi algorithm for the most likely latent state sequence.

    Args:
        pi_z:  initial distribution over latent states z, shape (N,)
        A_zz:  transition probabilities between latent states, shape (N, N)
        logB:  log emission probabilities, shape (T, N)

    Returns:
        z_star:  np.ndarray of shape (T,) with the MAP state index at each time.
        log_p_star: log p(z*, obs) for the best path.
    """
    T, N = logB.shape

    delta = np.zeros((T, N))   # log-prob of best path ending in state z at time t
    psi = np.zeros((T, N), dtype=int)  # argmax backpointers

    log_pi = np.log(pi_z + 1e-15)
    logA = np.log(A_zz + 1e-15)

    # Initialization
    delta[0] = log_pi + logB[0]
    psi[0] = 0

    # Recursion
    for t in range(1, T):
        # For each next state j, choose best previous i
        # delta[t-1, i] + logA[i, j]
        tmp = delta[t - 1][:, None] + logA  # (N, N)
        psi[t] = np.argmax(tmp, axis=0)
        delta[t] = tmp[psi[t], range(N)] + logB[t]

    # Termination
    log_p_star = np.max(delta[-1])
    z_T = np.argmax(delta[-1])

    # Backtracking
    z_star = np.zeros(T, dtype=int)
    z_star[-1] = z_T
    for t in reversed(range(T - 1)):
        z_star[t] = psi[t + 1, z_star[t + 1]]

    return z_star, log_p_star


def infer_posterior(obs, pi_z, A_zz, emissions):
    """
    Compute posterior distributions over Style and Action for a single sequence.

    Args:
        obs:       (T, obs_dim) observation sequence
        pi_z:      (N,) initial joint distribution over states
        A_zz:      (N, N) transition matrix
        emissions: trained GaussianEmissionModel

    Returns:
        gamma:        (T, N)   joint posteriors P(Z_t | obs)
        gamma_style:  (T, S)   P(Style_t | obs)
        gamma_action: (T, A)   P(Action_t | obs)
        loglik:       float    log p(obs)
    """
    S = emissions.num_style
    A = emissions.num_action
    N = S * A

    T = obs.shape[0]
    logB = np.zeros((T, N))
    for t in range(T):
        for z in range(N):
            s = z // A
            a = z % A
            logB[t, z] = emissions.log_likelihood(obs[t], style_idx=s, action_idx=a)

    gamma, xi, loglik = forward_backward(pi_z, A_zz, logB)

    gamma_style = np.zeros((T, S))
    gamma_action = np.zeros((T, A))
    for z in range(N):
        s = z // A
        a = z % A
        gamma_style[:, s] += gamma[:, z]
        gamma_action[:, a] += gamma[:, z]

    return gamma, gamma_style, gamma_action, loglik


def infer_viterbi_paths(obs, pi_z, A_zz, emissions):
    """
    Convenience wrapper: run Viterbi and decode (style, action) indices.

    Args:
        obs:       (T, obs_dim)
        pi_z:      (N,)
        A_zz:      (N, N)
        emissions: trained GaussianEmissionModel

    Returns:
        z_star:        (T,)       MAP joint state indices
        style_star:    (T,)       MAP style indices
        action_star:   (T,)       MAP action indices
        log_p_star:    float      log p(z*, obs)
    """
    S = emissions.num_style
    A = emissions.num_action
    N = S * A

    T = obs.shape[0]
    logB = np.zeros((T, N))
    for t in range(T):
        for z in range(N):
            s = z // A
            a = z % A
            logB[t, z] = emissions.log_likelihood(obs[t], style_idx=s, action_idx=a)

    z_star, log_p_star = viterbi(pi_z, A_zz, logB)

    style_star = z_star // A
    action_star = z_star % A

    return z_star, style_star, action_star, log_p_star