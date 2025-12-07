import numpy as np

from .trainer import forward_backward  

def viterbi(pi_z, A_zz, logB):
    """
    Run the Viterbi algorithm to find the most likely latent state sequence
    for a single observation trajectory in the joint HMM over Z_t = (Style_t, Action_t).

    Parameters
    pi_z : np.ndarray
        Initial distribution over joint latent states Z_0.
        Shape: (N,), where N = number of joint states (S * A).
    A_zz : np.ndarray
        State transition probability matrix between joint states. Each row should sum to 1.
        Shape: (N, N), where A_zz[i, j] = P(Z_{t+1} = j | Z_t = i).
    logB : np.ndarray
        Log emission likelihoods for this trajectory.
        Shape: (T, N), where logB[t, z] = log p(o_t | Z_t = z),
        T = sequence length.

    Returns
    z_star : np.ndarray
        Most probable (MAP) joint state sequence according to the model.
        Shape: (T,), where z_star[t] ∈ {0, ..., N-1} is the joint
        style–action index at time t.
    log_p_star : float
        Log probability of the best path together with the observations,
        i.e. log p(z_star, o_{0:T-1}).
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
    Compute posterior distributions over Style_t and Action_t for a single observation sequence using the current HMM/DBN parameters.
    This function:
      1. Builds the log emission matrix logB[t, z] = log p(o_t | Z_t = z) using the GaussianEmissionModel.
      2. Runs the forward–backward algorithm to obtain joint posteriors over Z_t = (style, action).
      3. Marginalises the joint posteriors to get separate posteriors over style and action at each time step.

    Parameters
    obs : np.ndarray
        Observation sequence for one vehicle/trajectory.
        Shape: (T, obs_dim), where:
            T       = number of time steps,
            obs_dim = number of continuous features per step.
    pi_z : np.ndarray
        Initial distribution over joint latent states Z_0.
        Shape: (N,), where N = S * A (S styles, A actions).
    A_zz : np.ndarray
        Transition probability matrix over joint states.
        Shape: (N, N), where A_zz[i, j] = P(Z_{t+1} = j | Z_t = i).
    emissions : GaussianEmissionModel
        Trained continuous emission model. Must expose:
            - num_style : int
            - num_action: int
            - log_likelihood(obs_t, style_idx, action_idx) -> float

    Returns
    gamma : np.ndarray
        Joint posterior over latent states.
        Shape: (T, N), where gamma[t, z] = P(Z_t = z | obs).
    gamma_style : np.ndarray
        Marginal posterior over driver style at each time step.
        Shape: (T, S), where gamma_style[t, s] = P(Style_t = s | obs).
    gamma_action : np.ndarray
        Marginal posterior over action at each time step.
        Shape: (T, A), where gamma_action[t, a] = P(Action_t = a | obs).
    loglik : float
        Log-likelihood of the entire observation sequence under the model:
        log p(obs).
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
    Run Viterbi decoding for a single observation sequence and decode the most likely style and action indices at each time step.

    This is a wrapper that:
      1. Builds log emissions logB[t, z] = log p(o_t | Z_t = z),
      2. Runs the Viterbi algorithm in the joint state space,
      3. Maps joint indices z_t back to (style_t, action_t).

    Parameters
    obs : np.ndarray
        Observation sequence for one vehicle/trajectory.
        Shape: (T, obs_dim), where:
            T       = number of time steps,
            obs_dim = number of continuous features per step.
    pi_z : np.ndarray
        Initial distribution over joint latent states Z_0.
        Shape: (N,), where N = S * A.
    A_zz : np.ndarray
        Transition probability matrix over joint states.
        Shape: (N, N), where A_zz[i, j] = P(Z_{t+1} = j | Z_t = i).
    emissions : GaussianEmissionModel
        Trained continuous emission model used to compute
        log p(o_t | style, action).

    Returns
    z_star : np.ndarray
        Most likely joint state sequence.
        Shape: (T,), where each entry is a joint index in {0, ..., N-1}.
    style_star : np.ndarray
        Most likely style index at each time step, derived from z_star.
        Shape: (T,), values in {0, ..., S-1}.
    action_star : np.ndarray
        Most likely action index at each time step, derived from z_star.
        Shape: (T,), values in {0, ..., A-1}.
    log_p_star : float
        Log probability of the best joint path and observations: log p(z_star, obs).
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