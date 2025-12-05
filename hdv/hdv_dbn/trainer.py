import numpy as np
from typing import List

from .model import HDVDBN
from .emissions import GaussianEmissionModel


def build_joint_transition_matrix(hdv_dbn):
    """
    Build a joint HMM representation over Z_t = (Style_t, Action_t).

    Returns:
        pi_z:   initial joint distribution over z (size S*A)
        A_zz:   transition matrix over z (shape (S*A, S*A))
    """
    S = hdv_dbn.num_style
    A = hdv_dbn.num_action
    num_states = S * A

    # Extract CPDs
    cpd_style0 = hdv_dbn.model.get_cpds(('Style', 0))
    cpd_action0 = hdv_dbn.model.get_cpds(('Action', 0))
    cpd_style1 = hdv_dbn.model.get_cpds(('Style', 1))
    cpd_action1 = hdv_dbn.model.get_cpds(('Action', 1))

    # initial joint pi(z) = P(Style_0, Action_0) = P(Style_0) P(Action_0 | Style_0)
    P_style0 = cpd_style0.values.reshape(S)              # (S,)
    # cpd_action0.values has shape (A, S): P(Action_0=a | Style_0=s)
    P_action0_given_style0 = cpd_action0.values          # (A, S)

    pi_z = np.zeros(num_states)
    for s in range(S):
        for a in range(A):
            z = s * A + a # flat index
            pi_z[z] = P_style0[s] * P_action0_given_style0[a, s]

    # transition A_zz' = P(Z_{t+1} = z' | Z_t = z)
    # From CPDs:
    #   P(Style_{t+1} | Style_t)
    #   P(Action_{t+1} | Action_t, Style_{t+1})
    P_style1_given_style0 = cpd_style1.values            # (S, S): rows=new, cols=old
    # cpd_action1.values has shape (A, A*S)
    # columns indexed by (Action_0, Style_1)
    P_action1_given_action0_style1 = cpd_action1.values  # (A, A*S)

    A_zz = np.zeros((num_states, num_states))
    for s in range(S):
        for a in range(A):
            z = s * A + a
            for s_next in range(S):
                # P(Style' | Style)
                p_s = P_style1_given_style0[s_next, s]
                for a_next in range(A):
                    z_next = s_next * A + a_next # flat index for next state.
                    # col index in cpd_action1: (Action_0=a, Style_1=s_next)
                    col = s_next * A + a
                    p_a = P_action1_given_action0_style1[a_next, col]
                    A_zz[z, z_next] = p_s * p_a

    # normalize rows (each row must sum to 1.) 
    # A[z,z′]=P(Z_t+1​=z′∣Z_t​=z)
    A_zz = A_zz / A_zz.sum(axis=1, keepdims=True)

    return pi_z, A_zz


def forward_backward(pi_z, A_zz, logB):
    """
    Standard scaled forward-backward for a single sequence.

    Args:
        pi_z:  initial distribution over latent states z, shape (N,)
        A_zz:  transition probabilities between latent states, shape (N, N)
        logB:  log emission probabilities, shape (T, N) where logB[t, z] = log p(o_t | z)

    Returns:
        gamma:  posterior probability of each state at each time, P(Z_t = z | obs), shape (T, N)
        xi:     posterior probability of each transition at each time, P(Z_t=z, Z_{t+1}=z' | obs), shape (T-1, N, N)
        loglik: log-likelihood of the whole observation sequence, log p(obs)
    """
    # N = number of latent joint states (style-action pairs).
    # T = length of this observation sequence (trajectory).
    T, N = logB.shape # extract number of time steps and states
    alpha = np.zeros((T, N)) # to store log forward messages at time t, state z.
    beta = np.zeros((T, N)) # to store log backward messages.
    c = np.zeros(T)  # scaling factors for normalising per-time-step log

    # Forward
    alpha[0] = np.log(pi_z + 1e-15) + logB[0] # + 1e-15 avoids log(0).
    c[0] = np.logaddexp.reduce(alpha[0]) # log-normalizer at t=0.
    alpha[0] -= c[0] # scaled alpha at t=0 now corresponds to a normalised distribution over states (in log domain).
    for t in range(1, T):
        # log(alpha_{t-1}) + log(A)
        tmp = alpha[t-1][:, None] + np.log(A_zz + 1e-15)     # (N, N)
        alpha[t] = np.logaddexp.reduce(tmp, axis=0) + logB[t]
        c[t] = np.logaddexp.reduce(alpha[t])
        alpha[t] -= c[t]

    # Backward
    beta[-1] = 0.0  # log(1)
    for t in reversed(range(T - 1)):
        tmp = np.log(A_zz + 1e-15) + logB[t+1][None, :] + beta[t+1][None, :]
        beta[t] = np.logaddexp.reduce(tmp, axis=1)
        beta[t] -= c[t+1]

    # Posteriors
    log_gamma = alpha + beta
    log_gamma = log_gamma - np.max(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma)
    gamma = gamma / gamma.sum(axis=1, keepdims=True)

    xi = np.zeros((T - 1, N, N))
    for t in range(T - 1):
        tmp = (
            alpha[t][:, None]
            + np.log(A_zz + 1e-15)
            + logB[t+1][None, :]
            + beta[t+1][None, :]
        )
        tmp -= np.max(tmp)
        xi_t = np.exp(tmp)
        xi_t /= xi_t.sum()
        xi[t] = xi_t

    loglik = c.sum()
    return gamma, xi, loglik


class HDVTrainer:
    """
    Glue code that:
      - uses HDVDBN to get latent dynamics,
      - uses GaussianEmissionModel for continuous obs,
      - runs EM to train both.
    """

    def __init__(self, obs_dim):
        self.hdv_dbn = HDVDBN()
        self.emissions = GaussianEmissionModel(obs_dim=obs_dim)

        self.S = self.hdv_dbn.num_style
        self.A = self.hdv_dbn.num_action
        self.num_states = self.S * self.A

        self.pi_z, self.A_zz = build_joint_transition_matrix(self.hdv_dbn)

    def em_train(self, obs_seqs, num_iters=100, tol=1e-3):
        """
        EM training for the DBN.

        Parameters:
        obs_seqs : list of np.ndarray 
            List of vehicle trajectories. Each trajectory (obs_seq) is a numpy array of shape (T_n, obs_dim).
            T_n = number of time steps for that vehicle
            obs_dim = number of continuous features per time step (vx, vy, ax, ay, etc.)
                obs_seqs = [
                    obs_seq_vehicle_1,   # shape (T1, obs_dim)
                    obs_seq_vehicle_2,   # shape (T2, obs_dim)
                    obs_seq_vehicle_3,   # shape (T3, obs_dim)
                    ...
                ]
        num_iters : int
            Maximum number of EM iterations.
        tol : float
            Convergence threshold on the change in total log-likelihood.
        """
        # log-likelihood from previous iteration. Initialise as -inf because at iteration 0 we don’t have a meaningful previous value. 
        # It ensures the first improvement is always positive.
        prev_loglik = -np.inf  
        for it in range(num_iters):
            gamma_all: List[np.ndarray] = [] # will hold one gamma matrix per trajectory.
            xi_all: List[np.ndarray] = [] # will hold one xi tensor per trajectory.
            total_loglik = 0.0 # accumulates the sum of log-likelihoods over all trajectories

            # E-step
            for obs in obs_seqs:
                T_n = obs.shape[0] # number of time steps for this trajectory.
                # Compute emission log-likelihoods logB[t,z] = log p(o_t | z)
                logB = np.zeros((T_n, self.num_states)) # matrix of emission log-likelihoods for this trajectory.
                for t in range(T_n):
                    for z in range(self.num_states):
                        s = z // self.A # style index
                        a = z % self.A # action index
                        logB[t, z] = self.emissions.log_likelihood(obs[t], style_idx=s, action_idx=a) 

                gamma, xi, loglik = forward_backward(self.pi_z, self.A_zz, logB)
                gamma_all.append(gamma)
                xi_all.append(xi)
                total_loglik += loglik

            # M-step: update pi_z, A_zz
            # Initial state distribution
            pi_new = np.zeros_like(self.pi_z)
            for gamma in gamma_all:
                pi_new += gamma[0] # gamma[0] = P(Z_0 | obs); ie. posterior distribution over initial state for that sequence
            pi_new /= pi_new.sum() # normalise to sum to 1.
            self.pi_z = pi_new # The new initial distribution is the average of posterior initial states over all trajectories.

            # Transition matrix
            A_new = np.zeros_like(self.A_zz)
            for xi in xi_all:
                A_new += xi.sum(axis=0) # sum over time -> expected transition counts
            # normalize rows: each row is P(Z_{t+1} | Z_t)
            A_new = A_new / A_new.sum(axis=1, keepdims=True)
            self.A_zz = A_new

            # Gaussian means/covariances per latent state
            self.emissions.update_from_posteriors(obs_seqs=obs_seqs, gamma_seqs=gamma_all)

            # Convergence check
            improvement = total_loglik - prev_loglik
            print(
                f"[EM] Iteration {it+1}/{num_iters}, "
                f"total log-likelihood: {total_loglik:.3f}, "
                f"improvement: {improvement:.6f}"
            )
            # Skip check on the very first iteration (prev_loglik = -inf)
            if it > 0 and improvement < tol:
                print(f"[EM] Converged at iteration {it+1} (Δloglik={improvement:.6f} < tol={tol}).")
                break

            prev_loglik = total_loglik

            
