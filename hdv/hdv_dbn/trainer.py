import numpy as np
from typing import List

from model import HDVDBN
from emissions import GaussianEmissionModel


def build_joint_transition_matrix(hdv_dbn):
    """
    Build a joint HMM transition representation over the combined latent state Z_t = (Style_t, Action_t).
    This function collapses the DBN defined over (Style_t, Action_t) into a standard HMM over joint
    states z, enabling the use of standard forward–backward inference.
    The joint initial distribution is constructed as:
        pi_z(s, a) = P(Style_0 = s) · P(Action_0 = a | Style_0 = s)
    The joint transition matrix is constructed as:
        A_zz[(s,a), (s',a')] = P(Style_{t+1} = s' | Style_t = s) · P(Action_{t+1} = a' | Action_t = a, Style_{t+1} = s')

    Parameters
    hdv_dbn : HDVDBN
        Trained or initialized HDVDBN object containing the DBN structure and Tabular CPDs for Style and Action variables.

    Returns
    pi_z : np.ndarray
        Initial joint distribution over joint states z = (Style, Action), 
        shape (S*A,), where S = number of styles and A = number of actions.
    A_zz : np.ndarray
        Joint state transition matrix,
        shape (S*A, S*A), where each row is a valid probability distribution:
        A_zz[z, z'] = P(Z_{t+1} = z' | Z_t = z).
    """
    S = int(hdv_dbn.num_style)
    A = int(hdv_dbn.num_action)
    num_states = S * A

    # Extract CPDs
    cpd_style0 = hdv_dbn.model.get_cpds(('Style', 0))
    cpd_action0 = hdv_dbn.model.get_cpds(('Action', 0))
    cpd_style1 = hdv_dbn.model.get_cpds(('Style', 1))
    cpd_action1 = hdv_dbn.model.get_cpds(('Action', 1))

    # initial joint pi(z) = P(Style_0, Action_0) = P(Style_0) P(Action_0 | Style_0)
    P_style0 = np.asarray(cpd_style0.values, dtype=float).reshape(S)      # (S,)
    # cpd_action0.values has shape (A, S): P(Action_0=a | Style_0=s)
    P_action0_given_style0 = np.asarray(cpd_action0.values, dtype=float)  # (A, S)

    pi_z = np.zeros(num_states)
    for s in range(S):
        for a in range(A):
            z = s * A + a # flat index
            pi_z[z] = P_style0[s] * P_action0_given_style0[a, s]

    # transition A_zz' = P(Z_{t+1} = z' | Z_t = z)
    # From CPDs:
    #   P(Style_{t+1} | Style_t)
    #   P(Action_{t+1} | Action_t, Style_{t+1})
    P_style1_given_style0 = np.asarray(cpd_style1.values, dtype=float).reshape(S, S) # (S, S): rows=new, cols=old
    # cpd_action1.values has shape (A, A*S)
    # columns indexed by (Action_0, Style_1)
    P_action1_given_action0_style1 = np.asarray(cpd_action1.values, dtype=float).reshape(A, A * S) # (A, A*S)

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
    row_sums = A_zz.sum(axis=1, keepdims=True)
    # Avoid division by zero: if a row is all zeros, keep it uniform instead of NaN
    zero_rows = (row_sums == 0)
    row_sums[zero_rows] = 1.0
    A_zz = A_zz / row_sums

    return pi_z, A_zz

def forward_backward(pi_z, A_zz, logB):
    """
    Run the scaled forward–backward algorithm for a single observation sequence.
    It computes:
        - gamma[t, z] = P(Z_t = z | o_{0:T-1})
        - xi[t, z, z'] = P(Z_t = z, Z_{t+1} = z' | o_{0:T-1})

    Parameters
    pi_z : np.ndarray
        Initial distribution over joint latent states z,
        shape (N,), where N = number of joint states.
    A_zz : np.ndarray
        Transition probability matrix between joint states,
        shape (N, N), where A_zz[z, z'] = P(Z_{t+1} = z' | Z_t = z).
    logB : np.ndarray
        Log emission likelihoods,
        shape (T, N), where logB[t, z] = log p(o_t | Z_t = z).

    Returns
    gamma : np.ndarray
        Posterior marginal probabilities over states,
        shape (T, N), where gamma[t, z] = P(Z_t = z | o_{0:T-1}).
    xi : np.ndarray
        Posterior transition probabilities,
        shape (T-1, N, N), where xi[t, z, z'] = P(Z_t = z, Z_{t+1} = z' | o_{0:T-1}).
    loglik : float
        Log-likelihood of the entire observation sequence: log p(o_{0:T-1}).
    """
    # N = number of latent joint states z = (Style, Action).
    # T = number of time steps in the observation sequence (trajectory).
    T, N = logB.shape # extract number of time steps and states
   
    # alpha[t, z] stores the *scaled log-forward message*:
    #   alpha[t, z] ≈ log P(Z_t = z | o_0:t)   (up to a normalization constant)
    alpha = np.zeros((T, N)) 
    # beta[t, z] stores the *scaled log-backward message*:
    #   beta[t, z] ≈ log P(o_{t+1:T-1} | Z_t = z)   (up to a normalization constant)
    beta = np.zeros((T, N)) 
    # c[t] stores the log normalization constant at time t:
    #   c[t] = log sum_z exp(alpha_unscaled[t, z])
    # These constants are required to:
    #   (1) prevent numerical underflow
    #   (2) compute the total log-likelihood exactly
    c = np.zeros(T)  

    # Initial forward message:
    #   alpha[0, z] = log P(Z_0 = z) + log P(o_0 | Z_0 = z)
    # +1e-15 avoids log(0) for impossible states
    alpha[0] = np.log(pi_z + 1e-15) + logB[0] 

    # Log normalizer at t=0:
    #   c[0] = log sum_z exp(alpha[0, z])
    c[0] = np.logaddexp.reduce(alpha[0]) 

    # Normalize alpha[0] in log-space: After this, exp(alpha[0]) sums to 1, i.e., alpha[0] represents a valid posterior over Z_0 (in log space)
    alpha[0] -= c[0] 

    # Forward recursion:
    # tmp[z_prev, z] = alpha[t-1, z_prev] + log P(Z_t = z | Z_{t-1} = z_prev)
    # alpha[t, z] = log sum_{z_prev} exp(tmp[z_prev, z]) + log P(o_t | Z_t = z)
    for t in range(1, T):
        # tmp = log(alpha_{t-1}) + log(A)
        tmp = alpha[t-1][:, None] + np.log(A_zz + 1e-15)  # alpha[t-1] is already in log-space.
        alpha[t] = np.logaddexp.reduce(tmp, axis=0) + logB[t]
        # Compute normalization constant at time t
        #   c[t] = log sum_z exp(alpha[t, z])
        c[t] = np.logaddexp.reduce(alpha[t])
        alpha[t] -= c[t] # # Normalize alpha[t] so it remains numerically stable

    # Initialization: beta[T-1, z] = log(1)
    # Because there are no future observations after time T-1
    beta[-1] = 0.0  

    # Backward recursion:
    # tmp[z, z_next] = log P(Z_{t+1} = z_next | Z_t = z) + log P(o_{t+1} | Z_{t+1} = z_next) + beta[t+1, z_next]
    # beta[t, z] = log sum_{z_next} exp(tmp[z, z_next])
    for t in reversed(range(T - 1)):
        tmp = np.log(A_zz + 1e-15) + logB[t+1][None, :] + beta[t+1][None, :]
        beta[t] = np.logaddexp.reduce(tmp, axis=1)
        # Apply the same scaling used in the forward pass
        # This ensures alpha and beta are on the same scale
        beta[t] -= c[t+1]

    # Posteriors: Combine forward and backward messages
    log_gamma = alpha + beta #   log_gamma[t, z] ∝ log P(Z_t = z | o_0:T-1)
    log_gamma = log_gamma - np.max(log_gamma, axis=1, keepdims=True) # Numerical stabilization: subtract max per time step before exponentiating
    gamma = np.exp(log_gamma) # Convert from log-space to probability space
    gamma = gamma / gamma.sum(axis=1, keepdims=True) # Normalize so that, for each t: sum_z gamma[t, z] = 1

    xi = np.zeros((T - 1, N, N)) # xi[t, z, z'] = P(Z_t = z, Z_{t+1} = z' | o_0:T-1)
    for t in range(T - 1):
        # Unnormalized joint log-probability:
        #   alpha[t, z] + log P(Z_{t+1} = z' | Z_t = z) + log P(o_{t+1} | Z_{t+1} = z') + beta[t+1, z']
        tmp = (
            alpha[t][:, None]
            + np.log(A_zz + 1e-15)
            + logB[t+1][None, :]
            + beta[t+1][None, :]
        )
        tmp -= np.max(tmp) # Stabilize before exponentiating
        xi_t = np.exp(tmp) # Convert to probability space
        xi_t /= xi_t.sum() # Normalize so sum_{z,z'} xi[t,z,z'] = 1
        xi[t] = xi_t # Store posterior transitions for time t

    loglik = c.sum() # Total log-likelihood of the observation sequence: log p(o_0:T-1) = sum_t c[t]
    
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

    def em_train(self, train_obs_seqs, val_obs_seqs=None, num_iters=100, tol=1e-3):
        """
        Train the joint DBN–Gaussian emission model using the EM algorithm.
        This method alternates between:
            - E-step: inference over joint latent states (Style, Action) using forward–backward to compute posterior responsibilities.
            - M-step:
                * update initial state distribution pi_z,
                * update transition matrix A_zz,
                * update Gaussian emission means and covariances.

        Parameters
        train_obs_seqs : list of np.ndarray
            List of observation sequences, one per vehicle, for training.
            Each element has shape (T_n, obs_dim), where:
                - T_n is the number of time steps for vehicle n,
                - obs_dim is the dimensionality of the observation vector
        val_obs_seqs : list of np.ndarray or None, optional
            Optional list of observation sequences used for validation. If provided, a validation log-likelihood is computed at each
            EM iteration and used for early stopping.
        num_iters : int, optional
            Maximum number of EM iterations. Default is 100.
        tol : float, optional
            Convergence threshold on the improvement in total log-likelihood between successive EM iterations. Default is 1e-3.
        
        Returns
        history : dict
            Dictionary with keys:
                - "train_loglik": list of total train log-likelihood per iter
                - "val_loglik": list of total val log-likelihood per iter (empty if no val data is provided)
        """
        history = {"train_loglik": [], "val_loglik": []}
        # If validation data is available, we use its log-likelihood for early stopping; otherwise we fall back to training LL.
        # log-likelihood from previous iteration. Initialise as -inf because at iteration 0 we don’t have a meaningful previous value. 
        # It ensures the first improvement is always positive.
        prev_criterion = -np.inf
        for it in range(num_iters):
            gamma_all: List[np.ndarray] = [] # will hold one gamma matrix per trajectory.
            xi_all: List[np.ndarray] = [] # will hold one xi tensor per trajectory.
            total_train_loglik = 0.0 # accumulates the sum of log-likelihoods over all trajectories

            # ----------------------
            # E-step on training data
            # ----------------------
            for obs in train_obs_seqs:
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
                total_train_loglik += loglik

            # ----------------------
            # M-step: update pi_z, A_zz
            # ----------------------
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
            self.emissions.update_from_posteriors(obs_seqs=train_obs_seqs, gamma_seqs=gamma_all)

            # ----------------------
            # Compute validation log-likelihood (if available)
            # ----------------------
            total_val_loglik = 0.0
            if val_obs_seqs is not None:
                for obs in val_obs_seqs:
                    T_n = obs.shape[0]
                    logB = np.zeros((T_n, self.num_states))
                    for t in range(T_n):
                        for z in range(self.num_states):
                            s = z // self.A
                            a = z % self.A
                            logB[t, z] = self.emissions.log_likelihood(
                                obs[t], style_idx=s, action_idx=a
                            )
                    # For validation we only need the log-likelihood
                    _, _, loglik_val = forward_backward(self.pi_z, self.A_zz, logB)
                    total_val_loglik += loglik_val
            
            # ----------------------
            # Bookkeeping and early stopping
            # ----------------------
            history["train_loglik"].append(total_train_loglik)
            if val_obs_seqs is not None:
                history["val_loglik"].append(total_val_loglik)
                criterion = total_val_loglik
                msg = (
                    f"[EM] Iteration {it+1}/{num_iters}  "
                    f"train LL = {total_train_loglik:.3f}  "
                    f"val LL = {total_val_loglik:.3f}  "
                    f"improvement = {criterion - prev_criterion:.6f}"
                )
            else:
                criterion = total_train_loglik
                msg = (
                    f"[EM] Iteration {it+1}/{num_iters}  "
                    f"train LL = {total_train_loglik:.3f}  "
                    f"improvement = {criterion - prev_criterion:.6f}"
                )
            print(msg)

            # Skip check on the very first iteration (prev_loglik = -inf)
            if it > 0 and (criterion - prev_criterion) < tol:
                print(
                    f"[EM] Early stop at iter {it+1}: "
                    f"improvement {criterion - prev_criterion:.6f} < tol={tol}."
                )
                break

            prev_criterion = criterion

        return history


    def save(self, path):
        """
        Save the learned joint transition parameters and Gaussian emissions to a .npz file.

        Parameters
        path : str or Path
            Target file path (e.g. 'models/dbn_highd.npz').
        """
        path = str(path)
        means, covs = self.emissions.to_arrays()
        np.savez_compressed(path, pi_z=self.pi_z, A_zz=self.A_zz, means=means, covs=covs)
        print(f"[HDVTrainer] Saved model parameters to {path}")

    @classmethod
    def load(cls, path):
        """
        Load a trained HDVTrainer instance from a .npz file.

        Parameters
        path : str or Path
            Path to the saved .npz file.

        Returns
        HDVTrainer
            A trainer instance with pi_z, A_zz and emission parameters restored.
        """
        path = str(path)
        data = np.load(path)
        pi_z = data["pi_z"]
        A_zz = data["A_zz"]
        means = data["means"]
        covs = data["covs"]

        obs_dim = means.shape[-1]
        trainer = cls(obs_dim=obs_dim)

        # Override initial values with loaded ones
        trainer.pi_z = pi_z
        trainer.A_zz = A_zz
        trainer.emissions.from_arrays(means, covs)

        print(f"[HDVTrainer] Loaded model parameters from {path}")
        return trainer    
