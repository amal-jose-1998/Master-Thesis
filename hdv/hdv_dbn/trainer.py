import numpy as np
from typing import List
from sklearn.cluster import MiniBatchKMeans
from tqdm.auto import tqdm
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .model import HDVDBN
from .emissions import GaussianEmissionModel, GaussianEmissionParams
from .config import TRAINING_CONFIG



def build_joint_transition_matrix(hdv_dbn):
    """
    Build a joint HMM transition representation over the combined latent state Z_t = (Style_t, Action_t).
    This function collapses the DBN defined over (Style_t, Action_t) into a standard HMM over joint
    states z, enabling the use of standard forward–backward inference.
    The joint initial distribution is constructed as:
        pi_z(s, a) = P(Style_0 = s) · P(Action_0 = a | Style_0 = s)
    The joint transition matrix is constructed as:
        A_zz[(s,a), (s',a')] = P(Style_{t+1} = s' | Style_t = s) · P(Action_{t+1} = a' | Action_t = a, Style_t = s, Style_{t+1} = s')

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

    # -------------------------
    # initial joint pi(z) = P(Style_0, Action_0) = P(Style_0) P(Action_0 | Style_0)
    # -------------------------
    P_style0 = np.asarray(cpd_style0.values, dtype=float).reshape(S)      # (S,)
    # cpd_action0.values has shape (A, S): P(Action_0=a | Style_0=s)
    P_action0_given_style0 = np.asarray(cpd_action0.values, dtype=float)  # (A, S)

    pi_z = np.zeros(num_states)
    for s in range(S):
        for a in range(A):
            z = s * A + a # flat index for (s, a)
            pi_z[z] = P_style0[s] * P_action0_given_style0[a, s]

    # -------------------------
    # transition A_zz' = P(Z_{t+1} = z' | Z_t = z)
    # -------------------------
    # From CPDs:
    #   P(Style_{t+1} | Style_t)
    #   P(Action_{t+1} | Action_t, Style_t, Style_{t+1})
    P_style1_given_style0 = np.asarray(cpd_style1.values, dtype=float).reshape(S, S) # (S, S): rows=new, cols=old
    # cpd_action1.values has shape (A, A*S*S)
    P_action1_given_action0_style0_style1 = np.asarray(cpd_action1.values, dtype=float).reshape(A, A * S * S)

    A_zz = np.zeros((num_states, num_states), dtype=float)
    for s in range(S):
        for a in range(A):
            z = s * A + a # index for current state
            for s_next in range(S):
                p_s = P_style1_given_style0[s_next, s]
                for a_next in range(A):
                    z_next = s_next * A + a_next # flat index for next state.
                    # Column index in cpd_action1 corresponding to (Action_0 = a, Style_0 = s, Style_1 = s_next)
                    # With evidence order [Action_0, Style_0, Style_1]
                    # and evidence_card [A, S, S], the mapping is: col = ((a * S) + s) * S + s_next
                    col = ((a * S) + s) * S + s_next
                    p_a = P_action1_given_action0_style0_style1[a_next, col]
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
        - xi_sum[z, z'] = sum_t P(Z_t = z, Z_{t+1} = z' | o_{0:T-1})

    Parameters
    pi_z : np.ndarray, shape (N,)
        Initial distribution over joint latent states z,  where N = number of joint states.
    A_zz : np.ndarray, shape (N, N)
        Transition probability matrix between joint states,  where A_zz[z, z'] = P(Z_{t+1} = z' | Z_t = z).
    logB : np.ndarray, shape (T, N)
        Log emission likelihoods, where logB[t, z] = log p(o_t | Z_t = z).

    Returns
    gamma : np.ndarray, shape (T, N)
        Posterior marginal probabilities over states, where gamma[t, z] = P(Z_t = z | o_{0:T-1}).
    xi_sum : np.ndarray, shape (N, N)
        Expected transition counts: xi_sum[z, z'] = sum_t P(Z_t=z, Z_{t+1}=z' | o).
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

    logA = np.log(A_zz + 1e-15)

    # ------------------------------------------------------------------
    # Initial forward message:
    # ------------------------------------------------------------------
    #   alpha[0, z] = log P(Z_0 = z) + log P(o_0 | Z_0 = z)
    # +1e-15 avoids log(0) for impossible states
    alpha[0] = np.log(pi_z + 1e-15) + logB[0] 
    # Log normalizer at t=0:
    #   c[0] = log sum_z exp(alpha[0, z])
    c[0] = np.logaddexp.reduce(alpha[0]) 
    # Normalize alpha[0] in log-space: After this, exp(alpha[0]) sums to 1, i.e., alpha[0] represents a valid posterior over Z_0 (in log space)
    alpha[0] -= c[0] 

    # ------------------------------------------------------------------
    # Forward recursion:
    # ------------------------------------------------------------------
    # tmp[z_prev, z] = alpha[t-1, z_prev] + log P(Z_t = z | Z_{t-1} = z_prev)
    # alpha[t, z] = log sum_{z_prev} exp(tmp[z_prev, z]) + log P(o_t | Z_t = z)
    for t in range(1, T):
        # tmp = log(alpha_{t-1}) + log(A)
        tmp = alpha[t-1][:, None] + logA  # alpha[t-1] is already in log-space.
        alpha[t] = np.logaddexp.reduce(tmp, axis=0) + logB[t]
        # Compute normalization constant at time t
        #   c[t] = log sum_z exp(alpha[t, z])
        c[t] = np.logaddexp.reduce(alpha[t])
        alpha[t] -= c[t] # # Normalize alpha[t] so it remains numerically stable

    # ------------------------------------------------------------------
    # Initialization: beta[T-1, z] = log(1)
    # ------------------------------------------------------------------
    # Because there are no future observations after time T-1
    beta[-1] = 0.0  

    # ------------------------------------------------------------------
    # Backward recursion:
    # ------------------------------------------------------------------
    # tmp[z, z_next] = log P(Z_{t+1} = z_next | Z_t = z) + log P(o_{t+1} | Z_{t+1} = z_next) + beta[t+1, z_next]
    # beta[t, z] = log sum_{z_next} exp(tmp[z, z_next])
    for t in reversed(range(T - 1)):
        tmp = logA + logB[t+1][None, :] + beta[t+1][None, :]
        beta[t] = np.logaddexp.reduce(tmp, axis=1)
        # Apply the same scaling used in the forward pass
        # This ensures alpha and beta are on the same scale
        beta[t] -= c[t+1]

    # ------------------------------------------------------------------
    # Posteriors: Combine forward and backward messages
    # ------------------------------------------------------------------
     # ----- State posteriors -----
    log_gamma = alpha + beta #   log_gamma[t, z] ∝ log P(Z_t = z | o_0:T-1)
    log_gamma = log_gamma - np.max(log_gamma, axis=1, keepdims=True) # Numerical stabilization: subtract max per time step before exponentiating
    gamma = np.exp(log_gamma) # Convert from log-space to probability space
    gamma = gamma / gamma.sum(axis=1, keepdims=True) # Normalize so that, for each t: sum_z gamma[t, z] = 1
    # ----- Transition posteriors (summed over time) -----
    xi_sum = np.zeros((N, N))
    for t in range(T - 1):
        # Unnormalized joint log-probability:
        #   alpha[t, z] + log P(Z_{t+1} = z' | Z_t = z) + log P(o_{t+1} | Z_{t+1} = z') + beta[t+1, z']
        tmp = (
            alpha[t][:, None]
            + logA
            + logB[t+1][None, :]
            + beta[t+1][None, :]
        )
        tmp -= np.max(tmp) # Stabilize before exponentiating
        xi_t = np.exp(tmp) # Convert to probability space
        xi_t /= xi_t.sum() # Normalize so sum_{z,z'} xi[t,z,z'] = 1
        xi_sum += xi_t

    loglik = c.sum() # Total log-likelihood of the observation sequence: log p(o_0:T-1) = sum_t c[t]
    
    return gamma, xi_sum, loglik


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

        self.scaler_mean = None  # shape (obs_dim,)
        self.scaler_std = None   # shape (obs_dim,)

    def em_train(self, train_obs_seqs, val_obs_seqs=None, wandb_run=None):
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
        wandb_run : wandb.sdk.wandb_run.Run or None, optional
            If provided, training metrics are logged to Weights & Biases at the end of each EM iteration.
        
        Returns
        history : dict
            Dictionary with keys:
                - "train_loglik": list of total train log-likelihood per iter
                - "val_loglik": list of total val log-likelihood per iter (empty if no val data is provided)
        """
        num_iters = TRAINING_CONFIG.em_num_iters
        tol = TRAINING_CONFIG.em_tol
        verbose = TRAINING_CONFIG.verbose
        use_progress = TRAINING_CONFIG.use_progress

        history = {"train_loglik": [], "val_loglik": []}

        # k-means initialisation of emission parameters
        self._init_emissions_kmeans(train_obs_seqs)

        prev_criterion = -np.inf # log-likelihood from previous iteration. Initialise as -inf because at iteration 0 we don’t have a meaningful previous value. 

        if verbose:
            print("\n==================== EM TRAINING START ====================\n")
            print(f"Number of style states:  {self.S}")
            print(f"Number of action states: {self.A}")
            print(f"Total joint states:      {self.num_states}")
            print(f"Training sequences:      {len(train_obs_seqs)}")
            if val_obs_seqs is not None:
                print(f"Validation sequences:    {len(val_obs_seqs)}")
            print("-----------------------------------------------------------\n")

        for it in range(num_iters):
            iter_start = time.perf_counter()
            if verbose:
                print(f"\n--------------- EM ITERATION {it+1} ----------------")

            # ----------------------
            # E-step on training data
            # ----------------------
            if verbose:
                print("E-step (train):")
            gamma_all, xi_all, total_train_loglik = self._e_step(
                obs_seqs=train_obs_seqs,
                use_progress=use_progress,
                verbose=verbose,
                it=it
            )

            # ----------------------
            # M-step: update pi_z, A_zz
            # ----------------------
            if verbose:
                print("M-step: updating π_z and A_zz...")
            delta_pi, delta_A, A_prev, A_new = self._m_step_transitions(
                gamma_all=gamma_all,
                xi_all=xi_all,
                verbose=verbose
            )
            
            # ----------------------
            # M-step: update emission parameters
            # ----------------------
            if verbose:
                print("  Updating emission parameters...")
            (state_weights_flat,
             total_responsibility_mass,
             state_weights_frac) = self._m_step_emissions(
                 train_obs_seqs=train_obs_seqs,
                 gamma_all=gamma_all,
                 use_progress=use_progress,
                 verbose=verbose
             )
            
            # ----------------------
            # Compute validation log-likelihood (if available)
            # ----------------------
            if verbose:
                print("\nValidation:")
            total_val_loglik, criterion = self._compute_validation_criterion(
                val_obs_seqs=val_obs_seqs,
                total_train_loglik=total_train_loglik,
                use_progress=use_progress,
                verbose=verbose,
                it=it
            )

            # ----------------------
            # Bookkeeping
            # ----------------------
            improvement = criterion - prev_criterion
            if verbose:
                print(f"  Criterion: {criterion:.3f}")
                print(f"  Improvement: {improvement:.6f}")
            history["train_loglik"].append(total_train_loglik)
            if val_obs_seqs is not None:
                history["val_loglik"].append(total_val_loglik)
            
            # ----------------------
            # WandB logging
            # ----------------------
            self._log_wandb_iteration(
                wandb_run=wandb_run,
                it=it,
                iter_start=iter_start,
                total_train_loglik=total_train_loglik,
                total_val_loglik=total_val_loglik,
                improvement=improvement,
                delta_pi=delta_pi,
                delta_A=delta_A,
                state_weights_flat=state_weights_flat,
                total_responsibility_mass=total_responsibility_mass,
                state_weights_frac=state_weights_frac,
                val_obs_seqs=val_obs_seqs,
                A_prev=A_prev,
                A_new=A_new,
            )
            
            # ----------------------
            # Early stopping
            # ----------------------
            # Skip check on the very first iteration (prev_loglik = -inf)
            if it > 0 and (criterion - prev_criterion) < tol:
                if verbose:
                    print("\n*** Early stopping triggered ***")
                break
            prev_criterion = criterion

        if verbose:
            print("\n===================== EM TRAINING END =====================")
        return history

    # ------------------------------------------------------------------
    # E-step helpers
    # ------------------------------------------------------------------
    def _compute_logB_for_sequence(self, obs):
        """Compute logB[t, z] = log p(o_t | z) for one trajectory."""
        T_n = obs.shape[0] # number of time steps for this trajectory.
        logB = np.zeros((T_n, self.num_states)) # matrix of emission log-likelihoods for this trajectory.
        for t in range(T_n):
            x_t = obs[t]
            for z in range(self.num_states):
                s = z // self.A
                a = z % self.A
                logB[t, z] = self.emissions.log_likelihood(
                    x_t, style_idx=s, action_idx=a
                )
        return logB

    def _e_step(self, obs_seqs, use_progress, verbose, it):
        """Run forward–backward on all training sequences."""
        gamma_all: List[np.ndarray] = [] # will hold one gamma matrix per trajectory.
        xi_all: List[np.ndarray] = [] # will hold one xi tensor per trajectory.
        total_train_loglik = 0.0 # accumulates the sum of log-likelihoods over all trajectories

        iterator = enumerate(obs_seqs)
        if use_progress:
            iterator = tqdm(
                iterator,
                total=len(obs_seqs),
                desc=f"E-step train (iter {it+1})",
                leave=False,
            )

        for i, obs in iterator:
            logB = self._compute_logB_for_sequence(obs) # Compute emission log-likelihoods logB[t,z] = log p(o_t | z)
            gamma, xi_sum, loglik = forward_backward(self.pi_z, self.A_zz, logB)

            if np.isnan(loglik):
                if verbose >= 2:
                    print(f"  Seq {i:03d}: loglik is NaN, skipping.")
                continue

            if verbose >= 2:
                print(f"  Seq {i:03d}: T={obs.shape[0]}, loglik={loglik:.3f}")

            gamma_all.append(gamma)
            xi_all.append(xi_sum)
            total_train_loglik += loglik

        if verbose:
            print(f"  Total train loglik: {total_train_loglik:.3f}")

        return gamma_all, xi_all, total_train_loglik

    # ------------------------------------------------------------------
    # M-step helpers: π_z, A_zz, emissions
    # ------------------------------------------------------------------
    def _m_step_transitions(self, gamma_all, xi_all, verbose):
        """Update π_z and A_zz from posterior statistics."""
        pi_prev = self.pi_z.copy()
        A_prev = self.A_zz.copy()

        # Initial state distribution
        pi_new = np.zeros_like(self.pi_z)
        for gamma in gamma_all:
            pi_new += gamma[0]  # posterior over Z_0
        pi_new /= pi_new.sum()  # normalise to sum to 1.
        self.pi_z = pi_new
        delta_pi = float(np.abs(pi_new - pi_prev).sum())

        # Transition matrix
        A_new = np.zeros_like(self.A_zz)
        for xi_sum in xi_all:
            A_new += xi_sum
        # normalize rows: each row is P(Z_{t+1} | Z_t)
        row_sums = A_new.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0 # Avoid division by zero
        A_new = A_new / row_sums
        self.A_zz = A_new
        delta_A = float(np.abs(A_new - A_prev).mean())

        if verbose:
            print(f"  Δπ_z (sum abs diff): {delta_pi:.6e}")
            print(f"  ΔA_zz (mean abs diff): {delta_A:.6e}")

        return delta_pi, delta_A, A_prev, A_new
    
    def _m_step_emissions(self, train_obs_seqs, gamma_all, use_progress, verbose):
        """
        Update Gaussian emission parameters from posteriors.
        Returns per-state responsibility mass and its normalised version.
        """
        
        state_weights_flat = self.emissions.update_from_posteriors(
            obs_seqs=train_obs_seqs,
            gamma_seqs=gamma_all,
            use_progress=use_progress,
            verbose=verbose,
        )
        total_responsibility_mass = float(state_weights_flat.sum())
        if total_responsibility_mass > 0.0:
            state_weights_frac = state_weights_flat / total_responsibility_mass
        else:
            state_weights_frac = np.zeros_like(state_weights_flat)

        if verbose >= 2:
            print("  Emission means (per state, first 3 dims):")
            for z in range(self.num_states):
                s = z // self.A
                a = z % self.A
                mean = self.emissions.params[s, a].mean
                print(f"    z={z:02d} (s={s},a={a}) mean[:3]={mean[:3]}")

        return state_weights_flat, total_responsibility_mass, state_weights_frac

    # ------------------------------------------------------------------
    # Validation / criterion helper
    # ------------------------------------------------------------------
    def _compute_validation_criterion(self, val_obs_seqs, total_train_loglik, use_progress, verbose, it):
        """
        Compute total validation log-likelihood and the criterion used for early stopping. If no validation set is provided, the training
        log-likelihood is used as criterion.
        """
        if val_obs_seqs is None:
            if verbose:
                print("No validation set provided; using train LL as criterion.")
            return 0.0, total_train_loglik

        if verbose:
            print("Validation E-step:")

        total_val_loglik = 0.0
        val_iter = enumerate(val_obs_seqs)
        if use_progress:
            val_iter = tqdm(
                val_iter,
                total=len(val_obs_seqs),
                desc=f"E-step val   (iter {it+1})",
                leave=False,
            )

        for j, obs in val_iter:
            logB = self._compute_logB_for_sequence(obs)
            _, _, loglik_val = forward_backward(self.pi_z, self.A_zz, logB) # For validation we only need the log-likelihood
            total_val_loglik += loglik_val

        if verbose:
            print(f"  Total val loglik: {total_val_loglik:.3f}")

        return total_val_loglik, total_val_loglik
    
    # ------------------------------------------------------------------
    # WandB logging helper
    # ------------------------------------------------------------------
    def _log_wandb_iteration(self, wandb_run, it, iter_start, total_train_loglik, total_val_loglik, improvement, delta_pi,
                             delta_A, state_weights_flat, total_responsibility_mass, state_weights_frac, val_obs_seqs, A_prev, A_new):
        """Log all per-iteration metrics to Weights & Biases."""
        if wandb_run is None:
            return

        import wandb

        iter_time = time.perf_counter() - iter_start

        # Emission stats
        means, covs = self.emissions.to_arrays()
        obs_dim = self.emissions.obs_dim
        means_2d = means.reshape(self.num_states, obs_dim)
        covs_2d = covs.reshape(self.num_states, obs_dim, obs_dim)
        cov_traces = np.trace(covs_2d, axis1=1, axis2=2)

        cov_logdets = []
        for z in range(self.num_states):
            s = z // self.A
            a = z % self.A
            sign, logdet = np.linalg.slogdet(self.emissions.params[s, a].cov)
            cov_logdets.append(logdet if sign > 0 else np.nan)

        mean_norms = np.linalg.norm(means_2d, axis=1)

        # π diagnostics
        pi = self.pi_z
        pi_entropy = float(-np.sum(pi * np.log(pi + 1e-15)))
        pi_max = float(pi.max())
        pi_min = float(pi.min())

        metrics = {
            "em_iter": it + 1,
            "time/iter_seconds": iter_time, # Iteration index and wall-clock time per EM iteration.
            "train/loglik": total_train_loglik, # Sum of log-likelihood over all training trajectories for that EM iteration.
            "train/delta_pi": float(delta_pi), # L1 change in initial state distribution
            "train/delta_A": float(delta_A), # Mean absolute difference between old and new transition matrices
            "train/log_delta_A": float(np.log10(delta_A + 1e-15)),
            "emissions/total_responsibility_mass": total_responsibility_mass, # Sum of γ over all trajectories, time steps, and states; ie. total number of time steps across all sequences
            "emissions/cov_logdet": cov_logdets,
            "pi/entropy": pi_entropy,
            "pi/max": pi_max,
            "pi/min": pi_min
        }

        if val_obs_seqs is not None:
            metrics["val/loglik"] = total_val_loglik # Sum of log-likelihood on the validation set
            metrics["val/improvement"] = improvement # val_loglik_now - val_loglik_prev
        else:
            metrics["val/loglik"] = np.nan
            metrics["train/improvement"] = improvement

        try:

            # π bar plot
            fig_pi, ax_pi = plt.subplots()
            ax_pi.bar(np.arange(self.num_states), pi)
            ax_pi.set_title("π_z distribution")
            ax_pi.set_xlabel("joint state z")
            ax_pi.set_ylabel("probability")
            metrics["pi/plot"] = wandb.Image(fig_pi) # Bar plot: π_z as a function of state index z.
            plt.close(fig_pi)

            # A_zz heatmap
            fig_A, ax_A = plt.subplots()
            im = ax_A.imshow(self.A_zz, aspect="auto")
            ax_A.set_title("Transition matrix A_zz")
            ax_A.set_xlabel("z'")
            ax_A.set_ylabel("z")
            fig_A.colorbar(im, ax=ax_A)
            metrics["A/heatmap"] = wandb.Image(fig_A) # Image of the transition matrix A_zz, row = current state, column = next state.
            plt.close(fig_A)

            # A_zz diagonal (stay probabilities)
            diag_A = np.diag(self.A_zz)
            fig_diag, ax_diag = plt.subplots()
            ax_diag.plot(np.arange(self.num_states), diag_A, marker="o")
            ax_diag.set_title("A_zz diagonal: P(Z_{t+1}=z | Z_t=z)")
            ax_diag.set_xlabel("joint state z")
            ax_diag.set_ylabel("stay probability")
            metrics["A/diag_plot"] = wandb.Image(fig_diag)
            plt.close(fig_diag)

            # ΔA heatmap (change in transition matrix)
            if A_prev is not None and A_new is not None:
                A_diff = A_new - A_prev
                if np.any(A_diff != 0.0):
                    fig_dA, ax_dA = plt.subplots()
                    vmax = np.max(np.abs(A_diff))
                    if vmax == 0:
                        vmax = 1.0
                    im_dA = ax_dA.imshow(A_diff, aspect="auto", vmin=-vmax, vmax=vmax)
                    ax_dA.set_title("ΔA_zz = A_new - A_prev")
                    ax_dA.set_xlabel("z'")
                    ax_dA.set_ylabel("z")
                    fig_dA.colorbar(im_dA, ax=ax_dA)
                    metrics["A/delta_heatmap"] = wandb.Image(fig_dA)
                    plt.close(fig_dA)

            # responsibility mass per state
            fig_w, ax_w = plt.subplots()
            ax_w.bar(np.arange(self.num_states), state_weights_flat)
            ax_w.set_title("State responsibility mass")
            ax_w.set_xlabel("joint state z")
            ax_w.set_ylabel("total γ mass")
            metrics["emissions/state_responsibility_mass_plot"] = wandb.Image(fig_w)
            plt.close(fig_w)

            # responsibility fraction per state 
            fig_wf, ax_wf = plt.subplots()
            ax_wf.bar(np.arange(self.num_states), state_weights_frac)
            ax_wf.set_title("State responsibility fraction")
            ax_wf.set_xlabel("joint state z")
            ax_wf.set_ylabel("fraction of total γ")
            metrics["emissions/state_responsibility_frac_plot"] = wandb.Image(fig_wf)
            plt.close(fig_wf)

            # mean norms per state 
            fig_mn, ax_mn = plt.subplots()
            ax_mn.plot(np.arange(self.num_states), mean_norms, marker="o")
            ax_mn.set_title("Emission mean norms ||μ_z||")
            ax_mn.set_xlabel("joint state z")
            ax_mn.set_ylabel("L2 norm")
            metrics["emissions/mean_norms_plot"] = wandb.Image(fig_mn)
            plt.close(fig_mn)

            # covariance trace per state 
            fig_ct, ax_ct = plt.subplots()
            ax_ct.plot(np.arange(self.num_states), cov_traces, marker="o")
            ax_ct.set_title("Covariance trace per state Tr(Σ_z)")
            ax_ct.set_xlabel("joint state z")
            ax_ct.set_ylabel("trace")
            metrics["emissions/cov_trace_plot"] = wandb.Image(fig_ct)
            plt.close(fig_ct)
        except Exception:
            pass

        wandb_run.log(metrics)

    # ------------------------------------------------------------------
    # k-means init, save, load 
    # ------------------------------------------------------------------
    def _init_emissions_kmeans(self, train_obs_seqs):
        """
        Initialise GaussianEmissionModel parameters k-means-style clustering, using a random subset of the training data for speed. 
        We treat each joint latent state z = (style, action) as a cluster.

        Parameters
        train_obs_seqs : list of np.ndarray
            Training observation sequences, each of shape (T_n, obs_dim).
        """
        max_samples = TRAINING_CONFIG.max_kmeans_samples
        seed = TRAINING_CONFIG.seed
        print("[HDVTrainer] Initialising emissions with (subsampled) k-means...")

        # Stack all time steps from all trajectories
        X_all = np.vstack(train_obs_seqs)  # shape (N_total, obs_dim)
        N_total, obs_dim = X_all.shape
        K = self.num_states

        # Choose a random subset for clustering
        rng = np.random.default_rng(seed)
        if N_total > max_samples:
            idx = rng.choice(N_total, size=max_samples, replace=False)
            X = X_all[idx]
            print(f"  Using subsample of {max_samples} out of {N_total} points "
                  f"for k-means initialisation.")
        else:
            X = X_all
            print(f"  Using all {N_total} points for k-means initialisation.")

        # Global covariance as a fallback if a cluster has too few points
        global_cov = np.cov(X, rowvar=False) + 1e-6 * np.eye(obs_dim)
        
        if np.any(np.isnan(global_cov)):
            print("  WARNING: Global covariance contains NaN. Using identity matrix.")
            global_cov = np.eye(obs_dim)

        # ---- Fast clustering ----
        mbk = MiniBatchKMeans(n_clusters=K, batch_size=2048, max_iter=100, n_init=5, random_state=seed)
        labels = mbk.fit_predict(X)        # labels for subsample
        centers = mbk.cluster_centers_     # shape (K, obs_dim)

        # For each cluster (joint state index z)
        for z in range(K):
            mask = (labels == z)
            num_points = mask.sum()

            if num_points < obs_dim + 1:
                # Too few points to estimate a full covariance reliably. so use global covariance and k-means center
                mean_z = centers[z]
                cov_z = global_cov.copy()
                print(f"  Cluster z={z:02d}: only {num_points} points, "
                      f"using global covariance as fallback.")
            else:
                X_z = X[mask]
                mean_z = X_z.mean(axis=0)
                cov_z = np.cov(X_z, rowvar=False) + 1e-6 * np.eye(obs_dim)

            s = z // self.A
            a = z % self.A

            self.emissions.params[s, a] = GaussianEmissionParams(mean=mean_z, cov=cov_z)

            if TRAINING_CONFIG.verbose >= 2:
                print(f"  Init mean for z={z:02d} (s={s}, a={a}): {mean_z}")
                sign, logdet = np.linalg.slogdet(cov_z)
                print(f"  Cov logdet for z={z:02d}: sign={sign}, logdet={logdet:.3e} (points={num_points})")

        print("[HDVTrainer] k-means initialisation done.")

    def save(self, path):
        """
        Save the learned joint transition parameters, Gaussian emissions, and feature scaler to a .npz file.

        Parameters
        path : str or Path
            Target file path (e.g. 'models/dbn_highd.npz').
        """
        path = str(path)
        means, covs = self.emissions.to_arrays()
        np.savez_compressed(path, pi_z=self.pi_z, A_zz=self.A_zz, means=means, covs=covs, scaler_mean=self.scaler_mean, scaler_std=self.scaler_std)
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

        # Try to load scaler if present
        if "scaler_mean" in data.files:
            trainer.scaler_mean = data["scaler_mean"]
        else:
            trainer.scaler_mean = None

        if "scaler_std" in data.files:
            trainer.scaler_std = data["scaler_std"]
        else:
            trainer.scaler_std = None

        print(f"[HDVTrainer] Loaded model parameters from {path}")
        return trainer    
