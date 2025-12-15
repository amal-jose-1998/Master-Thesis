"""
EM training loop for the HDV DBN with Gaussian emissions.

Key performance changes compared to the original NumPy version:
1) Remove Python loops over (t, z) when building logB:
      logB = emissions.loglik_all_states(obs)   # (T, N) in one GPU call
2) Run forward–backward in Torch using logsumexp (GPU-friendly).

Design choice:
- Transition structure (pi_z, A_zz) is built once from pgmpy CPDs on CPU, then
  moved to GPU for repeated inference during EM.
"""

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from tqdm.auto import tqdm
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .model import HDVDBN
from .emissions import GaussianEmissionModel, GaussianEmissionParams
from .config import TRAINING_CONFIG

# -----------------------------------------------------------------------------
# Numerical stability constants
# -----------------------------------------------------------------------------
EPSILON = 1e-6

# =============================================================================
# Transition matrix builder 
# =============================================================================
def build_joint_transition_matrix(hdv_dbn):
    """
    Build a joint HMM transition representation over the combined latent state:
        Z_t = (Style_t, Action_t)
    This collapses the DBN into an HMM over joint states z, enabling standard
    forward–backward inference.

    Joint initial distribution: pi_z(s, a) = P(Style_0=s) * P(Action_0=a | Style_0=s)

    Joint transition: A_zz[(s,a),(s',a')] = P(Style_{t+1}=s' | Style_t=s) *
                             P(Action_{t+1}=a' | Action_t=a, Style_t=s, Style_{t+1}=s')

    Parameters
    hdv_dbn : HDVDBN
        DBN model with CPDs for Style and Action at time 0 and 1.

    Returns
    pi_z : np.ndarray
        Shape (N,), N = S*A.
    A_zz : np.ndarray
        Shape (N, N), row-stochastic.
    """
    S = int(hdv_dbn.num_style)
    A = int(hdv_dbn.num_action)
    N = S * A

    # Extract CPDs
    cpd_style0 = hdv_dbn.model.get_cpds(("Style", 0))
    cpd_action0 = hdv_dbn.model.get_cpds(("Action", 0))
    cpd_style1 = hdv_dbn.model.get_cpds(("Style", 1))
    cpd_action1 = hdv_dbn.model.get_cpds(("Action", 1))

    # -------------------------
    # initial joint pi(z) = P(Style_0, Action_0) = P(Style_0) P(Action_0 | Style_0)
    # -------------------------
    P_style0 = np.asarray(cpd_style0.values, dtype=float).reshape(S)      # (S,)
    P_action0_given_style0 = np.asarray(cpd_action0.values, dtype=float) # (A, S)

    pi_sa = (P_action0_given_style0 * P_style0[None, :]).T   # (S, A)
    pi_z = pi_sa.reshape(N)                                   # (N,)
            
    # -------------------------
    # transition A_zz' = P(Z_{t+1} = z' | Z_t = z)
    # -------------------------
    # From CPDs:
    #   P(Style_{t+1} | Style_t)
    #   P(Action_{t+1} | Action_t, Style_t, Style_{t+1})
    P_style1_given_style0 = np.asarray(cpd_style1.values, dtype=float).reshape(S, S) # (S, S): rows=new, cols=old
    P_action1_given_action0_style0_style1 = np.asarray(cpd_action1.values, dtype=float).reshape(A, A * S * S) # (A, A*S*S)

    A_zz = np.zeros((N, N), dtype=float)
    for s in range(S):
        for a in range(A):
            z = s * A + a # index for current state
            for s_next in range(S):
                p_s = P_style1_given_style0[s_next, s]
                for a_next in range(A):
                    z_next = s_next * A + a_next # flat index for next state.
                    col = ((a * S) + s) * S + s_next # Column index in cpd_action1 corresponding to (Action_0 = a, Style_0 = s, Style_1 = s_next)
                    p_a = P_action1_given_action0_style0_style1[a_next, col]
                    A_zz[z, z_next] = p_s * p_a

    # normalize rows (each row must sum to 1.)  
    # A[z,z′]=P(Z_t+1​=z′∣Z_t​=z)
    row_sums = A_zz.sum(axis=1, keepdims=True)
    zero_rows = (row_sums == 0) # Avoid division by zero: if a row is all zeros, keep it uniform instead of NaN
    row_sums[zero_rows] = 1.0
    A_zz = A_zz / row_sums

    return pi_z, A_zz

# =============================================================================
# Torch forward-backward 
# =============================================================================
def forward_backward_torch(pi_z, A_zz, logB):
    """
    Forward–backward in log-domain using Torch.
    This computes (for one sequence):
      gamma[t,z] = P(Z_t=z | O_0:T-1)                  shape (T,N)
      xi_sum[z,z'] = sum_t P(Z_t=z, Z_{t+1}=z' | O)    shape (N,N)
      loglik = log p(O_0:T-1)                          scalar

    Parameters
    pi_z : torch.Tensor
        Initial distribution over joint states, shape (N,).
    A_zz : torch.Tensor
        Transition matrix, shape (N,N).
    logB : torch.Tensor
        Emission log-likelihoods, shape (T,N).

    Returns
    gamma : torch.Tensor
        Posterior marginals, shape (T,N).
    xi_sum : torch.Tensor
        Expected transition counts summed over time, shape (N,N).
    loglik : torch.Tensor
        Scalar log-likelihood of the sequence.
    """
    # N = number of latent joint states z = (Style, Action).
    # T = number of time steps in the observation sequence (trajectory).
    T, N = logB.shape
    device = logB.device
    dtype = logB.dtype

    log_pi = torch.log(pi_z + EPSILON)         # (N,)
    logA = torch.log(A_zz + EPSILON)           # (N,N)

    # ----- forward -----
    alpha = torch.empty((T, N), device=device, dtype=dtype)
    c = torch.empty((T,), device=device, dtype=dtype)

    alpha[0] = log_pi + logB[0]
    c[0] = torch.logsumexp(alpha[0], dim=0)
    alpha[0] = alpha[0] - c[0]

    for t in range(1, T):
        tmp = alpha[t - 1][:, None] + logA           # (N,N)
        alpha[t] = torch.logsumexp(tmp, dim=0) + logB[t]
        c[t] = torch.logsumexp(alpha[t], dim=0)
        alpha[t] = alpha[t] - c[t]

    loglik = c.sum()

    # ----- backward -----
    beta = torch.zeros((T, N), device=device, dtype=dtype)
    beta[T - 1] = 0.0

    for t in range(T - 2, -1, -1):
        tmp = logA + logB[t + 1][None, :] + beta[t + 1][None, :]
        beta[t] = torch.logsumexp(tmp, dim=1) - c[t + 1]

    # ----- gamma -----
    log_gamma = alpha + beta
    log_gamma = log_gamma - torch.logsumexp(log_gamma, dim=1, keepdim=True)
    gamma = torch.exp(log_gamma)

    # ----- xi_sum -----
    if T > 1:
        tmp = (
            alpha[:-1, :, None]          # (T-1, N, 1)
            + logA[None, :, :]           # (1,   N, N)
            + logB[1:, None, :]          # (T-1, 1, N)
            + beta[1:, None, :]          # (T-1, 1, N)
        )                                # -> (T-1, N, N)

        tmp = tmp - torch.logsumexp(tmp.reshape(T - 1, -1), dim=1).view(T - 1, 1, 1)
        xi_sum = torch.exp(tmp).sum(dim=0)  # (N, N)
    else:
        xi_sum = torch.zeros((N, N), device=device, dtype=dtype)

    return gamma, xi_sum, loglik

# =============================================================================
# Trainer
# =============================================================================
class HDVTrainer:
    """
    Trainer for the joint (Style, Action) HMM-equivalent model with Gaussian emissions.
    This class runs EM:
      - E-step: forward–backward per trajectory to compute gamma and xi_sum
      - M-step:
          * update pi_z and A_zz from expected counts
          * update Gaussian emissions using gamma responsibilities
    """

    def __init__(self, obs_dim):
        """
        Parameters
        obs_dim : int
            Observation dimensionality (e.g., 6 for [x, y, vx, vy, ax, ay]).
        """
        self.hdv_dbn = HDVDBN()
        self.emissions = GaussianEmissionModel(obs_dim=obs_dim)

        self.S = self.hdv_dbn.num_style
        self.A = self.hdv_dbn.num_action
        self.num_states = self.S * self.A

        # Build pi_z, A_zz once (CPU) then move to GPU
        pi_np, A_np = build_joint_transition_matrix(self.hdv_dbn)

        self.device = torch.device(getattr(TRAINING_CONFIG, "device", "cpu"))
        dtype_str = getattr(TRAINING_CONFIG, "dtype", "float32")
        self.dtype = torch.float32 if dtype_str == "float32" else torch.float64

        self.pi_z = torch.as_tensor(pi_np, device=self.device, dtype=self.dtype)
        self.A_zz = torch.as_tensor(A_np, device=self.device, dtype=self.dtype)

        # move emissions to GPU + build caches
        self.emissions.to_device(device=self.device, dtype=self.dtype)

        self.scaler_mean = None  # shape (obs_dim,)
        self.scaler_std = None   # shape (obs_dim,)

    # ------------------------------------------------------------------
    # EM training loop
    # ------------------------------------------------------------------
    def em_train(self, train_obs_seqs, val_obs_seqs=None, wandb_run=None):
        """
        Train the model using EM.

        Parameters
        train_obs_seqs : list[np.ndarray]
            Training trajectories. Each sequence has shape (T_n, obs_dim).
        val_obs_seqs : list[np.ndarray] | None
            Optional validation trajectories. If provided, validation log-likelihood
            is computed each EM iteration and used for early stopping.
        wandb_run : wandb.sdk.wandb_run.Run | None
            Optional Weights & Biases run object for logging.
        
        Returns
        history : dict
            Keys:
              - "train_loglik": list of total train log-likelihood per iteration
              - "val_loglik":   list of total val log-likelihood per iteration (empty if no val)
        """
        num_iters = TRAINING_CONFIG.em_num_iters
        tol = TRAINING_CONFIG.em_tol
        verbose = TRAINING_CONFIG.verbose
        use_progress = TRAINING_CONFIG.use_progress

        history = {"train_loglik": [], "val_loglik": []}

        # k-means initialisation of emission parameters
        self._init_emissions_kmeans(train_obs_seqs)
        self.emissions.to_device(device=self.device, dtype=self.dtype)
        
        prev_criterion = -np.inf # log-likelihood from previous iteration. Initialise as -inf because at iteration 0 we don’t have a meaningful previous value. 

        if verbose:
            print("\n==================== EM TRAINING START ====================\n")
            print(f"Device: {self.device} dtype={self.dtype}")
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
            gamma_all, xi_all, train_ll = self._e_step(
                obs_seqs=train_obs_seqs,
                use_progress=use_progress,
                verbose=verbose,
                it=it,
            )

            # ----------------------
            # M-step: update pi_z, A_zz
            # ----------------------
            if verbose:
                print("M-step: updating π_z and A_zz...")
            delta_pi, delta_A, A_prev, A_new = self._m_step_transitions(
                gamma_all=gamma_all,
                xi_all=xi_all,
                verbose=verbose,
            )
            
            # ----------------------
            # M-step: update emission parameters
            # ----------------------
            if verbose:
                print("  Updating emission parameters...")
            state_w, total_mass, state_frac = self._m_step_emissions(
                train_obs_seqs=train_obs_seqs,
                gamma_all=gamma_all,
                use_progress=use_progress,
                verbose=verbose,
            )
            
            # ----------------------
            # Compute validation log-likelihood (if available)
            # ----------------------
            if val_obs_seqs is None:
                val_ll = 0.0
                criterion = train_ll
                if verbose:
                    print("No validation set provided; using train LL as criterion.")
            else:
                if verbose:
                    print("Validation E-step:")
                val_ll = self._total_loglik_on_dataset(
                    obs_seqs=val_obs_seqs,
                    use_progress=use_progress,
                    desc=f"E-step val (iter {it+1})",
                )
                if verbose:
                    print(f"  Total val loglik: {val_ll:.3f}")
                criterion = val_ll

            # ----------------------
            # Bookkeeping and Early stopping
            # ----------------------
            improvement = criterion - prev_criterion
            history["train_loglik"].append(train_ll)
            if val_obs_seqs is not None:
                history["val_loglik"].append(val_ll)

            if verbose:
                print(f"  Criterion: {criterion:.3f}")
                print(f"  Improvement: {improvement:.6f}")

            if it > 0 and improvement < tol:
                if verbose:
                    print("\n*** Early stopping triggered ***")
                break
            prev_criterion = criterion
            
            # ----------------------
            # WandB logging
            # ----------------------
            self._log_wandb_iteration(
                wandb_run=wandb_run,
                it=it,
                iter_start=iter_start,
                total_train_loglik=train_ll,                           
                total_val_loglik=val_ll,                               
                improvement=improvement,
                delta_pi=delta_pi,
                delta_A=delta_A,
                state_weights_flat=state_w,                            
                total_responsibility_mass=total_mass,                  
                state_weights_frac=state_frac,                         
                val_obs_seqs=val_obs_seqs,
                A_prev=A_prev,
                A_new=A_new,
            )
            
        if verbose:
            print("\n===================== EM TRAINING END =====================")
        return history

    # ------------------------------------------------------------------
    # E-step helpers
    # ------------------------------------------------------------------
    def _compute_logB_for_sequence(self, obs):
        """
        Compute emission log-likelihoods for one trajectory:
            logB[t, z] = log p(o_t | Z_t=z)

        Parameters
        obs : np.ndarray
            Observation sequence, shape (T, obs_dim).

        Returns
        logB : torch.Tensor
            Emission log-likelihoods, shape (T, num_states), on self.device.
        """
        return self.emissions.loglik_all_states(obs)

    def _e_step(self, obs_seqs, use_progress, verbose, it):
        """
        Run forward–backward on all sequences (training set) and collect posteriors.
        
        Parameters
        obs_seqs : list[np.ndarray]
            List of sequences, each of shape (T_n, obs_dim).
        use_progress : bool
            Whether to show a tqdm progress bar.
        verbose : int
            Verbosity level.
        it : int
            EM iteration index (0-based), only used for tqdm labels.

        Returns
        gamma_all : list[torch.Tensor]
            Each element has shape (T_n, N) on GPU.
        xi_all : list[torch.Tensor]
            Each element has shape (N, N) on GPU.
        total_loglik : float
            Sum of log-likelihoods across all sequences (Python float).
        """
        gamma_all = []   
        xi_all = []       
        total_loglik = 0.0

        iterator = enumerate(obs_seqs)
        if use_progress:
            iterator = tqdm(
                iterator,
                total=len(obs_seqs),
                desc=f"E-step train (iter {it+1})",
                leave=False,
            )

        with torch.no_grad():
            for i, obs in iterator:
                logB = self._compute_logB_for_sequence(obs) # Compute emission log-likelihoods logB[t,z] = log p(o_t | z)
                gamma, xi_sum, loglik = forward_backward_torch(self.pi_z, self.A_zz, logB)

                if torch.isnan(loglik):
                    if verbose >= 2:
                        print(f"  Seq {i:03d}: loglik is NaN, skipping.")
                    continue

                gamma_all.append(gamma)     
                xi_all.append(xi_sum)       

                ll_i = float(loglik.detach().cpu().item())
                total_loglik += ll_i

                if verbose >= 2:
                    print(f"  Seq {i:03d}: T={obs.shape[0]}, loglik={ll_i:.3f}")

        if verbose:
            print(f"  Total train loglik: {total_loglik:.3f}")

        return gamma_all, xi_all, total_loglik

    def _total_loglik_on_dataset(self, obs_seqs, use_progress, desc):
        """
        Compute total log-likelihood over a dataset (train/val/test) without storing gamma/xi.
        This is used for validation scoring and avoids keeping unnecessary tensors in memory.

        Parameters
        obs_seqs : list[np.ndarray]
            Dataset sequences.
        use_progress : bool
            Whether to show tqdm.
        desc : str
            tqdm label.

        Returns
        total_ll : float
            Sum of per-sequence log-likelihoods.
        """
        total_ll = 0.0
        iterator = enumerate(obs_seqs)
        if use_progress:
            iterator = tqdm(iterator, total=len(obs_seqs), desc=desc, leave=False)

        with torch.no_grad():
            for _, obs in iterator:
                logB = self._compute_logB_for_sequence(obs)
                _, _, ll = forward_backward_torch(self.pi_z, self.A_zz, logB)  
                total_ll += float(ll.detach().cpu().item())
        return total_ll
    
    # ------------------------------------------------------------------
    # M-step helpers: π_z, A_zz, emissions
    # ------------------------------------------------------------------
    def _m_step_transitions(self, gamma_all, xi_all, verbose):
        """
        Update pi_z and A_zz from posterior expectations.

        Parameters
        gamma_all : list[torch.Tensor]
            Posterior marginals per sequence, each shape (T_n, N).
        xi_all : list[torch.Tensor]
            Expected transition counts per sequence, each shape (N, N), typically on CPU.
        verbose : int
            Verbosity level.

        Returns
        delta_pi : float
            L1 change in initial distribution (sum absolute difference).
        delta_A : float
            Mean absolute change in A_zz.
        A_prev : np.ndarray
            Previous transition matrix on CPU (for diagnostics/plots).
        A_new : np.ndarray
            Updated transition matrix on CPU (for diagnostics/plots).
        """
        pi_prev = self.pi_z.detach().cpu().numpy().copy()
        A_prev = self.A_zz.detach().cpu().numpy().copy()

        # pi_z update
        pi_new = torch.zeros_like(self.pi_z)
        for gamma in gamma_all:
            pi_new += gamma[0]
        pi_new = pi_new / (pi_new.sum() + EPSILON)

        # A_zz update
        A_new = torch.zeros_like(self.A_zz)
        for xi_sum in xi_all:
            A_new += xi_sum
        row_sums = A_new.sum(dim=1, keepdim=True)
        A_new = A_new / (row_sums + EPSILON)

        self.pi_z = pi_new
        self.A_zz = A_new

        delta_pi = float(np.abs(self.pi_z.detach().cpu().numpy() - pi_prev).sum())
        delta_A = float(np.abs(self.A_zz.detach().cpu().numpy() - A_prev).mean())

        if verbose:
            print(f"  Δπ_z (sum abs diff): {delta_pi:.6e}")
            print(f"  ΔA_zz (mean abs diff): {delta_A:.6e}")

        return delta_pi, delta_A, A_prev, self.A_zz.detach().cpu().numpy().copy()
    
    def _m_step_emissions(self, train_obs_seqs, gamma_all, use_progress, verbose):
        """
        Update Gaussian emission parameters from responsibilities.

        Parameters
        train_obs_seqs : list[np.ndarray]
            Training sequences.
        gamma_all : list[torch.Tensor]
            Posterior marginals per sequence, each shape (T_n, N).
        use_progress : bool
            Show progress bars inside the emission update.
        verbose : int
            Verbosity level.

        Returns
        state_weights_flat : np.ndarray
            Total responsibility mass per joint state, shape (N,).
        total_mass : float
            Sum over all state weights (should equal total number of time steps across sequences).
        state_frac : np.ndarray
            Normalized responsibility mass per state, shape (N,).
        """
        state_weights_flat = self.emissions.update_from_posteriors(obs_seqs=train_obs_seqs, gamma_seqs=gamma_all, use_progress=use_progress, verbose=verbose)
        total_mass = float(state_weights_flat.sum())
        state_frac = (state_weights_flat / total_mass) if total_mass > 0.0 else np.zeros_like(state_weights_flat)
        return state_weights_flat, total_mass, state_frac

    # ------------------------------------------------------------------
    # WandB logging helper
    # ------------------------------------------------------------------
    def _log_wandb_iteration(self, wandb_run, it, iter_start, total_train_loglik, total_val_loglik, improvement, delta_pi,
                             delta_A, state_weights_flat, total_responsibility_mass, state_weights_frac, val_obs_seqs, A_prev, A_new):
        """
        Log per-iteration metrics to Weights & Biases.

        Parameters
        wandb_run : wandb.sdk.wandb_run.Run | None
            WandB run object.
        it : int
            EM iteration index (0-based).
        iter_start : float
            Start time (perf_counter) of this EM iteration.
        total_train_loglik : float
            Total train log-likelihood for this iteration.
        total_val_loglik : float
            Total validation log-likelihood (np.nan if no validation).
        improvement : float
            Criterion improvement vs previous iteration.
        delta_pi, delta_A : float
            Transition parameter deltas.
        state_weights_flat : np.ndarray
            Responsibility mass per state.
        total_responsibility_mass : float
            Sum of responsibilities (diagnostic).
        state_weights_frac : np.ndarray
            Responsibility fractions per state.
        val_obs_seqs : list[np.ndarray] | None
            Validation set (used only to decide what to log).
        A_prev, A_new : np.ndarray | None
            Previous and updated A matrices for delta plotting.
        """
        if wandb_run is None:
            return

        import wandb

        iter_time = time.perf_counter() - iter_start

         # convert tensors to numpy for logging + plotting
        pi_np = self.pi_z.detach().cpu().numpy()
        A_np = self.A_zz.detach().cpu().numpy()  

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
        pi_entropy = float(-np.sum(pi_np * np.log(pi_np + 1e-15)))
        pi_max = float(pi_np.max())
        pi_min = float(pi_np.min())

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
            ax_pi.bar(np.arange(self.num_states), pi_np)
            ax_pi.set_title("π_z distribution")
            ax_pi.set_xlabel("joint state z")
            ax_pi.set_ylabel("probability")
            metrics["pi/plot"] = wandb.Image(fig_pi) # Bar plot: π_z as a function of state index z.
            plt.close(fig_pi)

            # A_zz heatmap
            fig_A, ax_A = plt.subplots()
            im = ax_A.imshow(A_np, aspect="auto")
            ax_A.set_title("Transition matrix A_zz")
            ax_A.set_xlabel("z'")
            ax_A.set_ylabel("z")
            fig_A.colorbar(im, ax=ax_A)
            metrics["A/heatmap"] = wandb.Image(fig_A) # Image of the transition matrix A_zz, row = current state, column = next state.
            plt.close(fig_A)

            # A_zz diagonal (stay probabilities)
            diag_A = np.diag(A_np)
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
        Initialise GaussianEmissionModel parameters using MiniBatchKMeans on a subsample.

        Parameters
        train_obs_seqs : list[np.ndarray]
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
        pi_np = self.pi_z.detach().cpu().numpy()
        A_np = self.A_zz.detach().cpu().numpy()
        means, covs = self.emissions.to_arrays()
        np.savez_compressed(path, pi_z=pi_np, A_zz=A_np, means=means, covs=covs, scaler_mean=self.scaler_mean, scaler_std=self.scaler_std)
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
        trainer.pi_z = torch.as_tensor(pi_z, device=trainer.device, dtype=trainer.dtype)  
        trainer.A_zz = torch.as_tensor(A_zz, device=trainer.device, dtype=trainer.dtype)  

        trainer.emissions.from_arrays(means, covs)
        trainer.emissions.to_device(device=trainer.device, dtype=trainer.dtype)  

        trainer.scaler_mean = data["scaler_mean"] if "scaler_mean" in data.files else None
        trainer.scaler_std = data["scaler_std"] if "scaler_std" in data.files else None

        print(f"[HDVTrainer] Loaded model parameters from {path}")
        return trainer  
