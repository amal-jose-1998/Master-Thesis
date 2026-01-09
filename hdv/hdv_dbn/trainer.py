"""
EM training loop for the HDV DBN with Mixed emissions (Gaussian + discrete).

Design choice:
- Transition structure (pi_z, A_zz) is initialised once from pgmpy CPDs on CPU, then
  updated by EM and stored on GPU for repeated inference.
"""

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from tqdm.auto import tqdm
import time

from .model import HDVDBN
from .emissions import MixedEmissionModel, GaussianParams
from .config import TRAINING_CONFIG
from .utils.wandb_logger import WandbLogger

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
                             P(Action_{t+1}=a' | Action_t=a, Style_{t+1}=s')

    Parameters
    hdv_dbn : HDVDBN
        DBN model with CPDs for Style and Action at time 0 and 1.

    Returns
    pi_z : np.ndarray
        Joint initial distribution. Shape (N,), N = S*A.
    A_zz : np.ndarray
        Joint transition matrix. Shape (N, N).
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

    pi_sum = float(pi_z.sum())
    if not np.isfinite(pi_sum) or pi_sum <= 0.0:
        pi_z = np.full((N,), 1.0 / N, dtype=float)
    else:
        pi_z = pi_z / pi_sum
            
    # -------------------------
    # transition A_zz' = P(Z_{t+1} = z' | Z_t = z)
    # -------------------------
    # From CPDs:
    #   P(Style_{t+1} | Style_t)
    #   P(Action_{t+1} | Action_t, Style_t, Style_{t+1})
    P_style1_given_style0 = np.asarray(cpd_style1.values, dtype=float).reshape(S, S) # (S, S): rows=new, cols=old
    P_action1_given_action0_style1 = np.asarray(cpd_action1.values, dtype=float).reshape(A, A * S) # (A, A*S)

    A_zz = np.zeros((N, N), dtype=float)
    for s in range(S):
        for a in range(A):
            z = s * A + a # index for current state
            for s_next in range(S):
                p_s = P_style1_given_style0[s_next, s]
                for a_next in range(A):
                    z_next = s_next * A + a_next # flat index for next state.
                    col = a * S + s_next  # Column for (Action_0=a, Style_1=s_next)
                    p_a = P_action1_given_action0_style1[a_next, col] 
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
    if T == 0:  
        gamma = torch.empty((0, N), device=logB.device, dtype=logB.dtype)  
        xi_sum = torch.zeros((N, N), device=logB.device, dtype=logB.dtype)  
        loglik = torch.tensor(0.0, device=logB.device, dtype=logB.dtype) 
        return gamma, xi_sum, loglik  
    
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
    Trainer for the joint (Style, Action) HMM-equivalent model with Mixed emissions.
    This class runs EM:
      - E-step: forward–backward per trajectory to compute gamma and xi_sum
      - M-step:
          * update pi_z and A_zz from expected counts
          * update Mixed emissions (Gaussian + Bernoulli + categorical lane_pos) using gamma
    """

    def __init__(self, obs_names):
        """
        Parameters
        obs_names : list[str]
            Names of observation features (for MixedEmissionModel).
        """
        self.hdv_dbn = HDVDBN()
        self.obs_names = list(obs_names)
        self.emissions = MixedEmissionModel(obs_names=self.obs_names)

        self.S = self.hdv_dbn.num_style
        self.A = self.hdv_dbn.num_action
        self.num_states = self.S * self.A

        # Build pi_z, A_zz once (CPU) then move to GPU
        pi_np, A_np = build_joint_transition_matrix(self.hdv_dbn)

        self.device = torch.device(getattr(TRAINING_CONFIG, "device", "cpu"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            if getattr(TRAINING_CONFIG, "verbose", 1):
                print("[HDVTrainer] WARNING: CUDA requested but not available. Falling back to CPU.")
            self.device = torch.device("cpu")
        dtype_str = getattr(TRAINING_CONFIG, "dtype", "float32")
        self.dtype = torch.float32 if dtype_str == "float32" else torch.float64

        self.pi_z = torch.as_tensor(pi_np, device=self.device, dtype=self.dtype)
        self.A_zz = torch.as_tensor(A_np, device=self.device, dtype=self.dtype)

        # move emissions to GPU + build caches
        self.emissions.to_device(device=self.device, dtype=self.dtype)

        self.scaler_mean = None  # shape (obs_dim,)
        self.scaler_std = None   # shape (obs_dim,)

        self.verbose = int(getattr(TRAINING_CONFIG, "verbose", 1))

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
        use_progress = TRAINING_CONFIG.use_progress
        patience = getattr(TRAINING_CONFIG, "early_stop_patience", 3)
        min_delta = getattr(TRAINING_CONFIG, "early_stop_min_delta_per_obs", 1e-4)
        delta_A_thresh = getattr(TRAINING_CONFIG, "early_stop_delta_A_thresh", 1e-5)
        bad_epochs = 0

        # Precompute number of observations (total timesteps)
        train_num_obs = int(sum(seq.shape[0] for seq in train_obs_seqs))
        if self.verbose:
            print(f"Train sequences: {len(train_obs_seqs)}  |  Train total timesteps: {train_num_obs}")
        val_num_obs = None
        if val_obs_seqs is not None:
            val_num_obs = int(sum(seq.shape[0] for seq in val_obs_seqs))
            if self.verbose:
                print(f"Validation sequences: {len(val_obs_seqs)}  |  Val total timesteps: {val_num_obs}")

        history = {"train_loglik": [], "val_loglik": []}

        # Initialisation of emission parameters
        self._init_emissions(train_obs_seqs)
        self.emissions.to_device(device=self.device, dtype=self.dtype)
        
        prev_criterion = None 

        if self.verbose:
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
            if self.verbose:
                print(f"\n--------------- EM ITERATION {it+1} ----------------")

            # ----------------------
            # E-step on training data
            # ----------------------
            if self.verbose:
                print("E-step (train):")
            gamma_all, xi_joint_all, xi_S_all, xi_a_all, train_ll, obs_used, obs_used_raw = self._e_step(
                                                                                            obs_seqs=train_obs_seqs,
                                                                                            use_progress=use_progress,
                                                                                            verbose=self.verbose,
                                                                                            it=it,
                                                                                            obs_seqs_raw=train_obs_seqs_raw,
                                                                                        )
            train_ll_per_obs = train_ll / max(train_num_obs, 1)

            # -----------------------------------------------------------
            # Trajectory-level diagnostics (from posteriors)
            # -----------------------------------------------------------
            switch_rates_train = self._expected_switch_rates_per_traj(gamma_all, xi_all)
            run_lengths_train, runlen_median_per_traj = self._run_lengths_from_gamma_argmax(gamma_all)
            ent_all_train, ent_mean_per_traj = self._posterior_entropy_from_gamma(gamma_all)
            # Semantics (scaled-space) for relative comparisons
            sem_feat_names, sem_means, sem_stds = self._posterior_weighted_key_feature_stats(obs_used, gamma_all)
            # Semantics (raw-space) for physical interpretation (meters, m/s, etc.)
            sem_means_raw, sem_stds_raw = None, None
            if obs_used_raw is not None:
                _, sem_means_raw, sem_stds_raw = self._posterior_weighted_key_feature_stats(obs_used_raw, gamma_all)

            # ----------------------
            # M-step: update pi_z, A_zz
            # ----------------------
            if self.verbose:
                print("M-step: updating π_z and A_zz...")
            delta_pi, delta_A, A_prev, A_new = self._m_step_transitions(
                                                    gamma_all=gamma_all,
                                                    xi_all=xi_all,
                                                    verbose=self.verbose,
                                                )
            
            # ----------------------
            # M-step: update emission parameters
            # ----------------------
            if self.verbose:
                print("M-step: Updating emission parameters...")
            state_w, total_mass, state_frac = self._m_step_emissions(
                                                    train_obs_seqs=train_obs_seqs,
                                                    gamma_all=gamma_all,
                                                    use_progress=use_progress,
                                                    verbose=self.verbose,
                                                )
            
            # ----------------------
            # Compute validation log-likelihood (if available)
            # ----------------------
            criterion_for_stop = None
            if val_obs_seqs is None:
                val_ll = 0.0
                criterion_for_stop = train_ll_per_obs
                if self.verbose:
                    print("No validation set provided; using train per-obs LL as criterion.")
            else:
                if self.verbose:
                    print("Validation E-step:")
                val_ll = self._total_loglik_on_dataset(
                            obs_seqs=val_obs_seqs,
                            use_progress=use_progress,
                            desc=f"E-step val (iter {it+1})",
                        )
                if self.verbose:
                    print(f"  Total val loglik: {val_ll:.3f}")
                # Scale-invariant criterion: average loglik per timestep
                criterion_for_stop = val_ll / max(val_num_obs, 1)
            
            # ----------------------
            # Bookkeeping and Early stopping
            # ----------------------
            history["train_loglik"].append(train_ll)
            if val_obs_seqs is not None:
                history["val_loglik"].append(val_ll)
            
            if not np.isfinite(criterion_for_stop):
                if self.verbose:
                    print("WARNING: Non-finite stopping criterion; skipping early stopping check this iteration.")
                WandbLogger.log_iteration(
                    trainer=self,
                    wandb_run=wandb_run,
                    it=it,
                    iter_start=iter_start,
                    total_train_loglik=train_ll,
                    total_val_loglik=val_ll,
                    improvement=np.nan,
                    criterion_for_stop=criterion_for_stop,
                    val_num_obs=val_num_obs,
                    train_num_obs=train_num_obs,
                    delta_pi=delta_pi,
                    delta_A=delta_A,
                    state_weights_flat=state_w,
                    total_responsibility_mass=total_mass,
                    state_weights_frac=state_frac,
                    val_obs_seqs=val_obs_seqs,
                    A_prev=A_prev,
                    A_new=A_new,
                    switch_rates_train=switch_rates_train,
                    run_lengths_train=run_lengths_train,
                    runlen_median_per_traj=runlen_median_per_traj,
                    ent_all_train=ent_all_train, 
                    ent_mean_per_traj=ent_mean_per_traj,
                    sem_feat_names=sem_feat_names, sem_means=sem_means, sem_stds=sem_stds,
                    sem_means_raw=sem_means_raw, sem_stds_raw=sem_stds_raw,
                )
                continue

            if prev_criterion is None:
                improvement = np.nan
            else:
                improvement = criterion_for_stop - prev_criterion

            if self.verbose:
                if val_obs_seqs is not None:
                    print(f"  Criterion (val per-obs): {criterion_for_stop:.6f}")
                else:
                    print(f"  Criterion (train per-obs): {criterion_for_stop:.6f}")
                print(f"  Improvement: {improvement:.6e}")

            if prev_criterion is not None:
                if improvement < min_delta:
                    bad_epochs += 1
                else:
                    bad_epochs = 0

                if bad_epochs >= patience and delta_A < delta_A_thresh:
                    if self.verbose:
                        print("\n*** Early stopping triggered (plateau + stable transitions) ***")
                    break
            prev_criterion = criterion_for_stop
            
            # ----------------------
            # WandB logging
            # ----------------------
            WandbLogger.log_iteration(
                trainer=self,
                wandb_run=wandb_run,
                it=it,
                iter_start=iter_start,
                total_train_loglik=train_ll,
                total_val_loglik=val_ll,
                improvement=improvement,
                criterion_for_stop=criterion_for_stop,
                val_num_obs=val_num_obs,
                train_num_obs=train_num_obs,
                delta_pi=delta_pi,
                delta_A=delta_A,
                state_weights_flat=state_w,
                total_responsibility_mass=total_mass,
                state_weights_frac=state_frac,
                val_obs_seqs=val_obs_seqs,
                A_prev=A_prev,
                A_new=A_new,
                switch_rates_train=switch_rates_train,
                run_lengths_train=run_lengths_train,
                runlen_median_per_traj=runlen_median_per_traj,
                ent_all_train=ent_all_train, 
                ent_mean_per_traj=ent_mean_per_traj,
                sem_feat_names=sem_feat_names, sem_means=sem_means, sem_stds=sem_stds,
                sem_means_raw=sem_means_raw, sem_stds_raw=sem_stds_raw,
            )

        if self.verbose:
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
        obs_used 
        """
        gamma_all = []   
        xi_all = []       
        obs_used = []
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
                if obs is None or obs.shape[0] == 0:
                    continue
                logB = self._compute_logB_for_sequence(obs) # Compute emission log-likelihoods logB[t,z] = log p(o_t | z)
                gamma, xi_sum, loglik = forward_backward_torch(self.pi_z, self.A_zz, logB)

                if torch.isnan(loglik):
                    if verbose >= 2:
                        print(f"  Seq {i:03d}: loglik is NaN, skipping.")
                    continue

                gamma_all.append(gamma)     
                xi_all.append(xi_sum)       
                obs_used.append(obs)

                ll_i = float(loglik.detach().cpu().item())
                total_loglik += ll_i

                if verbose >= 2:
                    print(f"  Seq {i:03d}: T={obs.shape[0]}, loglik={ll_i:.3f}")

        if verbose:
            print(f"  Total train loglik: {total_loglik:.3f}")

        return gamma_all, xi_all, total_loglik, obs_used
    
    def _expected_switch_rates_per_traj(self, gamma_all, xi_all):
        """
        Compute per-trajectory *expected* switch rates from posteriors.
        For a trajectory of length T, expected switch rate is:
            1 - (sum_t sum_k xi_t(k,k)) / (T-1)

        We only store xi summed over time (xi_sum), so:
            diag_mass = trace(xi_sum)
            expected_switch_rate = 1 - diag_mass/(T-1)
        """
        rates = []
        for gamma, xi_sum in zip(gamma_all, xi_all):
            T = int(gamma.shape[0])
            if T <= 1:
                rates.append(np.nan)
                continue
            diag_mass = float(torch.diagonal(xi_sum, 0).sum().detach().cpu().item())
            denom = float(T - 1)
            r = 1.0 - (diag_mass / max(denom, 1.0))
            if not np.isfinite(r):
                r = np.nan
            else:
                r = float(np.clip(r, 0.0, 1.0))
            rates.append(r)
        return np.asarray(rates, dtype=np.float64)

    def _run_lengths_from_gamma_argmax(self, gamma_all):
        """
        Compute run-lengths (segment durations) from hard labels:
            z_hat[t] = argmax_k gamma[t,k]

        Returns
        run_lengths : np.ndarray, shape (num_segments,)
            Lengths of contiguous segments across all trajectories.
        per_traj_median : np.ndarray, shape (num_traj,)
            Median run length within each trajectory (NaN if T==0).
        """
        all_runs = []
        traj_medians = []

        for gamma in gamma_all:
            T = int(gamma.shape[0])
            if T <= 0:
                traj_medians.append(np.nan)
                continue

            # hard path from marginals
            z_hat = torch.argmax(gamma, dim=1).detach().cpu().numpy().astype(np.int64)

            # compute contiguous run lengths
            runs = []
            cur = 1
            for t in range(1, T):
                if z_hat[t] == z_hat[t - 1]:
                    cur += 1
                else:
                    runs.append(cur)
                    cur = 1
            runs.append(cur)

            runs_np = np.asarray(runs, dtype=np.int64)
            all_runs.append(runs_np)
            traj_medians.append(float(np.median(runs_np)) if runs_np.size > 0 else np.nan)

        if len(all_runs) == 0:
            run_lengths = np.asarray([], dtype=np.int64)
        else:
            run_lengths = np.concatenate(all_runs, axis=0)

        return run_lengths.astype(np.int64), np.asarray(traj_medians, dtype=np.float64)
    
    def _posterior_entropy_from_gamma(self, gamma_all):
        """
        Compute normalized posterior entropy from gamma per timestep.

        Returns
        -------
        ent_all : np.ndarray, shape (sum_T,)
            Normalized entropies pooled over all timesteps in all trajectories. In [0,1].
        ent_mean_per_traj : np.ndarray, shape (num_traj,)
            Mean normalized entropy per trajectory (NaN if T==0).
        """
        K = int(self.num_states)
        logK = float(np.log(max(K, 2)))  # avoid divide-by-zero; K>=2 in practice

        ent_list = []
        ent_mean_traj = []

        for gamma in gamma_all:
            T = int(gamma.shape[0])
            if T <= 0:
                ent_mean_traj.append(np.nan)
                continue

            # gamma: (T,K). Ensure numerical stability.
            g = gamma.detach().cpu().numpy().astype(np.float64)
            g = np.clip(g, 1e-15, 1.0)
            g = g / g.sum(axis=1, keepdims=True)

            # H_t = -sum_k g*log(g)  -> (T,)
            H = -np.sum(g * np.log(g), axis=1)

            # normalized entropy in [0,1]
            Hn = H / logK
            ent_list.append(Hn)
            ent_mean_traj.append(float(np.mean(Hn)))

        ent_all = np.concatenate(ent_list, axis=0) if len(ent_list) else np.asarray([], dtype=np.float64)
        return ent_all.astype(np.float64), np.asarray(ent_mean_traj, dtype=np.float64)


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
            Expected transition counts per sequence, each shape (N, N).
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
        Update Mixed emission parameters from responsibilities.

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
            Sum over all state weights. Equals the total number of timesteps across all sequences that contributed posteriors (i.e., sequences not skipped).
        state_frac : np.ndarray
            Normalized responsibility mass per state, shape (N,).
        """
        state_weights_flat = self.emissions.update_from_posteriors(obs_seqs=train_obs_seqs, gamma_seqs=gamma_all, use_progress=use_progress, verbose=verbose)
        total_mass = float(state_weights_flat.sum())
        state_frac = (state_weights_flat / total_mass) if total_mass > 0.0 else np.zeros_like(state_weights_flat)
        return state_weights_flat, total_mass, state_frac

    # ------------------------------------------------------------------
    # emissions init, save, load 
    # ------------------------------------------------------------------
    def _init_emissions(self, train_obs_seqs):
        """
        Initialise MixedEmissionModel parameters using MiniBatchKMeans on continuous dims.
        - Gaussian part: KMeans on continuous dimensions only.
        - Bernoulli part: global mean of binary features.
        - Lane categorical: global empirical lane distribution with smoothing.

        Parameters
        train_obs_seqs : list[np.ndarray]
            Training observation sequences, each of shape (T_n, obs_dim).
        """
        max_samples = TRAINING_CONFIG.max_kmeans_samples
        seed = TRAINING_CONFIG.seed
        print("[HDVTrainer] Initialising emissions with (subsampled) k-means...")

        # Stack all time steps from all trajectories
        X_all = np.vstack(train_obs_seqs)  # (N_total, obs_dim)
        N_total = int(X_all.shape[0])
        K = int(self.num_states)

        # -----------------------------
        # Continuous part (Gaussian)
        # -----------------------------
        cont_idx = np.asarray(self.emissions.cont_idx, dtype=int)  
        if cont_idx.size == 0:
            raise RuntimeError("MixedEmissionModel has no continuous dimensions to initialise.")  

        X_all_cont = X_all[:, cont_idx]  # (N_total, cont_dim)
        cont_dim = int(X_all_cont.shape[1])

        # Choose a random subset for clustering
        rng = np.random.default_rng(seed)
        if N_total > max_samples:
            idx = rng.choice(N_total, size=max_samples, replace=False)
            Xc = X_all_cont[idx]
            print(f"  Using subsample of {max_samples} out of {N_total} points for k-means initialisation.")
        else:
            Xc = X_all_cont
            print(f"  Using all {N_total} points for k-means initialisation.")

        # global variance (diag) as fallback
        global_var = np.var(Xc, axis=0) + 1e-6  
        global_var = np.where(np.isfinite(global_var), global_var, 1.0) 
        global_var = np.maximum(global_var, 1e-6)  

        # ---- Fast clustering ----
        mbk = MiniBatchKMeans(
                n_clusters=K,
                batch_size=2048,
                max_iter=100,
                n_init=5,
                random_state=seed
            )
        labels = mbk.fit_predict(Xc)
        centers = mbk.cluster_centers_  # (K, cont_dim)

        # For each cluster (joint state index z)
        for z in range(K):
            mask = (labels == z)
            num_points = int(mask.sum())

            if num_points < cont_dim + 1:
                mean_z = centers[z]
                var_z = global_var.copy() 
                if TRAINING_CONFIG.verbose >= 1:
                    print(f"  Cluster z={z:02d}: only {num_points} points -> using global var fallback.")  
            else:
                X_z = Xc[mask]
                mean_z = X_z.mean(axis=0)
                var_z = np.var(X_z, axis=0) + 1e-6  # diagonal var

                # stability clamp
                var_z = np.where(np.isfinite(var_z), var_z, global_var)
                var_z = np.maximum(var_z, 1e-6)

            s, a = self.emissions.gauss._z_to_sa(z)
            self.emissions.gauss.params[s, a] = GaussianParams(mean=mean_z, var=var_z)  
        
        # -----------------------------
        # Discrete parts (Bernoulli + lane categorical)
        # -----------------------------
        if self.emissions.bin_dim > 0:  
            xb = X_all[:, self.emissions.bin_idx]
            xb = np.clip(xb, 0.0, 1.0)
            p0 = xb.mean(axis=0)
            p0 = np.clip(p0, 1e-3, 1.0 - 1e-3)
            self.emissions.bern_p = np.tile(p0[None, :], (K, 1))  
        # lane_pos init must ignore invalid lane_pos == -1 (or any out-of-range)
        lane_raw = X_all[:, self.emissions.lane_idx]
        lane_int = np.rint(lane_raw).astype(int)
        valid = (lane_int >= 0) & (lane_int < self.emissions.lane_K)
        if np.any(valid):
            lane_valid = lane_int[valid] 
            counts = np.bincount(lane_valid, minlength=self.emissions.lane_K).astype(np.float64)  
        else:
            # fallback to uniform if everything is invalid
            counts = np.ones((self.emissions.lane_K,), dtype=np.float64)  
        alpha = float(getattr(TRAINING_CONFIG, "cat_alpha", 1.0))
        p_lane = (counts + alpha) / (counts.sum() + alpha * self.emissions.lane_K)
        self.emissions.lane_p = np.tile(p_lane[None, :], (K, 1))

        self.emissions.invalidate_cache()
        self.emissions.to_device(device=self.device, dtype=self.dtype)

        print("[HDVTrainer] k-means initialisation done.")


    def save(self, path):
        """
        Save the learned joint transition parameters, Mixed emissions, and feature scaler to a .npz file.
        Saves:
          - Joint transitions: pi_z, A_zz
          - Emissions: MixedEmissionModel.to_arrays() (stored under em__*)
          - Scaler: global or classwise

        Parameters
        path : str or Path
            Target file path (e.g. 'models/dbn_highd.npz').
        """
        path = str(path)
        # -----------------------------
        # Transitions
        # -----------------------------
        if self.pi_z is None or self.A_zz is None:
            raise ValueError("Cannot save: transitions (pi_z/A_zz) are not initialized.")

        pi_np = self.pi_z.detach().cpu().numpy()
        A_np = self.A_zz.detach().cpu().numpy()

        if pi_np.ndim != 1:
            raise ValueError(f"pi_z must be 1D, got shape {pi_np.shape}")
        if A_np.ndim != 2 or A_np.shape[0] != A_np.shape[1]:
            raise ValueError(f"A_zz must be square 2D, got shape {A_np.shape}")
        if A_np.shape[0] != pi_np.shape[0]:
            raise ValueError(f"pi_z length {pi_np.shape[0]} does not match A_zz size {A_np.shape[0]}")

        payload = {
            "pi_z": pi_np,
            "A_zz": A_np,
        }

        # -----------------------------
        # Meta needed to reconstruct trainer
        # -----------------------------
        payload["obs_names"] = np.array(self.obs_names, dtype=object)
        payload["lane_num_categories"] = np.array([int(getattr(self.emissions, "lane_K", 0))], dtype=np.int64)
        if int(payload["lane_num_categories"][0]) <= 0:
            raise ValueError("lane_num_categories is invalid (<=0). Emissions may not be initialized correctly.")

        # -----------------------------
        # Emissions (strict keys)
        # -----------------------------
        required = {
            "obs_names",
            "lane_K",
            "cont_idx",
            "bin_idx",
            "lane_idx",
            "gauss_means",
            "gauss_vars",   
            "bern_p",
            "lane_p",
        }
        em_dict = self.emissions.to_arrays()  
        missing = [k for k in sorted(required) if k not in em_dict]
        if missing:
            raise ValueError(
                f"MixedEmissionModel.to_arrays() is missing keys: {missing}. "
                "Check emissions.py to_arrays()/from_arrays() implementation."
            )

        laneK_header = int(payload["lane_num_categories"][0])
        laneK_em = int(np.asarray(em_dict["lane_K"]).reshape(-1)[0])
        if laneK_header != laneK_em:
            raise ValueError(f"lane_K mismatch: header={laneK_header}, emissions={laneK_em}")

        for k, v in em_dict.items():      
            payload[f"em__{k}"] = v           

        # -----------------------------
        # Scaler (global vs classwise)
        # -----------------------------
        if isinstance(self.scaler_mean, dict) and isinstance(self.scaler_std, dict):
            classes = sorted(self.scaler_mean.keys())
            if set(classes) != set(self.scaler_std.keys()):
                raise ValueError("Classwise scaler keys mismatch between scaler_mean and scaler_std.")

            means_stack = np.stack([np.asarray(self.scaler_mean[c]) for c in classes], axis=0)
            stds_stack = np.stack([np.asarray(self.scaler_std[c]) for c in classes], axis=0)

            payload["scaler_mode"] = np.array(["classwise"], dtype=object)
            payload["scaler_classes"] = np.array(classes, dtype=object)
            payload["scaler_means"] = means_stack
            payload["scaler_stds"] = stds_stack
        else:
            payload["scaler_mode"] = np.array(["global"], dtype=object)
            payload["scaler_mean"] = np.asarray(self.scaler_mean) if self.scaler_mean is not None else np.array([])
            payload["scaler_std"] = np.asarray(self.scaler_std) if self.scaler_std is not None else np.array([])

        np.savez_compressed(path, **payload)
        print(f"[HDVTrainer] Saved model parameters to {path}")

    @classmethod
    def load(cls, path):
        """
        Load a trained HDVTrainer instance from a .npz file and restore Mixed emissions.
        Restores:
          - Joint transition parameters (pi_z, A_zz)
          - Mixed emission parameters (Gaussian + Bernoulli + categorical lane_pos)
          - Feature scaling parameters (global or classwise)

        Parameters
        path : str or Path
            Path to the saved .npz file.

        Returns
        HDVTrainer
            A trainer instance with pi_z, A_zz and emission parameters restored.
        """
        path = str(path)
        data = np.load(path, allow_pickle=True)

        # ------------------------------------------------------------------
        # Reconstruct trainer (MixedEmissionModel requires obs_names & lane_K). 
        # ------------------------------------------------------------------
        if "obs_names" not in data.files or "lane_num_categories" not in data.files:  
            raise ValueError(
                "Saved model is missing 'obs_names' or 'lane_num_categories'. "
                "Re-save the model with the updated save() implementation."
            )  

        obs_names = [str(x) for x in list(data["obs_names"])]  
        lane_K = int(np.asarray(data["lane_num_categories"]).ravel()[0])  

        trainer = cls(obs_names=obs_names, lane_num_categories=lane_K)  

        # ------------------------------------------------------------------
        # Restore transitions
        # ------------------------------------------------------------------
        if "pi_z" not in data.files or "A_zz" not in data.files:
            raise ValueError("Checkpoint is missing 'pi_z' or 'A_zz'.")

        trainer.pi_z = torch.as_tensor(data["pi_z"], device=trainer.device, dtype=trainer.dtype)
        trainer.A_zz = torch.as_tensor(data["A_zz"], device=trainer.device, dtype=trainer.dtype)

        # ------------------------------------------------------------------
        # Restore emissions (MixedEmissionModel)
        # ------------------------------------------------------------------
        em_payload = {}
        for k in data.files:
            if k.startswith("em__"):
                em_payload[k[len("em__"):]] = data[k]

        # Minimal required keys for the mixed model
        required_base = {"obs_names", "lane_K", "cont_idx", "bin_idx", "lane_idx", "gauss_means", "bern_p", "lane_p"}  
        missing_base = [k for k in sorted(required_base) if k not in em_payload]
        if missing_base:
            raise ValueError(
                f"Checkpoint is missing emission keys: {missing_base}. "
                "Re-save the model using the updated save() implementation."
            )

        # require at least one of vars/covs
        if ("gauss_vars" not in em_payload) and ("gauss_covs" not in em_payload):
            raise ValueError("Checkpoint must contain either 'gauss_vars' (new) or 'gauss_covs' (legacy).")  

        trainer.emissions.from_arrays(em_payload)
        trainer.emissions.to_device(device=trainer.device, dtype=trainer.dtype)

        # ------------------------------------------------------------------
        # Restore scaler
        # ------------------------------------------------------------------
        mode = str(np.asarray(data["scaler_mode"]).reshape(-1)[0]) if "scaler_mode" in data.files else "global"

        if mode == "classwise":
            for key in ("scaler_classes", "scaler_means", "scaler_stds"):
                if key not in data.files:
                    raise ValueError(f"Checkpoint marked classwise scaling but missing '{key}'.")

            classes = [str(c) for c in list(data["scaler_classes"])]
            means_stack = np.asarray(data["scaler_means"])
            stds_stack = np.asarray(data["scaler_stds"])
            if means_stack.shape[0] != len(classes) or stds_stack.shape[0] != len(classes):
                raise ValueError(
                    f"Scaler class count mismatch: classes={len(classes)}, "
                    f"means={means_stack.shape}, stds={stds_stack.shape}"
                )
            trainer.scaler_mean = {classes[i]: means_stack[i] for i in range(len(classes))}
            trainer.scaler_std = {classes[i]: stds_stack[i] for i in range(len(classes))}
        else:
            trainer.scaler_mean = data["scaler_mean"] if "scaler_mean" in data.files else None
            trainer.scaler_std = data["scaler_std"] if "scaler_std" in data.files else None

        print(f"[HDVTrainer] Loaded model parameters from {path}")
        return trainer
    
    def _posterior_weighted_key_feature_stats(self, obs_used, gamma_all):
        """
        Compute posterior-weighted mean ± std per state for a small set of derived key features (semantics).
        Features:
        - speed_mag = sqrt(vx^2 + vy^2)  (if vx,vy exist)
        - acc_mag   = sqrt(ax^2 + ay^2)  (if ax,ay exist)
        - For selected neighbor prefixes p:
                p_dx  conditioned on p_exists=1
                p_dvx conditioned on p_exists=1

        Returns
        feat_names : list[str]
        means : np.ndarray, shape (K, F)
        stds  : np.ndarray, shape (K, F)
        """
        K = int(self.num_states)

        # indices in obs vector
        name_to_idx = {n: i for i, n in enumerate(self.obs_names)}

        def idx(name: str):
            return name_to_idx.get(name, None)

        # ego indices
        i_vx, i_vy = idx("vx"), idx("vy")
        i_ax, i_ay = idx("ax"), idx("ay")

        compute_speed = (i_vx is not None and i_vy is not None)
        compute_acc   = (i_ax is not None and i_ay is not None)

        neighbor_prefixes = ["front", "left_front", "right_front"]#, "left_side", "right_side"]

        # Build list of (label, exists_idx, value_idx) entries we will compute
        # Each entry corresponds to one plotted feature column.
        feat_specs = []

        if compute_speed:
            feat_specs.append(("speed_mag", None, None))  
        if compute_acc:
            feat_specs.append(("acc_mag", None, None))    

        # Neighbor features: dx and dvx conditioned on exists=1
        for p in neighbor_prefixes:
            i_e = idx(f"{p}_exists")
            i_dx = idx(f"{p}_dx")
            i_dvx = idx(f"{p}_dvx")

            if i_e is None:
                continue  # cannot condition without exists

            if i_dx is not None:
                feat_specs.append((f"{p}_dx | {p}_exists=1", i_e, i_dx))
            if i_dvx is not None:
                feat_specs.append((f"{p}_dvx | {p}_exists=1", i_e, i_dvx))

        feat_names = [fs[0] for fs in feat_specs]
        F = len(feat_names)

        means = np.full((K, F), np.nan, dtype=np.float64)
        stds  = np.full((K, F), np.nan, dtype=np.float64)

        sum_w   = np.zeros((K, F), dtype=np.float64)
        sum_wx  = np.zeros((K, F), dtype=np.float64)
        sum_wx2 = np.zeros((K, F), dtype=np.float64)

        for obs, gamma in zip(obs_used, gamma_all):
            x = np.asarray(obs, dtype=np.float64)  # (T, D)
            g = gamma.detach().cpu().numpy().astype(np.float64)  # (T, K)

            # defensive normalization
            g = np.clip(g, 1e-15, 1.0)
            g = g / g.sum(axis=1, keepdims=True)

            # Precompute ego derived vectors once per trajectory (if needed)
            v = None
            a = None
            if compute_speed:
                v = np.sqrt(x[:, i_vx] ** 2 + x[:, i_vy] ** 2)  # (T,)
            if compute_acc:
                a = np.sqrt(x[:, i_ax] ** 2 + x[:, i_ay] ** 2)   # (T,)

            # Iterate columns and accumulate posterior-weighted moments
            for col, (label, i_e, i_val) in enumerate(feat_specs):
                if label == "speed_mag":
                    val = v
                    mask = None  # no conditioning
                elif label == "acc_mag":
                    val = a
                    mask = None
                else:
                    # conditioned neighbor feature
                    # mask: exists == 1
                    mask = (x[:, i_e] >= 0.5).astype(np.float64)
                    val = x[:, i_val]

                if val is None:
                    continue

                for k in range(K):
                    w = g[:, k]
                    if mask is not None:
                        w = w * mask
                    sw = w.sum()
                    if sw <= 0.0:
                        continue
                    sum_w[k, col]   += sw
                    sum_wx[k, col]  += (w * val).sum()
                    sum_wx2[k, col] += (w * val * val).sum()

        # finalize mean/std
        for k in range(K):
            for f in range(F):
                sw = sum_w[k, f]
                if sw > 1e-12:
                    mu = sum_wx[k, f] / sw
                    ex2 = sum_wx2[k, f] / sw
                    var = ex2 - mu * mu
                    if var < 0.0:
                        var = 0.0
                    means[k, f] = mu
                    stds[k, f] = np.sqrt(var)

        return feat_names, means, stds
