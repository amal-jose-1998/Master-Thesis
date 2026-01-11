"""EM training loop for the HDV DBN with Mixed emissions (Gaussian + discrete)."""

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
def build_structured_transition_params(hdv_dbn):
    """
    Extract structured transition parameters from the pgmpy DBN CPDs.
    We represent:
        pi_s0[s]              = P(s_0 = s)
        pi_a0_given_s0[s,a]  = P(a_0 = a | s_0 = s)
        A_s[s_prev, s]       = P(s_t = s | s_{t-1} = s_prev)
        A_a[s, a_prev, a]    = P(a_t = a | a_{t-1} = a_prev, s_t = s)

    Parameters
    hdv_dbn : HDVDBN
        DBN model with CPDs for Style and Action at time 0 and 1.

    Returns
    pi_s0 : np.ndarray, shape (S,)
    pi_a0_given_s0 : np.ndarray, shape (S, A)
    A_s : np.ndarray, shape (S, S)
    A_a : np.ndarray, shape (S, A, A)
        Interpreted as A_a[s_cur, a_prev, a_cur].
    """
    S = int(hdv_dbn.num_style)
    A = int(hdv_dbn.num_action)

    cpd_style0 = hdv_dbn.model.get_cpds(("Style", 0))
    cpd_action0 = hdv_dbn.model.get_cpds(("Action", 0))
    cpd_style1 = hdv_dbn.model.get_cpds(("Style", 1))
    cpd_action1 = hdv_dbn.model.get_cpds(("Action", 1))

    # -----------------------------
    # pi_s0
    # -----------------------------
    pi_s0 = np.asarray(cpd_style0.values, dtype=float).reshape(S)  # (S,)
    pi_s0_sum = float(pi_s0.sum())
    if not np.isfinite(pi_s0_sum) or pi_s0_sum <= 0.0:
        pi_s0 = np.full((S,), 1.0 / S, dtype=float)
    else:
        pi_s0 = pi_s0 / pi_s0_sum

    # -----------------------------
    # pi_a0_given_s0
    # cpd_action0.values is typically (A, S) = P(a0 | s0) with rows=a, cols=s
    # We transpose to (S, A).
    # -----------------------------
    P_a0_given_s0 = np.asarray(cpd_action0.values, dtype=float).reshape(A, S).T  # (S,A)
    # normalize per S
    row = P_a0_given_s0.sum(axis=1, keepdims=True)
    row[row <= 0.0] = 1.0
    pi_a0_given_s0 = P_a0_given_s0 / row

    # -----------------------------
    # A_s
    # cpd_style1.values is (S_new, S_old). Convert to (S_old, S_new) and normalize rows.
    # -----------------------------
    P_snew_given_sold = np.asarray(cpd_style1.values, dtype=float).reshape(S, S)  # rows=new, cols=old
    A_s = P_snew_given_sold.T  # (s_old, s_new) => (s_prev, s)
    row = A_s.sum(axis=1, keepdims=True)
    row[row <= 0.0] = 1.0
    A_s = A_s / row

    # -----------------------------
    # A_a
    # CPD: P(Action_1 | Action_0, Style_1)
    #
    # We reshape to vals[a_cur, a_prev, s_cur] and then fill:
    #   A_a[s_cur, a_prev, a_cur] = vals[a_cur, a_prev, s_cur]
    # (then normalize over a_cur per (s_cur, a_prev)).
    # -----------------------------
    vals = np.asarray(cpd_action1.values, dtype=float)
    # pgmpy may provide (A_cur, A_prev*S) or already multi-dim; handle both robustly
    if vals.ndim == 2:
        # (A_cur, A_prev*S) -> (A_cur, A_prev, S)
        vals = vals.reshape(A, A, S)
    elif vals.ndim == 3:
        # expected (A_cur, A_prev, S)
        if vals.shape != (A, A, S):
            vals = vals.reshape(A, A, S)
    else:
        # very unexpected; last-resort reshape
        vals = vals.reshape(A, A, S)

    A_a = np.zeros((S, A, A), dtype=float)  # (s_cur, a_prev, a_cur)
    for s_cur in range(S):
        for a_prev in range(A):
            probs = vals[:, a_prev, s_cur]  # (A_cur,)
            psum = float(probs.sum())
            if not np.isfinite(psum) or psum <= 0.0:
                A_a[s_cur, a_prev, :] = 1.0 / A
            else:
                A_a[s_cur, a_prev, :] = probs / psum

    return pi_s0, pi_a0_given_s0, A_s, A_a

# =============================================================================
# Torch forward-backward 
# =============================================================================
def forward_backward_torch(pi_s0, pi_a0_given_s0, A_s, A_a, logB_s_a):
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
        Log-likelihood of the observation sequence. Scalar.
        loglik = log p(O_0:T-1).
    """
    T, S, A = logB_s_a.shape
    device = logB_s_a.device
    dtype = logB_s_a.dtype

    if T == 0:
        gamma = torch.empty((0, S, A), device=device, dtype=dtype)
        xi_s_sum = torch.zeros((S, S), device=device, dtype=dtype)
        xi_a_sum = torch.zeros((S, A, A), device=device, dtype=dtype)
        loglik = torch.tensor(0.0, device=device, dtype=dtype)
        return gamma, xi_s_sum, xi_a_sum, loglik

    log_pi_s0 = torch.log(pi_s0 + EPSILON)                      # (S,)
    log_pi_a0_given_s0 = torch.log(pi_a0_given_s0 + EPSILON)    # (S,A)
    logAs = torch.log(A_s + EPSILON)                            # (S_prev,S_cur)
    logAa = torch.log(A_a + EPSILON)                            # (S_cur,A_prev,A_cur)

    # Helpful pre-permute:
    # For forward: need logAa indexed as (S_cur, A_prev, A_cur) already OK.
    # For xi vectorization: need (A_prev, S_cur, A_cur)
    logAa_ap_s_a = logAa.permute(1, 0, 2)                       # (A_prev, S_cur, A_cur)
    # For forward style addition: use logAs^T so we can broadcast (S_cur,S_prev,...)
    logAs_T = logAs.transpose(0, 1)                             # (S_cur,S_prev)

    # ------------------------------------------------------------------
    # Forward pass 
    # ------------------------------------------------------------------
    # alpha[t,s,a] normalized by c[t]
    alpha = torch.empty((T, S, A), device=device, dtype=dtype)
    c = torch.empty((T,), device=device, dtype=dtype)

    # t=0
    alpha0 = log_pi_s0[:, None] + log_pi_a0_given_s0 + logB_s_a[0] # (S,A)
    c[0] = torch.logsumexp(alpha0.reshape(-1), dim=0)
    alpha[0] = alpha0 - c[0]

    # forward recursion
    for t in range(1, T):
        prev = alpha[t - 1]  # (S_prev,A_prev)

        # tmp = prev + logAa for all s_cur:
        # prev[None, S_prev, A_prev, 1] + logAa[S_cur, 1, A_prev, A_cur]
        tmp = prev[None, :, :, None] + logAa[:, None, :, :]     # (S_cur, S_prev, A_prev, A_cur)
        m = torch.logsumexp(tmp, dim=2)     # sum over a_prev -> (S_cur, S_prev, A_cur)

        tmp2 = m + logAs_T[:, :, None]                # add style -> (S_cur, S_prev, A_cur)
        next_alpha = torch.logsumexp(tmp2, dim=1)     # sum over s_prev -> (S_cur, A_cur)

        a = next_alpha + logB_s_a[t]                  # (S,A)
        c[t] = torch.logsumexp(a.reshape(-1), dim=0)
        alpha[t] = a - c[t]

    loglik = c.sum()

    # ------------------------------------------------------------------
    # Backward pass 
    # ------------------------------------------------------------------
    beta = torch.zeros((T, S, A), device=device, dtype=dtype)
    beta[T - 1] = 0.0

    for t in range(T - 2, -1, -1):
        nb = logB_s_a[t + 1] + beta[t + 1]         # (S_cur, A_cur)

        # h[s_cur, a_prev] = logsumexp_{a_cur}( logAa[s_cur,a_prev,a_cur] + nb[s_cur,a_cur] )
        h = torch.logsumexp(logAa + nb[:, None, :], dim=2)           # (S_cur, A_prev)

        # beta[t, s_prev, a_prev] = logsumexp_{s_cur}( logAs[s_prev,s_cur] + h[s_cur,a_prev] ) - c[t+1]
        # Build (A_prev, S_prev, S_cur): logAs[None,S_prev,S_cur] + h.T[:,None,S_cur]
        tmp = logAs[None, :, :] + h.transpose(0, 1)[:, None, :]      # (A_prev, S_prev, S_cur)
        bt = torch.logsumexp(tmp, dim=2).transpose(0, 1)             # (S_prev, A_prev)

        beta[t] = bt - c[t + 1]

    # ------------------------------------------------------------------
    # Gamma
    # ------------------------------------------------------------------
    log_gamma = alpha + beta
    log_gamma = log_gamma - torch.logsumexp(log_gamma.reshape(T, -1), dim=1).view(T, 1, 1)
    gamma = torch.exp(log_gamma)                                     # (T,S,A)

    # ------------------------------------------------------------------
    # Xi sums
    # ------------------------------------------------------------------
    xi_s_sum = torch.zeros((S, S), device=device, dtype=dtype)
    xi_a_sum = torch.zeros((S, A, A), device=device, dtype=dtype)

    if T > 1:
        for t in range(T - 1):
            nb = logB_s_a[t + 1] + beta[t + 1]        # (S_cur, A_cur)

            # log_xi: (S_prev, A_prev, S_cur, A_cur)
            log_xi = (
                alpha[t][:, :, None, None]                           # (S, A, 1, 1)
                + logAs[:, None, :, None]                            # (S, 1, S, 1)
                + logAa_ap_s_a[None, :, :, :]                        # (1, A, S, A)
                + nb[None, None, :, :]                               # (1, 1, S, A)
            )

            # normalize
            Z = torch.logsumexp(log_xi.reshape(-1), dim=0)
            xi = torch.exp(log_xi - Z)                               # (S,A,S,A)

            # accumulate structured counts
            xi_s_sum += xi.sum(dim=(1, 3))          # sum over a_prev,a_cur -> (s_prev,s_cur)
            xi_a_sum += xi.sum(dim=0).permute(1, 0, 2)  # sum over s_prev -> (A_prev,s_cur,A_cur) => permute -> (s_cur,A_prev,A_cur)

    return gamma, xi_s_sum, xi_a_sum, loglik

# =============================================================================
# Trainer
# =============================================================================
class HDVTrainer:
    """
    Trainer for the structured DBN with factorized transitions and Mixed emissions.
    Latent state at time t is a pair:
        z_t = (s_t, a_t)
    where:
        s_t ∈ {0..S-1}  (driving style / regime)
        a_t ∈ {0..A-1}  (maneuver / action)

    Transition factorization:
        p(s_t | s_{t-1}) = A_s[s_prev, s]
        p(a_t | a_{t-1}, s_t) = A_a[s, a_prev, a]

    Initial distribution:
        p(s_0) = pi_s0[s]
        p(a_0 | s_0) = pi_a0_given_s0[s, a]

    Emissions (MixedEmissionModel), conditionally independent given z_t:
        - Diagonal Gaussian over continuous features
        - Independent Bernoulli over binary features
        - Categorical over lane_pos (K=5) and lc (K=3)

    EM procedure:
    - E-step: structured forward–backward per trajectory to compute
            gamma_t(s,a) = p(s_t=s, a_t=a | O)
            xi sums:
            xi_s_sum[s_prev, s]  = Σ_t p(s_t=s_prev, s_{t+1}=s | O)
            xi_a_sum[s, a_prev, a] = Σ_t p(s_{t+1}=s, a_t=a_prev, a_{t+1}=a | O)
        (A joint xi over z is also optionally accumulated for diagnostics only.)

    - M-step:
            pi_s0, pi_a0_given_s0, A_s, A_a updated from the expected counts above
            emission parameters updated from gamma (responsibilities)
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

        pi_s0_np, pi_a0_given_s0_np, As_np, Aa_np = build_structured_transition_params(self.hdv_dbn)


        self.device = torch.device(getattr(TRAINING_CONFIG, "device", "cpu"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            if getattr(TRAINING_CONFIG, "verbose", 1):
                print("[HDVTrainer] WARNING: CUDA requested but not available. Falling back to CPU.")
            self.device = torch.device("cpu")
        dtype_str = getattr(TRAINING_CONFIG, "dtype", "float32")
        self.dtype = torch.float32 if dtype_str == "float32" else torch.float64

        self.pi_s0 = torch.as_tensor(pi_s0_np, device=self.device, dtype=self.dtype)            # (S,)
        self.pi_a0_given_s0 = torch.as_tensor(pi_a0_given_s0_np, device=self.device, dtype=self.dtype)  # (S,A)
        self.A_s = torch.as_tensor(As_np, device=self.device, dtype=self.dtype)               # (S,S)
        self.A_a = torch.as_tensor(Aa_np, device=self.device, dtype=self.dtype)               # (S,A,A)

        # move emissions to GPU + build caches
        self.emissions.to_device(device=self.device, dtype=self.dtype)

        self.scaler_mean = None  # shape (obs_dim,)
        self.scaler_std = None   # shape (obs_dim,)

        self.verbose = int(getattr(TRAINING_CONFIG, "verbose", 1))

    # ------------------------------------------------------------------
    # EM training loop
    # ------------------------------------------------------------------
    def em_train(self, train_obs_seqs, val_obs_seqs=None, wandb_run=None, train_obs_seqs_raw=None, val_obs_seqs_raw=None,):
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
        train_obs_seqs_raw : list[np.ndarray] | None
            Raw trajectories. Each sequence has shape (T_n, obs_dim).
        val_obs_seqs_raw : list[np.ndarray] | None
            Raw trajectories. 
        
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
            gamma_all, xi_s_all, xi_a_all, train_ll, obs_used, obs_used_raw = self._e_step(
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
            run_lengths_train, runlen_median_per_traj = self._run_lengths_from_gamma_argmax(gamma_all)
            ent_all_train, ent_mean_per_traj = self._posterior_entropy_from_gamma(gamma_all)
            # Semantics (scaled-space) for relative comparisons
            sem_feat_names, sem_means, sem_stds = self._posterior_weighted_key_feature_stats(obs_used, gamma_all)
            # Semantics (raw-space) for physical interpretation (meters, m/s, etc.)
            sem_means_raw, sem_stds_raw = None, None
            if obs_used_raw is not None:
                _, sem_means_raw, sem_stds_raw = self._posterior_weighted_key_feature_stats(obs_used_raw, gamma_all)

            # ----------------------
            # M-step: update pi_s0, pi_a0|s0, A_s, A_a
            # ----------------------
            if self.verbose:
                print("M-step: updating pi_s0, pi_a0|s0, A_s, A_a...")
            delta_pi, delta_A, A_prev, A_new = self._m_step_transitions(
                                                    gamma_all=gamma_all,
                                                    xi_s_all=xi_s_all,
                                                    xi_a_all=xi_a_all,
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

    def _e_step(self, obs_seqs, use_progress, verbose, it, obs_seqs_raw=None):
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
        obs_seqs_raw : list[np.ndarray]
            List of raw sequences, each of shape (T_n, obs_dim).

        Returns
        gamma_all : list[torch.Tensor]
            Each element shape (T_n, N) where N=S*A (flattened joint), on self.device. This is kept for emission updates.
        xi_s_all : list[torch.Tensor]
            Each element shape (S, S), expected style transition counts.
        xi_a_all : list[torch.Tensor]
            Each element shape (S, A, A), expected action transition counts conditioned on next style.
        total_loglik : float
            Sum of log-likelihoods across all sequences (Python float).
        obs_used : list[np.ndarray]
            Sequences that were actually used (non-empty, finite ll).
        obs_used_raw : list[np.ndarray] | None
            Raw sequences aligned with obs_used, if provided.
        """
        if obs_seqs_raw is not None and len(obs_seqs_raw) != len(obs_seqs):
            raise ValueError("obs_seqs_raw must align with obs_seqs (same length/order).")

        gamma_all = []
        xi_s_all = []
        xi_a_all = []
        obs_used = []
        obs_used_raw = [] if (obs_seqs_raw is not None) else None
        total_loglik = 0.0

        iterator = enumerate(obs_seqs)
        if use_progress:
            iterator = tqdm(iterator, total=len(obs_seqs), desc=f"E-step train (iter {it+1})", leave=False,)

        with torch.no_grad():
            for i, obs in iterator:
        
                if obs is None or obs.shape[0] == 0:
                    continue

                # emissions are joint-indexed: (T, N). Reshape to (T, S, A) for structured FB.
                logB_flat = self._compute_logB_for_sequence(obs)  # (T, N); Compute emission log-likelihoods logB[t,z] = log p(o_t | z)
                T = int(logB_flat.shape[0])
                logB_s_a = logB_flat.view(T, int(self.S), int(self.A))  # (T,S,A)
                gamma_s_a, xi_s_sum, xi_a_sum, loglik = forward_backward_torch(
                                                                        self.pi_s0, self.pi_a0_given_s0, self.A_s, self.A_a, logB_s_a
                                                                    )
                
                if torch.isnan(loglik):
                    if verbose >= 2:
                        print(f"  Seq {i:03d}: loglik is NaN, skipping.")
                    continue

                gamma_flat = gamma_s_a.reshape(T, int(self.num_states))  # (T,N)

                gamma_all.append(gamma_flat)
                xi_s_all.append(xi_s_sum)
                xi_a_all.append(xi_a_sum)      

                obs_used.append(obs)
                if obs_used_raw is not None:
                    obs_used_raw.append(obs_seqs_raw[i])

                ll_i = float(loglik.detach().cpu().item())
                total_loglik += ll_i

                if verbose >= 2:
                    print(f"  Seq {i:03d}: T={obs.shape[0]}, loglik={ll_i:.3f}")

        if verbose:
            print(f"  Total train loglik: {total_loglik:.3f}")

        return gamma_all, xi_s_all, xi_a_all, total_loglik, obs_used, obs_used_raw

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
        Compute total log-likelihood over a dataset (train/val/test) without storing posteriors.
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
                logB_flat = self._compute_logB_for_sequence(obs)  # (T,N)
                T = int(logB_flat.shape[0])
                logB_s_a = logB_flat.view(T, int(self.S), int(self.A))
                _, _, _, ll = forward_backward_torch(
                                    self.pi_s0, self.pi_a0_given_s0, self.A_s, self.A_a, logB_s_a
                                )
                total_ll += float(ll.detach().cpu().item())
        return total_ll
    
    # ------------------------------------------------------------------
    # M-step helpers: π, A, emissions
    # ------------------------------------------------------------------
    def _m_step_transitions(self, gamma_all, xi_s_all, xi_a_all, verbose):
        """
        M-step for structured transitions.
        Learns:
            pi_s[s]                ∝ sum_seq gamma_seq[t=0, s, :]
            pi_a0_given_s0[s,a]    ∝ sum_seq gamma_seq[t=0, s, a]
            A_s[s_prev, s]         ∝ sum_seq xi_s_sum_seq[s_prev, s]
            A_a[s, a_prev, a]      ∝ sum_seq xi_a_sum_seq[s, a_prev, a]

        Parameters
        gamma_all : list[torch.Tensor]
            Each element shape (T_n, N=S*A), flattened.
        xi_s_all : list[torch.Tensor]
            Each element shape (S, S). style transition counts.
        xi_a_all : list[torch.Tensor]
            Each element shape (S, A, A). action transition counts conditioned on next style.
        verbose : int
            Verbosity level.

        Returns
        delta_pi : float
            L1 change in (pi_s, pi_a0_given_s0) concatenated.
        delta_A : float
            Mean absolute change across (A_s and A_a) entries.
        A_prev : dict[str, np.ndarray]
            Previous transitions (CPU) for diagnostics/plots.
        A_new : dict[str, np.ndarray]
            Updated transitions (CPU) for diagnostics/plots.
        """
        # snapshots for deltas/diagnostics
        prev = {
            "pi_s0": self.pi_s0.detach().cpu().numpy().copy(),
            "pi_a0_given_s0": self.pi_a0_given_s0.detach().cpu().numpy().copy(),
            "A_s": self.A_s.detach().cpu().numpy().copy(),
            "A_a": self.A_a.detach().cpu().numpy().copy(),
        }

        S = int(self.S)
        A = int(self.A)

        # update pi_s and pi_a0_given_s0 from gamma[t=0]
        pi_s0_new = torch.zeros((S,), device=self.device, dtype=self.dtype)
        pi_a0_given_s0_new = torch.zeros((S, A), device=self.device, dtype=self.dtype)
        for gamma_flat in gamma_all:
            if gamma_flat.shape[0] <= 0:
                continue
            g0 = gamma_flat[0].view(S, A)  # (S,A)
            pi_s0_new += g0.sum(dim=1)       # (S,)
            pi_a0_given_s0_new += g0                 # (S,A)
        
        # normalize
        pi_s0_new = pi_s0_new / (pi_s0_new.sum() + EPSILON)
        row = pi_a0_given_s0_new.sum(dim=1, keepdim=True)
        pi_a0_given_s0_new = pi_a0_given_s0_new / (row + EPSILON)

        # update A_s (MAP with sticky Dirichlet prior)
        As_counts = torch.zeros((S, S), device=self.device, dtype=self.dtype)
        for xi_s in xi_s_all:
            As_counts += xi_s
        alpha_s = float(getattr(TRAINING_CONFIG, "alpha_A_s", 0.1))
        kappa_s = float(getattr(TRAINING_CONFIG, "kappa_A_s", 25.0))  # sticky (slow)
        prior_s = torch.full((S, S), alpha_s, device=self.device, dtype=self.dtype)
        prior_s[torch.arange(S), torch.arange(S)] += kappa_s
        As_map = As_counts + prior_s
        As_new = As_map / (As_map.sum(dim=1, keepdim=True) + EPSILON)


        # update A_a (MAP with milder sticky Dirichlet prior)
        Aa_counts = torch.zeros((S, A, A), device=self.device, dtype=self.dtype)
        for xi_a in xi_a_all:
            Aa_counts += xi_a
        alpha_a = float(getattr(TRAINING_CONFIG, "alpha_A_a", 0.1))
        kappa_a = float(getattr(TRAINING_CONFIG, "kappa_A_a", 3.0))   # less sticky (fast)
        prior_a = torch.full((S, A, A), alpha_a, device=self.device, dtype=self.dtype)
        prior_a[:, torch.arange(A), torch.arange(A)] += kappa_a
        Aa_map = Aa_counts + prior_a
        Aa_new = Aa_map / (Aa_map.sum(dim=2, keepdim=True) + EPSILON)

        with torch.no_grad():
            rs = 1.0 - (torch.diag(As_counts).sum() / (As_counts.sum() + EPSILON))
            ra = 1.0 - (torch.diagonal(Aa_counts, dim1=1, dim2=2).sum() / (Aa_counts.sum() + EPSILON))
            if verbose:
                print(f"  switch-rate from xi: style={float(rs):.4f}, action={float(ra):.4f}")

        # write back
        self.pi_s0 = pi_s0_new
        self.pi_a0_given_s0 = pi_a0_given_s0_new
        self.A_s = As_new
        self.A_a = Aa_new

        # deltas
        pi_cat_prev = np.concatenate([prev["pi_s0"].ravel(), prev["pi_a0_given_s0"].ravel()], axis=0)
        pi_cat_new = np.concatenate(
            [self.pi_s0.detach().cpu().numpy().ravel(), self.pi_a0_given_s0.detach().cpu().numpy().ravel()],
            axis=0,
        )
        delta_pi = float(np.abs(pi_cat_new - pi_cat_prev).sum())

        dAs = np.abs(self.A_s.detach().cpu().numpy() - prev["A_s"]).mean()
        dAa = np.abs(self.A_a.detach().cpu().numpy() - prev["A_a"]).mean()
        delta_A = float(0.5 * (dAs + dAa))

        if verbose:
            print(f"  Δpi (pi_s0 + pi_a0|s0) L1: {delta_pi:.6e}")
            print(f"  ΔA_s mean abs: {float(dAs):.6e}")
            print(f"  ΔA_a mean abs: {float(dAa):.6e}")
            print(f"  ΔA (avg): {delta_A:.6e}")
        
        A_prev = {"A_s": prev["A_s"], "A_a": prev["A_a"]}
        A_new = {
            "A_s": self.A_s.detach().cpu().numpy().copy(),
            "A_a": self.A_a.detach().cpu().numpy().copy(),
        }

        return delta_pi, delta_A, A_prev, A_new
    
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

        if hasattr(self.emissions, "lc_idx") and self.emissions.lc_idx is not None:
            lc_raw = X_all[:, self.emissions.lc_idx]
            lc_int = np.rint(lc_raw).astype(int)

            # map {-1,0,1} -> {0,1,2}
            lc_mapped = lc_int + 1

            valid_lc = (lc_mapped >= 0) & (lc_mapped < self.emissions.lc_K)
            if np.any(valid_lc):
                lc_valid = lc_mapped[valid_lc]
                counts = np.bincount(lc_valid, minlength=self.emissions.lc_K).astype(np.float64)
            else:
                counts = np.ones((self.emissions.lc_K,), dtype=np.float64)

            alpha = float(getattr(TRAINING_CONFIG, "cat_alpha", 1.0))
            p_lc = (counts + alpha) / (counts.sum() + alpha * self.emissions.lc_K)
            self.emissions.lc_p = np.tile(p_lc[None, :], (K, 1))

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
        Save the learned structured transition parameters, Mixed emissions, and feature scaler to a .npz file.
        
        Parameters
        path : str or Path
            Target file path (e.g. 'models/dbn_highd.npz').
        """
        path = str(path)
        # -----------------------------
        # Transitions
        # -----------------------------
        # Validate structured transitions exist
        required = ("pi_s0", "pi_a0_given_s0", "A_s", "A_a")
        missing = [k for k in required if not hasattr(self, k) or getattr(self, k) is None]
        if missing:
            raise ValueError(f"Cannot save: missing structured transitions {missing}.")

        pi_s0_np = self.pi_s0.detach().cpu().numpy()
        pi_a0_given_s0_np = self.pi_a0_given_s0.detach().cpu().numpy()
        A_s_np = self.A_s.detach().cpu().numpy()
        A_a_np = self.A_a.detach().cpu().numpy()

        # basic shape checks
        S = int(self.S)
        A = int(self.A)
        if pi_s0_np.shape != (S,):
            raise ValueError(f"pi_S must have shape {(S,)}, got {pi_s0_np.shape}")
        if pi_a0_given_s0_np.shape != (S, A):
            raise ValueError(f"pi_a0_given_S0 must have shape {(S, A)}, got {pi_a0_given_s0_np.shape}")
        if A_s_np.shape != (S, S):
            raise ValueError(f"A_s must have shape {(S, S)}, got {A_s_np.shape}")
        if A_a_np.shape != (S, A, A):
            raise ValueError(f"A_a must have shape {(S, A, A)}, got {A_a_np.shape}")

        payload = {
            "S": np.array([S], dtype=np.int64),
            "A": np.array([A], dtype=np.int64),

            # structured transitions
            "pi_s0": pi_s0_np,
            "pi_a0_given_s0": pi_a0_given_s0_np,
            "A_s": A_s_np,
            "A_a": A_a_np,
        }

       # -----------------------------
        # Meta needed to reconstruct trainer/emissions
        # -----------------------------
        payload["obs_names"] = np.array(self.obs_names, dtype=object)
        lane_K = int(getattr(self.emissions, "lane_K", 0))
        payload["lane_num_categories"] = np.array([lane_K], dtype=np.int64)
        if int(payload["lane_num_categories"][0]) <= 0:
            raise ValueError("lane_num_categories is invalid (<=0). Emissions may not be initialized correctly.")
        lc_K = int(getattr(self.emissions, "lc_K", 0)) if hasattr(self.emissions, "lc_K") else 0
        payload["lc_num_categories"] = np.array([lc_K], dtype=np.int64)

        # -----------------------------
        # Emissions 
        # -----------------------------
        required = {
            "obs_names",
            "lane_K",
            "lc_K",
            "lc_p",
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
          - pi_s0, pi_a0_given_s0, A_s, A_a
          - Mixed emission parameters (Gaussian + Bernoulli + categorical lane_pos)
          - Feature scaling parameters (global or classwise)

        Parameters
        path : str or Path
            Path to the saved .npz file.

        Returns
        HDVTrainer
            Trainer instance with restored parameters.
        """
        path = str(path)
        data = np.load(path, allow_pickle=True)

        # ------------------------------------------------------------------
        # Reconstruct trainer (MixedEmissionModel requires obs_names & lane_K). 
        # ------------------------------------------------------------------
        if "obs_names" not in data.files or "lane_num_categories" not in data.files:  
            raise ValueError(
                "Saved model is missing 'obs_names' or 'lane_num_categories'. "
                "Re-save the model."
            )  

        obs_names = [str(x) for x in list(data["obs_names"])]  
        lane_K = int(np.asarray(data["lane_num_categories"]).ravel()[0])
        lc_K = int(np.asarray(data["lc_num_categories"]).ravel()[0]) if "lc_num_categories" in data.files else None

        trainer = cls(obs_names=obs_names)

        # ------------------------------------------------------------------
        # Restore transitions
        # ------------------------------------------------------------------

        if "pi_s0" in data.files and "A_s" in data.files and "A_a" in data.files and "pi_a0_given_s0" in data.files:
            trainer.pi_s0 = torch.as_tensor(data["pi_s0"], device=trainer.device, dtype=trainer.dtype)
            trainer.pi_a0_given_s0 = torch.as_tensor(data["pi_a0_given_s0"], device=trainer.device, dtype=trainer.dtype)
            trainer.A_s = torch.as_tensor(data["A_s"], device=trainer.device, dtype=trainer.dtype)
            trainer.A_a = torch.as_tensor(data["A_a"], device=trainer.device, dtype=trainer.dtype)

            # ensure num_states consistent 
            trainer.S = int(trainer.hdv_dbn.num_style)
            trainer.A = int(trainer.hdv_dbn.num_action)
            trainer.num_states = trainer.S * trainer.A

        else:
            raise ValueError(
                "Checkpoint is missing transitions. Expected pi_s0, pi_a0_given_s0, A_s and A_a"
            )

        # ------------------------------------------------------------------
        # Restore emissions (MixedEmissionModel)
        # ------------------------------------------------------------------
        em_payload = {}
        for k in data.files:
            if k.startswith("em__"):
                em_payload[k[len("em__"):]] = data[k]

        if not em_payload:
            raise ValueError("Checkpoint contains no emission payload (no keys starting with 'em__').")

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
        compute_speed_mag = (i_vx is not None and i_vy is not None)
        compute_acc_mag   = (i_ax is not None and i_ay is not None)

        # direct ego signals (only if present in obs_names)
        ego_direct = []
        for name in ["speed", "jerk_x", "vx", "vy", "ax", "ay"]:
            if idx(name) is not None:
                ego_direct.append(name)

        # Neighbors: ALL prefixes present in your baseline config
        neighbor_prefixes = [
            "front", "rear",
            "left_front", "left_side", "left_rear",
            "right_front", "right_side", "right_rear",
        ]

        neighbor_suffixes = ["dx", "dy", "dvx", "dvy", "thw", "ttc"]

         # Build list of feature specs:
        # each spec is (label, exists_idx_or_None, value_idx_or_None, special_type)
        # where special_type in {None, "ego_speed_mag", "ego_acc_mag", "direct"}
        feat_specs = []

        # Derived ego magnitudes
        if compute_speed_mag:
            feat_specs.append(("speed_mag", None, None, "ego_speed_mag"))
        if compute_acc_mag:
            feat_specs.append(("acc_mag", None, None, "ego_acc_mag"))

        # Direct ego channels
        for name in ego_direct:
            feat_specs.append((name, None, idx(name), "direct"))

        # Neighbor conditioned features
        for p in neighbor_prefixes:
            i_e = idx(f"{p}_exists")
            if i_e is None:
                continue  # can't condition without exists

            for suf in neighbor_suffixes:
                i_val = idx(f"{p}_{suf}")
                if i_val is None:
                    continue
                feat_specs.append((f"{p}_{suf} | {p}_exists=1", i_e, i_val, None))

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

            # Precompute derived ego vectors once per trajectory
            v_mag = None
            a_mag = None
            if compute_speed_mag:
                v_mag = np.sqrt(x[:, i_vx] ** 2 + x[:, i_vy] ** 2)
            if compute_acc_mag:
                a_mag = np.sqrt(x[:, i_ax] ** 2 + x[:, i_ay] ** 2)

            # Accumulate posterior-weighted moments
            for col, (label, i_e, i_val, special) in enumerate(feat_specs):
                mask = None
                val = None

                if special == "ego_speed_mag":
                    val = v_mag
                elif special == "ego_acc_mag":
                    val = a_mag
                elif special == "direct":
                    val = x[:, i_val]
                else:
                    # neighbor conditioned on exists==1
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