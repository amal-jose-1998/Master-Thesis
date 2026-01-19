"""EM training loop for the HDV DBN with Mixed emissions (Gaussian + Bernoulli)."""

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from tqdm.auto import tqdm
import time
from dataclasses import dataclass
from typing import Optional

from .model import HDVDBN
from .emissions import MixedEmissionModel, GaussianParams
from .config import TRAINING_CONFIG
from .utils.wandb_logger import WandbLogger
from .utils.trainer_diagnostics import _run_lengths_from_gamma_argmax, _posterior_entropy_from_gamma, _posterior_weighted_key_feature_stats
from .forward_backward import forward_backward_torch
from .utils.transitions import build_structured_transition_params

# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------
@dataclass
class EStepResult:
    gamma_all: list           # list[torch.Tensor] (T,N)
    xi_s_all: list            # list[torch.Tensor] (S,S)
    xi_a_all: list            # list[torch.Tensor] (S,A,A)
    total_loglik: float
    obs_used: list            # list[np.ndarray] (T,obs_dim)
    obs_used_raw: Optional[list]  # list[np.ndarray] | None (T,obs_dim)
    skipped_empty: int
    skipped_bad_logB: int
    skipped_bad_ll: int

@dataclass
class TransitionUpdate:
    delta_pi: float
    delta_A: float
    A_prev: dict
    A_new: dict

# -----------------------------------------------------------------------------
# Numerical stability constants
# -----------------------------------------------------------------------------
EPSILON = TRAINING_CONFIG.EPSILON if hasattr(TRAINING_CONFIG, "EPSILON") else 1e-6


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
        - Diagonal Gaussian over continuous window features (NaN-masked)
        - Independent Bernoulli over binary window features (e.g., lc_*_present)

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
        self.emissions = MixedEmissionModel(obs_names=self.obs_names, disable_discrete_obs=bool(getattr(TRAINING_CONFIG, "disable_discrete_obs", False)))
        self.S = self.hdv_dbn.num_style
        self.A = self.hdv_dbn.num_action
        self.num_states = self.S * self.A

        self.obs_index = {n: i for i, n in enumerate(self.obs_names)}
        self.lc_idx = [self.obs_index.get("lc_left_present"), self.obs_index.get("lc_right_present")] # indices of lane change indicators
        self.lc_idx = [i for i in self.lc_idx if i is not None] # filter out None

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
        self.lc_weight_invfreq = None  # to be initialized if needed

    # ------------------------------------------------------------------
    # EM training loop
    # ------------------------------------------------------------------
    def em_train(self, train_obs_seqs, val_obs_seqs=None, wandb_run=None, train_obs_seqs_raw=None):
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
            Used for computing physical-unit semantics in diagnostics.
        
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
        bad_epochs = 0 # counter for early stopping
        prev_criterion = None # for early stopping
        history = {"train_loglik": [], "val_loglik": []} # store loglik per iteration

        # Precompute number of observations (total timesteps)
        train_num_obs = int(sum(seq.shape[0] for seq in train_obs_seqs))
        if self.verbose:
            print(f"Train sequences: {len(train_obs_seqs)}  |  Train total timesteps: {train_num_obs}")
        val_num_obs = None
        if val_obs_seqs is not None:
            val_num_obs = int(sum(seq.shape[0] for seq in val_obs_seqs))
            if self.verbose:
                print(f"Validation sequences: {len(val_obs_seqs)}  |  Val total timesteps: {val_num_obs}")

        # Initialisation of emission parameters
        self._init_emissions(train_obs_seqs)

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

        if getattr(TRAINING_CONFIG, "lc_weight_mode", "none") != "none": # initialize lane-change weights only if needed
            self._init_lc_weight_invfreq(train_obs_seqs)

        for it in range(num_iters):
            iter_start = time.perf_counter()
            if self.verbose:
                print(f"\n--------------- EM ITERATION {it+1} ----------------")

            # ----------------------
            # E-step on training data
            # ----------------------
            if self.verbose:
                print("E-step (train):")
            e = self._e_step(
                    obs_seqs=train_obs_seqs,
                    use_progress=use_progress,
                    verbose=self.verbose,
                    it=it,
                    obs_seqs_raw=train_obs_seqs_raw,
                )
            
            gamma_all, xi_s_all, xi_a_all = e.gamma_all, e.xi_s_all, e.xi_a_all
            train_ll, obs_used, obs_used_raw = e.total_loglik, e.obs_used, e.obs_used_raw

            train_ll_per_obs = train_ll / max(train_num_obs, 1)

            # -----------------------------------------------------------
            # Trajectory-level diagnostics (from posteriors)
            # -----------------------------------------------------------
            run_lengths_train, runlen_median_per_traj = _run_lengths_from_gamma_argmax(gamma_all)
            ent_all_train, ent_mean_per_traj = _posterior_entropy_from_gamma(self.num_states, gamma_all) 
            # Semantics (scaled-space) for relative comparisons
            sem_feat_names, sem_means, sem_stds = _posterior_weighted_key_feature_stats(self.num_states, self.obs_names, obs_used, gamma_all)
            # Semantics (raw-space) for physical interpretation (meters, m/s, etc.)
            sem_means_raw, sem_stds_raw = None, None
            if obs_used_raw is not None:
                _, sem_means_raw, sem_stds_raw = _posterior_weighted_key_feature_stats(self.num_states, self.obs_names, obs_used_raw, gamma_all)

            # ----------------------
            # M-step: update pi_s0, pi_a0|s0, A_s, A_a
            # ----------------------
            if self.verbose:
                print("M-step: updating pi_s0, pi_a0|s0, A_s, A_a...")
            m = self._m_step_transitions(
                    gamma_all=gamma_all,
                    xi_s_all=xi_s_all,
                    xi_a_all=xi_a_all,
                    verbose=self.verbose,
                )
            delta_pi, delta_A, A_prev, A_new = m.delta_pi, m.delta_A, m.A_prev, m.A_new
            # ----------------------
            # M-step: update emission parameters
            # ----------------------
            if self.verbose:
                print("M-step: Updating emission parameters...")
            state_w, total_mass, state_frac = self._m_step_emissions(
                                                    train_obs_seqs=obs_used,
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
                    it=it+1,
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
                it=it+1,
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
    def _compute_lc_weights(self, obs):
        """
        Compute per-timestep weights based on lane-change indicator features.
    
        Parameters
        obs : np.ndarray
            Observation sequence, shape (T, obs_dim).
        
        Returns
        w : torch.Tensor
            Weights per timestep, shape (T,), on self.device.
        """
        T = obs.shape[0]

        # No LC features present in this dataset
        if not self.lc_idx:
            return torch.ones((T,), device=self.device, dtype=self.dtype)

        obs_np = np.asarray(obs)
        lc_cols = obs_np[:, self.lc_idx]  # (T, L)
        lc_active_np = np.any(lc_cols > 0.5, axis=1)  # (T,) bool

        # Base weights
        w = torch.ones((T,), device=self.device, dtype=self.dtype)

        # Which LC weight to use
        w_lc = self.lc_weight_invfreq
        w_lc = float(w_lc) if (w_lc is not None) else float(TRAINING_CONFIG.lc_weight)

        if lc_active_np.any():
            lc_active = torch.from_numpy(lc_active_np).to(device=self.device)
            w[lc_active] = w_lc

        return w # (T,) where lc timesteps have higher weights and non -lc timesteps have weight 1.0

    def _compute_logB(self, obs, skipped_bad_logB, i):
        # emissions are joint-indexed: (T, N). N = S * A
        logB_flat = self.emissions.loglikelihood(obs)  # (T, N); Compute emission log-likelihoods logB[t,z] = log p(o_t | z)
        
        if logB_flat.ndim != 2 or logB_flat.shape[1] != self.num_states or logB_flat.shape[0] != obs.shape[0]:
            raise ValueError(
                f"logB_flat has shape {tuple(logB_flat.shape)}, expected ({obs.shape[0]},{self.num_states}). "
                f"Check emission model vs num_states (S={self.S}, A={self.A})."
            )
        if not torch.isfinite(logB_flat).all(): # Skip if emissions contain non-finite values (avoids NaNs later)
            skipped_bad_logB += 1
            if self.verbose >= 2:
                bad = (~torch.isfinite(logB_flat)).sum().item()
                print(f"  Seq {i:03d}: logB has {bad} non-finite entries, skipping.")
            return None, skipped_bad_logB
        
        # Reshape logB to (T, S, A) for structured FB
        logB_s_a = logB_flat.view(obs.shape[0], self.S, self.A)   # logB[t,s,a] = log p(o_t | s,a); shape (T, S, A)

        return logB_s_a, skipped_bad_logB
    
    def _run_forward_backward(self, logB_s_a, mode):
        """
        Wrapper around forward_backward_torch that returns a uniform signature.

        Returns
        gamma_s_a : torch.Tensor  # (T,S,A)
        xi_s_sum  : torch.Tensor  # (S,S)
        xi_a_sum  : torch.Tensor  # (S,A,A)
        loglik    : torch.Tensor  # scalar
        xi_s_t    : torch.Tensor | None  # (T-1,S,S) if mode=="B" else None
        xi_a_t    : torch.Tensor | None  # (T-1,S,A,A) if mode=="B" else None
        """
        if mode == "B":
            gamma_s_a, xi_s_sum, xi_a_sum, loglik, xi_s_t, xi_a_t = forward_backward_torch(
                self.pi_s0,
                self.pi_a0_given_s0,
                self.A_s,
                self.A_a,
                logB_s_a,
                return_xi_t=True,
            )
            return gamma_s_a, xi_s_sum, xi_a_sum, loglik, xi_s_t, xi_a_t

        gamma_s_a, xi_s_sum, xi_a_sum, loglik = forward_backward_torch(
            self.pi_s0,
            self.pi_a0_given_s0,
            self.A_s,
            self.A_a,
            logB_s_a,
            return_xi_t=False,
        )
        return gamma_s_a, xi_s_sum, xi_a_sum, loglik, None, None


    def _apply_mode_B_weighting(self, gamma_s_a, xi_s_sum, xi_a_sum, xi_s_t, xi_a_t, w_t, T, xi_mode):
        """
        Apply lane-change weighting in mode == "B" to sufficient statistics.

        This modifies:
        - gamma_flat: node weights (T,N)
        - xi_s_sum  : edge-weighted expected style transition counts (S,S)
        - xi_a_sum  : edge-weighted expected action transition counts (S,A,A)

        Inputs (shapes):
        gamma_s_a : (T,S,A)
        xi_s_sum  : (S,S)          (unweighted sum from FB)
        xi_a_sum  : (S,A,A)        (unweighted sum from FB)
        xi_s_t    : (T-1,S,S)      (per-edge, only if T>=2)
        xi_a_t    : (T-1,S,A,A)    (per-edge, only if T>=2)
        w_t       : (T,)
        xi_mode   : "next" or "avg"

        Returns
        gamma_flat_weighted : (T,N)
        xi_s_sum_weighted   : (S,S)
        xi_a_sum_weighted   : (S,A,A)
        """
        if w_t is None:
            raise ValueError("mode=='B' requires w_t, got None")
        if xi_s_t is None or xi_a_t is None:
            raise ValueError("mode=='B' requires xi_*_t, got None")

        # 0) flatten gamma for emission updates
        gamma_flat = gamma_s_a.reshape(T, self.num_states)

        # 1) weight gamma per time step (DO NOT renormalize)
        gamma_flat = gamma_flat * w_t[:, None]  # (T,N); only those time steps with lc have higher weights, and hence higher responsibilities.

        # 2) weight xi per transition step
        if T >= 2:
            # edge weights have shape (T-1,)
            if xi_mode == "avg":
                w_edge = 0.5 * (w_t[:-1] + w_t[1:])  # (T-1,); treats the edge as equally “owned” by both endpoints.
            else:
                w_edge = w_t[1:]  # (T-1,); weight of the destination timestep controls the transition leading into it.

            # weighted expected transition counts
            xi_s_sum = (xi_s_t * w_edge[:, None, None]).sum(dim=0)               # (S,S)
            xi_a_sum = (xi_a_t * w_edge[:, None, None, None]).sum(dim=0)         # (S,A,A)
        else:
            # no transitions
            xi_s_sum = xi_s_sum * 0.0
            xi_a_sum = xi_a_sum * 0.0

        return gamma_flat, xi_s_sum, xi_a_sum


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

        S = int(self.S)
        A = int(self.A)
        N = int(self.num_states)

        # per sequence list of tensors
        gamma_all = []
        xi_s_all = []
        xi_a_all = []

        obs_used = [] # sequences that weren’t skipped.
        obs_used_raw = [] if (obs_seqs_raw is not None) else None # same as obs_used but raw-scale, only if provided.
        total_loglik = 0.0

        # to track why sequences were skipped
        skipped_empty = 0
        skipped_bad_logB = 0
        skipped_bad_ll = 0

        iterator = enumerate(obs_seqs)
        if use_progress:
            iterator = tqdm(iterator, total=len(obs_seqs), desc=f"E-step train (iter {it+1})", leave=False,)

        mode = getattr(TRAINING_CONFIG, "lc_weight_mode", "none")
        # choose edge weight scheme
        xi_mode = getattr(TRAINING_CONFIG, "lc_xi_weight", "next")

        debug_asserts = bool(getattr(TRAINING_CONFIG, "debug_asserts", False))

        with torch.no_grad():
            for i, obs in iterator: # iterates through each vehicle trajectory
                if obs is None or obs.shape[0] == 0: # no time steps in this sequence
                    skipped_empty += 1
                    continue

                T = obs.shape[0] # number of timesteps

                logB_s_a, skipped_bad_logB = self._compute_logB(obs, skipped_bad_logB, i)
                if logB_s_a is None:
                    continue

                w_t = self._compute_lc_weights(obs) if mode in ("A", "B") else None # (T,) where lc timesteps have higher weights and non -lc timesteps have weight 1.0
                
                # tempered likelihood (affects inference) 
                if mode == "A":
                    # logB[t,s,a] *= w_t[t]
                    logB_s_a = logB_s_a * w_t[:, None, None] # tempered likelihood, power likelihood, or annealed inference.
                
                # Run forward-backward 
                gamma_s_a, xi_s_sum, xi_a_sum, loglik, xi_s_t, xi_a_t = self._run_forward_backward(logB_s_a=logB_s_a, mode=mode)
                
                if mode == "B" and debug_asserts:
                    if xi_s_t is None or xi_a_t is None:
                        raise ValueError("mode=='B' requires xi_*_t, got None")

                    if T >= 2:
                        exp_xi_s_t_shape = (T - 1, S, S)
                        exp_xi_a_t_shape = (T - 1, S, A, A)

                        if xi_s_t.shape != exp_xi_s_t_shape:
                            raise ValueError(
                                f"xi_s_t has shape {tuple(xi_s_t.shape)}, expected {exp_xi_s_t_shape}"
                            )
                        if xi_a_t.shape != exp_xi_a_t_shape:
                            raise ValueError(
                                f"xi_a_t has shape {tuple(xi_a_t.shape)}, expected {exp_xi_a_t_shape}"
                            )
                        # Finite checks (avoid poisoning weighted sums)
                        if not torch.isfinite(xi_s_t).all():
                            raise ValueError("xi_s_t contains non-finite values")
                        if not torch.isfinite(xi_a_t).all():
                            raise ValueError("xi_a_t contains non-finite values")

                        xi_s_sum_from_t = xi_s_t.sum(dim=0)
                        if not torch.allclose(xi_s_sum, xi_s_sum_from_t, rtol=1e-4, atol=1e-6):
                            max_abs = (xi_s_sum - xi_s_sum_from_t).abs().max().item()
                            raise ValueError(
                                f"xi_s_sum != xi_s_t.sum(0) (max_abs_diff={max_abs:.3e})"
                            )

                        xi_a_sum_from_t = xi_a_t.sum(dim=0)
                        if not torch.allclose(xi_a_sum, xi_a_sum_from_t, rtol=1e-4, atol=1e-6):
                            max_abs = (xi_a_sum - xi_a_sum_from_t).abs().max().item()
                            raise ValueError(
                                f"xi_a_sum != xi_a_t.sum(0) (max_abs_diff={max_abs:.3e})"
                            )
                    else:
                        if xi_s_t.numel() != 0 or xi_a_t.numel() != 0:
                            raise ValueError("Expected empty xi_*_t tensors when T < 2")

                if loglik.ndim != 0:
                    raise ValueError(f"loglik must be scalar, got shape {tuple(loglik.shape)}")
                
                if not torch.isfinite(loglik):
                    skipped_bad_ll += 1
                    if verbose >= 2:
                        print(f"  Seq {i:03d}: loglik is non-finite, skipping.")
                    continue

                if gamma_s_a.shape != (T, S, A):
                    raise ValueError(
                        f"gamma_s_a has shape {tuple(gamma_s_a.shape)}, expected {(T, S, A)}"
                    )
                if xi_s_sum.shape != (S, S):
                    raise ValueError(f"xi_s_sum has shape {tuple(xi_s_sum.shape)}, expected {(S, S)}")
                if xi_a_sum.shape != (S, A, A):
                    raise ValueError(f"xi_a_sum has shape {tuple(xi_a_sum.shape)}, expected {(S, A, A)}")

                if mode == "B":
                    # weighted sufficient statistics (fractional counts) 
                    gamma_flat, xi_s_sum, xi_a_sum = self._apply_mode_B_weighting(
                        gamma_s_a=gamma_s_a,
                        xi_s_sum=xi_s_sum,
                        xi_a_sum=xi_a_sum,
                        xi_s_t=xi_s_t,
                        xi_a_t=xi_a_t,
                        w_t=w_t,
                        T=T,
                        xi_mode=xi_mode
                    )
                else:
                    gamma_flat = gamma_s_a.reshape(T, N) #  # flattened gamma for emission updates
                    
                gamma_all.append(gamma_flat) # (T,N) responsibilities per timestep over joint latent state; for emission M-step
                xi_s_all.append(xi_s_sum)    # (S,S) expected counts of style transitions; for A_s update
                xi_a_all.append(xi_a_sum)    # (S,A,A) expected counts of action transitions; for A_a update

                obs_used.append(obs)
                if obs_used_raw is not None: # to track which observations were used
                    obs_used_raw.append(obs_seqs_raw[i])

                ll_i = float(loglik.detach().cpu().item())
                total_loglik += ll_i

                if verbose >= 2:
                    print(f"  Seq {i:03d}: T={T}, loglik={ll_i:.3f}")

        if verbose:
            print(
                f"  Total train loglik: {total_loglik:.3f} | "
                f"used={len(obs_used)}/{len(obs_seqs)} | "
                f"skipped_empty={skipped_empty}, skipped_bad_logB={skipped_bad_logB}, skipped_bad_ll={skipped_bad_ll}"
            )

        return EStepResult(
            gamma_all=gamma_all,
            xi_s_all=xi_s_all,
            xi_a_all=xi_a_all,
            total_loglik=total_loglik,
            obs_used=obs_used,
            obs_used_raw=obs_used_raw,
            skipped_empty=skipped_empty,
            skipped_bad_logB=skipped_bad_logB,
            skipped_bad_ll=skipped_bad_ll,
        )

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
                if obs is None or obs.shape[0] == 0:
                    continue

                logB_flat = self.emissions.loglikelihood(obs)  # (T,N)
                if logB_flat.ndim != 2 or logB_flat.shape[1] != int(self.num_states):
                    raise ValueError(
                        f"logB_flat has shape {tuple(logB_flat.shape)}, expected (T,{int(self.num_states)})."
                    )
                if not torch.isfinite(logB_flat).all(): 
                    continue # Skip if emissions contain non-finite values

                T = int(logB_flat.shape[0])
                if T <= 0:
                    continue

                logB_s_a = logB_flat.view(T, int(self.S), int(self.A))
                _, _, _, ll = forward_backward_torch(
                                    self.pi_s0, self.pi_a0_given_s0, self.A_s, self.A_a, logB_s_a
                                )
                if torch.isfinite(ll) and (not torch.isnan(ll)):
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

        return TransitionUpdate(
            delta_pi=delta_pi,
            delta_A=delta_A,
            A_prev=A_prev,
            A_new=A_new,
        )
    
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

        Parameters
        train_obs_seqs : list[np.ndarray]
            Training observation sequences, each of shape (T_n, obs_dim).
        """
        max_samples = TRAINING_CONFIG.max_kmeans_samples
        seed = TRAINING_CONFIG.seed
        print("[HDVTrainer] Initialising emissions with (subsampled) k-means...")
        K = int(self.num_states) # total joint states
        rng = np.random.default_rng(seed)


        cont_idx = np.asarray(self.emissions.cont_idx, dtype=np.int64)  
        if cont_idx.size == 0:
            raise RuntimeError("MixedEmissionModel has no continuous dimensions to initialise.")  

        bin_idx = np.asarray(getattr(self.emissions, "bin_idx", []), dtype=np.int64)
        
        # -----------------------------
        # Continuous part (Gaussian)
        # -----------------------------
        pool = [] # to store a small sample of continuous rows for clustering (size ≤ max_samples)
        pool_n = 0 # counter of total rows seen so far
        cont_dim = None # no of continuous dims

        def try_add_rows(Xc_rows):
            """
            Add rows to reservoir pool up to max_samples (uniform reservoir sampling).
            The rows are added only with a probabiity of max_samples / total_rows_seen_so_far,
            ensuring that at the end, each row has equal probability of being included in the pool. 
            
            parameters
            Xc_rows : np.ndarray
                Continuous data rows to consider adding, shape (M, cont_dim).
            """
            nonlocal pool, pool_n
            for row in Xc_rows:
                pool_n += 1
                if len(pool) < max_samples: # pool not full yet
                    pool.append(row) # The first max_samples rows are always kept
                else: # pool is full => probabilistic replacement
                    j = rng.integers(0, pool_n)
                    if j < max_samples:
                        pool[j] = row

        for seq in train_obs_seqs:
            X = np.asarray(seq)
            Xc = X[:, cont_idx] # continuous part
            if cont_dim is None:
                cont_dim = int(Xc.shape[1])

            # Prefer fully-finite rows
            finite_rows = np.isfinite(Xc).all(axis=1)
            Xc_f = Xc[finite_rows]
            if Xc_f.size:
                try_add_rows(Xc_f)

        # If pool is empty (all rows had NaNs), fall back to partially-finite rows
        if len(pool) == 0:
            # Simple safe fallback: replace non-finite with 0.0 *per row* (no giant copies)
            for seq in train_obs_seqs:
                X = np.asarray(seq)
                Xc = X[:, cont_idx].copy()
                bad = ~np.isfinite(Xc)
                if bad.any():
                    Xc[bad] = 0.0
                try_add_rows(Xc)

        Xc = np.asarray(pool, dtype=np.float64) # shape (n_pool, cont_dim)
        if Xc.ndim != 2 or Xc.shape[0] < max(10 * K, (Xc.shape[1] if Xc.ndim == 2 else 1) + 1): # check if we have enough data
            # Extremely small data support: just use global mean/var on whatever we have.
            mu = np.nanmean(Xc, axis=0) if Xc.size else np.zeros((cont_idx.size,), dtype=np.float64)
            var = np.nanvar(Xc, axis=0) + 1e-6 if Xc.size else np.ones((cont_idx.size,), dtype=np.float64)
            var = np.where(np.isfinite(var), var, 1.0)
            var = np.maximum(var, 1e-6)
            for z in range(K): # assign all joint states to have same params for gaussian emissions
                s, a = self.emissions.gauss._z_to_sa(z)
                self.emissions.gauss.params[s, a] = GaussianParams(mean=mu, var=var)
        else:
            # Global variance fallback
            global_var = np.var(Xc, axis=0) + 1e-6
            global_var = np.where(np.isfinite(global_var), global_var, 1.0)
            global_var = np.maximum(global_var, 1e-6)

            mbk = MiniBatchKMeans(
                n_clusters=K, # one cluster per joint state (S×A)
                batch_size=2048,
                max_iter=100,
                n_init=5,
                random_state=seed,
            )
            labels = mbk.fit_predict(Xc) # Fits KMeans and returns a label for each sampled row. shape (n_pool,)
            centers = mbk.cluster_centers_ # Centers for each cluster. shape (K, cont_dim)

            # Assign per-cluster Gaussian params
            for z in range(K):
                mask = (labels == z) # boolean mask for rows assigned to cluster z
                num_points = int(mask.sum())
                if num_points < Xc.shape[1] + 1: # not enough points to compute a valid covariance
                    mean_z = centers[z]
                    var_z = global_var.copy() # use global variance as fallback
                else: # enough points to compute mean/var
                    X_z = Xc[mask]
                    mean_z = X_z.mean(axis=0) # shape (cont_dim,)
                    var_z = np.var(X_z, axis=0) + 1e-6
                    var_z = np.where(np.isfinite(var_z), var_z, global_var)
                    var_z = np.maximum(var_z, 1e-6) #shape (cont_dim,)

                s, a = self.emissions.gauss._z_to_sa(z)
                self.emissions.gauss.params[s, a] = GaussianParams(mean=mean_z, var=var_z) # each joint state gets its own Gaussian params for continuous dims
        
        # -----------------------------
        # Discrete parts (Bernoulli)
        # -----------------------------
        if getattr(self.emissions, "bin_dim", 0) > 0 and bin_idx.size > 0:
            sum_b = np.zeros((bin_idx.size,), dtype=np.float64) # number of “1”s observed
            cnt_b = np.zeros((bin_idx.size,), dtype=np.float64) # number of finite observations

            for seq in train_obs_seqs:
                X = np.asarray(seq) # shape (T, obs_dim)
                xb = X[:, bin_idx] 
                finite = np.isfinite(xb)
                xb_bin = (xb > 0.5) # binary 0/1

                # accumulate per-dim
                sum_b += (xb_bin & finite).sum(axis=0).astype(np.float64) # count of 1s
                cnt_b += finite.sum(axis=0).astype(np.float64) # count of finite entries

            p0 = np.divide(sum_b, np.maximum(cnt_b, 1.0)) # empirical mean per binary dim = P(X=1) = sum_b[d] / cnt_b[d]
            p0 = np.where(np.isfinite(p0), p0, 0.5) # fallback to 0.5 if division was invalid
            p0 = np.clip(p0, 1e-3, 1.0 - 1e-3) # avoid exact 0 or 1
            self.emissions.bern_p = np.tile(p0[None, :], (K, 1)) # same Bernoulli params for all joint states.

        self.emissions.invalidate_cache() # clear any cached values
        self.emissions.to_device(device=self.device, dtype=self.dtype) # move to device

        print("[HDVTrainer] k-means initialisation done.")

    def _init_lc_weight_invfreq(self, train_obs_seqs, clip=10.0):
        """
        Initialize the inverse frequency weight for left/right turn indicators.

        Parameters
        train_obs_seqs : list[np.ndarray]
            Training observation sequences, each of shape (T_n, obs_dim).
        clip : float
            Maximum allowed weight.
        """
        if not self.lc_idx:
            self.lc_weight_invfreq = 1.0
            return

        total = 0 # total number of timesteps seen
        lc = 0    # number of timesteps with any LC indicator active
        for obs in train_obs_seqs: # Iterate over vehicle trajectories
            if obs is None or obs.shape[0] == 0:
                continue
            x = torch.as_tensor(obs, device="cpu")  # on CPU for counting
            active = (x[:, self.lc_idx] > 0.5).any(dim=1) # check if any LC indicator is active at each timestep
            total += active.numel() # total timesteps
            lc += int(active.sum()) # timesteps with LC active

        if total == 0:
            self.lc_weight_invfreq = 1.0
            return

        p_lc = lc / total            # proportion of timesteps with LC active
        w = 1.0 / max(p_lc, EPSILON) # inverse frequency weighting

        # normalize so average weight ~ 1. Done to avoid overall scaling of log-likelihoods.
        w = w / (p_lc * w + (1.0 - p_lc))

        self.lc_weight_invfreq = float(min(w, clip))

        if self.verbose:
            print(
                f"[LC weighting] "
                f"total_timesteps={total}, "
                f"lc_timesteps={lc}, "
                f"p_lc={p_lc:.6e}, "
                f"raw_invfreq={1.0 / max(p_lc, EPSILON):.3f}, "
                f"normalized_w_lc={w:.3f}, "
                f"clipped_w_lc={self.lc_weight_invfreq:.3f}"
            )


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
        # ----------------------------
        payload["obs_names"] = np.array(self.obs_names, dtype=object)

        # -----------------------------
        # Emissions 
        # -----------------------------
        required = {
            "obs_names",
            "cont_idx",
            "bin_idx",
            "bernoulli_names",
            "gauss_means",
            "gauss_vars",
            "bern_p",
        }
        em_dict = self.emissions.to_arrays()  
        missing = [k for k in sorted(required) if k not in em_dict]
        if missing:
            raise ValueError(
                f"MixedEmissionModel.to_arrays() is missing keys: {missing}. "
                "Check emissions.py to_arrays()/from_arrays() implementation."
            )

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
        # Reconstruct trainer 
        # ------------------------------------------------------------------
        if "obs_names" not in data.files:
            raise ValueError(
                "Saved model is missing 'obs_names'. "
                "Re-save the model."
            )  

        obs_names = [str(x) for x in list(data["obs_names"])] 
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
    
    