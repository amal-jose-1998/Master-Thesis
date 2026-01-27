"""EM training loop for the HDV DBN with Mixed emissions (Gaussian + Bernoulli)."""

import numpy as np
import torch
from tqdm.auto import tqdm
import time
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from .model import HDVDBN
from .hierarchical_emissions import MixedEmissionModel as HierarchicalMixedEmissionModel
from .poe_emissions import MixedEmissionModel as PoEMixedEmissionModel
from .config import TRAINING_CONFIG
from .utils.wandb_logger import WandbLogger
from .utils.trainer_diagnostics import (
    #run_lengths_from_gamma_sa_argmax,
    #posterior_entropy_from_gamma,
    posterior_weighted_feature_stats
)
from .forward_backward import forward_backward_torch
from .utils.transitions import build_structured_transition_params
from .utils.initialise_mixed_emissions import init_emissions

# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------
@dataclass
class EStepResult:
    gamma_all: list           # list[torch.Tensor] (T,S,A)
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
    delta_pi_s0: float
    delta_pi_a0_given_s0: float
    delta_A_s: float
    delta_A_a: float

# -----------------------------------------------------------------------------
# Numerical stability constants
# -----------------------------------------------------------------------------
EPSILON = TRAINING_CONFIG.EPSILON if hasattr(TRAINING_CONFIG, "EPSILON") else 1e-6
learn_pi0 = bool(getattr(TRAINING_CONFIG, "learn_pi0", False))
pi0_alpha = float(getattr(TRAINING_CONFIG, "pi0_alpha", 0.0))  # pseudo-count smoothing

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

    def __init__(self, obs_names, emission_model="poe"):
        """
        Parameters
        obs_names : list[str]
            Names of observation features (for MixedEmissionModel).
        """
        self.hdv_dbn = HDVDBN()
        self.obs_names = list(obs_names)

        em_mode = emission_model or getattr(TRAINING_CONFIG, "emission_model", "poe")
        em_mode = str(em_mode).lower().strip()
        disable_disc = bool(getattr(TRAINING_CONFIG, "disable_discrete_obs", False))

        if em_mode == "poe":
            self.emissions = PoEMixedEmissionModel(obs_names=self.obs_names, disable_discrete_obs=disable_disc)
        elif em_mode == "hierarchical":
            self.emissions = HierarchicalMixedEmissionModel(obs_names=self.obs_names, disable_discrete_obs=disable_disc)
        else:
            raise ValueError(f"Unknown emission_model='{em_mode}'. Use 'poe' or 'hierarchical'.")

        self.emission_model = em_mode
        
        self.S = int(self.hdv_dbn.num_style)
        self.A = int(self.hdv_dbn.num_action)
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
    def em_train(self, train_obs_seqs, val_obs_seqs=None, wandb_run=None, train_obs_seqs_raw=None, checkpoint_dir=None, checkpoint_every=5,):
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
        # Normalize checkpoint settings
        ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        if ckpt_dir is not None:
            ckpt_dir.mkdir(parents=True, exist_ok=True)

        ckpt_every = None if checkpoint_every is None else int(checkpoint_every)
        if ckpt_every is not None and ckpt_every <= 0:
            ckpt_every = None

        num_iters = TRAINING_CONFIG.em_num_iters
        use_progress = TRAINING_CONFIG.use_progress
        patience = getattr(TRAINING_CONFIG, "early_stop_patience", 3)
        min_delta = getattr(TRAINING_CONFIG, "early_stop_min_delta_per_obs", 1e-4)
        delta_A_thresh = getattr(TRAINING_CONFIG, "early_stop_delta_A_thresh", 1e-5)
        delta_pi_thresh = getattr(TRAINING_CONFIG, "early_stop_delta_pi_thresh", 1e-5)
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
        init_emissions(train_obs_seqs, self.emissions, self.device, self.dtype)

        if self.verbose:
            print("\n==================== EM TRAINING START ====================\n")
            print(f"Device: {self.device} dtype={self.dtype}")
            print(f"Number of style states:  {self.S}")
            print(f"Number of action states: {self.A}")
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

            # -----------------------------------------------------------
            # Trajectory-level diagnostics (from posteriors)
            # -----------------------------------------------------------
            #run_lengths_train, runlen_median_per_traj = run_lengths_from_gamma_sa_argmax(gamma_all)
            #ent_all_train, ent_mean_per_traj = posterior_entropy_from_gamma(gamma_all)
            # Semantics (scaled-space) for relative comparisons
            #sem_feat_names, sem_means, sem_stds = posterior_weighted_feature_stats(
            #                                                obs_names=self.obs_names,
            #                                                obs_seqs=obs_used,
            #                                                gamma_sa_seqs=gamma_all,
            #                                                S=int(self.S),
            #                                                A=int(self.A),
            #                                            )
            # Semantics (raw-space) for physical interpretation (meters, m/s, etc.)
            sem_means_raw, sem_stds_raw = None, None
            if obs_used_raw is not None:
                sem_feat_names, sem_means_raw, sem_stds_raw = posterior_weighted_feature_stats(
                                                    obs_names=self.obs_names,
                                                    obs_seqs=obs_used_raw,
                                                    gamma_sa_seqs=gamma_all,
                                                    S=int(self.S),
                                                    A=int(self.A),
                                                )

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
            # ----------------------
            # M-step: update emission parameters
            # ----------------------
            # snapshot BEFORE emission update
            em_prev = self.emissions.to_arrays()
            if self.verbose:
                print("M-step: Updating emission parameters...")
            state_weights_sa, total_mass, state_frac_sa = self._m_step_emissions(
                                                    train_obs_seqs=obs_used,
                                                    gamma_all=gamma_all,
                                                    use_progress=use_progress,
                                                    verbose=self.verbose,
                                                )
            # snapshot AFTER emission update
            em_new = self.emissions.to_arrays()
            em_deltas = self._emission_delta(em_prev, em_new)
            if self.verbose >= 2:
                print(f"  emission_delta_mean: {em_deltas.get('emission_delta_mean', np.nan):.6e}")
            # ----------------------
            # Compute validation log-likelihood (if available)
            # ----------------------
            criterion_for_stop = None
            if val_obs_seqs is None:
                val_ll = 0.0
                criterion_for_stop = float(train_ll) / max(int(train_num_obs), 1)
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
                criterion_for_stop = float(val_ll) / max(int(val_num_obs), 1)
            
            # ----------------------
            # Bookkeeping 
            # ----------------------
            history["train_loglik"].append(train_ll)
            if val_obs_seqs is not None:
                history["val_loglik"].append(val_ll)

            # ----------------------
            # Improvement computation 
            # ----------------------
            criterion_finite = np.isfinite(criterion_for_stop)
            prev_finite = (prev_criterion is not None) and np.isfinite(prev_criterion)
            if criterion_finite and prev_finite:
                improvement = float(criterion_for_stop - prev_criterion)
            else:
                improvement = np.nan
            
            # ----------------------
            # Early stopping check
            # ----------------------
            should_break = False
            transitions_stable = None
            pi_stable = None

            # Only run early-stopping logic if criterion is finite and we have a previous value
            if criterion_finite and prev_finite:
                if improvement < min_delta:
                    bad_epochs += 1
                else:
                    bad_epochs = 0
            
                transitions_stable = (m.delta_A_s < delta_A_thresh) and (m.delta_A_a < delta_A_thresh)
                pi_stable = (m.delta_pi_s0 < delta_pi_thresh) and (m.delta_pi_a0_given_s0 < delta_pi_thresh)

                if bad_epochs >= patience:
                    should_break = True

            else:
                # Non-finite criterion or first iteration: don't early-stop
                if not np.isfinite(criterion_for_stop) and self.verbose:
                    print("WARNING: Non-finite stopping criterion; early stopping disabled this iteration.")

            # ----------------------
            # Console prints 
            # ----------------------
            if self.verbose:
                tag = "val" if (val_obs_seqs is not None) else "train"
                if criterion_finite:
                    print(f"  Criterion ({tag} per-obs): {criterion_for_stop:.6f}")
                else:
                    print(f"  Criterion ({tag} per-obs): {criterion_for_stop}")

                if np.isfinite(improvement):
                    print(f"  Improvement: {improvement:.6e}")
                else:
                    print(f"  Improvement: {improvement}")

                # Early-stop debug line 
                if transitions_stable is not None and pi_stable is not None:
                    print(f"  bad_epochs={bad_epochs}/{patience} | "
                        f"stable_A={transitions_stable} (th={delta_A_thresh:g}) | "
                        f"stable_pi={pi_stable} (th={delta_pi_thresh:g})")

            # ----------------------
            # WandB logging
            # ----------------------
            criterion_source = "val" if (val_obs_seqs is not None) else "train"
            train_ll_per_obs = float(train_ll) / max(int(train_num_obs), 1)
            val_ll_per_obs = (float(val_ll) / max(int(val_num_obs), 1)) if (val_num_obs is not None) else np.nan

            log_kwargs = {
                # identity / timing
                "trainer": self,
                "wandb_run": wandb_run,
                "it": it,
                "iter_start": iter_start,

                # likelihood / stopping
                "total_train_ll": float(train_ll),
                "train_ll_per_obs": float(train_ll_per_obs),
                "total_val_ll": float(val_ll),
                "val_ll_per_obs": float(val_ll_per_obs) if np.isfinite(val_ll_per_obs) else np.nan,
                "criterion_for_stop": float(criterion_for_stop) if np.isfinite(criterion_for_stop) else np.nan,
                "criterion_source": criterion_source,
                "improvement": float(improvement) if np.isfinite(improvement) else np.nan,

                # dataset sizes
                "train_num_obs": int(train_num_obs),
                "val_num_obs": int(val_num_obs) if (val_num_obs is not None) else None,

                # e-step health
                "train_total_seqs": int(len(train_obs_seqs)),
                "train_used_seqs": int(len(e.obs_used)),
                "skipped_empty": int(e.skipped_empty),
                "skipped_bad_logB": int(e.skipped_bad_logB),
                "skipped_bad_ll": int(e.skipped_bad_ll),

                # deltas (blockwise + aggregate)
                "delta_pi_s0": float(m.delta_pi_s0),
                "delta_pi_a0_given_s0": float(m.delta_pi_a0_given_s0),
                "delta_A_s": float(m.delta_A_s),
                "delta_A_a": float(m.delta_A_a),
                "log_emission_delta_mean": float(em_deltas.get("emission_delta_mean", np.nan)),
                # hierarchical keys (if using hierarchical emissions)
                "log_delta_gauss_mean": float(em_deltas.get("gauss_mean", np.nan)),
                "log_delta_gauss_var":  float(em_deltas.get("gauss_var", np.nan)),
                "log_delta_bern_p":     float(em_deltas.get("bern_p", np.nan)),
                # PoE keys (if using PoE emissions)
                "log_delta_style_gauss_mean":  float(em_deltas.get("style_gauss_mean", np.nan)),
                "log_delta_style_gauss_var":   float(em_deltas.get("style_gauss_var", np.nan)),
                "log_delta_action_gauss_mean": float(em_deltas.get("action_gauss_mean", np.nan)),
                "log_delta_action_gauss_var":  float(em_deltas.get("action_gauss_var", np.nan)),
                "log_delta_style_bern_p":      float(em_deltas.get("style_bern_p", np.nan)),
                "log_delta_action_bern_p":     float(em_deltas.get("action_bern_p", np.nan)),

                # early stop state (optional; helps debugging)
                "bad_epochs": int(bad_epochs),
                "transitions_stable": bool(transitions_stable) if transitions_stable is not None else None,
                "pi_stable": bool(pi_stable) if pi_stable is not None else None,

                # matrices (for heatmaps) — current values only
                "pi_s0": self.pi_s0.detach().cpu().numpy(),
                "pi_a0_given_s0": self.pi_a0_given_s0.detach().cpu().numpy(),
                "A_s": self.A_s.detach().cpu().numpy(),
                "A_a": self.A_a.detach().cpu().numpy(),

                # posteriors/diagnostics
                "state_weights_sa": state_weights_sa,
                "total_responsibility_mass": float(total_mass),
                "state_frac_sa": state_frac_sa,

                #"run_lengths_train": run_lengths_train,
                #"runlen_median_per_traj": runlen_median_per_traj,
                #"ent_all_train": ent_all_train,
                #"ent_mean_per_traj": ent_mean_per_traj,

                "sem_feat_names": sem_feat_names,
                #"sem_means": sem_means,
                #"sem_stds": sem_stds,
                "sem_means_raw": sem_means_raw,
                "sem_stds_raw": sem_stds_raw,
            }

            WandbLogger.log_iteration(**log_kwargs)

            # ----------------------
            # Periodic checkpoint saving
            # ----------------------
            if ckpt_dir is not None and ckpt_every is not None:
                if ((it + 1) % ckpt_every) == 0:
                    ckpt_path = ckpt_dir / f"ckpt_iter{it+1:04d}.npz"
                    self.save(ckpt_path)

            # Update prev_criterion after logging
            prev_criterion = criterion_for_stop

            if should_break:
                if self.verbose:
                    print("\n*** Early stopping triggered (log-likelihood plateau) ***")
                break

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
        logB_sa = self.emissions.loglikelihood(obs)  # (T,S,A); Compute emission log-likelihoods logB[t,s,a] = log p(o_t | s,a)
        
        if logB_sa.ndim != 3 or logB_sa.shape != (obs.shape[0], self.S, self.A):
            raise ValueError(
                f"logB_sa has shape {tuple(logB_sa.shape)}, expected ({obs.shape[0]},{self.S},{self.A}). "
                "Check emission model S/A vs trainer S/A."
            )
        if not torch.isfinite(logB_sa).all(): # Skip if emissions contain non-finite values (avoids NaNs later)
            skipped_bad_logB += 1
            if self.verbose >= 2:
                bad = (~torch.isfinite(logB_sa)).sum().item()
                print(f"  Seq {i:03d}: logB has {bad} non-finite entries, skipping.")
            return None, skipped_bad_logB
        
        return logB_sa, skipped_bad_logB
    
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


    def _apply_mode_B_weighting(self, gamma_sa, xi_s_sum, xi_a_sum, xi_s_t, xi_a_t, w_t, T, xi_mode):
        """
        Apply lane-change weighting in mode == "B" to sufficient statistics.

        This modifies:
        - gamma_flat: node weights (T,N)
        - xi_s_sum  : edge-weighted expected style transition counts (S,S)
        - xi_a_sum  : edge-weighted expected action transition counts (S,A,A)

        Inputs (shapes):
        gamma_sa : (T,S,A)
        xi_s_t    : (T-1,S,S)      (per-edge, only if T>=2)
        xi_a_t    : (T-1,S,A,A)    (per-edge, only if T>=2)
        w_t       : (T,)
        xi_mode   : "next" or "avg"

        Returns
        gamma_sa_weighted   : (T,S,A)
        xi_s_sum_weighted   : (S,S)
        xi_a_sum_weighted   : (S,A,A)
        """
        if w_t is None:
            raise ValueError("mode=='B' requires w_t, got None")
        if xi_s_t is None or xi_a_t is None:
            raise ValueError("mode=='B' requires xi_*_t, got None")

        # 1) weight gamma per time step 
        gamma_sa = gamma_sa * w_t[:, None, None]  # (T,S,A); only those time steps with lc have higher weights, and hence higher responsibilities.

        # 2) weight xi per transition step
        if T >= 2:
            # edge weights have shape (T-1,)
            if xi_mode == "avg":
                w_edge = 0.5 * (w_t[:-1] + w_t[1:])  # (T-1,); treats the edge as equally “owned” by both endpoints.
            else:
                w_edge = w_t[1:]  # (T-1,); weight of the destination timestep controls the transition leading into it.

            # weighted expected transition counts
            xi_s_sum = (xi_s_t * w_edge[:, None, None]).sum(dim=0)                  # (S,S)
            xi_a_sum = (xi_a_t * w_edge[:, None, None, None]).sum(dim=0)         # (S,A,A)
        else:
            # no transitions
            xi_s_sum = xi_s_sum * 0.0
            xi_a_sum = xi_a_sum * 0.0

        return gamma_sa, xi_s_sum, xi_a_sum


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
            Each element shape (T_n, S, A) on self.device. This is kept for emission updates.
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
                gamma_sa, xi_s_sum, xi_a_sum, loglik, xi_s_t, xi_a_t = self._run_forward_backward(logB_s_a=logB_s_a, mode=mode)
                
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

                if gamma_sa.shape != (T, S, A):
                    raise ValueError(
                        f"gamma_s_a has shape {tuple(gamma_sa.shape)}, expected {(T, S, A)}"
                    )
                if xi_s_sum.shape != (S, S):
                    raise ValueError(f"xi_s_sum has shape {tuple(xi_s_sum.shape)}, expected {(S, S)}")
                if xi_a_sum.shape != (S, A, A):
                    raise ValueError(f"xi_a_sum has shape {tuple(xi_a_sum.shape)}, expected {(S, A, A)}")

                if mode == "B":
                    # weighted sufficient statistics (fractional counts) 
                    gamma_sa, xi_s_sum, xi_a_sum = self._apply_mode_B_weighting(
                        gamma_sa=gamma_sa,
                        xi_s_sum=xi_s_sum,
                        xi_a_sum=xi_a_sum,
                        xi_s_t=xi_s_t,
                        xi_a_t=xi_a_t,
                        w_t=w_t,
                        T=T,
                        xi_mode=xi_mode
                    )
                
                gamma_all.append(gamma_sa)   # (T,S,A) responsibilities per timestep over joint latent state; for emission M-step
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

                logB_sa = self.emissions.loglikelihood(obs)  # (T,S,A)
                if logB_sa.ndim != 3 or logB_sa.shape[1:] != (int(self.S), int(self.A)):
                    raise ValueError(
                        f"logB_sa has shape {tuple(logB_sa.shape)}, expected (T,{int(self.S)},{int(self.A)})."
                    )
                if not torch.isfinite(logB_sa).all():
                    continue # Skip if emissions contain non-finite values

                T = int(logB_sa.shape[0])
                if T <= 0:
                    continue

                _, _, _, ll = forward_backward_torch(self.pi_s0, self.pi_a0_given_s0, self.A_s, self.A_a, logB_sa)

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
            Each element shape (T_n, S, A)
        xi_s_all : list[torch.Tensor]
            Each element shape (S, S). style transition counts.
        xi_a_all : list[torch.Tensor]
            Each element shape (S, A, A). action transition counts conditioned on next style.
        verbose : int
            Verbosity level.

        Returns
        TransitionUpdate
            Contains mean-absolute deltas for:
            - pi_s0, pi_a0_given_s0
            - A_s, A_a
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

        # update pi_s0 and pi_a0_given_s0 from gamma[t=0]
        # For windowed / arbitrarily-cropped sequences, t=0 is not a meaningful “start of behavior”.
        # In that case it is common to fix the initial distribution to a non-informative choice
        # (e.g., uniform) rather than fitting it to segmentation artifacts.
        
        if learn_pi0:
            pi_s0_new = torch.zeros((S,), device=self.device, dtype=self.dtype)
            pi_a0_given_s0_new = torch.zeros((S, A), device=self.device, dtype=self.dtype)
            for gamma_sa in gamma_all:
                if gamma_sa.shape[0] <= 0:
                    continue
                g0 = gamma_sa[0]  # (S,A)
                if g0.shape != (S, A):
                    raise ValueError(f"gamma_sa[0] has shape {tuple(g0.shape)}, expected {(S,A)}")
                pi_s0_new += g0.sum(dim=1)     # (S,)
                pi_a0_given_s0_new += g0       # (S,A)
            
            # optional smoothing (Dirichlet pseudo-counts)
            if pi0_alpha > 0.0:
                pi_s0_new = pi_s0_new + pi0_alpha
                pi_a0_given_s0_new = pi_a0_given_s0_new + pi0_alpha

            # normalize
            pi_s0_new = pi_s0_new / (pi_s0_new.sum() + EPSILON)
            row = pi_a0_given_s0_new.sum(dim=1, keepdim=True)
            pi_a0_given_s0_new = pi_a0_given_s0_new / (row + EPSILON)
        
        else:
            # fixed non-informative initial distributions
            pi_s0_new = torch.full((S,), 1.0 / S, device=self.device, dtype=self.dtype)
            pi_a0_given_s0_new = torch.full((S, A), 1.0 / A, device=self.device, dtype=self.dtype)

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
        pi_s0_prev           = prev["pi_s0"]
        pi_a0_given_s0_prev  = prev["pi_a0_given_s0"]
        As_prev              = prev["A_s"]
        Aa_prev              = prev["A_a"]

        pi_s0_new_np           = self.pi_s0.detach().cpu().numpy()
        pi_a0_given_s0_new_np  = self.pi_a0_given_s0.detach().cpu().numpy()
        As_new_np              = self.A_s.detach().cpu().numpy()
        Aa_new_np              = self.A_a.detach().cpu().numpy()

        delta_pi_s0 = float(np.abs(pi_s0_new_np - pi_s0_prev).mean())                              # mean abs
        delta_pi_a  = float(np.abs(pi_a0_given_s0_new_np  - pi_a0_given_s0_prev ).mean())          # mean abs

        delta_A_s = float(np.abs(As_new_np - As_prev).mean())                 # mean abs
        delta_A_a = float(np.abs(Aa_new_np - Aa_prev).mean())                 # mean abs


        if verbose:
            print(f"  Δpi_s0 mean abs   : {delta_pi_s0:.6e}")
            print(f"  Δpi_a0|s0 mean abs: {delta_pi_a:.6e}")
            print(f"  ΔA_s mean abs     : {delta_A_s:.6e}")
            print(f"  ΔA_a mean abs     : {delta_A_a:.6e}")

        return TransitionUpdate(
            delta_pi_s0=delta_pi_s0,
            delta_pi_a0_given_s0=delta_pi_a,
            delta_A_s=delta_A_s,
            delta_A_a=delta_A_a
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
        state_weights_sa : np.ndarray (S,A)
            Total responsibility mass per joint (s,a). Useful for logging.
        total_mass : float
            Sum over all state weights. Equals the total number of timesteps across all sequences that contributed posteriors (i.e., sequences not skipped).
        state_frac_sa : np.ndarray (S,A)
            Normalized joint masses.
        """
        if self.emission_model == "poe":
            masses = self.emissions.update_from_posteriors(
                obs_seqs=train_obs_seqs,
                gamma_sa_seqs=gamma_all,
                lr=float(getattr(TRAINING_CONFIG, "poe_em_lr", 1e-2)),
                steps=int(getattr(TRAINING_CONFIG, "poe_em_steps", 10)),
                use_progress=use_progress,
                verbose=verbose,
            )
        else:
            # hierarchical closed-form M-step; no lr/steps
            masses = self.emissions.update_from_posteriors(
                obs_seqs=train_obs_seqs,
                gamma_sa_seqs=gamma_all,
                use_progress=use_progress,
                verbose=verbose,
            )

        state_weights_sa = np.asarray(masses["mass_joint"], dtype=np.float64)  # (S,A)
        total_mass = float(state_weights_sa.sum())
        state_frac_sa = (state_weights_sa / total_mass) if total_mass > 0.0 else np.zeros_like(state_weights_sa)
        return state_weights_sa, total_mass, state_frac_sa

    # ------------------------------------------------------------------
    # emissions init, save, load 
    # ------------------------------------------------------------------
    def _emission_delta(self, prev_em_dict, new_em_dict):
        """Compute per-key mean-absolute deltas between two emission snapshots, plus an aggregate mean."""
        deltas = {}

        # Keys that are not learnable parameters (metadata / indices / names)
        exclude = {
            "obs_names",
            "bernoulli_names",
            "cont_idx",
            "bin_idx",
        }

        keys = sorted(set(prev_em_dict.keys()) & set(new_em_dict.keys()))
        for k in keys:
            if k in exclude:
                continue

            a = np.asarray(prev_em_dict[k])
            b = np.asarray(new_em_dict[k])

            # Only compare numeric arrays with identical shape
            if a.shape != b.shape or a.size == 0:
                continue
            if not (np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number)):
                continue

            deltas[k] = float(np.mean(np.abs(b - a)))

        # Aggregate scalar: mean of per-key deltas
        deltas["emission_delta_mean"] = float(np.mean(list(deltas.values()))) if deltas else np.nan
        return deltas

    def _init_lc_weight_invfreq(self, train_obs_seqs, clip=25.0):
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

        self.lc_weight_invfreq = float(min(w, clip))

        if self.verbose:
            print(
                f"[LC weighting] "
                f"total_timesteps={total}, "
                f"lc_timesteps={lc}, "
                f"p_lc={p_lc:.6e}, "
                f"raw_invfreq=w_lc={1.0 / max(p_lc, EPSILON):.3f}, "
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
            raise ValueError(f"pi_s0 must have shape {(S,)}, got {pi_s0_np.shape}")
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

            "emission_model": np.array([self.emission_model], dtype=object),
        }

        # -----------------------------
        # Emissions 
        # -----------------------------
        if self.emission_model == "poe":
            required = {
                "obs_names",
                "cont_idx",
                "bin_idx",
                "bernoulli_names",
                "style_gauss_mean",
                "style_gauss_var",
                "action_gauss_mean",
                "action_gauss_var",
                "style_bern_p",
                "action_bern_p",
            }
        elif self.emission_model == "hierarchical":
            required = {
                "obs_names",
                "cont_idx",
                "bin_idx",
                "bernoulli_names",
                "gauss_mean",
                "gauss_var",
                "bern_p",
            }
        else:
            raise ValueError(f"Unknown emission_model='{self.emission_model}' in save().")
        
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
          - Mixed emission parameters (Gaussian + Bernoulli)
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
        # Restore emissions (MixedEmissionModel)
        # ------------------------------------------------------------------
        em_payload = {k[len("em__"):]: data[k] for k in data.files if k.startswith("em__")}
        if not em_payload:
            raise ValueError("Checkpoint contains no emission payload (no keys starting with 'em__').")

        obs_names = list(np.asarray(em_payload["obs_names"], dtype=object).tolist())

        em_mode = "poe"
        if "emission_model" in data.files:
            em_mode = str(np.asarray(data["emission_model"]).reshape(-1)[0])

        trainer = cls(obs_names=obs_names, emission_model=em_mode)

        trainer.emissions.from_arrays(em_payload)
        trainer.emissions.to_device(device=trainer.device, dtype=trainer.dtype)

        # ------------------------------------------------------------------
        # Restore transitions
        # ------------------------------------------------------------------
        needed = {"pi_s0", "pi_a0_given_s0", "A_s", "A_a"}
        if not needed.issubset(set(data.files)):
            raise ValueError(f"Checkpoint missing transitions. Expected keys: {sorted(needed)}")

        trainer.pi_s0 = torch.as_tensor(data["pi_s0"], device=trainer.device, dtype=trainer.dtype)
        trainer.pi_a0_given_s0 = torch.as_tensor(data["pi_a0_given_s0"], device=trainer.device, dtype=trainer.dtype)
        trainer.A_s = torch.as_tensor(data["A_s"], device=trainer.device, dtype=trainer.dtype)
        trainer.A_a = torch.as_tensor(data["A_a"], device=trainer.device, dtype=trainer.dtype)

        # ensure num_states consistent 
        trainer.S = int(trainer.hdv_dbn.num_style)
        trainer.A = int(trainer.hdv_dbn.num_action)
        trainer.num_states = trainer.S * trainer.A

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
    
    