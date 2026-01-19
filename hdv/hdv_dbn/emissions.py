"""
Window-level mixed emissions for the HDV DBN with joint latent state z=(style, action).
Observation vector o_t contains:
  (1) Continuous window features: modeled by a diagonal Gaussian per joint state.
  (2) Bernoulli features: explicitly listed in config.BERNOULLI_FEATURES
      (e.g., lc_left_present / lc_right_present).

Two operating modes controlled by TRAINING_CONFIG.disable_discrete_obs:
  - disable_discrete_obs = False (full / weighted):
      log p(o_t | z) = log N_diag(o_t[cont]; μ_z, diag(var_z))
                     + w_bern * Σ_d log Bern(o_t[bin_d]; p_zd)

  - disable_discrete_obs = True (continuous-only likelihood):
      log p(o_t | z) = log N_diag(o_t[cont]; μ_z, diag(var_z))
"""

import math
import numpy as np
import torch
from dataclasses import dataclass
from tqdm.auto import tqdm

from .config import DBN_STATES, TRAINING_CONFIG, BERNOULLI_FEATURES

# ---------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------
EPSILON = 1e-8

# =============================================================================
# Parameter containers
# =============================================================================
@dataclass(frozen=True)
class GaussianParams:
    """
    Mean and *diagonal* variance for one Gaussian.
    We model the continuous emission as a diagonal Gaussian per joint state:
        p(x | z) = N(x; mean_z, diag(var_z))

    Attributes
    mean : np.ndarray
        Mean vector, shape (D,).
    var : np.ndarray
        Variance vector (diagonal of covariance), shape (D,).
    """
    mean: np.ndarray
    var: np.ndarray


# =============================================================================
# Gaussian emission model (continuous features)
# =============================================================================
class GaussianEmissionModel:
    """
    Diagonal Gaussian emission model (continuous features), with optional gating masks.

    Gating (mask):
      For each time t and dimension d, a mask m[t,d] in {0,1} indicates whether that
      continuous dimension is valid/observed. Log-likelihood and M-step statistics
      only include dimensions where m[t,d] == 1.

    Provides:
      - loglik_all_states(x): (T, N) log-likelihoods for all states
      - update_from_posteriors(...): M-step for mu/cov
    """
    def __init__(self, obs_dim):
        """
        Initialize the Gaussian emission model with zero means and identity covariances.

        Parameters
        obs_dim : int
            Dimensionality of the observation vector o_t.
        """
        self.obs_dim = obs_dim # dimension of input feature vector

        self.style_states = DBN_STATES.driving_style
        self.action_states = DBN_STATES.action
        self.num_style = len(self.style_states)
        self.num_action = len(self.action_states)
        self.num_states = self.num_style * self.num_action

        # Explicit param table in (style, action) layout
        self.params = np.empty((self.num_style, self.num_action), dtype=object) # shape (S, A)
        for s in range(self.num_style):
            for a in range(self.num_action):
                self.params[s, a] = GaussianParams(
                    mean=np.zeros(self.obs_dim, dtype=np.float64),
                    var=np.ones(self.obs_dim, dtype=np.float64),
                )

        # Torch caches
        self._device = torch.device("cpu")
        self._dtype = torch.float32
        self._means_t = None    # (N,D)
        self._var_t = None      # (N,D)
        self._inv_var_t = None  # (N,D)
        self._log_var_t = None  # (N,D)

    # =========================================================================
    # Internal mapping helpers
    # =========================================================================
    def _z_to_sa(self, z):
        """
        Map a flattened joint index z into (style_idx, action_idx).
            z_idx = style_idx * num_action + action_idx
        Solving for style_idx and action_idx gives:
            style_idx = z_idx // num_action
            action_idx = z_idx % num_action

        Parameters
        z : int
            Joint state index in [0, num_states).

        Returns
        (style_idx, action_idx) : Tuple[int, int]
        """
        s = int(z // self.num_action) # integer division for style index because style is the major index (dim 0)
        a = int(z % self.num_action) # modulo for action index because action is the minor index (dim 1)
        return s, a
    
    # =========================================================================
    # Torch device / cache management
    # =========================================================================
    def to_device(self, device, dtype):
        """
        Build caches on the target device/dtype. Uses Cholesky.

        Parameters
        device : str | torch.device
            Torch device (e.g., "cuda", "cpu").

        dtype : torch.dtype
            Floating-point dtype.
        """
        self._device = torch.device(device)
        self._dtype = dtype

        N, D = self.num_states, self.obs_dim
        means = np.zeros((N, D), dtype=np.float64)
        var = np.ones((N, D), dtype=np.float64)

        for z in range(N):
            s, a = self._z_to_sa(z)
            means[z] = self.params[s, a].mean
            var[z] = self.params[s, a].var

        means_t = torch.as_tensor(means, device=self._device, dtype=self._dtype)
        var_t = torch.as_tensor(var, device=self._device, dtype=self._dtype)

        # numerical floors
        min_diag = float(getattr(TRAINING_CONFIG, "min_cov_diag", 1e-5))
        jitter = float(getattr(TRAINING_CONFIG, "emission_jitter", 1e-6))
        var_t = torch.clamp(var_t, min=min_diag) + jitter

        self._means_t = means_t
        self._var_t = var_t
        self._inv_var_t = 1.0 / var_t
        self._log_var_t = torch.log(var_t)

    def invalidate_cache(self):
        """
        Invalidate Torch caches. They will be rebuilt on next use.
        """
        self._means_t = None
        self._var_t = None
        self._inv_var_t = None
        self._log_var_t = None

    # =========================================================================
    # Likelihood evaluation
    # =========================================================================
    def loglikelihood(self, obs_cont, mask=None):
        """
        Vectorized evaluation of log-likelihoods for ALL states and ALL time steps.
        Computes:
            logB[t, z] = log p(o_t | z) = log N_diag(o_t; μ_z, diag(var_z)) = sum_d log N(o_t[d]; μ_zd, var_zd)
        for t = 0..T-1 and z = 0..N-1, where N = num_style * num_action.

        Parameters
        obs_cont : np.ndarray | torch.Tensor
            Observation sequence of shape (T, obs_dim) with continuous features only.
        mask : (T,D) or None
            Optional gating mask (0/1). If provided, only masked-in dims contribute.

        Returns
        logB : torch.Tensor
            Log-likelihood matrix with shape (T, num_states), stored on the configured Torch device.
        """
        if self.obs_dim == 0:
            raise RuntimeError("GaussianEmissionModel called with obs_dim==0. Caller should skip Gaussian path.")

        if self._means_t is None or self._inv_var_t is None or self._log_var_t is None:
            raise RuntimeError(
                "GaussianEmissionModel caches not initialized. "
                "Call emissions.to_device() after setting parameters."
            )

        x = obs_cont
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, device=self._device, dtype=self._dtype)
        else:
            x = x.to(device=self._device, dtype=self._dtype)

        if x.ndim != 2:
            raise ValueError(f"Expected obs_cont with shape (T,D), got {tuple(x.shape)}")
        T, D = x.shape
        if D != self.obs_dim:
            raise ValueError(f"GaussianEmissionModel expected obs_dim={self.obs_dim}, got {D}")

        if mask is None:
            m = None
        else:
            m = mask
            if not torch.is_tensor(m):
                m = torch.as_tensor(m, device=self._device, dtype=self._dtype)
            else:
                m = m.to(device=self._device, dtype=self._dtype)
            if m.shape != x.shape:
                raise ValueError(f"mask shape {tuple(m.shape)} must match obs_cont shape {tuple(x.shape)}")
            m = (m > 0.5).to(dtype=self._dtype) # Force it to {0,1} float mask.

        # Diagonal Gaussian logpdf per dimension:
        # log N(x; mu, var) = -0.5 * [log(2π) + log(var) + (x-mu)^2 / var]
        diff = x[:, None, :] - self._means_t[None, :, :]                                 # diff[t,z,d]=x[t,d]−μ[z,d]; shape (T,N,D)
        quad = (diff * diff) * self._inv_var_t[None, :, :]                               # quad[t,z,d] = (x[t,d]−μ[z,d])^2 / var[z,d]; shape (T,N,D)
        per_dim = -0.5 * (math.log(2.0 * math.pi) + self._log_var_t[None, :, :] + quad)  # per_dim[t,z,d] = log N(x[t,d]; μ[z,d], var[z,d]); shape (T,N,D)

        if m is not None: # apply mask
            per_dim = per_dim * m[:, None, :]

        return per_dim.sum(dim=2)  # sum over d; shape (T,N)

    # =========================================================================
    # EM M-step update
    # =========================================================================
    def update_from_posteriors(self, obs_seqs, gamma_seqs, mask_seqs=None, use_progress=True, verbose=0):
        """
        M-step update for Gaussian params using responsibilities gamma[t,z] (with optional masks).
            gamma[t, z] = P(Z_t = z | O_1:T)
        With masks m[t,d]:
            W[z,d]   = Σ_t γ[t,z] m[t,d]
            μ[z,d]   = Σ_t γ[t,z] m[t,d] x[t,d]   / W[z,d]
            E[x^2]   = Σ_t γ[t,z] m[t,d] x[t,d]^2 / W[z,d]
            var[z,d] = E[x^2] - μ[z,d]^2

        Parameters
        obs_seqs : list of np.ndarray
            List of observation sequences. Each element has shape (T_n, obs_dim).
        gamma_seqs : list of np.ndarray
            List of responsibilities. Each element has shape (T_n, num_states).
        verbose : int
            0 = no prints,
            1 = per-iteration summary,
            2 = detailed (more debug prints).
        use_progress : bool
            If True, show progress bars for the emission M-step.

        Returns
        weights_flat : np.ndarray
            1D array of length (num_style * num_action) with total responsibility mass per joint state z = (style, action).
        """
        N, D = self.num_states, self.obs_dim
        if D == 0:
            return np.zeros((N,), dtype=np.float64)

        dtype_str = getattr(TRAINING_CONFIG, "dtype", "float32")
        dtype = torch.float32 if dtype_str == "float32" else torch.float64
        device = torch.device(getattr(TRAINING_CONFIG, "device", "cpu"))
        self.to_device(device=device, dtype=dtype)

        # init accumulators
        weights_z = torch.zeros((N,), device=device, dtype=dtype)        # (N,)
        weights_zd = torch.zeros((N, D), device=device, dtype=dtype)     # (N,D)
        sum_x = torch.zeros((N, D), device=device, dtype=dtype)          # (N,D)
        sum_x2 = torch.zeros((N, D), device=device, dtype=dtype)         # (N,D)

        if mask_seqs is None:
            mask_seqs = [None] * len(obs_seqs)
        if len(mask_seqs) != len(obs_seqs):
            raise ValueError("mask_seqs must be None or have the same length as obs_seqs")

        it = zip(obs_seqs, gamma_seqs, mask_seqs)
        if use_progress:
            it = tqdm(it, total=len(obs_seqs), desc="M-step emissions (gauss-diag)", leave=False)

        for obs, gamma, mask in it:
            x = obs if torch.is_tensor(obs) else torch.as_tensor(obs, device=device, dtype=dtype)
            g = gamma if torch.is_tensor(gamma) else torch.as_tensor(gamma, device=device, dtype=dtype)
            if x.device != device or x.dtype != dtype:
                x = x.to(device=device, dtype=dtype)  
            if g.device != device or g.dtype != dtype:
                g = g.to(device=device, dtype=dtype)

            if x.ndim != 2 or x.shape[1] != D:
                raise ValueError(f"Expected obs_cont shape (T,{D}), got {tuple(x.shape)}")
            if g.ndim != 2 or g.shape[1] != N:
                raise ValueError(f"Expected gamma shape (T,{N}), got {tuple(g.shape)}")
            if x.shape[0] != g.shape[0]:
                raise ValueError(f"T mismatch: obs T={x.shape[0]} vs gamma T={g.shape[0]}")

            # state mass
            weights_z += g.sum(dim=0) # state mass accumulation over sequences; shape (N,)

            # mask
            if mask is None:
                m = torch.ones((x.shape[0], D), device=device, dtype=dtype)
            else:
                m = mask if torch.is_tensor(mask) else torch.as_tensor(mask, device=device, dtype=dtype)
                m = m.to(device=device, dtype=dtype)
                if m.shape != x.shape:
                    raise ValueError(f"mask shape {tuple(m.shape)} must match obs_cont shape {tuple(x.shape)}")
                m = (m > 0.5).to(dtype=dtype)

            # Weighted sums, per (z,d)
            weights_zd += g.T @ m
            sum_x += g.T @ (m * x)
            sum_x2 += g.T @ (m * (x * x))

        mean_new = sum_x / (weights_zd + EPSILON)                 # (N,D)
        ex2 = sum_x2 / (weights_zd + EPSILON)                     # (N,D)
        var_new = ex2 - mean_new * mean_new                       # (N,D)

        min_diag = float(getattr(TRAINING_CONFIG, "min_cov_diag", 1e-5))
        jitter = float(getattr(TRAINING_CONFIG, "emission_jitter", 1e-6))
        min_mass = float(getattr(TRAINING_CONFIG, "gauss_min_state_mass", 50.0))

        #get previous params (for low-mass states)
        sa_pairs = [self._z_to_sa(z) for z in range(N)]
        prev_means = torch.as_tensor(
            np.stack([self.params[s, a].mean for (s, a) in sa_pairs], axis=0),
            device=device, dtype=dtype
        )
        prev_vars = torch.as_tensor(
            np.stack([self.params[s, a].var for (s, a) in sa_pairs], axis=0),
            device=device, dtype=dtype
        )

        # vectorized low-mass mask + observed-dim mask
        mass_ok = (weights_z >= min_mass)[:, None]          # (N,1) boolean; need min_mass to update
        obs_ok = (weights_zd > 1.0)                         # (N,D) boolean; need >1 sample to compute variance
        upd = mass_ok & obs_ok                              # (N,D); only update these entries

        mean = prev_means.clone()
        var = prev_vars.clone()

        mean[upd] = mean_new[upd]
        var[upd] = torch.clamp(var_new[upd], min=min_diag) + jitter

        
        mean_np = mean.detach().cpu().numpy()
        var_np = var.detach().cpu().numpy()
        for z in range(N):
            s, a = self._z_to_sa(z)
            self.params[s, a] = GaussianParams(mean=mean_np[z], var=var_np[z])

        self.invalidate_cache()
        return weights_z.detach().cpu().numpy()

    # =========================================================================
    # Saving / loading helpers
    # =========================================================================
    def to_arrays(self):
        """
        Export Gaussian parameters as dense NumPy arrays in (style, action) layout.

        Returns
        means : np.ndarray
            Shape (num_style, num_action, obs_dim)

        vars : np.ndarray
            Shape (num_style, num_action, obs_dim)
        """
        means = np.zeros((self.num_style, self.num_action, self.obs_dim), dtype=np.float64)
        vars_ = np.zeros((self.num_style, self.num_action, self.obs_dim), dtype=np.float64)
        for s in range(self.num_style):
            for a in range(self.num_action):
                p = self.params[s, a]
                means[s, a] = p.mean
                vars_[s, a] = p.var
        return means, vars_

    def from_arrays(self, means, covs_or_vars):
        """
        Load Gaussian parameters from dense arrays in (style, action) layout.

        Parameters
        means : np.ndarray
            Shape (num_style, num_action, obs_dim)

        covs_or_vars : np.ndarray
            Shape (num_style, num_action, obs_dim, obs_dim) or (num_style, num_action, obs_dim)
        """
        exp_means = (self.num_style, self.num_action, self.obs_dim)
        if means.shape != exp_means:
            raise ValueError(f"means has shape {means.shape}, expected {exp_means}")

        # accept either vars (S,A,D) or legacy covs (S,A,D,D)
        if covs_or_vars.ndim == 3:
            vars_ = covs_or_vars
        elif covs_or_vars.ndim == 4:
            vars_ = np.diagonal(covs_or_vars, axis1=-2, axis2=-1)
        else:
            raise ValueError("covs_or_vars must have ndim 3 (vars) or 4 (covs)")

        exp_vars = (self.num_style, self.num_action, self.obs_dim)
        if vars_.shape != exp_vars:
            raise ValueError(f"vars has shape {vars_.shape}, expected {exp_vars}")

        for s in range(self.num_style):
            for a in range(self.num_action):
                self.params[s, a] = GaussianParams(
                    mean=np.asarray(means[s, a], dtype=np.float64),
                    var=np.asarray(vars_[s, a], dtype=np.float64),
                )
        self.invalidate_cache()

# =============================================================================
# Mixed emission model (continuous + Bernoulli + categorical)
# =============================================================================
class MixedEmissionModel:
    """
    Hybrid emission model over the full observation vector (T, F), split by names:
      - Continuous dims: diagonal Gaussian with NaN-masking
      - Bernoulli dims: independent Bernoulli per joint state

    Public API:
      - loglik_all_states(obs) -> (T, N) torch tensor
      - update_from_posteriors(obs_seqs, gamma_seqs, ...) -> weights (N,)
      - to_arrays / from_arrays for checkpointing
    """
    def __init__(self, obs_names, disable_discrete_obs=False, bernoulli_names=None,):
        """
        Initialize the mixed emission model.

        Parameters
        obs_names : Sequence[str]
            List of observation feature names, in order.
        disable_discrete_obs : bool
            If True, only continuous features are used in likelihoods.
        bernoulli_names : Sequence[str] | None
            List of names for Bernoulli features. If None, uses config.BERNOULLI_FEATURES.

        Raises
        ValueError
            If lane_name or lc_name is not found in obs_names.
        """
        self.obs_names = list(obs_names)
        self.obs_dim = len(self.obs_names) 

        self.style_states = DBN_STATES.driving_style
        self.action_states = DBN_STATES.action
        self.num_style = len(self.style_states)
        self.num_action = len(self.action_states)
        self.num_states = self.num_style * self.num_action

        self.disable_discrete_obs = bool(disable_discrete_obs)

        if bernoulli_names is None:
            bernoulli_names = list(BERNOULLI_FEATURES)
        self.bernoulli_names = list(bernoulli_names)

        name_to_idx = {n: i for i, n in enumerate(self.obs_names)}
        self.bin_idx = [int(name_to_idx[n]) for n in self.bernoulli_names if n in name_to_idx]
        bin_set = set(self.bin_idx)
        self.bin_dim = len(self.bin_idx)
        # Continuous dims are everything except Bernoulli dims.
        self.cont_idx = [i for i in range(self.obs_dim) if i not in bin_set]
        self.cont_dim = len(self.cont_idx)

        # components
        self.gauss = GaussianEmissionModel(obs_dim=self.cont_dim)
        # Bernoulli params: (N,B)
        self.bern_p = np.full((self.num_states, self.bin_dim), 0.5, dtype=np.float64) # shape (N, bin_dim) all 0.5

        # setup torch caches for discrete distributions
        self._device = torch.device("cpu")
        self._dtype = torch.float32
        self._bern_p_t = None     # (N,B), where B = bin_dim and N = num_states

    def invalidate_cache(self):
        """Invalidate Torch caches. They will be rebuilt on next use."""
        self._bern_p_t = None
        self.gauss.invalidate_cache()

    def to_device(self, device, dtype):
        """ 
        Build caches on the target device/dtype.
        
        Parameters
        device : str | torch.device
            Torch device (e.g., "cuda", "cpu").
        dtype : torch.dtype
            Floating-point dtype.
        """
        self._device = torch.device(device)
        self._dtype = dtype
        self.gauss.to_device(device=device, dtype=dtype)

        if self.bin_dim > 0:
            self._bern_p_t = torch.as_tensor(self.bern_p, device=self._device, dtype=self._dtype).clamp(EPSILON, 1.0 - EPSILON)
        else:
            self._bern_p_t = None
    
    def _ensure_caches(self):
        need = (self.gauss._means_t is None) or (self.gauss._inv_var_t is None) or (self.gauss._log_var_t is None)
        need = need or (self.bin_dim > 0 and self._bern_p_t is None)
        if need:
            dtype_str = getattr(TRAINING_CONFIG, "dtype", "float32")
            dtype = torch.float32 if dtype_str == "float32" else torch.float64
            device = torch.device(getattr(TRAINING_CONFIG, "device", "cpu"))
            self.to_device(device=device, dtype=dtype)

    # =========================================================================
    # Likelihood evaluation
    # =========================================================================
    def loglikelihood_parts(self, obs): 
        """
        Compute emission log-likelihood parts for one trajectory, for all states:
          - Gaussian part (continuous dims) -> (T, N)
          - Bernoulli part (binary dims)    -> (T, N)  

        Parameters
        obs : np.ndarray | torch.Tensor
            Observation sequence of shape (T, obs_dim). 

        Returns
        (logp_gauss, logp_bern) each torch.Tensor of shape (T, N).
        """ 
        self._ensure_caches()

        x = obs
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, device=self._device, dtype=self._dtype)
        else:
            x = x.to(device=self._device, dtype=self._dtype)

        if x.ndim != 2 or x.shape[1] != self.obs_dim:
            raise ValueError(f"Expected obs shape (T,{self.obs_dim}), got {tuple(x.shape)}")

        # N = num_states 
        # T = time steps
        # B = bin_dim
        # Dc = self.cont_dim

        # (1) Gaussian on continuous dims with finite-masking
        if self.cont_dim > 0:
            x_cont_raw = x[:, self.cont_idx]  # (T,Dc); continuous dims
            finite = torch.isfinite(x_cont_raw)
            mask_cont = finite.to(dtype=self._dtype)
            x_cont = torch.where(finite, x_cont_raw, torch.zeros_like(x_cont_raw))
            logp_gauss = self.gauss.loglikelihood(x_cont, mask=mask_cont) # (T,N)
         
        else:
            T = x.shape[0]
            logp_gauss = torch.zeros((T, self.num_states), device=self._device, dtype=self._dtype)

        # (2) Bernoulli on bin dims
        if self.bin_dim > 0:
            xb_raw = x[:, self.bin_idx]  # (T,B)
            xb = (xb_raw > 0.5).to(dtype=self._dtype)  # treat as binary
            p = self._bern_p_t[None, :, :]             # cached Bernoulli probabilities for each joint state; (1,N,B)
            xb = xb[:, None, :]                        # (T,1,B)
            # Compute log p(x[t,b] | z) for Bernoulli:
            # log Bern(x; p) = x * log(p) + (1-x) * log(1-p)
            logp_bern_full = xb * torch.log(p) + (1.0 - xb) * torch.log(1.0 - p)  # logp_bern_full[t,z,b] = log p(x[t,b] | z); shape (T,N,B)
            logp_bern = logp_bern_full.sum(dim=2)      # logp_bern[t,z] = sum_b log p(x[t,b] | z); shape (T,N)
        else:
            logp_bern = torch.zeros_like(logp_gauss) # (T,N)

        return logp_gauss, logp_bern
    

    def loglikelihood(self, obs):
        """
        Main emission evaluator.
        logB[t,z] = log p(o_t | z).

        Parameters
        obs : np.ndarray | torch.Tensor
            Observation sequence of shape (T, obs_dim).
        
        Returns
        logB : torch.Tensor
            Log-likelihood matrix with shape (T, num_states), stored on the configured Torch device.    
        """
        logp_gauss, logp_bern = self.loglikelihood_parts(obs)
        if self.disable_discrete_obs:
            logB = logp_gauss
        else:
            w_bern = float(getattr(TRAINING_CONFIG, "bern_weight", 1.0))
            logB = logp_gauss + w_bern * logp_bern

        return logB

    # =========================================================================
    # EM M-step update
    # =========================================================================
    def update_from_posteriors(self, obs_seqs, gamma_seqs, use_progress=True, verbose=0):
        """
        M-step update for mixed emissions from responsibilities.
          - Bernoulli: masked MLE (only where values are finite; binary threshold at 0.5)
          - Gaussian: masked diagonal update (mask = isfinite for continuous dims)

        Parameters
        obs_seqs : list of np.ndarray
            List of observation sequences. Each element has shape (T_n, obs_dim).
        gamma_seqs : list of np.ndarray
            List of responsibilities. Each element has shape (T_n, num_states).
        verbose : int
            0 = no prints,
            1 = per-iteration summary,
            2 = detailed (more debug prints).
        use_progress : bool
            If True, show progress bars for the emission M-step.

        Returns
        weights_np : (N,)
            Total responsibility mass per joint state.
        """
        dtype_str = getattr(TRAINING_CONFIG, "dtype", "float32")
        dtype = torch.float32 if dtype_str == "float32" else torch.float64
        device = torch.device(getattr(TRAINING_CONFIG, "device", "cpu"))
        self.to_device(device=device, dtype=dtype)

        N = self.num_states
        # allocate accumulators
        if (not self.disable_discrete_obs) and self.bin_dim > 0:
            sum_bin = torch.zeros((N, self.bin_dim), device=device, dtype=dtype)
            sum_bin_w = torch.zeros((N, self.bin_dim), device=device, dtype=dtype)
        else:
            sum_bin = None
            sum_bin_w = None

        cont_obs_seqs = []
        cont_mask_seqs = []
        gamma_cont_seqs = []

        it = zip(obs_seqs, gamma_seqs)
        if use_progress:
            it = tqdm(it, total=len(obs_seqs), desc="M-step emissions (mixed)", leave=False)

        for obs, gamma in it:
            x = obs if torch.is_tensor(obs) else torch.as_tensor(obs, device=device, dtype=dtype)
            g = gamma if torch.is_tensor(gamma) else torch.as_tensor(gamma, device=device, dtype=dtype)
            x = x.to(device=device, dtype=dtype)
            g = g.to(device=device, dtype=dtype)

            if x.ndim != 2 or x.shape[1] != self.obs_dim:
                raise ValueError(f"Expected obs shape (T,{self.obs_dim}), got {tuple(x.shape)}")
            if g.ndim != 2 or g.shape[1] != N:
                raise ValueError(f"Expected gamma shape (T,{N}), got {tuple(g.shape)}")
            if x.shape[0] != g.shape[0]:
                raise ValueError(f"T mismatch: obs T={x.shape[0]} vs gamma T={g.shape[0]}")
            
            gamma_cont_seqs.append(g)

            # Bernoulli MLE (finite-masked)
            if (sum_bin is not None) and self.bin_dim > 0:
                xb_raw = x[:, self.bin_idx]
                finite_b = torch.isfinite(xb_raw)
                xb = (xb_raw > 0.5).to(dtype=dtype) * finite_b.to(dtype=dtype)
                m = finite_b.to(dtype=dtype)
                sum_bin += g.T @ xb
                sum_bin_w += g.T @ m

            # Continuous for Gaussian: finite-mask per entry (NaNs kept out)
            if self.cont_dim > 0:
                cont_raw = x[:, self.cont_idx]
                finite_c = torch.isfinite(cont_raw)
                cont_x = torch.where(finite_c, cont_raw, torch.zeros_like(cont_raw))
                cont_m = finite_c.to(dtype=dtype)
                cont_obs_seqs.append(cont_x)
                cont_mask_seqs.append(cont_m)
            else:
                cont_obs_seqs.append(x[:, :0])
                cont_mask_seqs.append(None)

        # Gaussian update
        weights_np = self.gauss.update_from_posteriors(
            obs_seqs=cont_obs_seqs,
            gamma_seqs=gamma_cont_seqs,
            mask_seqs=cont_mask_seqs,
            use_progress=False,
            verbose=verbose,
        )

        # Bernoulli update
        if (sum_bin is not None) and self.bin_dim > 0:
            p = sum_bin / (sum_bin_w + EPSILON)
            p = p.clamp(EPSILON, 1.0 - EPSILON)
            self.bern_p = p.detach().cpu().numpy()

        self.invalidate_cache()
        return weights_np

    # -----------------------------------------------------------------
    # Serialization helpers
    # -----------------------------------------------------------------
    def to_arrays(self):
        """
        Export mixed emission parameters as NumPy arrays in a dictionary.
        
        Returns
        payload : Dict[str, np.ndarray]
            Dictionary of NumPy arrays representing the model parameters.
        """
        g_means, g_vars = self.gauss.to_arrays()

        # backward-compatible diagonal covariance export
        S, A, D = g_vars.shape
        g_covs = np.zeros((S, A, D, D), dtype=np.float64)
        diag = np.arange(D)
        g_covs[:, :, diag, diag] = g_vars

        payload = dict(
            obs_names=np.array(self.obs_names, dtype=object),
            cont_idx=np.array(self.cont_idx, dtype=np.int64),
            bin_idx=np.array(self.bin_idx, dtype=np.int64),
            bernoulli_names=np.array(self.bernoulli_names, dtype=object),
            gauss_means=g_means,
            gauss_vars=g_vars,
            gauss_covs=g_covs,
            bern_p=np.asarray(self.bern_p, dtype=np.float64),
        )

        return payload

    def from_arrays(self, payload):
        """
        Load mixed emission parameters from a dictionary of NumPy arrays.
        
        Parameters
        payload : Dict[str, np.ndarray]
            Dictionary of NumPy arrays representing the model parameters.
        
        Raises
        ValueError
            If required keys are missing or have incorrect shapes.
        """
        self.obs_names = list(payload["obs_names"].tolist())
        self.obs_dim = len(self.obs_names)

        # Bernoulli names
        if "bernoulli_names" in payload:
            self.bernoulli_names = list(np.asarray(payload["bernoulli_names"], dtype=object).tolist())
        else:
            self.bernoulli_names = list(BERNOULLI_FEATURES)
        
        # Indices
        if "cont_idx" in payload and "bin_idx" in payload:
            self.cont_idx = list(np.asarray(payload["cont_idx"], dtype=np.int64).tolist())
            self.bin_idx = list(np.asarray(payload["bin_idx"], dtype=np.int64).tolist())
        else:
            # rebuild indices from names (best-effort)
            name_to_idx = {n: i for i, n in enumerate(self.obs_names)}
            self.bin_idx = [int(name_to_idx[n]) for n in self.bernoulli_names if n in name_to_idx]
            bin_set = set(self.bin_idx)
            self.cont_idx = [i for i in range(self.obs_dim) if i not in bin_set]

        self.cont_dim = len(self.cont_idx)
        self.bin_dim = len(self.bin_idx)

        # rebuild components
        self.gauss = GaussianEmissionModel(obs_dim=self.cont_dim)

        g_means = payload["gauss_means"]
        if "gauss_vars" in payload:
            g_second = payload["gauss_vars"]
        else:
            g_second = payload["gauss_covs"]  # legacy
        self.gauss.from_arrays(g_means, g_second)

        self.bern_p = np.asarray(payload.get("bern_p", np.full((self.num_states, self.bin_dim), 0.5)), dtype=np.float64)
        if self.bern_p.shape != (self.num_states, self.bin_dim):
            raise ValueError(f"bern_p has shape {self.bern_p.shape}, expected {(self.num_states, self.bin_dim)}")

        self.invalidate_cache()