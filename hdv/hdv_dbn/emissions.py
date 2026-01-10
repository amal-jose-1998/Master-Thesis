"""
Mixed emission model for an HMM / DBN-equivalent formulation with joint latent state z.

Observation vector o_t is partitioned into:
  (1) Continuous features  o_t[cont]  -> Diagonal Gaussian per state (with optional gating masks)
  (2) Binary features      o_t[bin_d] -> Independent Bernoulli per state (e.g., *_exists)
  (3) Categorical features  o_t[cat_c] -> Categorical per state (e.g., lane_pos, lc)

Likelihood factorization (conditional independence given z):
  Factorization (conditional independence given z):
    p(o_t | z) =
        N_diag(o_t[cont] ; mu_z, diag(var_z))   [with gating for missing relative dims]
        * Π_d Bern(o_t[bin_d] ; p_zd)
        * Π_c Cat(o_t[cat_c] ; pi_zc)
"""
import math
import numpy as np
import torch
from dataclasses import dataclass
from tqdm.auto import tqdm

from .config import DBN_STATES, TRAINING_CONFIG, CATEGORICAL_FEATURE_SIZES

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

        Parameters
        z : int
            Joint state index in [0, num_states).

        Returns
        (style_idx, action_idx) : Tuple[int, int]
        """
        s = int(z // self.num_action)
        a = int(z % self.num_action)
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
    def loglik_all_states(self, obs_cont, mask=None):
        """
        Vectorized evaluation of log-likelihoods for ALL states and ALL time steps.
        Computes:
            logB[t, z] = log p(o_t | z)
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
            # no continuous dims => zero contribution
            x = obs_cont
            T = int(x.shape[0]) if hasattr(x, "shape") else 0
            return torch.zeros((T, self.num_states), device=self._device, dtype=self._dtype)

        if self._means_t is None or self._inv_var_t is None or self._log_var_t is None:
            dtype_str = getattr(TRAINING_CONFIG, "dtype", "float32")
            dtype = torch.float32 if dtype_str == "float32" else torch.float64
            device = torch.device(getattr(TRAINING_CONFIG, "device", "cpu"))
            self.to_device(device=device, dtype=dtype)

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
            m = (m > 0.5).to(dtype=self._dtype)

        # log N(x; mu, var) = -0.5 * [log(2π) + log(var) + (x-mu)^2 / var]
        diff = x[:, None, :] - self._means_t[None, :, :]            # (T,N,D)
        quad = (diff * diff) * self._inv_var_t[None, :, :]          # (T,N,D)
        per_dim = -0.5 * (math.log(2.0 * math.pi) + self._log_var_t[None, :, :] + quad)  # (T,N,D)

        if m is not None:
            per_dim = per_dim * m[:, None, :]

        return per_dim.sum(dim=2)  # (T,N)

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

            # state mass
            w = g.sum(dim=0)  # (N,)
            weights_z += w

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
        mass_ok = (weights_z >= min_mass)[:, None]          # (N,1) boolean
        obs_ok = (weights_zd > 1.0)                         # (N,D) boolean
        upd = mass_ok & obs_ok                              # (N,D)

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

      - Continuous indices: GaussianEmissionModel (diagonal) with gating for relative dims
      - Binary indices (*_exists): independent Bernoulli per state
      - Categorical indices (default: lane_pos): categorical per state

    Public API:
      - loglik_all_states(obs) -> (T, N) torch tensor
      - update_from_posteriors(obs_seqs, gamma_seqs, ...) -> weights (N,)
      - to_arrays / from_arrays for checkpointing
    """

    def __init__(self, obs_names, lane_name="lane_pos", lc_name = "lc", exists_suffix="_exists"):
        """
        Initialize the mixed emission model.

        Parameters
        obs_names : Sequence[str]
            List of observation feature names, in order.
        lane_name : str
            Name of the categorical lane position feature.
        lc_name : str
            Name of the categorical lane change feature.
        exists_suffix : str
            Suffix for binary existence mask features.

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

        if lane_name not in self.obs_names:
            raise ValueError(f"Categorical feature '{lane_name}' not found in obs_names.")

        self.lane_name = lane_name
        self.lane_idx = int(self.obs_names.index(lane_name))  # lane_pos index

        self.lc_name = lc_name
        if self.lc_name not in self.obs_names:
            raise ValueError(f"Categorical feature '{self.lc_name}' not found in obs_names.")
        self.lc_idx = int(self.obs_names.index(self.lc_name))

        self.exists_suffix = exists_suffix  
        use_exists_bern = bool(getattr(TRAINING_CONFIG, "exists_as_bernoulli", True)) 

        all_exists_idx = [i for i, n in enumerate(self.obs_names) if n.endswith(exists_suffix)]
        self.bin_idx = all_exists_idx if use_exists_bern else []  
        bin_set = set(self.bin_idx)

        # continuous = everything except categorical and binary
        self.cont_idx = [
                            i for i in range(self.obs_dim)
                            if (i != self.lane_idx) and (i != self.lc_idx) and (i not in bin_set)
                        ]
        
        # Gating map for neighbor-relative continuous features 
        # Any continuous feature ending with _dx/_dy/_dvx/_dvy will be gated by its corresponding *_exists
        rel_suffixes = ("_dx", "_dy", "_dvx", "_dvy")
        name_to_idx = {n: i for i, n in enumerate(self.obs_names)}
        self._cont_rel_gate = []  # list of (cont_local_idx, exists_global_idx)
        for local_j, global_j in enumerate(self.cont_idx):
            fname = self.obs_names[global_j]
            for suf in rel_suffixes:
                if fname.endswith(suf):
                    prefix = fname[: -len(suf)]
                    ex_name = f"{prefix}_exists"
                    if ex_name in name_to_idx:
                        self._cont_rel_gate.append((local_j, int(name_to_idx[ex_name])))
                    break

        # store dimensions
        self.cont_dim = len(self.cont_idx)
        self.bin_dim = len(self.bin_idx)
        self.lane_K = int(CATEGORICAL_FEATURE_SIZES.get("lane_pos", 5)) # lane_pos has 5 categories (0,1,2,3,4)
        if self.lane_K < 2:
            raise ValueError("lane_num_categories must be >= 2")
        self.lc_K = int(CATEGORICAL_FEATURE_SIZES.get("lc", 3)) # lc has 3 categories (-1,0,+1 mapped to 0,1,2)

        # components
        self.gauss = GaussianEmissionModel(obs_dim=self.cont_dim)

        # initialize discrete parameters
        self.bern_p = np.full((self.num_states, self.bin_dim), 0.5, dtype=np.float64) # shape (N, bin_dim) all 0.5
        self.lane_p = np.full((self.num_states, self.lane_K), 1.0 / self.lane_K, dtype=np.float64) # shape (N, lane_K) uniform 1/lane_K
        self.lc_p   = np.full((self.num_states, self.lc_K),   1.0 / self.lc_K,   dtype=np.float64) # shape (N, lc_K) uniform 1/lc_K


        # setup torch caches for discrete distributions
        self._device = torch.device("cpu")
        self._dtype = torch.float32
        self._bern_p_t = None     # (N,B), where B = bin_dim and N = num_states
        self._lane_logp_t = None  # (N,K), where K = lane_K and N = num_states
        self._lc_logp_t = None    # (N,K), where K = lc_K and N = num_states

    def invalidate_cache(self):
        """Invalidate Torch caches. They will be rebuilt on next use."""
        self._bern_p_t = None
        self._lane_logp_t = None
        self._lc_logp_t = None 
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

        lane_p_t = torch.as_tensor(self.lane_p, device=self._device, dtype=self._dtype).clamp(EPSILON, 1.0 - EPSILON)
        self._lane_logp_t = torch.log(lane_p_t)
        lc_p_t = torch.as_tensor(self.lc_p, device=self._device, dtype=self._dtype).clamp(EPSILON, 1.0 - EPSILON)
        self._lc_logp_t = torch.log(lc_p_t)

    def loglik_parts_all_states(self, obs): 
        """
        Compute emission log-likelihood parts for one trajectory, for all states:
          - Gaussian part (continuous dims) -> (T, N)
          - Bernoulli part (binary dims)    -> (T, N)   
          - Categorical part                -> (T, N)   

        Returns
        (logp_gauss, logp_bern, logp_lane, logp_lc) each torch.Tensor of shape (T, N).
        """ 
        if (
            self._lane_logp_t is None
            or self._lc_logp_t is None
            or (self.bin_dim > 0 and self._bern_p_t is None)
            or self.gauss._means_t is None
            or self.gauss._inv_var_t is None
            or self.gauss._log_var_t is None
        ):
            dtype_str = getattr(TRAINING_CONFIG, "dtype", "float32")
            dtype = torch.float32 if dtype_str == "float32" else torch.float64
            device = torch.device(getattr(TRAINING_CONFIG, "device", "cpu"))
            self.to_device(device=device, dtype=dtype)

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
        # Tv = valid time steps for lane_pos

        # (1) Gaussian part
        x_cont = x[:, self.cont_idx] if self.cont_dim > 0 else x[:, :0] # (T,cont_dim)
        if self.cont_dim > 0 and self._cont_rel_gate:
            mask_cont = torch.ones((x.shape[0], self.cont_dim), device=x.device, dtype=self._dtype)
            for local_j, ex_idx in self._cont_rel_gate:
                mask_cont[:, local_j] = (x[:, ex_idx] > 0.5).to(dtype=self._dtype)
        else:
            mask_cont = None
        logp_gauss = self.gauss.loglik_all_states(x_cont, mask=mask_cont)  # (T,N) 

        # (2) Bernoulli part (summed over binary features)
        if self.bin_dim > 0:
            xb = x[:, self.bin_idx].clamp(0.0, 1.0)       # (T,B)
            p = self._bern_p_t[None, :, :]                # (1,N,B)
            xb = xb[:, None, :]                           # (T,1,B)
            logp_bern_full = xb * torch.log(p) + (1.0 - xb) * torch.log(1.0 - p)  # (T,N,B)
            logp_bern = logp_bern_full.sum(dim=2)         # (T,N)
        else:
            logp_bern = torch.zeros_like(logp_gauss)

        # (3) Lane categorical part (0 for invalid frames)
        logp_lane = torch.zeros_like(logp_gauss)
        lane_col = x[:, self.lane_idx]  # (T,)
        lane_col = torch.where(torch.isfinite(lane_col), lane_col, torch.tensor(-1.0, device=x.device, dtype=x.dtype))
        lane_raw = lane_col.round().to(torch.long)               # (T,)
        valid = (lane_raw >= 0) & (lane_raw < self.lane_K) # valid lane categories only
        if valid.any():
            lane_valid = lane_raw[valid]                        # (Tv,)
            logp_lane_valid = self._lane_logp_t[:, lane_valid]  # (N,Tv)
            logp_lane[valid] = logp_lane_valid.T                # (Tv,N)

        # (4) Lane-change categorical part (0 for invalid frames)
        logp_lc = torch.zeros_like(logp_gauss)
        lc_col = x[:, self.lc_idx]  # (T,)
        lc_col = torch.where(torch.isfinite(lc_col), lc_col, torch.tensor(-99.0, device=x.device, dtype=x.dtype))
        lc_raw = lc_col.round().to(torch.long)  # expected values: -1, 0, +1
        # map {-1,0,+1} -> {0,1,2}
        lc_mapped = lc_raw + 1
        valid_lc = (lc_mapped >= 0) & (lc_mapped < self.lc_K)
        if valid_lc.any():
            lc_valid = lc_mapped[valid_lc]             # (Tv,)
            logp_lc_valid = self._lc_logp_t[:, lc_valid]  # (N,Tv)
            logp_lc[valid_lc] = logp_lc_valid.T        # (Tv,N)

        return logp_gauss, logp_bern, logp_lane, logp_lc

    def loglik_all_states(self, obs):
        """
        This is the main emission evaluator. Compute 
        logB[t,z] = log p(o_t | z) = logp_gauss + logp_bern + logp_lane 
        for all t and all states z.

        Parameters
        obs : np.ndarray | torch.Tensor
            Observation sequence of shape (T, obs_dim).
        
        Returns
        logB : torch.Tensor
            Log-likelihood matrix with shape (T, num_states), stored on the configured Torch device.    
        """
        logp_gauss, logp_bern, logp_lane, logp_lc = self.loglik_parts_all_states(obs) 
        w_bern = float(getattr(TRAINING_CONFIG, "bern_weight", 1.0))
        w_lane = float(getattr(TRAINING_CONFIG, "lane_weight", 1.0))
        w_lc   = float(getattr(TRAINING_CONFIG, "lc_weight", 1.0))
        logp = logp_gauss + w_bern * logp_bern + w_lane * logp_lane + w_lc * logp_lc
        return logp

    def update_from_posteriors(self, obs_seqs, gamma_seqs, use_progress, verbose):
        """
        M-step update for mixed emissions from responsibilities.

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
        sum_bin = torch.zeros((N, self.bin_dim), device=device, dtype=dtype) if self.bin_dim > 0 else None # for Bernoulli MLE
        lane_counts = torch.zeros((N, self.lane_K), device=device, dtype=dtype) # for categorical MLE
        lc_counts   = torch.zeros((N, self.lc_K),   device=device, dtype=dtype)

        cont_obs_seqs = []
        cont_mask_seqs = []
        gamma_cont_seqs = []

        it = zip(obs_seqs, gamma_seqs)
        if use_progress:
            it = tqdm(it, total=len(obs_seqs), desc="M-step emissions (mixed)", leave=False)

        for obs, gamma in it:
            x = torch.as_tensor(obs, device=device, dtype=dtype)
            g = gamma if torch.is_tensor(gamma) else torch.as_tensor(gamma, device=device, dtype=dtype)
            g = g.to(device=device, dtype=dtype)
            gamma_cont_seqs.append(g) 

            # Bernoulli update
            if self.bin_dim > 0:
                xb = (x[:, self.bin_idx] > 0.5).to(dtype=dtype)
                sum_bin += g.T @ xb                      # (N,B)

            # Categorical update for lane_pos (ignore invalid lane_pos == -1)
            lane_col = x[:, self.lane_idx]
            lane_col = torch.where(torch.isfinite(lane_col), lane_col, torch.tensor(-1.0, device=x.device, dtype=x.dtype))
            lane_raw = lane_col.round().to(torch.long)
            valid = (lane_raw >= 0) & (lane_raw < self.lane_K)
            if valid.any():
                lane_valid = lane_raw[valid]                               # (Tv,)
                g_valid = g[valid]                                         # (Tv,N)
                lane_oh = torch.nn.functional.one_hot(lane_valid, num_classes=self.lane_K).to(dtype=dtype)    # (Tv,K)
                lane_counts += g_valid.T @ lane_oh                          # (N,K)
            
            # Categorical update for lc (ignore invalid / NaNs)
            lc_col = x[:, self.lc_idx]
            lc_col = torch.where(torch.isfinite(lc_col), lc_col, torch.tensor(-99.0, device=x.device, dtype=x.dtype))
            lc_raw = lc_col.round().to(torch.long)      # {-1,0,+1}
            lc_mapped = lc_raw + 1                      # -> {0,1,2}
            valid_lc = (lc_mapped >= 0) & (lc_mapped < self.lc_K)
            if valid_lc.any():
                lc_valid = lc_mapped[valid_lc]  # (Tv,)
                g_valid = g[valid_lc]           # (Tv,N)
                lc_oh = torch.nn.functional.one_hot(lc_valid, num_classes=self.lc_K).to(dtype=dtype)  # (Tv,K)
                lc_counts += g_valid.T @ lc_oh
            
            # Continuous + mask for gated diagonal Gaussian
            if self.cont_dim > 0:
                cont_x = x[:, self.cont_idx].detach().cpu().numpy()  
                cont_obs_seqs.append(cont_x)

                if self._cont_rel_gate:
                    m = torch.ones((x.shape[0], self.cont_dim), device=device, dtype=dtype)
                    for local_j, ex_idx in self._cont_rel_gate:
                        m[:, local_j] = (x[:, ex_idx] > 0.5).to(dtype=dtype)
                    cont_mask_seqs.append(m.detach().cpu().numpy())
                else:
                    cont_mask_seqs.append(None)
            else:
                cont_obs_seqs.append(np.zeros((x.shape[0], 0), dtype=np.float64))
                cont_mask_seqs.append(None)

        # Gaussian update (continuous, diagonal, gated)
        weights_np = self.gauss.update_from_posteriors(
            obs_seqs=cont_obs_seqs,
            gamma_seqs=gamma_cont_seqs,
            mask_seqs=cont_mask_seqs,
            use_progress=False,
            verbose=verbose,
        )
        
        weights_t = torch.as_tensor(weights_np, device=device, dtype=dtype)  # (N,)

        # Bernoulli MLE
        if self.bin_dim > 0:
            p = sum_bin / (weights_t[:, None] + EPSILON)
            p = p.clamp(EPSILON, 1.0 - EPSILON)
            self.bern_p = p.detach().cpu().numpy()

        # Categorical (Dirichlet smoothing)
        alpha = float(getattr(TRAINING_CONFIG, "cat_alpha", 1.0))
        lane_p = (lane_counts + alpha)
        lane_p = lane_p / (lane_p.sum(dim=1, keepdim=True) + EPSILON)
        self.lane_p = lane_p.detach().cpu().numpy()
        lc_p = (lc_counts + alpha)
        lc_p = lc_p / (lc_p.sum(dim=1, keepdim=True) + EPSILON)
        self.lc_p = lc_p.detach().cpu().numpy()

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

        # Backward-compatible diagonal covariance export
        S, A, D = g_vars.shape
        g_covs = np.zeros((S, A, D, D), dtype=np.float64)
        diag_idx = np.arange(D)
        g_covs[:, :, diag_idx, diag_idx] = g_vars
        
        return dict(
            obs_names=np.array(self.obs_names, dtype=object),
            lane_name=np.array([self.lane_name], dtype=object),
            lane_K=np.array([self.lane_K], dtype=np.int64),
            lc_name=np.array([self.lc_name], dtype=object),
            lc_K=np.array([self.lc_K], dtype=np.int64),
            cont_idx=np.array(self.cont_idx, dtype=np.int64),
            bin_idx=np.array(self.bin_idx, dtype=np.int64),
            lane_idx=np.array([self.lane_idx], dtype=np.int64),
            lc_idx=np.array([self.lc_idx], dtype=np.int64),
            gauss_means=g_means,
            gauss_vars=g_vars,
            gauss_covs=g_covs,
            bern_p=np.asarray(self.bern_p, dtype=np.float64),
            lane_p=np.asarray(self.lane_p, dtype=np.float64),
            lc_p=np.asarray(self.lc_p, dtype=np.float64),
        )

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

        self.lane_name = str(np.asarray(payload.get("lane_name", np.array(["lane_pos"], dtype=object))).reshape(-1)[0])
        self.lane_K = int(np.asarray(payload["lane_K"]).reshape(-1)[0])

        self.lc_name = str(np.asarray(payload.get("lc_name", np.array(["lc"], dtype=object))).reshape(-1)[0])
        self.lc_K = int(np.asarray(payload.get("lc_K", np.array([3], dtype=np.int64))).reshape(-1)[0])
        
        self.cont_idx = list(np.asarray(payload["cont_idx"], dtype=np.int64).tolist())
        self.bin_idx = list(np.asarray(payload["bin_idx"], dtype=np.int64).tolist())
        self.lane_idx = int(np.asarray(payload["lane_idx"]).reshape(-1)[0])
        self.lc_idx = int(np.asarray(payload.get("lc_idx", np.array([self.obs_names.index("lc")], dtype=np.int64))).reshape(-1)[0])

        self.cont_dim = len(self.cont_idx)
        self.bin_dim = len(self.bin_idx)

        # rebuild gating map
        rel_suffixes = ("_dx", "_dy", "_dvx", "_dvy")
        name_to_idx = {n: i for i, n in enumerate(self.obs_names)}
        self._cont_rel_gate = []
        for local_j, global_j in enumerate(self.cont_idx):
            fname = self.obs_names[global_j]
            for suf in rel_suffixes:
                if fname.endswith(suf):
                    prefix = fname[: -len(suf)]
                    ex_name = f"{prefix}_exists"
                    if ex_name in name_to_idx:
                        self._cont_rel_gate.append((local_j, int(name_to_idx[ex_name])))
                    break

        self.gauss = GaussianEmissionModel(obs_dim=self.cont_dim)

        g_means = payload["gauss_means"]
        if "gauss_vars" in payload:
            g_second = payload["gauss_vars"]
        else:
            g_second = payload["gauss_covs"]  # legacy checkpoints
        self.gauss.from_arrays(g_means, g_second)

        self.bern_p = np.asarray(payload["bern_p"], dtype=np.float64)
        
        self.lane_p = np.asarray(payload["lane_p"], dtype=np.float64)
        if self.lane_p.shape != (self.num_states, self.lane_K):
            raise ValueError(f"lane_p has shape {self.lane_p.shape}, expected {(self.num_states, self.lane_K)}")
        
        self.lc_p = np.asarray(
            payload.get("lc_p", np.full((self.num_states, self.lc_K), 1.0 / self.lc_K)),
            dtype=np.float64,
        )
        if self.lc_p.shape != (self.num_states, self.lc_K):
            raise ValueError(f"lc_p has shape {self.lc_p.shape}, expected {(self.num_states, self.lc_K)}")

        self.invalidate_cache()