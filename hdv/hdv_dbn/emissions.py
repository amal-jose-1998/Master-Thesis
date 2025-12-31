"""
Mixed emission model for an HMM / DBN-equivalent formulation with joint latent state z.

Observation vector o_t is partitioned into:
  (1) Continuous features  o_t[cont]  -> Multivariate Gaussian per state
  (2) Binary features      o_t[bin_d]  -> Independent Bernoulli per state (e.g., *_exists)
  (3) Categorical feature  o_t[cat_c]  -> Categorical per state (e.g., lane_pos)

Likelihood factorization (conditional independence given z):
  Factorization (conditional independence given z):
    p(o_t | z) =
        N(o_t[cont] ; mu_z, Sigma_z)
        * Π_d Bern(o_t[bin_d] ; p_zd)
        * Π_c Cat(o_t[cat_c] ; pi_zc)
"""
import math
import numpy as np
import torch
from dataclasses import dataclass
from tqdm.auto import tqdm

from .config import DBN_STATES, TRAINING_CONFIG

# ---------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------
EPSILON = 1e-8
MAX_JITTER_ATTEMPTS = 4

# =============================================================================
# Parameter containers
# =============================================================================
@dataclass(frozen=True)
class GaussianParams:
    """
    Mean/covariance for one multivariate Gaussian.

    Attributes
    mean : np.ndarray
        Mean vector of shape (D,)
    cov : np.ndarray
        Covariance matrix of shape (D,D)
    """
    mean: np.ndarray  
    cov: np.ndarray   


# =============================================================================
# Gaussian emission model (continuous features)
# =============================================================================
class GaussianEmissionModel:
    """
    Multivariate Gaussian per joint state z = (style, action).

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
                                        cov=np.eye(self.obs_dim, dtype=np.float64),
                                    )

        # Torch caches
        self._device = torch.device("cpu")
        self._dtype = torch.float32
        self._means_t = None     # (N,D), where N = num_states and D = obs_dim
        self._cov_inv_t = None   # (N,D,D)
        self._logdet_t = None    # (N,)

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
        Build caches on the target device/dtype.
        Uses Cholesky when possible; falls back to pseudo-inverse for hard cases.

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
        covs = np.zeros((N, D, D), dtype=np.float64)

        for z in range(N):
            s, a = self._z_to_sa(z)
            means[z] = self.params[s, a].mean
            covs[z] = self.params[s, a].cov

        means_t = torch.as_tensor(means, device=self._device, dtype=self._dtype)
        covs_t = torch.as_tensor(covs, device=self._device, dtype=self._dtype)

        cov_inv_t = torch.empty_like(covs_t)
        logdet_t = torch.empty((N,), device=self._device, dtype=self._dtype)

        eye = torch.eye(D, device=self._device, dtype=self._dtype)
        base_jitter = float(getattr(TRAINING_CONFIG, "emission_jitter", 1e-6))
        
        # Precompute cov_inv and logdet with jittered Cholesky
        for z in range(N):
            C = covs_t[z]
            jitter = base_jitter
            ok = False
            for _ in range(MAX_JITTER_ATTEMPTS):
                try:
                    L = torch.linalg.cholesky(C + jitter * eye)
                    cov_inv_t[z] = torch.cholesky_inverse(L)
                    logdet_t[z] = 2.0 * torch.log(torch.diagonal(L)).sum()
                    ok = True
                    break
                except Exception:
                    jitter *= 10.0

            if not ok:
                C2 = C + (1e-3 * eye)
                cov_inv_t[z] = torch.linalg.pinv(C2)
                sign, ld = torch.slogdet(C2)
                logdet_t[z] = ld if sign > 0 else torch.tensor(0.0, device=self._device, dtype=self._dtype)

        self._means_t = means_t
        self._cov_inv_t = cov_inv_t
        self._logdet_t = logdet_t

    def invalidate_cache(self):
        """
        Invalidate Torch caches. They will be rebuilt on next use.
        """
        self._means_t = None
        self._cov_inv_t = None
        self._logdet_t = None

    # =========================================================================
    # Likelihood evaluation
    # =========================================================================
    def loglik_all_states(self, obs_cont):
        """
        Vectorized evaluation of log-likelihoods for ALL states and ALL time steps.
        Computes:
            logB[t, z] = log p(o_t | z)
        for t = 0..T-1 and z = 0..N-1, where N = num_style * num_action.

        Parameters
        obs_cont : np.ndarray | torch.Tensor
            Observation sequence of shape (T, obs_dim) with continuous features only.

        Returns
        logB : torch.Tensor
            Log-likelihood matrix with shape (T, num_states), stored on the configured Torch device.
        """
        if self.obs_dim == 0:
            # no continuous dims => zero contribution
            x = obs_cont
            T = int(x.shape[0]) if hasattr(x, "shape") else 0
            return torch.zeros((T, self.num_states), device=self._device, dtype=self._dtype)

        if self._means_t is None or self._cov_inv_t is None or self._logdet_t is None:
            dtype_str = getattr(TRAINING_CONFIG, "dtype", "float32")
            dtype = torch.float32 if dtype_str == "float32" else torch.float64
            device = torch.device(getattr(TRAINING_CONFIG, "device", "cpu"))
            self.to_device(device=device, dtype=dtype)

        x = obs_cont
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, device=self._device, dtype=self._dtype)
        else:
            x = x.to(device=self._device, dtype=self._dtype)

        T, D = x.shape
        if D != self.obs_dim:
            raise ValueError(f"GaussianEmissionModel expected obs_dim={self.obs_dim}, got {D}")

        diff = x[:, None, :] - self._means_t[None, :, :]               # (T,N,D) => xt​−μz​
        tmp = torch.einsum("tnd,ndk->tnk", diff, self._cov_inv_t)      # (T,N,D)
        maha = torch.einsum("tnk,tnd->tn", tmp, diff)                  # (T,N)

        const = float(self.obs_dim) * math.log(2.0 * math.pi)
        return -0.5 * (const + self._logdet_t[None, :] + maha)  # (T,N)

    # =========================================================================
    # EM M-step update
    # =========================================================================
    def update_from_posteriors(self, obs_seqs, gamma_seqs, use_progress, verbose):
        """
        M-step update for Gaussian params using responsibilities gamma[t,z].
            gamma[t, z] = P(Z_t = z | O_1:T)
        For each state z, the updates are:
            μ_z = (∑_t gamma[t,z] o_t) / (∑_t gamma[t,z])
            Σ_z = (∑_t gamma[t,z] (o_t - μ_z)(o_t - μ_z)^T) / (∑_t gamma[t,z])

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
        weights = torch.zeros((N,), device=device, dtype=dtype) # shape (N,) total responsibility per state
        sum_x = torch.zeros((N, D), device=device, dtype=dtype) # shape (N,D) weighted sum of observations
        sum_xx = torch.zeros((N, D, D), device=device, dtype=dtype) # shape (N,D,D) weighted sum of outer products

        it = zip(obs_seqs, gamma_seqs)
        if use_progress:
            it = tqdm(it, total=len(obs_seqs), desc="M-step emissions (gauss)", leave=False)

        for obs, gamma in it:
            x = torch.as_tensor(obs, device=device, dtype=dtype)
            g = gamma if torch.is_tensor(gamma) else torch.as_tensor(gamma, device=device, dtype=dtype)
            g = g.to(device=device, dtype=dtype)

            w = g.sum(dim=0)              # (N,)
            weights += w
            sum_x += g.T @ x              # (N,D)
            sum_xx += torch.einsum("tn,td,te->nde", g, x, x)  # (N,D,D)

        mean = sum_x / (weights[:, None] + EPSILON) 
        cov = (sum_xx / (weights[:, None, None] + EPSILON)) - torch.einsum("nd,ne->nde", mean, mean) 

        min_diag = float(getattr(TRAINING_CONFIG, "min_cov_diag", 1e-5))
        cov = cov.clone()
        diag = torch.diagonal(cov, dim1=-2, dim2=-1)
        cov.diagonal(dim1=-2, dim2=-1).copy_(torch.clamp(diag, min=min_diag))
        cov = cov + (float(getattr(TRAINING_CONFIG, "emission_jitter", 1e-6)) * torch.eye(D, device=device, dtype=dtype))[None, :, :]

        mean_np = mean.detach().cpu().numpy()
        cov_np = cov.detach().cpu().numpy()

        for z in range(N):
            s, a = self._z_to_sa(z)
            self.params[s, a] = GaussianParams(mean=mean_np[z], cov=cov_np[z])

        self.invalidate_cache()
        return weights.detach().cpu().numpy()

    # =========================================================================
    # Saving / loading helpers
    # =========================================================================
    def to_arrays(self):
        """
        Export Gaussian parameters as dense NumPy arrays in (style, action) layout.

        Returns
        means : np.ndarray
            Shape (num_style, num_action, obs_dim)

        covs : np.ndarray
            Shape (num_style, num_action, obs_dim, obs_dim)
        """
        means = np.zeros((self.num_style, self.num_action, self.obs_dim), dtype=np.float64)
        covs = np.zeros((self.num_style, self.num_action, self.obs_dim, self.obs_dim), dtype=np.float64)
        for s in range(self.num_style):
            for a in range(self.num_action):
                p = self.params[s, a]
                means[s, a] = p.mean
                covs[s, a] = p.cov
        return means, covs

    def from_arrays(self, means, covs):
        """
        Load Gaussian parameters from dense arrays in (style, action) layout.

        Parameters
        means : np.ndarray
            Shape (num_style, num_action, obs_dim)

        covs : np.ndarray
            Shape (num_style, num_action, obs_dim, obs_dim)
        """
        exp_means = (self.num_style, self.num_action, self.obs_dim)
        exp_covs = (self.num_style, self.num_action, self.obs_dim, self.obs_dim)

        if means.shape != exp_means:
            raise ValueError(f"means has shape {means.shape}, expected {exp_means}")
        if covs.shape != exp_covs:
            raise ValueError(f"covs has shape {covs.shape}, expected {exp_covs}")
        
        for s in range(self.num_style):
            for a in range(self.num_action):
                self.params[s, a] = GaussianParams(
                                        mean=np.asarray(means[s, a], dtype=np.float64),
                                        cov=np.asarray(covs[s, a], dtype=np.float64),
                                    )
        self.invalidate_cache()

# =============================================================================
# Mixed emission model (continuous + Bernoulli + categorical)
# =============================================================================
class MixedEmissionModel:
    """
    Hybrid emission model over the full observation vector (T, F), split by names:

      - Continuous indices: GaussianEmissionModel
      - Binary indices (*_exists): independent Bernoulli per state
      - Categorical indices (default: lane_pos): categorical per state

    Public API:
      - loglik_all_states(obs) -> (T, N) torch tensor
      - update_from_posteriors(obs_seqs, gamma_seqs, ...) -> weights (N,)
      - to_arrays / from_arrays for checkpointing
    """

    def __init__(self, obs_names, lane_num_categories, lane_name="lane_pos", exists_suffix="_exists"):
        """
        Initialize the mixed emission model.

        Parameters
        obs_names : Sequence[str]
            List of observation feature names, in order.
        lane_num_categories : int
            Number of categories for the lane_pos categorical feature.
        lane_name : str
            Name of the categorical lane position feature.
        exists_suffix : str
            Suffix for binary existence mask features.

        Raises
        ValueError
            If lane_name is not found in obs_names.
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
        self.lane_idx = int(self.obs_names.index(lane_name)) # index of lane_pos feature

        self.bin_idx = [i for i, n in enumerate(self.obs_names) if n.endswith(exists_suffix)] # indices where name ends with _exists

        # continuous = everything except categorical and binary
        bin_set = set(self.bin_idx)
        self.cont_idx = [i for i in range(self.obs_dim) if i != self.lane_idx and i not in bin_set]
        
        # store dimensions
        self.cont_dim = len(self.cont_idx)
        self.bin_dim = len(self.bin_idx)
        self.lane_K = int(lane_num_categories) # should be 3 for valid lanes {0,1,2}
        if self.lane_K < 2:
            raise ValueError("lane_num_categories must be >= 2")

        # components
        self.gauss = GaussianEmissionModel(obs_dim=self.cont_dim)

        # initialize discrete parameters
        self.bern_p = np.full((self.num_states, self.bin_dim), 0.5, dtype=np.float64) # shape (N, bin_dim) all 0.5
        self.lane_p = np.full((self.num_states, self.lane_K), 1.0 / self.lane_K, dtype=np.float64) # shape (N, K) uniform 1/K

        # setup torch caches for discrete distributions
        self._device = torch.device("cpu")
        self._dtype = torch.float32
        self._bern_p_t = None     # (N,B), where B = bin_dim and N = num_states
        self._lane_logp_t = None  # (N,K), where K = lane_K and N = num_states

    def invalidate_cache(self):
        """Invalidate Torch caches. They will be rebuilt on next use."""
        self._bern_p_t = None
        self._lane_logp_t = None
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

        lane_p_t = torch.as_tensor(self.lane_p, device=self._device, dtype=self._dtype).clamp(EPSILON, 1.0)
        self._lane_logp_t = torch.log(lane_p_t)

    def loglik_all_states(self, obs):
        """
        This is the main emission evaluator. Compute logB[t,z] = log p(o_t | z) for all t and all states z.

        Parameters
        obs : np.ndarray | torch.Tensor
            Observation sequence of shape (T, obs_dim).
        
        Returns
        logB : torch.Tensor
            Log-likelihood matrix with shape (T, num_states), stored on the configured Torch device.    
        """
        if (
            self._lane_logp_t is None
            or (self.bin_dim > 0 and self._bern_p_t is None)
            or self.gauss._means_t is None
            or self.gauss._cov_inv_t is None
            or self.gauss._logdet_t is None
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
        logp = self.gauss.loglik_all_states(x_cont)  # (T,N) 

        # (2) Bernoulli part for *_exists
        if self.bin_dim > 0:
            xb = x[:, self.bin_idx].clamp(0.0, 1.0)          # (T,B)
            p = self._bern_p_t[None, :, :]                   # (1,N,B)
            xb = xb[:, None, :]                              # (T,1,B)
            # Compute per time/state/feature
            logp_bern = xb * torch.log(p) + (1.0 - xb) * torch.log(1.0 - p)  # (T,N,B)
            # sums over feature, adds to logp
            logp = logp + logp_bern.sum(dim=2)               # (T,N)

        # (3) Categorical lane_pos
        lane_col = x[:, self.lane_idx]                          # (T,)
        # treat non-finite values as invalid (maps to missing)
        lane_col = torch.where(torch.isfinite(lane_col), lane_col, torch.tensor(-1.0, device=x.device, dtype=x.dtype))
        
        lane_raw = lane_col.round().to(torch.long)               # (T,)
        valid = (lane_raw >= 0) & (lane_raw < self.lane_K)       # valid lane categories only
        if valid.any():
            lane_valid = lane_raw[valid]                       # (Tv,)
            logp_lane_valid = self._lane_logp_t[:, lane_valid] # (N,Tv)
            # only add for valid frames; invalid frames add 0 (uninformative)
            logp[valid] = logp[valid] + logp_lane_valid.T

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
        weights = torch.zeros((N,), device=device, dtype=dtype)
        sum_bin = torch.zeros((N, self.bin_dim), device=device, dtype=dtype) if self.bin_dim > 0 else None
        lane_counts = torch.zeros((N, self.lane_K), device=device, dtype=dtype)

        cont_obs_seqs = []

        it = zip(obs_seqs, gamma_seqs)
        if use_progress:
            it = tqdm(it, total=len(obs_seqs), desc="M-step emissions (mixed)", leave=False)

        for obs, gamma in it:
            x = torch.as_tensor(obs, device=device, dtype=dtype)
            g = gamma if torch.is_tensor(gamma) else torch.as_tensor(gamma, device=device, dtype=dtype)
            g = g.to(device=device, dtype=dtype)

            weights += g.sum(dim=0)

            if self.bin_dim > 0:
                xb = x[:, self.bin_idx].clamp(0.0, 1.0)  # (T,B)
                sum_bin += g.T @ xb                      # (N,B)

            # ignore invalid lane_pos == -1
            lane_col = x[:, self.lane_idx]
            lane_col = torch.where(torch.isfinite(lane_col), lane_col, torch.tensor(-1.0, device=x.device, dtype=x.dtype))
            lane_raw = lane_col.round().to(torch.long)
            valid = (lane_raw >= 0) & (lane_raw < self.lane_K)

            if valid.any():
                lane_valid = lane_raw[valid]                               # (Tv,)
                g_valid = g[valid]                                         # (Tv,N)
                lane_oh = torch.nn.functional.one_hot(lane_valid, num_classes=self.lane_K).to(dtype=dtype)                                          # (Tv,K)
                lane_counts += g_valid.T @ lane_oh                          # (N,K)

            if self.cont_dim > 0:
                cont_obs_seqs.append(x[:, self.cont_idx].detach().cpu().numpy())
            else:
                cont_obs_seqs.append(np.zeros((x.shape[0], 0), dtype=np.float64))

        # Gaussian update (continuous)
        weights_np = self.gauss.update_from_posteriors(
                                    obs_seqs=cont_obs_seqs,
                                    gamma_seqs=gamma_seqs,
                                    use_progress=use_progress,
                                    verbose=verbose,
                                )

        # Bernoulli update
        if self.bin_dim > 0:
            p = sum_bin / (weights[:, None] + EPSILON)  # (N,B)
            p = p.clamp(EPSILON, 1.0 - EPSILON)
            self.bern_p = p.detach().cpu().numpy()

        # Categorical update with additive smoothing
        alpha = float(getattr(TRAINING_CONFIG, "cat_alpha", 1.0))
        lane_p = (lane_counts + alpha)
        lane_p = lane_p / (lane_p.sum(dim=1, keepdim=True) + EPSILON)  # (N,K)
        self.lane_p = lane_p.detach().cpu().numpy()

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
        g_means, g_covs = self.gauss.to_arrays()
        return dict(
            obs_names=np.array(self.obs_names, dtype=object),
            lane_name=np.array([self.lane_name], dtype=object),
            lane_K=np.array([self.lane_K], dtype=np.int64),
            cont_idx=np.array(self.cont_idx, dtype=np.int64),
            bin_idx=np.array(self.bin_idx, dtype=np.int64),
            lane_idx=np.array([self.lane_idx], dtype=np.int64),
            gauss_means=g_means,
            gauss_covs=g_covs,
            bern_p=np.asarray(self.bern_p, dtype=np.float64),
            lane_p=np.asarray(self.lane_p, dtype=np.float64),
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

        self.cont_idx = list(np.asarray(payload["cont_idx"], dtype=np.int64).tolist())
        self.bin_idx = list(np.asarray(payload["bin_idx"], dtype=np.int64).tolist())
        self.lane_idx = int(np.asarray(payload["lane_idx"]).reshape(-1)[0])

        self.cont_dim = len(self.cont_idx)
        self.bin_dim = len(self.bin_idx)

        self.gauss = GaussianEmissionModel(obs_dim=self.cont_dim)
        self.gauss.from_arrays(payload["gauss_means"], payload["gauss_covs"])

        self.bern_p = np.asarray(payload["bern_p"], dtype=np.float64)
        self.lane_p = np.asarray(payload["lane_p"], dtype=np.float64)

        self.invalidate_cache()