"""
Gaussian emission model for the HDV DBN / HMM-equivalent formulation.
This module implements continuous emissions of the form:
    p(o_t | z_t) = N(o_t ; μ_z, Σ_z)

where the joint latent state is:
    z = (Style, Action)

Key design goals
1) Numerical stability:
   - Work in log-domain when evaluating likelihoods.
   - Ensure covariances remain positive definite via diagonal floors and jitter.

2) Performance:
   - Provide a vectorized method to compute log-likelihoods for *all* states z
     for an entire trajectory in one call:
         logB = loglik_all_states(obs)   # shape (T, num_states)
"""

import math
import numpy as np
import torch
from dataclasses import dataclass
from tqdm.auto import tqdm

from .config import DBN_STATES, TRAINING_CONFIG

# Numerical stability constants
EPSILON = 1e-6
MAX_JITTER_ATTEMPTS = 4


@dataclass
class GaussianEmissionParams:
    """
    Parameters of one multivariate Gaussian emission distribution.
    Each joint latent state z = (style, action) has one Gaussian:
        o ~ N(mean, cov)

    Attributes
    mean : np.ndarray
        Mean vector of the Gaussian distribution with shape ``(obs_dim,)``.
    cov : np.ndarray
        Covariance matrix of the Gaussian distribution with shape ``(obs_dim, obs_dim)``.
        This matrix is expected to be symmetric positive definite.
    """
    mean: np.ndarray      # shape (obs_dim,)
    cov: np.ndarray       # shape (obs_dim, obs_dim)


class GaussianEmissionModel:
    """
    Multivariate Gaussian emission model p(o_t | Style_t, Action_t).
    The model maintains one Gaussian per (style, action) pair:
        z = (s, a), s in styles, a in actions
        p(o | z) = N(o ; μ_{s,a}, Σ_{s,a})
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

        # Parameter table: one Gaussian per (style, action) pair
        self.params = np.empty((self.num_style, self.num_action), dtype=object) # Shape (num_style, num_action)
        for s in range(self.num_style):
            for a in range(self.num_action):
                self.params[s, a] = GaussianEmissionParams(
                    mean=np.zeros(self.obs_dim, dtype=np.float64),
                    cov=np.eye(self.obs_dim, dtype=np.float64)
                )

        # Torch execution configuration (set lazily)
        self._device: torch.device = torch.device("cpu")
        self._dtype: torch.dtype = torch.float32

        # Cached torch tensors for fast likelihood evaluation
        self._means_t = None     # (N, D)
        self._covs_t = None      # (N, D, D)
        self._chol_t = None      # (N, D, D) lower-triangular
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
        s = z // self.num_action
        a = z % self.num_action
        return (s, a)

    def _sa_to_z(self, style_idx, action_idx):
        """
        Map (style_idx, action_idx) into flattened joint index z.

        Parameters
        style_idx : int 
            Index of the style state.
        action_idx : int
            Index of the action state.

        Returns
        z : int
            Joint state index.
        """
        return style_idx * self.num_action + action_idx
    
    # =========================================================================
    # Torch device / cache management
    # =========================================================================
    def to_device(self, device, dtype):
        """
        Move emission parameter caches to a Torch device and dtype.

        This does not change the canonical storage (self.params), which remains
        as NumPy arrays for easy saving/loading. Instead, it creates cached Torch
        tensors used for fast GPU evaluation.

        Parameters
        device : str | torch.device
            Torch device (e.g., "cuda", "cpu").

        dtype : torch.dtype
            Floating-point dtype.
        """
        self._device = torch.device(device)
        self._dtype = dtype

        means_np, covs_np = self.to_arrays_flat()
        self._means_t = torch.as_tensor(means_np, device=self._device, dtype=self._dtype)   # (N,D)
        self._covs_t = torch.as_tensor(covs_np, device=self._device, dtype=self._dtype)     # (N,D,D)

        self._precompute_cache()

    def _precompute_cache(self):
        """
        Precompute per-state Cholesky factors and log determinants.

        This is essential for fast repeated evaluation of log-likelihoods during
        the E-step, where the same covariances are used for many observations.

        The method:
        - applies a diagonal floor
        - adds increasing jitter if Cholesky fails
        - stores:
            chol[z]  = L  such that Σ = L L^T
            logdet[z] = log |Σ|
        """
        assert self._covs_t is not None, "Call to_device() before precomputing cache."
        covs = self._covs_t # Shape: (num_states, obs_dim, obs_dim)
        N, D, _ = covs.shape # Each covs[z] is a D×D matrix

        # Diagonal floor
        min_diag = float(getattr(TRAINING_CONFIG, "min_cov_diag", 1e-5))
        covs = covs.clone()
        diag = torch.diagonal(covs, dim1=-2, dim2=-1)
        covs.diagonal(dim1=-2, dim2=-1).copy_(torch.clamp(diag, min=min_diag))

        # Cholesky with jitter
        base_jitter = float(getattr(TRAINING_CONFIG, "jitter", 1e-6))
        eye = torch.eye(D, device=covs.device, dtype=covs.dtype).unsqueeze(0)  # (1,D,D)

        chol = None
        for k in range(MAX_JITTER_ATTEMPTS):
            jitter = base_jitter * (10.0 ** k)
            try:
                chol = torch.linalg.cholesky(covs + jitter * eye)
                break
            except RuntimeError:
                chol = None

        if chol is None:
            # Final attempt with large jitter
            chol = torch.linalg.cholesky(covs + (base_jitter * (10.0 ** MAX_JITTER_ATTEMPTS)) * eye)

        logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)  # (N,)

        self._covs_t = covs
        self._chol_t = chol # Cholesky factor
        self._logdet_t = logdet # Log determinant

    def invalidate_cache(self):
        """
        Invalidate Torch caches.
        Call this after any direct modifications to self.params to force the model
        to rebuild device tensors and cached Cholesky factors.
        """
        self._means_t = None
        self._covs_t = None
        self._chol_t = None
        self._logdet_t = None

    # =========================================================================
    # Likelihood evaluation
    # =========================================================================
    def log_likelihood(self, obs, style_idx, action_idx):
        """
        Compute the log-likelihood of a single observation vector.
        Specifically, this evaluates: 
            log p(obs | Style=style_idx, Action=action_idx) 
        using the corresponding multivariate Gaussian parameters.

        This is the original scalar evaluation method. It is kept for backwards
        compatibility but is NOT recommended inside (t,z) Python loops.
        
        Parameters
        obs : np.ndarray
            Observation vector with shape ``(obs_dim,)``.
        style_idx : int
            Index of the style state.
        action_idx : int
            Index of the action state.

        Returns
        float
            Log-likelihood under the selected Gaussian.
        """
        p = self.params[style_idx, action_idx] 
        x = obs - p.mean 
        d = self.obs_dim

        base_cov = p.cov
        cov = base_cov
        success = False

        for attempt in range(MAX_JITTER_ATTEMPTS):
            try:
                sign, logdet = np.linalg.slogdet(cov)
                if sign <= 0:
                    raise np.linalg.LinAlgError("Covariance matrix not positive definite.")
                sol = np.linalg.solve(cov, x)
                success = True
                # If we had to add jitter (attempt > 0), store the stabilised cov
                if attempt > 0:
                    p.cov = cov
                break
            except np.linalg.LinAlgError:
                # Increase diagonal jitter: 1e-6, 1e-5, 1e-4, 1e-3
                jitter = 10.0 ** (-6 + attempt)
                cov = base_cov + jitter * np.eye(d)
        if not success:
            # Extreme fallback: use identity covariance
            cov = np.eye(d)
            p.cov = cov
            sign, logdet = 1.0, 0.0  # det(I) = 1 => logdet = 0
            sol = x  # solving I * sol = x gives sol = x

        quad = np.dot(x, sol)
        log_likelihood = -0.5 * (d * np.log(2 * np.pi) + logdet + quad)
        
        if np.isnan(log_likelihood):
            print(
                "[GaussianEmissionModel] WARNING: NaN log-likelihood encountered, "
                "falling back to large negative value (-1e10)."
            )
            log_likelihood = -1e10
        
        return log_likelihood 
    
    def loglik_all_states(self, obs_seq):
        """
        Vectorized evaluation of log-likelihoods for ALL states and ALL time steps.
        Computes:
            logB[t, z] = log p(o_t | z)
        for t = 0..T-1 and z = 0..N-1, where N = num_style * num_action.

        Parameters
        obs_seq : np.ndarray | torch.Tensor
            Observation sequence of shape (T, obs_dim).

        Returns
        logB : torch.Tensor
            Log-likelihood matrix with shape (T, num_states), stored on the configured Torch device.
        """
        if self._means_t is None or self._chol_t is None or self._logdet_t is None:
            dtype_str = getattr(TRAINING_CONFIG, "dtype", "float32")
            dtype = torch.float32 if dtype_str == "float32" else torch.float64
            device = getattr(TRAINING_CONFIG, "device", "cpu")
            self.to_device(device=device, dtype=dtype)

        assert self._means_t is not None and self._chol_t is not None and self._logdet_t is not None

        if torch.is_tensor(obs_seq):
            x = obs_seq.to(device=self._device, dtype=self._dtype)
        else:
            x = torch.as_tensor(obs_seq, device=self._device, dtype=self._dtype)

        if torch.isnan(x).any():
            raise ValueError("NaN detected in obs_seq passed to GaussianEmissionModel.loglik_all_states(). "
                            "Fill missing context values (e.g., dx/dvx) with 0 using masks before training/eval.")

        # Shapes:
        # x: (T, D)
        # means: (N, D)
        # chol: (N, D, D)
        T, D = x.shape
        N = self.num_states

        means = self._means_t
        chol = self._chol_t
        logdet = self._logdet_t

        # diff: (T, N, D)
        diff = x[:, None, :] - means[None, :, :]

        # Solve L y = diff^T per state (batched triangular solve)
        # rhs: (N, D, T)
        rhs = diff.permute(1, 2, 0) # it is the right hand side of the equation 
        y = torch.linalg.solve_triangular(chol, rhs, upper=False) # it solves for y in L y = diff^T

        # mahalanobis: (T, N)
        maha = (y ** 2).sum(dim=1).transpose(0, 1)

        const = D * math.log(2.0 * math.pi)
        logB = -0.5 * (const + logdet[None, :] + maha)
        return logB

    # =========================================================================
    # EM M-step update
    # =========================================================================
    def update_from_posteriors(self, obs_seqs, gamma_seqs, use_progress, verbose):
        """
        Update Gaussian emission parameters using posterior state probabilities.
        This method implements the emission M-step given posterior probabilities:
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
        obs_dim = self.obs_dim
        num_states = self.num_states

        dtype_str = getattr(TRAINING_CONFIG, "dtype", "float32")
        dtype = torch.float32 if dtype_str == "float32" else torch.float64
        device = torch.device(getattr(TRAINING_CONFIG, "device", "cpu"))
        self.to_device(device=device, dtype=dtype)

        # init accumulators
        weights = torch.zeros((num_states,), device=device, dtype=dtype)           # (N,)
        sum_x = torch.zeros((num_states, obs_dim), device=device, dtype=dtype)    # (N,D)
        sum_xx = torch.zeros((num_states, obs_dim, obs_dim), device=device, dtype=dtype)  # (N,D,D)

        iterator = zip(obs_seqs, gamma_seqs)
        if use_progress:
            iterator = tqdm(iterator, total=len(obs_seqs), desc="M-step emissions (accumulate)", leave=False)

        for obs, gamma in iterator: 
            # obs: (T,D) on GPU
            x = torch.as_tensor(obs, device=device, dtype=dtype)
            # gamma: (T,N) on GPU
            if torch.is_tensor(gamma):
                g = gamma.to(device=device, dtype=dtype)
            else:
                g = torch.as_tensor(gamma, device=device, dtype=dtype)
            T_n = x.shape[0]
            if g.shape != (T_n, num_states):
                raise ValueError(f"gamma shape {tuple(g.shape)} does not match (T_n,num_states)=({T_n},{num_states}).")

            # weights[z] += sum_t g[t,z]
            weights += g.sum(dim=0)

            # sum_x[z,:] += sum_t g[t,z] * x[t,:]
            sum_x += g.transpose(0, 1) @ x  # (N,T)@(T,D)=(N,D)

            # sum_xx[z,:,:] += sum_t g[t,z] * x[t]x[t]^T
            outer_t = torch.einsum("ti,tj->tij", x, x)       # (T,D,D)
            sum_xx += torch.einsum("tz,tij->zij", g, outer_t)
        
        # Compute means and covariances
        eps = EPSILON
        mean = sum_x / (weights[:, None] + eps)  # (N,D)

        cov = sum_xx / (weights[:, None, None] + eps) - torch.einsum("ni,nj->nij", mean, mean)

        # Diagonal floor + small jitter
        min_diag = float(getattr(TRAINING_CONFIG, "min_cov_diag", 1e-5))
        diag = torch.diagonal(cov, dim1=-2, dim2=-1)
        cov = cov.clone()
        cov.diagonal(dim1=-2, dim2=-1).copy_(torch.clamp(diag, min=min_diag))
        jitter = float(getattr(TRAINING_CONFIG, "emission_jitter", 1e-6))
        cov = cov + jitter * torch.eye(obs_dim, device=device, dtype=dtype).unsqueeze(0)

        # Write back to self.params in (style, action) layout on CPU as NumPy
        mean_np = mean.detach().cpu().numpy()
        cov_np = cov.detach().cpu().numpy()

        for z in range(num_states):
            s, a = self._z_to_sa(z)
            if mean_np[z].shape != (obs_dim,) or cov_np[z].shape != (obs_dim, obs_dim):
                raise RuntimeError("Internal shape error in emission parameter update.")
            self.params[s, a] = GaussianEmissionParams(mean=mean_np[z], cov=cov_np[z])

        # Refresh caches for the next E-step
        self.to_device(device=device, dtype=dtype)

        weights_np = weights.detach().cpu().numpy()
        total_weight = float(weights_np.sum())

        if verbose >= 1:
            print(f"  [GaussianEmissionModel] Emission update done. Total responsibility mass = {total_weight:.3e}")
            if total_weight > 0.0:
                frac = weights_np / total_weight
                print("     Responsibility mass per joint state:")
                for z, (w, f) in enumerate(zip(weights_np, frac)):
                    s, a = self._z_to_sa(z)
                    print(f"      z={z:02d} (s={s}, a={a}) mass={w:.0f}  frac={f:.4f}")

        if verbose >= 2 and total_weight > 0.0:
            print("     Example means for first few states:")
            shown = 0
            for z in range(num_states):
                s, a = self._z_to_sa(z)
                m = self.params[s, a].mean
                print(f"        (s={s}, a={a}) mean[:3] = {m[:3]}")
                shown += 1
                if shown >= 3:
                    break

        return weights_np

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

    def to_arrays_flat(self):
        """
        Export Gaussian parameters in flattened joint-state layout.

        Returns
        means_flat : np.ndarray
            Shape (num_states, obs_dim), where z = s*num_action + a

        covs_flat : np.ndarray
            Shape (num_states, obs_dim, obs_dim)
        """
        means_sa, covs_sa = self.to_arrays()
        means_flat = means_sa.reshape(self.num_states, self.obs_dim)
        covs_flat = covs_sa.reshape(self.num_states, self.obs_dim, self.obs_dim)
        return means_flat, covs_flat

    def from_arrays(self, means, covs):
        """
        Load Gaussian parameters from dense arrays in (style, action) layout.

        Parameters
        means : np.ndarray
            Shape (num_style, num_action, obs_dim)

        covs : np.ndarray
            Shape (num_style, num_action, obs_dim, obs_dim)
        """
        if means.shape != (self.num_style, self.num_action, self.obs_dim):
            raise ValueError(f"means has shape {means.shape}, expected {(self.num_style, self.num_action, self.obs_dim)}")
        if covs.shape != (self.num_style, self.num_action, self.obs_dim, self.obs_dim):
            raise ValueError(f"covs has shape {covs.shape}, expected {(self.num_style, self.num_action, self.obs_dim, self.obs_dim)}")

        for s in range(self.num_style):
            for a in range(self.num_action):
                self.params[s, a] = GaussianEmissionParams(mean=means[s, a], cov=covs[s, a])

        self.invalidate_cache()