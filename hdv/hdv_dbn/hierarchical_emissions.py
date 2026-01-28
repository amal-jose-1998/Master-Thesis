"""
Conditional / hierarchical emissions for structured DBN:

  Style_t  in {0..S-1}
  Action_t in {0..A-1}
  Obs_t  depends on Action_t only in the graph, but emission parameters are conditioned on (s,a):

    p(o_t | a_t, s_t) = Emission(o_t; theta[s_t, a_t])
"""

import math

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import DBN_STATES, TRAINING_CONFIG, BERNOULLI_FEATURES

EPSILON = TRAINING_CONFIG.EPSILON if hasattr(TRAINING_CONFIG, "EPSILON") else 1e-6


# =============================================================================
# Helpers
# =============================================================================

def _get_device_dtype():
    dtype_str = getattr(TRAINING_CONFIG, "dtype", "float32")
    dtype = torch.float32 if dtype_str == "float32" else torch.float64
    device = torch.device(getattr(TRAINING_CONFIG, "device", "cpu"))
    return device, dtype


def _as_torch(x, device, dtype):
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


# =============================================================================
# Diagonal Gaussian
# =============================================================================

class DiagGaussianExpert:
    """
    Diagonal Gaussian for joint state (s,a) over D continuous dims.
    Params:
      mean: (S,A,D)
      var : (S,A,D)  diagonal variances
    """
    def __init__(self, S, A, D):
        self.S, self.A, self.D = int(S), int(A), int(D)
        self.mean = np.zeros((self.S, self.A, self.D), dtype=np.float64)
        self.var  = np.ones((self.S, self.A, self.D), dtype=np.float64)

        self._device = torch.device("cpu")
        self._dtype = torch.float32

        # cached tensors
        self._mean_t = None   # (S,A,D)
        self._var_t = None    # (S,A,D)
        self._inv_var_t = None
        self._log_var_t = None

    def invalidate_cache(self):
        self._mean_t = None
        self._var_t = None
        self._inv_var_t = None
        self._log_var_t = None

    def to_device(self, device, dtype):
        self._device = torch.device(device)
        self._dtype = dtype

        if self.D == 0:
            self.invalidate_cache()
            return

        mean_t = torch.as_tensor(self.mean, device=self._device, dtype=self._dtype)
        var_t = torch.as_tensor(self.var, device=self._device, dtype=self._dtype)

        min_diag = float(getattr(TRAINING_CONFIG, "min_cov_diag", 1e-5))
        jitter = float(getattr(TRAINING_CONFIG, "emission_jitter", 1e-6))
        var_t = torch.clamp(var_t, min=min_diag) + jitter # Numerical safety: variance clamping + jitter

        self._mean_t = mean_t
        self._var_t = var_t
        self._inv_var_t = 1.0 / var_t
        self._log_var_t = torch.log(var_t)

    def loglikelihood(self, x_cont, mask=None):
        """
        x_cont: (T,D)
        mask  : (T,D) in {0,1} float (optional)

        Returns:
          logp: (T,S,A)
        """
        if self.D == 0:
            T = int(x_cont.shape[0])
            return torch.zeros((T, self.S, self.A), device=self._device, dtype=self._dtype)

        if self._mean_t is None or self._inv_var_t is None or self._log_var_t is None:
            raise RuntimeError("DiagGaussian cache not initialized. Call to_device().")

        x = x_cont.to(device=self._device, dtype=self._dtype)
        if mask is not None:
            m = (mask.to(device=self._device, dtype=self._dtype) > 0.5).to(dtype=self._dtype)
        else:
            m = None

        # per-dim diagonal Gaussian:
        # log N(x; mu, var) = -0.5 * [log(2π) + log(var) + (x-mu)^2/var]
        # broadcast:
        # x:      (T,1,1,D)
        # mean:   (1,S,A,D)
        # var:    (1,S,A,D)
        diff = x[:, None, None, :] - self._mean_t[None, :, :, :]              # (T,S,A,D)
        quad = (diff * diff) * self._inv_var_t[None, :, :, :]                 # (T,S,A,D)
        per_dim = -0.5 * (math.log(2.0 * math.pi) + self._log_var_t[None, :, :, :] + quad)

        if m is not None:
             per_dim = per_dim * m[:, None, None, :]                           # mask (T,1,1,D)

        return per_dim.sum(dim=3)                                             # (T,S,A)

    def m_step(self, cont_seqs, gamma_seqs, mask_seqs=None, use_progress=True):
        """
        Weighted M-step using joint responsibilities gamma[t,s,a]

         For each (s,a,d):
          W[s,a,d]   = sum_t gamma[t,s,a] * m[t,d]
          mu[s,a,d]  = sum_t gamma[t,s,a] * m[t,d] * x[t,d]  / (W+eps)
          ex2[s,a,d] = sum_t gamma[t,s,a] * m[t,d] * x[t,d]^2 / (W[s,a,d] + eps)
          var[s,a,d] = ex2[s,a,d] - mu[s,a,d]^2

        Low-mass states are left unchanged (by gauss_min_state_mass).
        """
        if self.D == 0:
            return

        device, dtype = _get_device_dtype()
        self.to_device(device=device, dtype=dtype)

        S, A, D = self.S, self.A, self.D

        sum_w = torch.zeros((S, A, D), device=device, dtype=dtype)
        sum_x = torch.zeros((S, A, D), device=device, dtype=dtype)
        sum_x2 = torch.zeros((S, A, D), device=device, dtype=dtype)
        mass_sa = torch.zeros((S, A), device=device, dtype=dtype)

        if mask_seqs is None:
            mask_seqs = [None] * len(cont_seqs) # If no masks provided, assume “all valid” for each sequence.
        if len(mask_seqs) != len(cont_seqs):
            raise ValueError("mask_seqs must be None or same length as cont_seqs")

        it = zip(cont_seqs, gamma_seqs, mask_seqs)
        if use_progress:
            it = tqdm(it, total=len(cont_seqs), desc="M-step hierarchical Gaussian", leave=False)

        for x, g, m in it: # Loop over sequences
            x = _as_torch(x, device=device, dtype=dtype)                      # (T,D)
            g = _as_torch(g, device=device, dtype=dtype)                      # (T,S,A)

            if x.ndim != 2 or x.shape[1] != D:
                raise ValueError(f"Expected cont x (T,{D}), got {tuple(x.shape)}")
            T = int(x.shape[0])
            if g.ndim != 3 or g.shape != (T, S, A):
                raise ValueError(f"Expected gamma (T,S,A)=({T},{S},{A}), got {tuple(g.shape)}")
            if x.shape[0] != g.shape[0]:
                raise ValueError("T mismatch between x and gamma")
           
            mass_sa += g.sum(dim=0)                                        # (S,A); Used to decide if a joint state has enough data to update.

            # Build the mask
            if m is None:
                mm = torch.ones_like(x, device=device, dtype=dtype) # (T,D); every entry is treated as observed.
            else:
                mm = _as_torch(m, device=device, dtype=dtype)
                if mm.shape != x.shape:
                    raise ValueError("mask shape must match x shape")
                mm = (mm > 0.5).to(dtype=dtype)

            # Expand for joint weighting
            # g: (T,S,A) -> (T,S,A,1)
            g4 = g[:, :, :, None]


            sum_w  += (g4 * mm[:, None, None, :]).sum(dim=0)                  # (S,A,D); masked “effective sample count” per state/dim.
            sum_x  += (g4 * (mm[:, None, None, :] * x[:, None, None, :])).sum(dim=0)
            sum_x2 += (g4 * (mm[:, None, None, :] * (x[:, None, None, :] ** 2))).sum(dim=0)  

        mean_new = sum_x / (sum_w + EPSILON)
        ex2 = sum_x2 / (sum_w + EPSILON)
        var_new = ex2 - mean_new * mean_new

        min_diag = float(getattr(TRAINING_CONFIG, "min_cov_diag", 1e-5))
        jitter  = float(getattr(TRAINING_CONFIG, "emission_jitter", 1e-6))
        min_mass = float(getattr(TRAINING_CONFIG, "gauss_min_state_mass", 50.0))

        mean_old = torch.as_tensor(self.mean, device=device, dtype=dtype)
        var_old  = torch.as_tensor(self.var,  device=device, dtype=dtype)

        # update only (s,a) pairs with enough data and enough observed weight
        upd_sa = (mass_sa >= min_mass)[:, :, None] & (sum_w > 1.0)

        mean = mean_old.clone()
        var  = var_old.clone()
        mean[upd_sa] = mean_new[upd_sa]
        var[upd_sa]  = torch.clamp(var_new[upd_sa], min=min_diag) + jitter

        self.mean = mean.detach().cpu().numpy()
        self.var  = var.detach().cpu().numpy()
        self.invalidate_cache()


# =============================================================================
# Bernoulli 
# =============================================================================

class BernoulliExpert:
    """
    Independent Bernoulli expert for joint state (s,a) over B binary dims.
    Params:
      p: (S,A,B)
    """
    def __init__(self, S, A, B):
        self.S, self.A, self.B = int(S), int(A), int(B)
        self.p = np.full((self.S, self.A, self.B), 0.5, dtype=np.float64) # maximum uncertainty (uninformative prior-ish)

        self._device = torch.device("cpu")
        self._dtype = torch.float32
        self._p_t = None  # (S,A,B)

    def invalidate_cache(self):
        self._p_t = None

    def to_device(self, device, dtype):
        self._device = torch.device(device)
        self._dtype = dtype
        if self.B == 0:
            self._p_t = None
            return
        p = torch.as_tensor(self.p, device=self._device, dtype=self._dtype)
        self._p_t = p.clamp(EPSILON, 1.0 - EPSILON)

    def loglikelihood(self, x_bin):
        """
        x_bin: (T,B) float in {0,1}
        Returns: (T,S,A)
        """
        if self.B == 0:
            T = int(x_bin.shape[0])
            return torch.zeros((T, self.S, self.A), device=self._device, dtype=self._dtype)

        if self._p_t is None:
            raise RuntimeError("BernoulliExpert cache not initialized. Call to_device().")

        xb = x_bin.to(device=self._device, dtype=self._dtype)                 # (T,B)
        xb = xb[:, None, None, :]                                            # (T,1,1,B)
        p  = self._p_t[None, :, :, :]                                        # (1,S,A,B)

        logp = xb * torch.log(p) + (1.0 - xb) * torch.log(1.0 - p)           # (T,S,A,B)
        return logp.sum(dim=3)                                               # (T,S,A)

    def m_step(self, xbin_raw_seqs, gamma_seqs, finite_mask_seqs=None, use_progress=True):
        """
        M-step for Bernoulli with optional finite mask (in case xbin contains NaNs).
          W[s,a,j] = sum_t gamma[t,s,a] * m[t,j]
          p[s,a,j] = sum_t gamma[t,s,a] * m[t,j] * b[t,j] / (W[s,a,j] + eps)
        """
        if self.B == 0:
            return

        device, dtype = _get_device_dtype()
        self.to_device(device=device, dtype=dtype)

        S, A, B = self.S, self.A, self.B
        sum_x = torch.zeros((S, A, B), device=device, dtype=dtype)
        sum_w = torch.zeros((S, A, B), device=device, dtype=dtype)

        if finite_mask_seqs is None:
            finite_mask_seqs = [None] * len(xbin_raw_seqs)
        if len(finite_mask_seqs) != len(xbin_raw_seqs):
            raise ValueError("finite_mask_seqs must be None or same length as xbin_raw_seqs")

        it = zip(xbin_raw_seqs, gamma_seqs, finite_mask_seqs)
        if use_progress:
            it = tqdm(it, total=len(xbin_raw_seqs), desc="M-step hierarchical Bernoulli", leave=False)

        for xb_raw, g, fm in it:
            xb_raw = _as_torch(xb_raw, device=device, dtype=dtype)            # (T,B)
            g =  _as_torch(g, device=device, dtype=dtype)              # (T,S,A)

            T = int(xb_raw.shape[0])
            if xb_raw.ndim != 2 or xb_raw.shape[1] != B:
                raise ValueError(f"Expected xb_raw (T,{B}), got {tuple(xb_raw.shape)}")
            if g.ndim != 3 or g.shape != (T, S, A):
                raise ValueError(f"Expected gamma_sa (T,S,A)=({T},{S},{A}), got {tuple(g.shape)}")
            if xb_raw.shape[0] != g.shape[0]:
                raise ValueError("T mismatch between xb_raw and gamma")

            if fm is None:
                finite = torch.isfinite(xb_raw)
            else:
                finite = _as_torch(fm, device=device, dtype=dtype) > 0.5

            m = finite.to(dtype=dtype)                                       # (T,B)
            b = (xb_raw > 0.5).to(dtype=dtype) * m                            # (T,B)

            g4 = g[:, :, :, None]                                          # (T,S,A,1)
            sum_x += (g4 * b[:, None, None, :]).sum(dim=0)                    # (S,A,B)
            sum_w += (g4 * m[:, None, None, :]).sum(dim=0)                    # (S,A,B)

        p = (sum_x / (sum_w + EPSILON)).clamp(EPSILON, 1.0 - EPSILON)
        self.p = p.detach().cpu().numpy()
        self.invalidate_cache()


# =============================================================================
# Hierarchical Mixed Emission Model
# =============================================================================

class MixedEmissionModel:
    """
    Single emission model p(o_t | s_t, a_t) with parameters theta[s,a].

    Splits features into:
      - Continuous dims: everything except bernoulli features
      - Bernoulli dims : names in bernoulli_names (default: config.BERNOULLI_FEATURES)
    """
    def __init__(self, obs_names, disable_discrete_obs=False, bernoulli_names=None):
        self.obs_names = list(obs_names)
        self.obs_dim = len(self.obs_names)

        self.style_states = DBN_STATES.driving_style
        self.action_states = DBN_STATES.action
        self.S = len(self.style_states)
        self.A = len(self.action_states)

        self.disable_discrete_obs = bool(disable_discrete_obs)

        if bernoulli_names is None:
            bernoulli_names = list(BERNOULLI_FEATURES)
        self.bernoulli_names = list(bernoulli_names)

        name_to_idx = {n: i for i, n in enumerate(self.obs_names)}
        self.bin_idx = [int(name_to_idx[n]) for n in self.bernoulli_names if n in name_to_idx]
        bin_set = set(self.bin_idx)

        self.bin_dim = len(self.bin_idx)
        self.cont_idx = [i for i in range(self.obs_dim) if i not in bin_set]
        self.cont_dim = len(self.cont_idx)

        # Experts
        self.gauss = DiagGaussianExpert(self.S, self.A, self.cont_dim)
        self.bern  = BernoulliExpert(self.S, self.A, self.bin_dim)

        # runtime cache config
        self._device = torch.device("cpu")
        self._dtype = torch.float32

    def invalidate_cache(self):
        self.gauss.invalidate_cache()
        self.bern.invalidate_cache()

    def to_device(self, device, dtype):
        self._device = torch.device(device)
        self._dtype = dtype
        self.gauss.to_device(device, dtype)
        self.bern.to_device(device, dtype)

    def _ensure_device(self):
        device, dtype = _get_device_dtype()
        # If experts weren't materialized on the configured device yet, push them.
        need = (self.cont_dim > 0) and (self.gauss._mean_t is None)
        need = need or ((self.bin_dim > 0) and (self.bern._p_t is None))
        if need or self._device != device or self._dtype != dtype:
            self.to_device(device, dtype)

    # -------------------------------------------------------------------------
    # Likelihood
    # -------------------------------------------------------------------------
    def loglikelihood(self, obs):
        """
        Returns:
          logB_sa: (T,S,A)
        """
        self._ensure_device()
        device, dtype = self._device, self._dtype

        x = _as_torch(obs, device=device, dtype=dtype)
        if x.ndim != 2 or x.shape[1] != self.obs_dim:
            raise ValueError(f"Expected obs shape (T,{self.obs_dim}), got {tuple(x.shape)}")

        T = int(x.shape[0])

        # Continuous with finite mask
        if self.cont_dim > 0:
            x_cont_raw = x[:, self.cont_idx]                                  # (T,Dc)
            finite = torch.isfinite(x_cont_raw)
            mask_c = finite.to(dtype=dtype)
            x_cont = torch.where(finite, x_cont_raw, torch.zeros_like(x_cont_raw))
            log_gauss = self.gauss.loglikelihood(x_cont, mask=mask_c)      # (T,S,A)
        else:
            log_gauss = torch.zeros((T, self.S, self.A), device=device, dtype=dtype)

        if self.disable_discrete_obs or self.bin_dim == 0:
            return log_gauss

        xb_raw = x[:, self.bin_idx]                                           # (T,B)
        xb = (xb_raw > 0.5).to(dtype=dtype)
        log_bern = self.bern.loglikelihood(xb)                             # (T,S,A)

        w_bern = float(getattr(TRAINING_CONFIG, "bern_weight", 1.0))
        return log_gauss + w_bern * log_bern
    
    # -------------------------------------------------------------------------
    # EM M-step
    # -------------------------------------------------------------------------
    def update_from_posteriors(self, obs_seqs, gamma_sa_seqs, use_progress=True):
        device, dtype = _get_device_dtype()
        self.to_device(device, dtype)

        cont_obs_seqs, cont_mask_seqs = [], []
        xb_raw_seqs, xb_finite_seqs = [], []

        it = zip(obs_seqs, gamma_sa_seqs)
        if use_progress:
            it = tqdm(it, total=len(obs_seqs), desc="M-step emissions (hierarchical)", leave=False)

        for obs, gamma_sa in it:
            x = _as_torch(obs, device=device, dtype=dtype)
            g = _as_torch(gamma_sa, device=device, dtype=dtype)

            if x.ndim != 2 or x.shape[1] != self.obs_dim:
                raise ValueError(f"Expected obs shape (T,{self.obs_dim}), got {tuple(x.shape)}")

            T = int(x.shape[0])
            if g.ndim != 3 or g.shape != (T, self.S, self.A):
                raise ValueError(f"Expected gamma_sa (T,S,A)=({T},{self.S},{self.A}), got {tuple(g.shape)}")

            # Continuous: finite mask per entry
            if self.cont_dim > 0:
                cont_raw = x[:, self.cont_idx]
                finite_c = torch.isfinite(cont_raw)
                cont_x = torch.where(finite_c, cont_raw, torch.zeros_like(cont_raw))
                cont_m = finite_c.to(dtype=dtype)
            else:
                cont_x = x[:, :0]
                cont_m = torch.ones((T, 0), device=device, dtype=dtype)

            cont_obs_seqs.append(cont_x)
            cont_mask_seqs.append(cont_m)

            # Bernoulli: keep raw and finite mask (in case any NaNs exist)
            if (not self.disable_discrete_obs) and self.bin_dim > 0:
                xb_raw = x[:, self.bin_idx]
                xb_raw_seqs.append(xb_raw)
                xb_finite_seqs.append(torch.isfinite(xb_raw).to(dtype=dtype))

        # Expert updates
        self.gauss.m_step(cont_obs_seqs, gamma_sa_seqs, mask_seqs=cont_mask_seqs, use_progress=False)

        if (not self.disable_discrete_obs) and self.bin_dim > 0:
            self.bern.m_step(xb_raw_seqs, gamma_sa_seqs, finite_mask_seqs=xb_finite_seqs, use_progress=False)

        self.invalidate_cache()

        # Return masses for debugging/logging
        mass_joint = torch.zeros((self.S, self.A), device=device, dtype=dtype)
        for g in gamma_sa_seqs:
            gg = _as_torch(g, device=device, dtype=dtype)
            mass_joint += gg.sum(dim=0)
        
        mass_style = mass_joint.sum(dim=1)   # (S,)
        mass_action = mass_joint.sum(dim=0)  # (A,)

        return {
            "mass_style": mass_style.detach().cpu().numpy(),
            "mass_action": mass_action.detach().cpu().numpy(),
            "mass_joint": mass_joint.detach().cpu().numpy(),
        }

    # -------------------------------------------------------------------------
    # Serialization helpers
    # -------------------------------------------------------------------------
    def to_arrays(self):
        """
        Export emission parameters in a dict.
        """
        payload = dict(
            obs_names=np.array(self.obs_names, dtype=object),
            cont_idx=np.array(self.cont_idx, dtype=np.int64),
            bin_idx=np.array(self.bin_idx, dtype=np.int64),
            bernoulli_names=np.array(self.bernoulli_names, dtype=object),

            gauss_mean=np.asarray(self.gauss.mean, dtype=np.float64),  # (S,A,Dc)
            gauss_var=np.asarray(self.gauss.var, dtype=np.float64),    # (S,A,Dc)
            bern_p=np.asarray(self.bern.p, dtype=np.float64),          # (S,A,B)
        )
        return payload

    def from_arrays(self, payload):
        """
        Load emission parameters from dict.
        """
        self.obs_names = list(np.asarray(payload["obs_names"], dtype=object).tolist())
        self.obs_dim = len(self.obs_names)

        if "bernoulli_names" in payload:
            self.bernoulli_names = list(np.asarray(payload["bernoulli_names"], dtype=object).tolist())
        else:
            self.bernoulli_names = list(BERNOULLI_FEATURES)

        self.cont_idx = list(np.asarray(payload["cont_idx"], dtype=np.int64).tolist())
        self.bin_idx  = list(np.asarray(payload["bin_idx"],  dtype=np.int64).tolist())
        self.cont_dim = len(self.cont_idx)
        self.bin_dim  = len(self.bin_idx)

        # Recreate experts with correct dims
        self.gauss = DiagGaussianExpert(self.S, self.A, self.cont_dim)
        self.bern  = BernoulliExpert(self.S, self.A, self.bin_dim)

        self.gauss.mean = np.asarray(payload["gauss_mean"], dtype=np.float64)
        self.gauss.var  = np.asarray(payload["gauss_var"],  dtype=np.float64)
        self.bern.p     = np.asarray(payload.get("bern_p", np.full((self.S, self.A, self.bin_dim), 0.5)), dtype=np.float64)

        # Basic sanity checks
        if self.gauss.mean.shape != (self.S, self.A, self.cont_dim):
            raise ValueError("gauss_mean shape mismatch")
        if self.gauss.var.shape != (self.S, self.A, self.cont_dim):
            raise ValueError("gauss_var shape mismatch")
        if self.bern.p.shape != (self.S, self.A, self.bin_dim):
            raise ValueError("bern_p shape mismatch")

        self.invalidate_cache()