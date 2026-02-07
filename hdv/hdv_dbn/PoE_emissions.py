"""
PoE (Product-of-Experts) emissions for a structured DBN with hidden nodes:

  Style_t  in {0..S-1}
  Action_t in {0..A-1}

Each expert sees the full observation o_t:

  Style expert : p(o_t | s_t)
  Action expert: p(o_t | a_t)

PoE combination:
  p(o_t | s,a) ∝ p(o_t | s) * p(o_t | a)

In log-space:
  logB_sa[t,s,a] = logB_s[t,s] + logB_a[t,a]
"""

import math
import numpy as np
import torch
from tqdm.auto import tqdm

from .config import DBN_STATES, TRAINING_CONFIG, BERNOULLI_FEATURES
from .utils.poe_utils import (
    get_device_dtype,
    as_torch,
    build_unconstrained_emission_params_and_opt,
    accumulate_Q_and_mass,
    regularizer,
    snapshot_params,
    restore_params,
)


EPSILON = TRAINING_CONFIG.EPSILON if hasattr(TRAINING_CONFIG, "EPSILON") else 1e-6

# =============================================================================
# Diagonal Gaussian expert (masked)
# =============================================================================

class DiagGaussianExpert:
    """
    Diagonal Gaussian expert for K discrete states over D continuous dimensions.

    Params:
      mean: (K,D)
      var : (K,D)   diagonal variances

    Log-likelihood for one time step (with mask m in {0,1}):
      log p(x_t | k) = sum_d m[t,d] * log N(x[t,d]; mean[k,d], var[k,d])
    where, m[t,d] is the mask
    """
    def __init__(self, K, D):
        self.K = int(K)
        self.D = int(D)

        self.mean = np.zeros((self.K, self.D), dtype=np.float64)
        self.var  = np.ones((self.K, self.D), dtype=np.float64)

        self._device = torch.device("cpu")
        self._dtype = torch.float32

        # cached tensors
        self._mean_t = None   # (K,D)
        self._var_t = None    # (K,D)
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
          logp: (T,K)
        """
        if self.D == 0:
            T = int(x_cont.shape[0])
            return torch.zeros((T, self.K), device=self._device, dtype=self._dtype)

        if self._mean_t is None or self._inv_var_t is None or self._log_var_t is None:
            raise RuntimeError("DiagGaussianExpert cache not initialized. Call to_device().")

        x = x_cont.to(device=self._device, dtype=self._dtype)
        if mask is not None:
            m = mask.to(device=self._device, dtype=self._dtype)
            m = (m > 0.5).to(dtype=self._dtype)
        else:
            m = None

        # per-dim diagonal Gaussian:
        # log N(x; mu, var) = -0.5 * [log(2π) + log(var) + (x-mu)^2/var]
        diff = x[:, None, :] - self._mean_t[None, :, :]                 # (T,K,D)
        quad = (diff * diff) * self._inv_var_t[None, :, :]              # (T,K,D)
        per_dim = -0.5 * (math.log(2.0 * math.pi) + self._log_var_t[None, :, :] + quad)

        if m is not None:
            per_dim = per_dim * m[:, None, :]

        return per_dim.sum(dim=2)  # (T,K)

    def m_step(self, cont_seqs, gamma_seqs, mask_seqs=None, use_progress=True):
        """
        M-step with masked sufficient statistics.

        For each state k and dimension d:
          W[k,d]   = sum_t gamma[t,k] * m[t,d]
          mu[k,d]  = sum_t gamma[t,k] * m[t,d] * x[t,d]  / (W[k,d] + eps)
          ex2[k,d] = sum_t gamma[t,k] * m[t,d] * x[t,d]^2 / (W[k,d] + eps)
          var[k,d] = ex2[k,d] - mu[k,d]^2

        Low-mass states are left unchanged (by gauss_min_state_mass).
        """
        if self.D == 0:
            return

        device, dtype = get_device_dtype()
        self.to_device(device=device, dtype=dtype)

        K, D = self.K, self.D

        sum_wd = torch.zeros((K, D), device=device, dtype=dtype)
        sum_x  = torch.zeros((K, D), device=device, dtype=dtype)
        sum_x2 = torch.zeros((K, D), device=device, dtype=dtype)
        mass_k = torch.zeros((K,), device=device, dtype=dtype)

        if mask_seqs is None:
            mask_seqs = [None] * len(cont_seqs) # If no masks provided, assume “all valid” for each sequence.
        if len(mask_seqs) != len(cont_seqs):
            raise ValueError("mask_seqs must be None or same length as cont_seqs")

        it = zip(cont_seqs, gamma_seqs, mask_seqs)
        if use_progress:
            it = tqdm(it, total=len(cont_seqs), desc="M-step PoE Gaussian", leave=False)

        for x, g, m in it: # Loop over sequences
            x = as_torch(x, device=device, dtype=dtype)
            g = as_torch(g, device=device, dtype=dtype)

            if x.ndim != 2 or x.shape[1] != D:
                raise ValueError(f"Expected cont x (T,{D}), got {tuple(x.shape)}")
            if g.ndim != 2 or g.shape[1] != K:
                raise ValueError(f"Expected gamma (T,{K}), got {tuple(g.shape)}")
            if x.shape[0] != g.shape[0]:
                raise ValueError("T mismatch between x and gamma")

            mass_k += g.sum(dim=0) # Used later to decide if a state has enough data to update.

            # Build the mask
            if m is None:
                mm = torch.ones_like(x, device=device, dtype=dtype) # every entry is treated as observed.
            else:
                mm = as_torch(m, device=device, dtype=dtype)
                if mm.shape != x.shape:
                    raise ValueError("mask shape must match x shape")
                mm = (mm > 0.5).to(dtype=dtype)

            sum_wd += g.T @ mm             # masked “effective sample count” per state/dim.
            sum_x  += g.T @ (mm * x)       
            sum_x2 += g.T @ (mm * (x * x))  

        mean_new = sum_x / (sum_wd + EPSILON)
        ex2 = sum_x2 / (sum_wd + EPSILON)
        var_new = ex2 - mean_new * mean_new

        min_diag = float(getattr(TRAINING_CONFIG, "min_cov_diag", 1e-5))
        jitter = float(getattr(TRAINING_CONFIG, "emission_jitter", 1e-6))
        min_mass = float(getattr(TRAINING_CONFIG, "gauss_min_state_mass", 50.0))

        mean_old = torch.as_tensor(self.mean, device=device, dtype=dtype)
        var_old  = torch.as_tensor(self.var,  device=device, dtype=dtype)

        # update only where we have enough responsibility mass and enough observed weight
        upd = (mass_k >= min_mass)[:, None] & (sum_wd > 1.0)

        mean = mean_old.clone()
        var  = var_old.clone()
        mean[upd] = mean_new[upd]
        var[upd]  = torch.clamp(var_new[upd], min=min_diag) + jitter

        self.mean = mean.detach().cpu().numpy()
        self.var  = var.detach().cpu().numpy()
        self.invalidate_cache()


# =============================================================================
# Bernoulli expert
# =============================================================================

class BernoulliExpert:
    """
    Independent Bernoulli expert for K discrete states over B binary dims.

    Params:
      p: (K,B)

    Log-likelihood:
      log p(b_t | k) = sum_j [ b[t,j] log p[k,j] + (1-b[t,j]) log (1-p[k,j]) ]
    """
    def __init__(self, K, B):
        self.K = int(K)
        self.B = int(B)

        self.p = np.full((self.K, self.B), 0.5, dtype=np.float64) # maximum uncertainty (uninformative prior-ish)

        self._device = torch.device("cpu")
        self._dtype = torch.float32
        self._p_t = None  # (K,B)

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
        Returns: (T,K)
        """
        if self.B == 0:
            T = int(x_bin.shape[0])
            return torch.zeros((T, self.K), device=self._device, dtype=self._dtype)

        if self._p_t is None:
            raise RuntimeError("BernoulliExpert cache not initialized. Call to_device().")

        xb = x_bin.to(device=self._device, dtype=self._dtype)
        xb = xb[:, None, :]                 # (T,1,B)
        p  = self._p_t[None, :, :]          # (1,K,B)

        logp = xb * torch.log(p) + (1.0 - xb) * torch.log(1.0 - p)  # (T,K,B)
        return logp.sum(dim=2)  # (T,K)

    def m_step(self, xbin_raw_seqs, gamma_seqs, finite_mask_seqs=None, use_progress=True):
        """
        M-step for Bernoulli with optional finite mask (in case xbin contains NaNs).
          W[k,j] = sum_t gamma[t,k] * m[t,j]
          p[k,j] = sum_t gamma[t,k] * m[t,j] * b[t,j] / (W[k,j] + eps)
        """
        if self.B == 0:
            return

        device, dtype = get_device_dtype()
        self.to_device(device=device, dtype=dtype)

        K, B = self.K, self.B
        sum_x = torch.zeros((K, B), device=device, dtype=dtype)
        sum_w = torch.zeros((K, B), device=device, dtype=dtype)

        if finite_mask_seqs is None:
            finite_mask_seqs = [None] * len(xbin_raw_seqs)
        if len(finite_mask_seqs) != len(xbin_raw_seqs):
            raise ValueError("finite_mask_seqs must be None or same length as xbin_raw_seqs")

        it = zip(xbin_raw_seqs, gamma_seqs, finite_mask_seqs)
        if use_progress:
            it = tqdm(it, total=len(xbin_raw_seqs), desc="M-step PoE Bernoulli", leave=False)

        for xb_raw, g, fm in it:
            xb_raw = as_torch(xb_raw, device=device, dtype=dtype)
            g = as_torch(g, device=device, dtype=dtype)

            if xb_raw.ndim != 2 or xb_raw.shape[1] != B:
                raise ValueError(f"Expected xb_raw (T,{B}), got {tuple(xb_raw.shape)}")
            if g.ndim != 2 or g.shape[1] != K:
                raise ValueError(f"Expected gamma (T,{K}), got {tuple(g.shape)}")
            if xb_raw.shape[0] != g.shape[0]:
                raise ValueError("T mismatch between xb_raw and gamma")

            if fm is None:
                finite = torch.isfinite(xb_raw)
            else:
                finite = as_torch(fm, device=device, dtype=dtype) > 0.5

            m = finite.to(dtype=dtype)
            b = (xb_raw > 0.5).to(dtype=dtype) * m

            sum_x += g.T @ b
            sum_w += g.T @ m

        p = (sum_x / (sum_w + EPSILON)).clamp(EPSILON, 1.0 - EPSILON)
        self.p = p.detach().cpu().numpy()
        self.invalidate_cache()


# =============================================================================
# PoE Mixed Emission Model (Style expert + Action expert)
# =============================================================================

class MixedEmissionModel:
    """
    PoE emission model over the full observation vector (T,F).

    Splits features into:
      - Continuous dims: everything except bernoulli features
      - Bernoulli dims : names in bernoulli_names (default: config.BERNOULLI_FEATURES)

    Experts:
      - Style expert  (K=S): DiagGaussian + Bernoulli
      - Action expert (K=A): DiagGaussian + Bernoulli

    Likelihood:
      logB_s[t,s] = log p(o_t | s)
      logB_a[t,a] = log p(o_t | a)
      logB_sa[t,s,a] = logB_s[t,s] + logB_a[t,a]
    """
    def __init__(self, obs_names, disable_discrete_obs=False, bernoulli_names=None):
        self.obs_names = list(obs_names)
        self.obs_dim = len(self.obs_names)

        self.style_states = DBN_STATES.driving_style
        self.action_states = DBN_STATES.action
        self.num_style = len(self.style_states)   # S
        self.num_action = len(self.action_states) # A

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
        self.style_gauss = DiagGaussianExpert(K=self.num_style, D=self.cont_dim)
        self.action_gauss = DiagGaussianExpert(K=self.num_action, D=self.cont_dim)
        self.style_bern = BernoulliExpert(K=self.num_style, B=self.bin_dim)
        self.action_bern = BernoulliExpert(K=self.num_action, B=self.bin_dim)

        # runtime cache config
        self._device = torch.device("cpu")
        self._dtype = torch.float32

    def invalidate_cache(self):
        self.style_gauss.invalidate_cache()
        self.action_gauss.invalidate_cache()
        self.style_bern.invalidate_cache()
        self.action_bern.invalidate_cache()

    def to_device(self, device, dtype):
        self._device = torch.device(device)
        self._dtype = dtype
        self.style_gauss.to_device(device, dtype)
        self.action_gauss.to_device(device, dtype)
        self.style_bern.to_device(device, dtype)
        self.action_bern.to_device(device, dtype)

    def _ensure_device(self):
        device, dtype = get_device_dtype()
        # If experts weren't materialized on the configured device yet, push them.
        need = (self.cont_dim > 0) and (self.style_gauss._mean_t is None or self.action_gauss._mean_t is None)
        need = need or ((self.bin_dim > 0) and (self.style_bern._p_t is None or self.action_bern._p_t is None))
        if need or self._device != device or self._dtype != dtype:
            self.to_device(device, dtype)

    # -------------------------------------------------------------------------
    # Likelihood
    # -------------------------------------------------------------------------
    def loglikelihood_experts(self, obs):
        """
        Returns:
          logB_s: (T,S)
          logB_a: (T,A)

        Each expert sees the full observation, but is conditioned on its own hidden variable.
        """
        self._ensure_device()
        device, dtype = self._device, self._dtype

        x = as_torch(obs, device=device, dtype=dtype)
        if x.ndim != 2 or x.shape[1] != self.obs_dim:
            raise ValueError(f"Expected obs shape (T,{self.obs_dim}), got {tuple(x.shape)}")

        # Continuous path with finite masking
        if self.cont_dim > 0:
            x_cont_raw = x[:, self.cont_idx]                  # (T,Dc)
            finite = torch.isfinite(x_cont_raw)
            mask_c = finite.to(dtype=dtype)
            x_cont = torch.where(finite, x_cont_raw, torch.zeros_like(x_cont_raw))

            log_s_gauss = self.style_gauss.loglikelihood(x_cont, mask=mask_c)    # (T,S)
            log_a_gauss = self.action_gauss.loglikelihood(x_cont, mask=mask_c)  # (T,A)
        else:
            T = int(x.shape[0])
            log_s_gauss = torch.zeros((T, self.num_style), device=device, dtype=dtype)
            log_a_gauss = torch.zeros((T, self.num_action), device=device, dtype=dtype)

        if self.disable_discrete_obs or self.bin_dim == 0:
            return log_s_gauss, log_a_gauss

        # Bernoulli path (binary threshold at 0.5)
        xb_raw = x[:, self.bin_idx]                  # (T,B)
        xb = (xb_raw > 0.5).to(dtype=dtype)          # (T,B)

        log_s_bern = self.style_bern.loglikelihood(xb)    # (T,S)
        log_a_bern = self.action_bern.loglikelihood(xb)   # (T,A)

        w_bern = float(getattr(TRAINING_CONFIG, "bern_weight", 1.0))
        return (log_s_gauss + w_bern * log_s_bern), (log_a_gauss + w_bern * log_a_bern)

    def loglikelihood(self, obs):
        """
        Fully normalized PoE emission.

        Returns:
            logB_sa: (T,S,A) where
                logB_sa[t,s,a] = log p_s(x_t|s) + log p_a(x_t|a) - log Z_sa(t,s,a)
        """
        logB_s, logB_a = self.loglikelihood_experts(obs)  # (T,S), (T,A)
        self._ensure_device()
        device, dtype = self._device, self._dtype

        x = as_torch(obs, device=device, dtype=dtype)
        T = int(x.shape[0])
        S, A = self.num_style, self.num_action

        # ----------------------------
        # Continuous logZ (masked)
        # ----------------------------
        if self.cont_dim > 0:
            # (T,Dc)
            x_cont_raw = x[:, self.cont_idx]
            finite_c = torch.isfinite(x_cont_raw)
            m_c = finite_c.to(dtype=dtype)  # (T,Dc)

            # params: (S,Dc), (A,Dc)
            mu_s = self.style_gauss._mean_t          # (S,Dc)
            var_s = self.style_gauss._var_t          # (S,Dc)
            mu_a = self.action_gauss._mean_t         # (A,Dc)
            var_a = self.action_gauss._var_t         # (A,Dc)

            # broadcast to (S,A,Dc)
            dmu = mu_s[:, None, :] - mu_a[None, :, :]           # (S,A,Dc)
            v = var_s[:, None, :] + var_a[None, :, :]           # (S,A,Dc)

            # log N(mu_s ; mu_a, v) per dim: -(1/2)[log(2π)+log v + dmu^2/v]
            logZc_per_dim = -0.5 * (math.log(2.0 * math.pi) + torch.log(v) + (dmu * dmu) / v)  # (S,A,Dc)

            # apply time mask: sum_d m_c[t,d]*logZc_per_dim[s,a,d]
            # -> (T,S,A)
            logZc = torch.einsum("td,sad->tsa", m_c, logZc_per_dim)

        else:
            logZc = torch.zeros((T, S, A), device=device, dtype=dtype)

        # ----------------------------
        # Bernoulli logZ (masked)
        # ----------------------------
        if (not self.disable_discrete_obs) and (self.bin_dim > 0):
            xb_raw = x[:, self.bin_idx]                 # (T,B)
            finite_b = torch.isfinite(xb_raw)
            m_b = finite_b.to(dtype=dtype)              # (T,B)

            p_s = self.style_bern._p_t                  # (S,B)
            p_a = self.action_bern._p_t                 # (A,B)

            # Z_j(s,a) = p_s p_a + (1-p_s)(1-p_a)  -> (S,A,B)
            Zj = p_s[:, None, :] * p_a[None, :, :] + (1.0 - p_s[:, None, :]) * (1.0 - p_a[None, :, :])
            # numerical safety
            Zj = Zj.clamp(EPSILON, 1.0)

            logZj = torch.log(Zj)                       # (S,A,B)
            logZb = torch.einsum("tb,sab->tsa", m_b, logZj)  # (T,S,A)
        else:
            logZb = torch.zeros((T, S, A), device=device, dtype=dtype)

        logZ = logZc + logZb  # (T,S,A)

        return logB_s[:, :, None] + logB_a[:, None, :] - logZ

    # -------------------------------------------------------------------------
    # EM M-step
    # -------------------------------------------------------------------------
    def update_from_posteriors(self, obs_seqs, gamma_sa_seqs, lr=1e-2, steps=10, use_progress=True, verbose=0, chunk_size=None):
        """
        Joint gradient-based M-step for fully normalized PoE emissions.
        Maximizes the exact PoE Q-function:
        sum_{seq} sum_t sum_{s,a} gamma[t,s,a] * (log p_s(x|s) + log p_a(x|a) - log Z_sa(t))

        No separable updates (style/action are coupled via Z_sa).

        Parameters
        obs_seqs : list[np.ndarray | torch.Tensor]
            Each obs has shape (T, obs_dim)
        gamma_sa_seqs : list[torch.Tensor]
            Each gamma_sa has shape (T, S, A) and is NOT required to be normalized
            (mode-B weighting intentionally breaks per-timestep normalization).

        Returns
        dict with:
        - mass_style:  (S,) total responsibility mass per style
        - mass_action: (A,) total responsibility mass per action
        - mass_joint:  (S,A) total responsibility mass per (style,action)
        """
        self._ensure_device()
        device, dtype = get_device_dtype()
        S, A = self.num_style, self.num_action
        Dc, B = self.cont_dim, self.bin_dim

        # ----------------------------
        # Stabilization / regularization hyperparams 
        # ----------------------------
        lam_mu = float(getattr(TRAINING_CONFIG, "poe_em_lam_mu", 1e-5)) # penalizes large means (prevents drift)
        lam_logvar = float(getattr(TRAINING_CONFIG, "poe_em_lam_logvar", 1e-4)) # discourages variance collapsing to ~0 or exploding.
        lam_logit = float(getattr(TRAINING_CONFIG, "poe_em_lam_logit", 1e-5)) # prevents Bernoulli logits saturating (probabilities going to 0/1).
        weight_decay = float(getattr(TRAINING_CONFIG, "poe_em_weight_decay", 0.0)) # optimizer-level L2 penalty.
        inner_patience = int(getattr(TRAINING_CONFIG, "poe_em_inner_patience", 3)) # early stopping inside the M-step loop if loss isn’t improving.

        # ----------------------------
        # Unconstrained parameterization
        # ----------------------------
        bundle = build_unconstrained_emission_params_and_opt(
            style_gauss_mean=self.style_gauss.mean,
            action_gauss_mean=self.action_gauss.mean,
            style_gauss_var=self.style_gauss.var,
            action_gauss_var=self.action_gauss.var,
            device=device,
            dtype=dtype,
            disable_discrete_obs=self.disable_discrete_obs,
            bin_dim=B,
            weight_decay=weight_decay,
            lr=lr,
            style_bern_p=None if self.disable_discrete_obs else getattr(self.style_bern, "p", None),
            action_bern_p=None if self.disable_discrete_obs else getattr(self.action_bern, "p", None),
        )

        mu_s = bundle["mu_s"]
        mu_a = bundle["mu_a"]
        rho_s = bundle["rho_s"]
        rho_a = bundle["rho_a"]
        logit_s = bundle["logit_s"]
        logit_a = bundle["logit_a"]
        params = bundle["params"]
        opt = bundle["opt"] # optimiser
        vmin = bundle["vmin"]
        jitter = bundle["jitter"]

        ctx = {
            "device": device, "dtype": dtype,
            "S": S, "A": A, "Dc": Dc, "B": B,
            "cont_idx": self.cont_idx, "bin_idx": self.bin_idx,
            "vmin": vmin, "jitter": jitter, "EPSILON": EPSILON,
            "mu_s": mu_s, "mu_a": mu_a,
            "rho_s": rho_s, "rho_a": rho_a,
            "logit_s": logit_s, "logit_a": logit_a,
            "lam_mu": lam_mu, "lam_logvar": lam_logvar, "lam_logit": lam_logit,
        }

        if chunk_size is None:
            cs = int(getattr(TRAINING_CONFIG, "poe_em_chunk_size", 0))
            chunk_size = None if cs <= 0 else cs

        best_snap = None
        best_loss = None
        bad_steps = 0
        patience = int(getattr(TRAINING_CONFIG, "poe_em_inner_patience", 0))

        # ----------------------------
        # Optimize Q_emit by gradient ascent
        # ----------------------------
        it = range(steps)
        if use_progress:
            it = tqdm(it, total=steps, desc="M-step PoE (joint grad)", leave=False)

        for k in it:
            opt.zero_grad(set_to_none=True)
            Q_sum, mass = accumulate_Q_and_mass(obs_seqs, gamma_sa_seqs, ctx, chunk_size=chunk_size)
            if float(mass.detach().cpu().item()) < 1e-6:
                if verbose >= 0:
                    print("[PoE M-step] mass ~ 0, skipping emission update.")
                break

            reg = regularizer(ctx)
            loss = -Q_sum + reg
            loss.backward()

            torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
            opt.step()

            # best-snapshot + optional inner early-stop
            loss_val = float(loss.detach().cpu().item())
            if best_loss is None or loss_val < best_loss:
                best_loss = loss_val
                best_snap = snapshot_params(ctx)
                bad_steps = 0
            else:
                bad_steps += 1
                if patience > 0 and bad_steps >= patience:
                    break

        if best_snap is not None:
            restore_params(ctx, best_snap)

        # ----------------------------
        # Write back to numpy storage + refresh caches
        # ----------------------------
        with torch.no_grad():
            var_s = torch.nn.functional.softplus(ctx["rho_s"]) + ctx["vmin"] + ctx["jitter"]
            var_a = torch.nn.functional.softplus(ctx["rho_a"]) + ctx["vmin"] + ctx["jitter"]
            self.style_gauss.mean = ctx["mu_s"].detach().cpu().numpy()
            self.action_gauss.mean = ctx["mu_a"].detach().cpu().numpy()
            self.style_gauss.var = var_s.detach().cpu().numpy()
            self.action_gauss.var = var_a.detach().cpu().numpy()

            if (ctx["logit_s"] is not None) and (ctx["logit_a"] is not None):
                ps = torch.sigmoid(ctx["logit_s"]).clamp(EPSILON, 1.0 - EPSILON)
                pa = torch.sigmoid(ctx["logit_a"]).clamp(EPSILON, 1.0 - EPSILON)
                self.style_bern.p = ps.detach().cpu().numpy()
                self.action_bern.p = pa.detach().cpu().numpy()

        self.invalidate_cache()
        self.to_device(device=device, dtype=dtype)

        # Masses for logging (same as your current implementation)
        mass_joint = torch.zeros((S, A), device=device, dtype=dtype)
        for g in gamma_sa_seqs:
            mass_joint += as_torch(g, device=device, dtype=dtype).sum(dim=0)
        return {
            "mass_style": mass_joint.sum(dim=1).detach().cpu().numpy(),
            "mass_action": mass_joint.sum(dim=0).detach().cpu().numpy(),
            "mass_joint": mass_joint.detach().cpu().numpy(),
        }

    # -------------------------------------------------------------------------
    # Serialization helpers
    # -------------------------------------------------------------------------
    def to_arrays(self):
        """
        Export PoE emission parameters in a dict.
        """
        payload = dict(
            obs_names=np.array(self.obs_names, dtype=object),
            cont_idx=np.array(self.cont_idx, dtype=np.int64),
            bin_idx=np.array(self.bin_idx, dtype=np.int64),
            bernoulli_names=np.array(self.bernoulli_names, dtype=object),

            style_gauss_mean=np.asarray(self.style_gauss.mean, dtype=np.float64),   # (S,Dc)
            style_gauss_var=np.asarray(self.style_gauss.var, dtype=np.float64),     # (S,Dc)
            action_gauss_mean=np.asarray(self.action_gauss.mean, dtype=np.float64), # (A,Dc)
            action_gauss_var=np.asarray(self.action_gauss.var, dtype=np.float64),   # (A,Dc)

            style_bern_p=np.asarray(self.style_bern.p, dtype=np.float64),           # (S,B)
            action_bern_p=np.asarray(self.action_bern.p, dtype=np.float64),         # (A,B)
        )
        return payload

    def from_arrays(self, payload):
        """
        Load PoE emission parameters from dict.
        """
        self.obs_names = list(np.asarray(payload["obs_names"], dtype=object).tolist())
        self.obs_dim = len(self.obs_names)

        if "bernoulli_names" in payload:
            self.bernoulli_names = list(np.asarray(payload["bernoulli_names"], dtype=object).tolist())
        else:
            self.bernoulli_names = list(BERNOULLI_FEATURES)

        self.cont_idx = list(np.asarray(payload["cont_idx"], dtype=np.int64).tolist())
        self.bin_idx = list(np.asarray(payload["bin_idx"], dtype=np.int64).tolist())
        self.cont_dim = len(self.cont_idx)
        self.bin_dim = len(self.bin_idx)

        # Recreate experts with correct dims
        self.style_gauss = DiagGaussianExpert(K=self.num_style, D=self.cont_dim)
        self.action_gauss = DiagGaussianExpert(K=self.num_action, D=self.cont_dim)
        self.style_bern = BernoulliExpert(K=self.num_style, B=self.bin_dim)
        self.action_bern = BernoulliExpert(K=self.num_action, B=self.bin_dim)

        self.style_gauss.mean = np.asarray(payload["style_gauss_mean"], dtype=np.float64)
        self.style_gauss.var  = np.asarray(payload["style_gauss_var"], dtype=np.float64)
        self.action_gauss.mean = np.asarray(payload["action_gauss_mean"], dtype=np.float64)
        self.action_gauss.var  = np.asarray(payload["action_gauss_var"], dtype=np.float64)

        self.style_bern.p = np.asarray(payload.get("style_bern_p", np.full((self.num_style, self.bin_dim), 0.5)), dtype=np.float64)
        self.action_bern.p = np.asarray(payload.get("action_bern_p", np.full((self.num_action, self.bin_dim), 0.5)), dtype=np.float64)

        # Basic sanity checks
        if self.style_gauss.mean.shape != (self.num_style, self.cont_dim):
            raise ValueError("style_gauss_mean shape mismatch")
        if self.action_gauss.mean.shape != (self.num_action, self.cont_dim):
            raise ValueError("action_gauss_mean shape mismatch")
        if self.style_bern.p.shape != (self.num_style, self.bin_dim):
            raise ValueError("style_bern_p shape mismatch")
        if self.action_bern.p.shape != (self.num_action, self.bin_dim):
            raise ValueError("action_bern_p shape mismatch")

        self.invalidate_cache()