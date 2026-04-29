"""
Tied action-emission model for the HDV DBN.

This model keeps the hidden state z_t = (s_t, a_t), but it does not learn
one full emission for every (style, action) pair.

Instead, it uses two feature blocks:

    style features  -> p(o_style,t  | s_t)
    action features -> p(o_action,t | a_t)

Then:

    log p(o_t | s_t, a_t)
        = log p(o_style,t  | s_t) + log p(o_action,t | a_t)

This ties the action emission across styles. Therefore, action_0, action_1,
action_2, and action_3 keep the same emission meaning in both style states.
"""
import numpy as np
import torch
from tqdm.auto import tqdm

from .config import (
    DBN_STATES,
    TRAINING_CONFIG,
    BERNOULLI_FEATURES,
    TIED_ACTION_FEATURES,
    TIED_STYLE_FEATURES,
)
from .poe_emissions import DiagGaussianExpert, BernoulliExpert
from .utils.poe_utils import as_torch



# =============================================================================
# Helper functions
# =============================================================================
def _unique_keep_order(names):
    """Remove duplicates but keep the original order."""
    out = []
    seen = set()

    for n in names:
        if n not in seen:
            out.append(n)
            seen.add(n)

    return out


# =============================================================================
# Tied action mixed emission model
# =============================================================================
class MixedEmissionModel:
    """
    Tied action-emission model.
    The style expert sees only style/context features.
    The action expert sees only action/maneuver features.

    Style emission:
        p(o_style | s)

    Action emission:
        p(o_action | a)

    Combined emission:
        logB[t, s, a] = log p(o_style,t | s) + log p(o_action,t | a)
    
    Returns:
        logB_sa with shape (T, S, A)
    """
    emission_kind = "tied_action"

    def __init__(self, obs_names, disable_discrete_obs=False, bernoulli_names=None):
        self.obs_names = list(obs_names)
        self.obs_dim = len(self.obs_names)

        self.style_states = DBN_STATES.driving_style
        self.action_states = DBN_STATES.action

        self.num_style = len(self.style_states)
        self.num_action = len(self.action_states)

        self.disable_discrete_obs = bool(disable_discrete_obs)

        if bernoulli_names is None:
            bernoulli_names = list(BERNOULLI_FEATURES)
        self.bernoulli_names = list(bernoulli_names)

        name_to_idx = {n: i for i, n in enumerate(self.obs_names)}

        # -------------------------------------------------------------
        # Resolve feature split
        # -------------------------------------------------------------
        self.action_feature_names = _unique_keep_order(
            [n for n in TIED_ACTION_FEATURES if n in name_to_idx]
        )
        self.style_feature_names = _unique_keep_order(
            [n for n in TIED_STYLE_FEATURES if n in name_to_idx]
        )

        # Safety: no feature should be used in both blocks.
        action_set = set(self.action_feature_names)
        self.style_feature_names = [
            n for n in self.style_feature_names
            if n not in action_set
        ]

        if not self.action_feature_names:
            raise ValueError("No valid action features found for tied_action emission.")

        if not self.style_feature_names:
            raise ValueError("No valid style features found for tied_action emission.")


        self.action_idx = [name_to_idx[n] for n in self.action_feature_names]
        self.style_idx = [name_to_idx[n] for n in self.style_feature_names]

        bern_idx = {
            name_to_idx[n]
            for n in self.bernoulli_names
            if n in name_to_idx
        }

        self.action_bin_idx = [i for i in self.action_idx if i in bern_idx]
        self.style_bin_idx = [i for i in self.style_idx if i in bern_idx]

        self.action_cont_idx = [i for i in self.action_idx if i not in bern_idx]
        self.style_cont_idx = [i for i in self.style_idx if i not in bern_idx]

        self.action_cont_dim = len(self.action_cont_idx)
        self.style_cont_dim = len(self.style_cont_idx)

        self.action_bin_dim = len(self.action_bin_idx)
        self.style_bin_dim = len(self.style_bin_idx)

        if self.action_cont_dim <= 0:
            raise ValueError("tied_action action block has no continuous features.")

        if self.style_cont_dim <= 0:
            raise ValueError("tied_action style block has no continuous features.")


        # -------------------------------------------------------------
        # Experts
        # -------------------------------------------------------------
        self.style_gauss = DiagGaussianExpert(
            K=self.num_style,
            D=self.style_cont_dim,
        )
        self.action_gauss = DiagGaussianExpert(
            K=self.num_action,
            D=self.action_cont_dim,
        )

        self.style_bern = BernoulliExpert(
            K=self.num_style,
            B=self.style_bin_dim,
        )

        self.action_bern = BernoulliExpert(
            K=self.num_action,
            B=self.action_bin_dim,
        )

        self._device = torch.device("cpu")
        self._dtype = torch.float32

    # -------------------------------------------------------------------------
    # Device/cache handling
    # -------------------------------------------------------------------------

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

    def _ensure_device(self, device=None, dtype=None):
        device = self._device if device is None else torch.device(device)
        dtype = self._dtype if dtype is None else dtype

        need = False
        need = need or (self.style_cont_dim > 0 and self.style_gauss._mean_t is None)
        need = need or (self.action_cont_dim > 0 and self.action_gauss._mean_t is None)
        need = need or (self.style_bin_dim > 0 and self.style_bern._p_t is None)
        need = need or (self.action_bin_dim > 0 and self.action_bern._p_t is None)
        if need or self._device != device or self._dtype != dtype:
            self.to_device(device=device, dtype=dtype)

    # -------------------------------------------------------------------------
    # Feature extraction
    # -------------------------------------------------------------------------

    def _extract_cont(self, x, idx):
        T = int(x.shape[0])

        if len(idx) == 0:
            z = torch.empty((T, 0), device=x.device, dtype=x.dtype)
            return z, z

        raw = x[:, idx]
        finite = torch.isfinite(raw)

        mask = finite.to(dtype=x.dtype)
        clean = torch.where(finite, raw, torch.zeros_like(raw))

        return clean, mask


    def _extract_bin(self, x, idx):
        T = int(x.shape[0])

        if len(idx) == 0:
            z = torch.empty((T, 0), device=x.device, dtype=x.dtype)
            return z, z

        raw = x[:, idx]
        finite = torch.isfinite(raw)

        mask = finite.to(dtype=x.dtype)
        clean = torch.where(finite, raw, torch.zeros_like(raw))
        binary = (clean > 0.5).to(dtype=x.dtype)

        return binary, mask

    # ------------------------------------------------------------------
    # Likelihood
    # ------------------------------------------------------------------
    def loglikelihood(self, obs, device=None, dtype=None):
        """
        Compute emission log-likelihoods.

        Returns:
            logB_sa: torch.Tensor, shape (T, S, A)
        """
        self._ensure_device(device=device, dtype=dtype)

        device = self._device
        dtype = self._dtype

        x = as_torch(obs, device=device, dtype=dtype)

        if x.ndim != 2 or x.shape[1] != self.obs_dim:
            raise ValueError(
                f"Expected obs shape (T,{self.obs_dim}), got {tuple(x.shape)}"
            )
        
        # -------------------------------------------------------------
        # Style likelihood
        # -------------------------------------------------------------
        style_cont, style_cont_mask = self._extract_cont(x, self.style_cont_idx)
        log_style = self.style_gauss.loglikelihood(style_cont, mask=style_cont_mask)

        # -------------------------------------------------------------
        # Action likelihood
        # -------------------------------------------------------------
        action_cont, action_cont_mask = self._extract_cont(x, self.action_cont_idx)
        log_action = self.action_gauss.loglikelihood(action_cont, mask=action_cont_mask)

        # -------------------------------------------------------------
        # Optional Bernoulli likelihood
        # -------------------------------------------------------------
        if not self.disable_discrete_obs:
            w_bern = float(getattr(TRAINING_CONFIG, "bern_weight", 1.0))

            if self.style_bin_dim > 0:
                style_bin, _ = self._extract_bin(x, self.style_bin_idx)
                log_style = log_style + w_bern * self.style_bern.loglikelihood(
                    style_bin
                )
            if self.action_bin_dim > 0:
                action_bin, _ = self._extract_bin(x, self.action_bin_idx)
                log_action = log_action + w_bern * self.action_bern.loglikelihood(
                    action_bin
                )

        # Combine into joint emission likelihood.
        return log_style[:, :, None] + log_action[:, None, :]

    # -------------------------------------------------------------------------
    # EM M-step
    # -------------------------------------------------------------------------
    def update_from_posteriors(self, obs_seqs, gamma_sa_seqs, use_progress=True, device=None, dtype=None):
        """
        Update style/action emissions from posterior responsibilities.

        Important:
            gamma_s = sum_a gamma_sa
            gamma_a = sum_s gamma_sa

        This is the parameter tying step.
        """
        self._ensure_device(device=device, dtype=dtype)

        device = self._device
        dtype = self._dtype

        style_cont_seqs = []
        style_cont_masks = []

        action_cont_seqs = []
        action_cont_masks = []

        style_bin_seqs = []
        style_bin_masks = []

        action_bin_seqs = []
        action_bin_masks = []

        gamma_s_seqs = []
        gamma_a_seqs = []

        iterator = zip(obs_seqs, gamma_sa_seqs)
        if use_progress:
            iterator = tqdm(iterator, total=len(obs_seqs), desc="M-step emissions (tied_action)", leave=False)

        for obs, gamma_sa in iterator:
            x = as_torch(obs, device=device, dtype=dtype)
            g = as_torch(gamma_sa, device=device, dtype=dtype)

            T = int(x.shape[0])

            if x.ndim != 2 or x.shape[1] != self.obs_dim:
                raise ValueError(f"Expected obs shape (T,{self.obs_dim}), got {tuple(x.shape)}")

            if g.ndim != 3 or g.shape != (T, self.num_style, self.num_action):
                raise ValueError(
                    f"Expected gamma_sa shape ({T},{self.num_style},{self.num_action}), "
                    f"got {tuple(g.shape)}"
                )

            # This is the key tied-action step: we sum out the other variable to get the responsibilities for each block
            gamma_s = g.sum(dim=2)  # (T, S)
            gamma_a = g.sum(dim=1)  # (T, A)

            gamma_s_seqs.append(gamma_s)
            gamma_a_seqs.append(gamma_a)

            style_cont, style_cont_mask = self._extract_cont(x, self.style_cont_idx)
            action_cont, action_cont_mask = self._extract_cont(x, self.action_cont_idx)

            style_cont_seqs.append(style_cont)
            style_cont_masks.append(style_cont_mask)

            action_cont_seqs.append(action_cont)
            action_cont_masks.append(action_cont_mask)

            if not self.disable_discrete_obs:
                style_bin, style_bin_mask = self._extract_bin(x, self.style_bin_idx)
                action_bin, action_bin_mask = self._extract_bin(x, self.action_bin_idx)

                style_bin_seqs.append(style_bin)
                style_bin_masks.append(style_bin_mask)

                action_bin_seqs.append(action_bin)
                action_bin_masks.append(action_bin_mask)

        self.style_gauss.m_step(
            style_cont_seqs,
            gamma_s_seqs,
            mask_seqs=style_cont_masks,
            device=device,
            dtype=dtype,
            use_progress=False,
        )

        self.action_gauss.m_step(
            action_cont_seqs,
            gamma_a_seqs,
            mask_seqs=action_cont_masks,
            device=device,
            dtype=dtype,
            use_progress=False,
        )

        if not self.disable_discrete_obs:
            if self.style_bin_dim > 0:
                self.style_bern.m_step(
                    style_bin_seqs,
                    gamma_s_seqs,
                    finite_mask_seqs=style_bin_masks,
                    device=device,
                    dtype=dtype,
                    use_progress=False,
                )

            if self.action_bin_dim > 0:
                self.action_bern.m_step(
                    action_bin_seqs,
                    gamma_a_seqs,
                    finite_mask_seqs=action_bin_masks,
                    device=device,
                    dtype=dtype,
                    use_progress=False,
                )

        self.invalidate_cache()

        mass_joint = torch.zeros((self.num_style, self.num_action), device=device, dtype=dtype)

        for g in gamma_sa_seqs:
            mass_joint += as_torch(g, device=device, dtype=dtype).sum(dim=0)

        return {
            "mass_style": mass_joint.sum(dim=1).detach().cpu().numpy(),
            "mass_action": mass_joint.sum(dim=0).detach().cpu().numpy(),
            "mass_joint": mass_joint.detach().cpu().numpy(),
        }

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------
    def to_arrays(self):
        """
        Export emission parameters to numpy arrays.
        """
        return {
            "obs_names": np.array(self.obs_names, dtype=object),
            "bernoulli_names": np.array(self.bernoulli_names, dtype=object),

            "style_feature_names": np.array(self.style_feature_names, dtype=object),
            "action_feature_names": np.array(self.action_feature_names, dtype=object),

            "style_idx": np.array(self.style_idx, dtype=np.int64),
            "action_idx": np.array(self.action_idx, dtype=np.int64),

            "style_cont_idx": np.array(self.style_cont_idx, dtype=np.int64),
            "action_cont_idx": np.array(self.action_cont_idx, dtype=np.int64),

            "style_bin_idx": np.array(self.style_bin_idx, dtype=np.int64),
            "action_bin_idx": np.array(self.action_bin_idx, dtype=np.int64),

            "style_gauss_mean": np.asarray(self.style_gauss.mean, dtype=np.float64),
            "style_gauss_var": np.asarray(self.style_gauss.var, dtype=np.float64),

            "action_gauss_mean": np.asarray(self.action_gauss.mean, dtype=np.float64),
            "action_gauss_var": np.asarray(self.action_gauss.var, dtype=np.float64),

            "style_bern_p": np.asarray(self.style_bern.p, dtype=np.float64),
            "action_bern_p": np.asarray(self.action_bern.p, dtype=np.float64),
        }

    def from_arrays(self, payload):
        """
        Load emission parameters from numpy arrays.
        """
        self.obs_names = list(np.asarray(payload["obs_names"], dtype=object).tolist())
        self.obs_dim = len(self.obs_names)

        self.bernoulli_names = list(np.asarray(payload["bernoulli_names"], dtype=object).tolist())

        self.style_feature_names = list(np.asarray(payload["style_feature_names"], dtype=object).tolist())
        self.action_feature_names = list(np.asarray(payload["action_feature_names"], dtype=object).tolist())

        self.style_idx = list(np.asarray(payload["style_idx"], dtype=np.int64).tolist())
        self.action_idx = list(np.asarray(payload["action_idx"], dtype=np.int64).tolist())

        self.style_cont_idx = list(np.asarray(payload["style_cont_idx"], dtype=np.int64).tolist())
        self.action_cont_idx = list(np.asarray(payload["action_cont_idx"], dtype=np.int64).tolist())

        self.style_bin_idx = list(np.asarray(payload["style_bin_idx"], dtype=np.int64).tolist())
        self.action_bin_idx = list(np.asarray(payload["action_bin_idx"], dtype=np.int64).tolist())

        self.style_cont_dim = len(self.style_cont_idx)
        self.action_cont_dim = len(self.action_cont_idx)
        self.style_bin_dim = len(self.style_bin_idx)
        self.action_bin_dim = len(self.action_bin_idx)

        self.style_gauss = DiagGaussianExpert(self.num_style, self.style_cont_dim)
        self.action_gauss = DiagGaussianExpert(self.num_action, self.action_cont_dim)

        self.style_bern = BernoulliExpert(self.num_style, self.style_bin_dim)
        self.action_bern = BernoulliExpert(self.num_action, self.action_bin_dim)

        self.style_gauss.mean = np.asarray(payload["style_gauss_mean"], dtype=np.float64)
        self.style_gauss.var = np.asarray(payload["style_gauss_var"], dtype=np.float64)

        self.action_gauss.mean = np.asarray(payload["action_gauss_mean"], dtype=np.float64)
        self.action_gauss.var = np.asarray(payload["action_gauss_var"], dtype=np.float64)

        self.style_bern.p = np.asarray(payload["style_bern_p"], dtype=np.float64)
        self.action_bern.p = np.asarray(payload["action_bern_p"], dtype=np.float64)

        if self.style_gauss.mean.shape != (self.num_style, self.style_cont_dim):
            raise ValueError("style_gauss_mean shape mismatch")
        if self.action_gauss.mean.shape != (self.num_action, self.action_cont_dim):
            raise ValueError("action_gauss_mean shape mismatch")
        if self.style_bern.p.shape != (self.num_style, self.style_bin_dim):
            raise ValueError("style_bern_p shape mismatch")
        if self.action_bern.p.shape != (self.num_action, self.action_bin_dim):
            raise ValueError("action_bern_p shape mismatch")

        self.invalidate_cache()