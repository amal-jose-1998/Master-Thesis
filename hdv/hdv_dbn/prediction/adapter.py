from dataclasses import dataclass
import numpy as np
import torch


@dataclass(frozen=True)
class StructuredParams:
    """Structured DBN parameters required for filtering and rollout."""
    pi_s0: torch.Tensor           # (S,)
    pi_a0_given_s0: torch.Tensor  # (S,A)
    A_s: torch.Tensor             # (S,S)
    A_a: torch.Tensor             # (S,A,A)


class HdvDbnAdapter:
    """Thin wrapper around the trained DBN model for online belief + rollout predict"""
    def __init__(self, dbn, S, A, device=None, dtype=None):
        self.dbn = dbn
        self.S = int(S)
        self.A = int(A)
        
        dbn_device = getattr(dbn, "device", None)
        dbn_dtype = getattr(dbn, "dtype", None)

        self.device = torch.device(device) if device is not None else (
            torch.device(dbn_device) if dbn_device is not None else torch.device("cpu")
        )
        self.dtype = dtype if dtype is not None else (dbn_dtype if dbn_dtype is not None else torch.float32)

        if not hasattr(dbn, "emissions") or not hasattr(dbn.emissions, "loglikelihood"):
            raise AttributeError("dbn must expose `.emissions.loglikelihood(obs_seq)` returning (T,S,A).")

        for nm in ("pi_s0", "pi_a0_given_s0", "A_s", "A_a"):
            if not hasattr(dbn, nm):
                raise AttributeError(f"dbn is missing required attribute `{nm}` (expected HDVTrainer-like object).")

        self._params = None
        self._P_joint = None  # (SA, SA)

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _as_torch(self, x):
        t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        t = t.to(dtype=self.dtype, device=self.device)
        return t

    # ----------------------------
    # Emissions: log p(o_t | s,a)
    # ----------------------------
    def loglikelihood_sequence(self, obs_seq):
        """
        compute log-likelihood table for a whole trajectory once.

        obs_seq: (T, D)
        returns: logB (T, S, A)
        """
        x = self._as_torch(obs_seq)
        logB = self.dbn.emissions.loglikelihood(x)
        logB = self._as_torch(logB)

        if logB.ndim != 3:
            raise ValueError(f"Expected logB ndim=3 (T,S,A), got shape {tuple(logB.shape)}")
        if logB.shape[1] != self.S or logB.shape[2] != self.A:
            raise ValueError(f"Expected logB shape (T,{self.S},{self.A}), got {tuple(logB.shape)}")
        return logB
    
    def loglikelihood_sa(self, o_t):
        """Compute log p(o_t | s,a), shape (S,A), for a single timestep."""
        o_t = self._as_torch(o_t).unsqueeze(0)  # (1, D)
        logB = self.loglikelihood_sequence(o_t)  # (1,S,A)
        return logB[0] # (S, A)

    # ----------------------------
    # Structured transitions + priors
    # ----------------------------
    def params(self):
        """
        Fetch and cache (pi_s0, pi_a0_given_s0, A_s, A_a) as torch tensors on adapter device/dtype.
        """
        if self._params is not None:
            return self._params

        p = StructuredParams(
            pi_s0=self._as_torch(self.dbn.pi_s0),
            pi_a0_given_s0=self._as_torch(self.dbn.pi_a0_given_s0),
            A_s=self._as_torch(self.dbn.A_s),
            A_a=self._as_torch(self.dbn.A_a),
        )
        # sanity
        if p.pi_s0.shape != (self.S,):
            raise ValueError(f"pi_s0 must be {(self.S,)}, got {tuple(p.pi_s0.shape)}")
        if p.pi_a0_given_s0.shape != (self.S, self.A):
            raise ValueError(f"pi_a0_given_s0 must be ({self.S},{self.A}), got {tuple(p.pi_a0_given_s0.shape)}")
        if p.A_s.shape != (self.S, self.S):
            raise ValueError(f"A_s must be ({self.S},{self.S}), got {tuple(p.A_s.shape)}")
        if p.A_a.shape != (self.S, self.A, self.A):
            raise ValueError(f"A_a must be ({self.S},{self.A},{self.A}), got {tuple(p.A_a.shape)}")

        self._params = p
        return p

    # ---------------------------------------------------------------------
    # Joint transition matrix for rollout
    # ---------------------------------------------------------------------
    def joint_transition_matrix(self, cache=True):
        """
        Build joint transition matrix P of shape (S*A, S*A):
          P[z_prev, z_next] = p(z_next | z_prev)
        where z = s*A + a.
        """
        if cache and self._P_joint is not None:
            return self._P_joint

        p = self.params()
        A_s = p.A_s  # (S,S)
        A_a = p.A_a  # (S,A,A) indexed by s_next

        # P4[s_prev, a_prev, s_next, a_next] = A_s[s_prev,s_next] * A_a[s_next,a_prev,a_next]
        P4 = A_s[:, None, :, None] * A_a.permute(1, 0, 2)[None, :, :, :]  # (S,A,S,A)
        P = P4.reshape(self.S * self.A, self.S * self.A)

        # Defensive normalization (should already be normalized if A_s/A_a are)
        row_sums = P.sum(dim=1, keepdim=True)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6, rtol=1e-6):
            print("[adapter] joint transition matrix row sums not 1; renormalizing")
        P = P / row_sums.clamp_min(1e-12)

        if cache:
            self._P_joint = P
        return P

    # ----------------------------
    # Index helpers
    # ----------------------------
    def z_index(self, s, a):
        return int(s) * self.A + int(a)

    def unindex(self, z):
        z = int(z)
        return z // self.A, z % self.A
