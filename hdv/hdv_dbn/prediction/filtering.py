"""Bayes filtering for discrete DBN latent state z=(style, action)."""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class BeliefState:
    """
    Filtering belief in log-space over latent state (S, A) or (B, S, A).

    Attributes:
        log_prob: Logarithm of the belief probabilities; shape (S, A) or (B, S, A).
        t: Time step index corresponding to the belief state.    
    """
    log_prob: torch.Tensor
    t: int

    @property
    def prob(self):
        """Normalized belief in probability space (shape preserved)."""
        x = self.log_prob
        max_log = x.amax(dim=(-2, -1), keepdim=True) # for numerical stability when exponentiating
        exp_shifted = torch.exp(x - max_log) # shift log-probabilities by max to prevent overflow, shape (S, A) or (B, S, A)
        denom = exp_shifted.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-30) # sum over S and A to get normalization constant, shape (1, 1) or (B, 1, 1)
        return exp_shifted / denom # normalized probabilities, shape (S, A) or (B, S, A)


class StructuredDBNFilter:
    """Bayes filter for discrete DBN with latent state z=(style, action) and known parameters."""
    def __init__(self, *, pi_s0, pi_a0_given_s0, A_s, A_a, device=None, dtype=None):
        """
        Initialize the filter with DBN parameters.

        parameters:
            pi_s0: Initial state distribution p(s0); shape (S,).
            pi_a0_given_s0: Initial action distribution p(a0|s0); shape (S, A).
            A_s: State transition probabilities p(s_t|s_{t-1}); shape (S, S).
            A_a: Action transition probabilities p(a_t|s_t, a_{t-1}); shape (S, A, A).
            device: Optional torch device for storing parameters.
            dtype: Optional torch dtype for storing parameters.
        """
        self.device = device or torch.device("cpu")
        self.dtype = dtype or torch.float32

        self.pi_s0 = torch.as_tensor(pi_s0, device=self.device, dtype=self.dtype)
        self.pi_a0_given_s0 = torch.as_tensor(pi_a0_given_s0, device=self.device, dtype=self.dtype)
        self.A_s = torch.as_tensor(A_s, device=self.device, dtype=self.dtype)
        self.A_a = torch.as_tensor(A_a, device=self.device, dtype=self.dtype)

        self.S = int(self.pi_s0.shape[0])
        self.A = int(self.pi_a0_given_s0.shape[1])

        self._log_pi_s0 = torch.clamp(torch.log(self.pi_s0), min=-1e6)
        self._log_pi_a0_given_s0 = torch.clamp(torch.log(self.pi_a0_given_s0), min=-1e6)
        self._log_A_s = torch.clamp(torch.log(self.A_s), min=-1e6)
        self._log_A_a = torch.clamp(torch.log(self.A_a), min=-1e6)

    def initial_belief(self):
        """Initial joint belief p(s0, a0) in log-space; shape (S, A)."""
        log_belief = self._log_pi_s0[:, None] + self._log_pi_a0_given_s0
        return BeliefState(log_belief, t=0)

    def predict(self, belief_t: BeliefState):
        """
        Prediction step to compute p(s_{t+1}, a_{t+1} | o_{0:t}) in log-space given belief at time t.
        The prediction is computed as:
            p(s_{t+1}, a_{t+1} | o_{0:t}) = sum_{s_t, a_t} p(s_{t+1} | s_t) p(a_{t+1} | s_{t+1}, a_t) p(s_t, a_t | o_{0:t})
        
        parameters:
            belief_t: BeliefState at time t, containing log probabilities over (S, A) and time index t.
        
        returns:
            BeliefState at time t+1, containing log probabilities over (S, A) and time index t+1.
        """
        log_belief = belief_t.log_prob # shape (S, A) or (B, S, A)

        # unify shapes
        if log_belief.ndim == 2: 
            log_belief = log_belief.unsqueeze(0)  # (1,S,A)
            squeeze_out = True
        elif log_belief.ndim == 3: # (B,S,A)
            squeeze_out = False
        else:
            raise ValueError(f"belief_t.log_prob must have ndim 2 or 3, got {log_belief.ndim}")

        # Stage A: sum over previous style s_t to get intermediate log probabilities over (S_next, A_prev)
        m = torch.logsumexp(
            self._log_A_s[None, :, :, None] + log_belief[:, :, None, :],  # (B,S_prev,S_next,A_prev)
            dim=1,                                                        # sum over S_prev
        )  # (B,S_next,A_prev)

        # Stage B: apply p(a_next | s_next, a_prev) and sum over previous action a_t
        log_pred = torch.logsumexp(
            m[:, :, :, None] + self._log_A_a[None, :, :, :],  # (B,S_next,A_prev,A_next)
            dim=2,
        )  # (B,S_next,A_next)

        out = log_pred.squeeze(0) if squeeze_out else log_pred
        return BeliefState(out, t=belief_t.t + 1)

    def update(self, belief_pred: BeliefState, emission_loglik: torch.Tensor):
        """
        Update step to compute posterior belief p(s_t, a_t | o_{0:t}) in log-space given predicted belief and emission log-likelihood.
        The update is computed as:
            p(s_t, a_t | o_{0:t}) = p(o_t | s_t, a_t) p(s_t, a_t | o_{0:t-1}) / p(o_t | o_{0:t-1})
        where p(o_t | o_{0:t-1}) is the normalization constant computed by summing over all (s_t, a_t).

        parameters:
            belief_pred: Predicted BeliefState at time t, containing log probabilities over (S, A) and time index t.
            emission_loglik: Log-likelihood of the current observation given (s_t, a_t); shape (S, A), (B, S, A), or (B, T, S, A) where T is the number of time steps in the batch.

        returns:
            BeliefState at time t, containing updated log probabilities over (S, A) and time index t.
        """
        # Predicted (prior) belief for time t in log-space: log p(z_t | o_{1:t-1})
        log_belief_pred = belief_pred.log_prob # shape (S, A) or (B, S, A)

        # Emission log-likelihood(s) for the current observation: log p(o_t | z_t)
        log_emit = emission_loglik # shape (S, A), (B, S, A), or (B, T, S, A)

        # -----------------------------
        # 1) Normalize emission shape
        # -----------------------------
        # If emissions are provided for a whole sequence (B,T,S,A), use only the latest step
        if log_emit.ndim == 4:              
            log_emit = log_emit[:, -1]     # (B,S,A)
        
        # If belief is batched (B,S,A) but emissions are unbatched (S,A), this is usually a bug:
        # independent streams must have their own per-stream likelihoods.
        elif log_emit.ndim == 2 and log_belief_pred.ndim == 3:
            raise ValueError(
                "belief_pred is batched but emission_loglik is unbatched; "
                "pass per-stream emissions with shape (B, S, A)."
            )
        
        # If emissions are batched (B,S,A) but belief is unbatched (S,A),
        # only allow B==1 (otherwise ambiguous which emission corresponds to the single belief). which emission corresponds to the single belief
        elif log_emit.ndim == 3 and log_belief_pred.ndim == 2: 
            if log_emit.shape[0] != 1:
                raise ValueError(
                    "emission_loglik has batch dimension > 1 while belief_pred is unbatched; "
                    "pass a single (S, A) emission or batched belief_pred."
                )
            log_emit = log_emit.squeeze(0)        # shape (S, A)

        # Reject any other unexpected emission shapes early.
        elif log_emit.ndim not in (2, 3):                      
            raise ValueError(f"emission_loglik must have ndim 2/3/4, got {log_emit.ndim}")

        # -----------------------------------------
        # 2) Align belief/emission shapes (B vs 1)
        # -----------------------------------------
        # If belief is unbatched (S,A) but emissions are (1,S,A), upgrade belief to (1,S,A) for uniform math.
        if log_belief_pred.ndim == 2 and log_emit.ndim == 3: 
            if log_emit.shape[0] != 1: # if not exactly 1, this is likely a bug: either the belief should be batched or the emissions should not have a batch dimension
                raise ValueError(
                    "belief_pred is unbatched but emission_loglik has batch dimension > 1; "
                    "pass unbatched emission (S, A) or batched belief_pred."
                )
            log_belief_pred = log_belief_pred.unsqueeze(0) # shape (1, S, A)

        # If belief is (1,S,A) but emissions are unbatched (S,A), upgrade emissions to (1,S,A).
        if log_belief_pred.ndim == 3 and log_emit.ndim == 2: # batched belief with unbatched emissions is only valid for batch size 1
            if log_belief_pred.shape[0] != 1:
                raise ValueError(
                    "belief_pred has batch dimension > 1 while emission_loglik is unbatched; "
                    "pass a single (S, A) emission or batched belief_pred."
                )
            log_emit = log_emit.unsqueeze(0) # shape (1, S, A)

        # After all conversions, shapes must match exactly to avoid accidental broadcasting.
        if tuple(log_emit.shape) != tuple(log_belief_pred.shape): 
            raise ValueError(
                f"Emission shape {tuple(log_emit.shape)} incompatible with belief_pred {tuple(log_belief_pred.shape)}"
            )

        # -----------------------------
        # 3) Bayes update in log-space
        # -----------------------------
        # Unnormalized posterior:
        #   log p(z_t | o_{1:t}) ∝ log p(o_t | z_t) + log p(z_t | o_{1:t-1})
        log_unnorm = log_emit + log_belief_pred

        # Normalization constant (logZ) per stream:
        #   logZ = log sum_{s,a} exp(log_unnorm[s,a])
        logZ = torch.logsumexp(log_unnorm, dim=(-2, -1), keepdim=True)

        # Normalized posterior in log-space:
        log_post = log_unnorm - logZ

        # If the caller provided an unbatched belief (S,A) but we temporarily promoted to (1,S,A),
        # squeeze back to match the original unbatched API.
        if belief_pred.log_prob.ndim == 2 and log_post.ndim == 3:
            log_post = log_post.squeeze(0)

        return BeliefState(log_post, t=belief_pred.t)