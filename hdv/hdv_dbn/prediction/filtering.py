"""
Bayes Filtering for Discrete-Latent DBNs.
Implements the recursive Bayes filter (predict + update) for discrete joint latents.

1. Predict step: use transitions to move belief forward in time.
2. Update step: multiply by how likely the observation is under each (s,a) and normalize.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch


EPS = 1e-8


@dataclass(frozen=True)
class BeliefState:
    """
    Filtering posterior belief at time t.
    
    Attributes
    log_prob : torch.Tensor
        Log of unnormalized belief, shape (*latent_shape).
        For joint (S, A) latent: log_prob.shape = (S, A).
    t : int
        Time index I it was last updated.
    """
    log_prob: torch.Tensor
    t: int
    
    @property
    def prob(self):
        """Normalized belief in probability space."""
        # log_sum_exp trick for numerical stability
        max_log = torch.max(self.log_prob)
        exp_shifted = torch.exp(self.log_prob - max_log)
        return exp_shifted / torch.sum(exp_shifted)


class BayesFilter(ABC):
    """
    Abstract base for Bayes filtering on discrete latent spaces.
    Subclasses implement:
      - transition parameters
      - initial belief
      - predict/update steps
    """
    
    @abstractmethod
    def initial_belief(self):
        """p(z_0) → BeliefState."""
        pass
    
    @abstractmethod
    def predict(self, belief_t):
        """
        Predict step: p(z_{t+1} | o_{1:t}) = sum_{z_t} p(z_{t+1}|z_t) * p(z_t | o_{1:t}).
        """
        pass
    
    @abstractmethod
    def update(self, belief_pred, emission_loglik):
        """
        Update step: p(z_t | o_{1:t}) ∝ p(o_t | z_t) * p(z_t | o_{1:t-1}).
        
        Parameters
        belief_pred : BeliefState
            Prior p(z_t | o_{1:t-1}) from predict step (in log domain).
        emission_loglik : torch.Tensor
            Log emission p(o_t | z), shape matching latent space.
        
        Returns
        BeliefState
            Posterior with normalized log_prob.
        """
        pass


class StructuredDBNFilter(BayesFilter):
    """
    Bayes filter for factorized discrete DBN (style, action).
    Latent: z_t = (s_t, a_t) where s_t ∈ {0, ..., S-1}, a_t ∈ {0, ..., A-1}.
    
    Transitions:
      p(s_{t+1} | s_t) = A_s[s_t, s_{t+1}]
      p(a_{t+1} | a_t, s_{t+1}) = A_a[s_{t+1}, a_t, a_{t+1}]
    
    Joint: p(z_{t+1} | z_t) = p(s_{t+1} | s_t) * p(a_{t+1} | a_t, s_{t+1})
    
    Parameters
    pi_s0 : torch.Tensor, shape (S,)
        Initial style prior.
    pi_a0_given_s0 : torch.Tensor, shape (S, A)
        Initial action prior conditioned on style.
    A_s : torch.Tensor, shape (S, S)
        Style transition matrix.
    A_a : torch.Tensor, shape (S, A, A)
        Action transition tensor: A_a[s_next, a_prev, a_next].
    device : torch.device, optional
        GPU/CPU placement.
    dtype : torch.dtype, optional
        Precision (default: float32).
    """
    
    def __init__(self, *, pi_s0, pi_a0_given_s0, A_s, A_a, device=None, dtype=None):
        self.device = device or torch.device("cpu")
        self.dtype = dtype or torch.float32
        
        # Move to device/dtype
        self.pi_s0 = torch.as_tensor(pi_s0, device=self.device, dtype=self.dtype)
        self.pi_a0_given_s0 = torch.as_tensor(pi_a0_given_s0, device=self.device, dtype=self.dtype)
        self.A_s = torch.as_tensor(A_s, device=self.device, dtype=self.dtype)
        self.A_a = torch.as_tensor(A_a, device=self.device, dtype=self.dtype)
        
        self.S = int(self.pi_s0.shape[0])
        self.A = int(self.pi_a0_given_s0.shape[1])
        
        # Pre-log for numerical stability
        self._log_pi_s0 = torch.clamp(torch.log(self.pi_s0), min=-1e6)
        self._log_pi_a0_given_s0 = torch.clamp(torch.log(self.pi_a0_given_s0), min=-1e6)
        self._log_A_s = torch.clamp(torch.log(self.A_s), min=-1e6)
        self._log_A_a = torch.clamp(torch.log(self.A_a), min=-1e6)
    
    def initial_belief(self):
        """
        p(z_0) = p(s_0) * p(a_0 | s_0).
        
        Returns log belief of shape (S, A).
        """
        # p_s0[:, None] * p_a0[None, :] broadcasts to (S, A)
        log_belief = self._log_pi_s0[:, None] + self._log_pi_a0_given_s0
        return BeliefState(log_belief, t=0)
    
    def predict(self, belief_t):
        """
        Predict:
            p(s', a' | o_{1:t}) = sum_{s,a} p(s'|s) p(a'|a, s') p(s,a | o_{1:t})

        Computed in log-space via log-sum-exp.
        
        Shapes
        log_belief_t : (S, A)
        log_A_s      : (S, S) where log_A_s[s, s']
        log_A_a      : (S, A, A) where log_A_a[s', a, a']

        Returns
        BeliefState
            Prior belief at time t+1, log-space, shape (S, A).
            (This is a predicted prior, not yet updated with emissions.)
        """
        log_belief = belief_t.log_prob  # log p(s,a | o_{1:t}), shape (S, A).
        
        # Step 1:
        # For each s' and a_prev, compute:
        #   m[s', a_prev] = logsumexp_s ( log_A_s[s, s'] + log_belief[s, a_prev] )
        #
        # Shapes:
        #   log_A_s: (S, S)
        #   log_belief: (S, A)
        # We want output m: (S', A_prev) = (S, A)
        m = torch.logsumexp(
            self._log_A_s.unsqueeze(2) + log_belief.unsqueeze(1),  # (S, S', A)
            dim=0                                                   # sum over s
        )  # (S', A_prev)

        # Step 2:
        # For each s' and a_next:
        #   log_pred[s', a_next] = logsumexp_{a_prev} ( m[s', a_prev] + log_A_a[s', a_prev, a_next] )
        #
        log_pred = torch.logsumexp(
            m.unsqueeze(2) + self._log_A_a,  # (S', A_prev, A_next)
            dim=1                             # sum over a_prev
        )  # (S', A_next)

        return BeliefState(log_pred, t=belief_t.t + 1)
    
    def update(self, belief_pred, emission_loglik):
        """
        Update: p(z_t | O_{1:t}) ∝ p(O_t | z_t) * p(z_t | O_{1:t-1}).
        
        Parameters
        belief_pred : BeliefState
            Prior from predict step, shape (S, A).
        emission_loglik : torch.Tensor
            Log emission, shape (S, A) or (T, S, A) if a sequence.
            If (T, S, A), uses the last (t=T-1) slice.
        
        Returns
        BeliefState
            Updated belief with normalized log_prob.
        """
        log_belief_pred = belief_pred.log_prob  # (S, A)
        
        # Extract emission for this timestep if needed
        if emission_loglik.ndim == 3:
            log_emit = emission_loglik[-1]
        else:
            log_emit = emission_loglik
        
        log_unnorm = log_emit + log_belief_pred
        log_norm = log_unnorm - torch.logsumexp(log_unnorm.reshape(-1), dim=0)
        return BeliefState(log_norm, t=belief_pred.t)