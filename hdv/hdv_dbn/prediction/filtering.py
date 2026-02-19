"""
Bayes Filtering for Discrete-Latent DBNs. Implements the recursive Bayes filter (predict + update) for discrete joint latents.
1. Predict step: use transitions to move belief forward in time.
2. Update step: multiply by how likely the observation is under each (s,a) and normalize.
"""

from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class BeliefState:
    """
    Filtering posterior belief at time t.
    
    Attributes
    log_prob : torch.Tensor
        Log of unnormalized belief. For joint (S, A) latent: log_prob.shape = (S, A).
    t : int
        Time index the belief corresponds to.
    """
    log_prob: torch.Tensor # log-space belief over the joint latent (s,a) at time t; shape (S, A).
    t: int # time index corresponding to this belief state (after processing observation at time t).
    
    @property
    def prob(self):
        """Normalized belief in probability space."""
        # Converts log-space belief into normalized probability-space belief.
        max_log = torch.max(self.log_prob) # for numerical stability: subtract max log-prob before exponentiating to avoid overflow.
        exp_shifted = torch.exp(self.log_prob - max_log) # shift log-probs so max is at 0, then exponentiate to get unnormalized probabilities.
        return exp_shifted / torch.sum(exp_shifted) # normalize to sum to 1


class StructuredDBNFilter():
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
        
        # Precompute logs for numerical stability
        self._log_pi_s0 = torch.clamp(torch.log(self.pi_s0), min=-1e6) # Clamps to at least -1e6 to avoid -inf if any entry is 0.
        self._log_pi_a0_given_s0 = torch.clamp(torch.log(self.pi_a0_given_s0), min=-1e6)
        self._log_A_s = torch.clamp(torch.log(self.A_s), min=-1e6)
        self._log_A_a = torch.clamp(torch.log(self.A_a), min=-1e6)
    
    def initial_belief(self):
        """
        build the initial joint belief:  p(z_0) = p(s_0) * p(a_0 | s_0).
        
        Returns log belief of shape (S, A).
        """
        # p_s0[:, None] * p_a0[None, :] broadcasts to (S, A)
        log_belief = self._log_pi_s0[:, None] + self._log_pi_a0_given_s0
        return BeliefState(log_belief, t=0) # belief at time 0.
    
    def predict(self, belief_t: BeliefState):
        """
        Do transition “predict” step to compute prior for next step.
        Predict:
            p(s', a' | o_{1:t}) = sum_{s,a} p(s'|s) p(a'|a, s') p(s,a | o_{1:t})
        
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
        # log_belief[s,a] = log p(s_t=s, a_t=a | O_{1:t})
        
        # Step 1: sum out previous style s to get intermediate m[s', a_prev] = sum_s log p(s'|s) + log p(s,a_prev | o_{1:t})
        m = torch.logsumexp(
            self._log_A_s.unsqueeze(2) + log_belief.unsqueeze(1),  # (S, S', A)
            dim=0                                                   # sum over s
        )  # (S', A_prev)

        # Step 2: sum out previous action a_prev to get log p(s', a' | o_{1:t}) = sum_{a_prev} m[s', a_prev] + log p(a'|a_prev, s')
        log_pred = torch.logsumexp(
            m.unsqueeze(2) + self._log_A_a,  # (S', A_prev, A_next)
            dim=1                             # sum over a_prev
        )  # (S', A_next)
        # log_pred is log prior for (s_{t+1}, a_{t+1}) given observations up to time t.
        return BeliefState(log_pred, t=belief_t.t + 1) # Increments time index because this is belief for the next step.
    
    def update(self, belief_pred: BeliefState, emission_loglik: torch.Tensor):
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
        log_belief_pred = belief_pred.log_prob  # log prior over (S,A).
        
        # Extract emission for this timestep if needed
        if emission_loglik.ndim == 3:
            log_emit = emission_loglik[-1] # If given a sequence of emissions, take the last one corresponding to current time t.
        else:
            log_emit = emission_loglik # (S, A) log emission for current time t.
        
        log_unnorm = log_emit + log_belief_pred # Log version of multiplying prior by likelihood.
        # Computes normalization constant: log p(O_t | O_{1:t-1}) = log sum_{s,a} p(O_t | s,a) * p(s,a | O_{1:t-1}).
        log_norm = log_unnorm - torch.logsumexp(log_unnorm.reshape(-1), dim=0) # Normalize to get log probabilities that sum to 1.
        return BeliefState(log_norm, t=belief_pred.t) # Updated belief at the same time index as the predict step (t), but now incorporates the new observation.