"""
Online predictor combining filtering + one-step forecast.
Encapsulates Bayes filter recursion and prediction logic.

1. Update belief using the new observation:
    “Given what I saw now, what’s my probability over (style, action) right now?”
2. Once warmup is finished:
    a. Predict next:
        “If I roll forward one step using the transition model, what’s the most likely next (style, action)?”
"""

from dataclasses import dataclass
import torch

from .filtering import StructuredDBNFilter, BeliefState


@dataclass(frozen=True)
class PredictionOutput:
    """Result of one-step prediction (joint latent only)."""
    belief_state: BeliefState              # Current posterior p(z_t | O_{1:t})
    pred_logprob: torch.Tensor             # Next-step logits p(z_{t+1} | O_{1:t}), shape (S, A)


class OnlinePredictor:
    """
    Stateful online filtering and one-step prediction.
    Maintains belief through a trajectory, predicts next latent state.
    
    Parameters
    model : GenerativeModel
        Model with transition params and emissions.
    warmup_steps : int, default=10
        Number of filtering steps before we produce predictions.
        (Posterior needs time to leave prior.)
    device : torch.device, optional
        Force device placement.
    dtype : torch.dtype, optional
        Force precision.
    """
    
    def __init__(self, model, warmup_steps=10, device=None, dtype=None):
        self.model = model
        self.warmup_steps = int(warmup_steps)
        self.device = device or model.device
        self.dtype = dtype or model.dtype
        
        # Initialize filter with model parameters
        pi_s0, pi_a0_s0, A_s, A_a = model.initial_belief()
        self.filter = StructuredDBNFilter(
            pi_s0=pi_s0,
            pi_a0_given_s0=pi_a0_s0,
            A_s=A_s,
            A_a=A_a,
            device=self.device,
            dtype=self.dtype,
        )
        
        self.S = self.filter.S
        self.A = self.filter.A
        
        # Initializes internal state
        self._belief = None
        self._step_count = 0
        self._is_warmup = True
    
    def reset(self):
        """Reset to initial belief and step counter."""
        self._belief = self.filter.initial_belief()
        self._step_count = 0
        self._is_warmup = True
    
    def update(self, obs_t):
        """
        Bayes filter step: predict + update given observation.
        
        Parameters
        obs_t : torch.Tensor
            Single observation, shape (F,) or (1, F).
        
        Returns
        BeliefState
            Updated belief p(z_t | O_{1:t}).
        """
        if self._belief is None:
            self.reset()
        
        # Reshape if needed
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        
        obs_t = torch.as_tensor(obs_t, device=self.device, dtype=self.dtype)
        
        # Predict: p(z_t | O_{1:t-1})
        belief_pred = self.filter.predict(self._belief)
        
        # Emission likelihood for this observation
        emission_ll = self.model.emission_loglik(obs_t)  # (1, S, A)
        
        # Update: p(z_t | O_{1:t})
        belief_post = self.filter.update(belief_pred, emission_ll.squeeze(0))
        
        self._belief = belief_post
        # Update warmup counters
        self._step_count += 1
        if self._step_count >= self.warmup_steps:
            self._is_warmup = False
        
        return belief_post
    
    def predict_next(self):
        """
        Predict next latent state given current belief.
        Returns: p(z_{t+1} | O_{1:t}) via one-step forecast.
        
        Raises
        RuntimeError
            If called during warmup or before any update().
        
        Returns
        PredictionOutput
            Joint prediction (style x action logits/probs) only.
        """
        if self._belief is None:
            raise RuntimeError("No observations yet. Call update() first.")
        
        if self._is_warmup:
            raise RuntimeError(
                f"Still in warmup ({self._step_count} < {self.warmup_steps} steps). "
                "Predictions not reliable yet."
            )
        
        # One-step forecast: p(z_{t+1} | O_{1:t}) = sum_z p(z_{t+1}|z) * p(z|O_{1:t})
        belief_next = self.filter.predict(self._belief)  # (S, A) in log space

        # Joint logits/probabilities for next latent (style x action)
        logprob = belief_next.log_prob  # (S, A)

        return PredictionOutput(
            belief_state=self._belief,
            pred_logprob=logprob,
        )
    
    def predict_horizon(self, H, return_full=False):
        """
        Predict latent marginals $H$ steps ahead without new observations.
        Rolls the filter forward H times using only transitions (no emission updates).
        
        Parameters
        H : int
            Horizon (number of steps ahead).
        return_full : bool
            If True, return all H predictions. Else return only the H-th.
        
        Returns
        torch.Tensor
            Shape (H, S, A) if return_full, else (S, A).
        """
        if self._belief is None:
            raise RuntimeError("No observations yet. Call update() first.")
        
        predictions = []
        curr_belief = self._belief
        
        for _ in range(H):
            curr_belief = self.filter.predict(curr_belief)
            predictions.append(curr_belief.prob)
        
        if return_full:
            return torch.stack(predictions)  # (H, S, A)
        else:
            return predictions[-1]  # (S, A)
    
    @property
    def is_ready(self):
        """True if past warmup and ready to predict."""
        return not self._is_warmup
    
    @property
    def current_belief(self):
        """Current posterior belief (read-only)."""
        return self._belief
