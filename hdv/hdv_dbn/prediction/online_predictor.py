"""
Online predictor combining filtering + one-step forecast. Encapsulates Bayes filter recursion and prediction logic.
1. Update belief using the new observation:
    “Given what I saw now, what’s my probability over (style, action) right now?”
2. Once warmup is finished:
    a. Predict next:
        “If I roll forward one step using the transition model, what’s the most likely next (style, action)?”
"""

from dataclasses import dataclass
import torch

from .filtering import StructuredDBNFilter, BeliefState
from .model_interface import HDVDbnModel


@dataclass(frozen=True)
class PredictionOutput:
    """Result of one-step prediction (joint latent only) in log-space."""
    belief_state: BeliefState              # current posterior belief after seeing observation at time t: p(z_t | O_{1:t}); shape (S, A)
    pred_logprob: torch.Tensor             # forecast distribution for the next step: p(z_{t+1} | O_{1:t}); shape (S, A)


class OnlinePredictor:
    """
    Stateful online filtering and one-step prediction. Maintains belief through a trajectory, predicts next latent state.
    
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
    
    def __init__(self, model: HDVDbnModel, warmup_steps=5, device=None, dtype=None):
        self.model = model # interface to the underlying DBN model (provides transitions and emissions)
        self.warmup_steps = int(warmup_steps) # number of updates before allowing predictions.
        self.device = device or model.device
        self.dtype = dtype or model.dtype
        
        # Initialize filter with model parameters
        pi_s0, pi_a0_s0, A_s, A_a = model.initial_belief() # Get initial belief components from the model to construct the filter.
        self.filter = StructuredDBNFilter( # Initializes the Bayes filter with the model's initial belief components.
            pi_s0=pi_s0,
            pi_a0_given_s0=pi_a0_s0,
            A_s=A_s,
            A_a=A_a,
            device=self.device,
            dtype=self.dtype,
        )
        
        # Stores latent sizes inferred from priors.
        self.S = self.filter.S
        self.A = self.filter.A
        
        # Initializes internal state
        self._belief = None # will hold BeliefState after the first observation update is processed.
        self._step_count = 0 # how many updates have happened (used to track warmup).
        self._is_warmup = True # blocks predictions until enough updates have passed.
    
    def reset(self):
        """Reset to initial belief and step counter."""
        self._belief = self.filter.initial_belief()
        self._step_count = 0
        self._is_warmup = True
    
    def update(self, obs_t: torch.Tensor):
        """
        Bayes filter recursion step for one observation: predict + update given observation.
        
        Parameters
        obs_t : torch.Tensor
            Single observation, shape (F,) or (1, F).
        
        Returns
        BeliefState
            Updated belief p(z_t | O_{1:t}).
        """
        if self._belief is None: # If this is the first update, initialize starting belief.
            self.reset()
        
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0) # If user passes (F,), convert to (1,F). Emissions expect a batch dimension, even if it's just one observation.
        
        obs_t = torch.as_tensor(obs_t, device=self.device, dtype=self.dtype)
        
        # Predict step (transition): p(z_t | O_{1:t-1}) = sum_{z_{t-1}} p(z_t|z_{t-1}) * p(z_{t-1}|O_{1:t-1})
        belief_pred = self.filter.predict(self._belief) # Computes the prior for the current time using transitions
        
        # Emission likelihood for this observation
        emission_ll = self.model.emission_loglik(obs_t)  # (1, S, A); Calls the emission model to compute: p(o_t | z_t) for all z_t=(s_t, a_t).
        
        # Update step (conditioning on observation): p(z_t | O_{1:t}) = p(o_t | z_t) * p(z_t | O_{1:t-1}) / p(o_t | O_{1:t-1})
        # Squeeze the batch/time dimension from (1, S, A) to (S, A) for a single observation before updating the belief state.
        belief_post = self.filter.update(belief_pred, emission_ll.squeeze(0))
        
        self._belief = belief_post # Updates internal belief to posterior at time t.
        # Update warmup counters
        self._step_count += 1
        if self._step_count >= self.warmup_steps:
            self._is_warmup = False
        
        return belief_post
    
    def predict_next(self):
        """
        This produces the one-step forecast from the current posterior
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
        
        if self._is_warmup: # Blocks predictions early so you don’t evaluate on a belief that’s basically just the prior.
            raise RuntimeError(
                f"Still in warmup ({self._step_count} < {self.warmup_steps} steps). "
                "Predictions not reliable yet."
            )
        
        # Applies transitions once more, but without any emission update: p(z_{t+1} | O_{1:t}) = sum_z_t p(z_{t+1}|z_t) * p(z_t|O_{1:t})
        belief_next = self.filter.predict(self._belief)  # BeliefState at time t+1 containing log_prob over (S,A).

        # Joint logits/probabilities for next latent (style x action)
        logprob = belief_next.log_prob  # Extracts the (S,A) log distribution.

        return PredictionOutput(
            belief_state=self._belief, # current belief at t (posterior)
            pred_logprob=logprob, # forecast distribution for t+1 (logits/log-probs).
        )
    
    def predict_horizon(self, H, return_full=True):
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
        
        for _ in range(H): # Repeatedly applies the transition predict step H times to roll forward the belief without any new observations.
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
