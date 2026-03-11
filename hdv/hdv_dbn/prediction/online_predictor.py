"""Online predictors for single-stream and batched DBN inference."""
from dataclasses import dataclass
import torch

try:
    from .filtering import StructuredDBNFilter, BeliefState
    from .model_interface import HDVDbnModel
except ImportError:
    from filtering import StructuredDBNFilter, BeliefState
    from model_interface import HDVDbnModel


@dataclass(frozen=True)
class PredictionOutput:
    """One-step prediction output for a single stream."""
    belief_state: BeliefState # Current posterior belief state after update
    pred_logprob: torch.Tensor # Predicted log-probability for next time step (shape (S, A))


@dataclass(frozen=True)
class BatchedPredictionOutput:
    """One-step prediction output for batched streams."""
    belief_logprob: torch.Tensor # Current posterior log-belief for all streams (shape (B, S, A))
    pred_logprob: torch.Tensor # Predicted log-probability for next time step (shape (B, S, A)); non active-ready streams are masked to -inf
    ready_mask: torch.Tensor # Boolean mask indicating which streams are past warmup and ready for prediction (shape (B,))
    active_mask: torch.Tensor # Boolean mask indicating which streams are active in the current batch (shape (B,))


class OnlinePredictor:
    """Online filtering and one-step prediction for a single stream of observations."""
    def __init__(self, model: HDVDbnModel, warmup_steps=5, device=None, dtype=None):
        """Initialize the OnlinePredictor with a given HDVDbnModel and optional parameters.
        
        parameters:
        - model: An instance of HDVDbnModel that defines the DBN structure and parameters.
        - warmup_steps: Number of initial steps to run before predictions are considered reliable.
        - device: Optional torch device to run computations on (e.g., 'cpu' or 'cuda').
        - dtype: Optional torch dtype for computations (e.g., torch.float32).
        """
        self.model = model
        self.warmup_steps = int(warmup_steps)
        self.device = device or model.device
        self.dtype = dtype or model.dtype

        pi_s0, pi_a0_s0, A_s, A_a = model.initial_belief() # Get initial belief parameters from the model
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

        self._belief = None # shape (S, A) log-probability of current belief state
        self._step_count = 0
        self._is_warmup = True
    
    def reset(self):
        """Reset to initial belief and step counter."""
        self._belief = self.filter.initial_belief() # Reset belief to initial state
        self._step_count = 0
        self._is_warmup = True
    
    def update(self, obs_t: torch.Tensor):
        """
        Run one predict-update step with observation shape (F,) or (1,F).
        
        parameters:
        - obs_t: Observation at current time step, either as a 1D tensor of shape (F,) or a 2D tensor of shape (1, F).

        returns:
        - belief_post: Updated belief state after incorporating the new observation.    
        """
        if self._belief is None:
            self.reset() # Initialize belief if this is the first update call

        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0) # Ensure obs_t has shape (1, F) for consistency
        elif obs_t.ndim != 2:
            raise ValueError(f"obs_t must have shape (F,) or (1,F), got {tuple(obs_t.shape)}")

        obs_t = torch.as_tensor(obs_t, device=self.device, dtype=self.dtype)

        belief_pred = self._belief if self._step_count == 0 else self.filter.predict(self._belief) # first observation uses initial prior; later ones use predicted prior
        emission_ll = self.model.emission_loglik(obs_t) # log-likelihood of obs_t under the model's emission distribution
        belief_post = self.filter.update(belief_pred, emission_ll) # posterior belief after incorporating obs_t

        self._belief = belief_post # Update internal belief state to the new posterior
        self._step_count += 1 
        if self._step_count >= self.warmup_steps: # After warmup_steps, predictions are considered reliable
            self._is_warmup = False

        return belief_post
    
    def predict_next(self):
        """
        Predict one-step-ahead latent log-distribution from current posterior.

        returns:
        - PredictionOutput containing the current belief state and the predicted log-probability for the next time step.
        
        """
        if self._belief is None:
            raise RuntimeError("No observations yet. Call update() first.")

        if self._is_warmup:
            raise RuntimeError(
                f"Still in warmup ({self._step_count} < {self.warmup_steps} steps). "
                "Predictions not reliable yet."
            )

        belief_next = self.filter.predict(self._belief) # Predict next time step's belief state based on current belief

        return PredictionOutput(
            belief_state=self._belief,
            pred_logprob=belief_next.log_prob, 
        )
    
    def predict_horizon(self, H, return_full=True):
        """Roll transitions forward H steps without new emissions."""
        if self._belief is None:
            raise RuntimeError("No observations yet. Call update() first.")

        predictions = []
        curr_belief = self._belief

        for _ in range(H):
            curr_belief = self.filter.predict(curr_belief)
            predictions.append(curr_belief.prob)

        if return_full:
            return torch.stack(predictions)
        return predictions[-1]
    
    @property
    def is_ready(self):
        """True if past warmup and ready to predict."""
        return not self._is_warmup
    
    @property
    def current_belief(self):
        """Current posterior belief (read-only)."""
        return self._belief


class BatchedOnlinePredictor:
    """Online predictor for batched streams of observations, with independent DBN inference per stream."""
    def __init__(self, model: HDVDbnModel, warmup_steps=5, device=None, dtype=None):
        """
        Initialize the BatchedOnlinePredictor with a given HDVDbnModel and optional parameters.

        parameters:
        - model: An instance of HDVDbnModel that defines the DBN structure and parameters.
        - warmup_steps: Number of initial steps to run before predictions are considered reliable for each stream.
        - device: Optional torch device to run computations on (e.g., 'cpu' or 'cuda').
        - dtype: Optional torch dtype for computations (e.g., torch.float32).   
        """
        self.model = model
        self.warmup_steps = int(warmup_steps)
        self.device = device or model.device
        self.dtype = dtype or model.dtype

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

        self._belief_logprob = None  # (B, S, A) log-probability of current belief state for each stream in the batch
        self._step_count = None # (B,) count of updates for each stream to track warmup status
        self._is_warmup = None # (B,) boolean mask indicating which streams are still in warmup phase

    def reset(self, batch_size):
        """
        Reset all batched streams to initial belief. 
        
        parameters:
        - batch_size: Number of independent streams in the batch to reset.
        """
        B = int(batch_size)
        if B <= 0:
            raise ValueError(f"batch_size must be > 0, got {B}")

        init = self.filter.initial_belief().log_prob # shape (S, A)
        if init.ndim != 2:
            raise ValueError(f"initial_belief log_prob must be 2D, got {tuple(init.shape)}")
        self._belief_logprob = init.unsqueeze(0).repeat(B, 1, 1).contiguous() # shape (B, S, A)
        self._step_count = torch.zeros(B, device=self.device, dtype=torch.long) # Initialize step counts to zero for all streams
        self._is_warmup = torch.ones(B, device=self.device, dtype=torch.bool) # All streams start in warmup phase

    def _normalize_obs(self, obs_t: torch.Tensor):
        """
        Normalize input observation to shape (B, F) and ensure it's a torch tensor on the correct device and dtype.

        parameters:
        - obs_t: Input observation, either as a 1D tensor of shape (F,) for a single stream or a 2D tensor of shape (B, F) for batched streams.

        returns:
        - obs_t: Normalized observation tensor of shape (B, F) on the correct device and dtype.
        """
        if not torch.is_tensor(obs_t):
            obs_t = torch.as_tensor(obs_t, device=self.device, dtype=self.dtype)
        else:
            if obs_t.device != self.device or obs_t.dtype != self.dtype:
                obs_t = obs_t.to(device=self.device, dtype=self.dtype)

        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0) # If obs_t is shape (F,), treat it as a single stream and add batch dimension to make it (1, F)
        elif obs_t.ndim != 2:
            raise ValueError(f"obs_t must have shape (F,) or (B,F), got {tuple(obs_t.shape)}")

        return obs_t

    def _normalize_active_mask(self, B, active_mask):
        """
        Normalize the active_mask to a boolean tensor of shape (B,) on the correct device.

        parameters:
        - B: Batch size (number of streams) to validate against.
        - active_mask: Optional input mask indicating which streams are active. Can be None, a list/array, or a torch tensor.
        
        returns:
        - active_mask: Normalized boolean tensor of shape (B,) where True indicates active streams and False indicates inactive streams. 
        If input is None, returns a tensor of all True (all streams active).
        """
        if active_mask is None:
            return torch.ones(B, device=self.device, dtype=torch.bool)

        if not torch.is_tensor(active_mask):
            active_mask = torch.as_tensor(active_mask, device=self.device)
        else:
            active_mask = active_mask.to(device=self.device)

        active_mask = active_mask.bool().reshape(-1)
        if active_mask.shape[0] != B:
            raise ValueError(f"active_mask length {active_mask.shape[0]} does not match batch size {B}")
        return active_mask

    def update(self, obs_t: torch.Tensor, active_mask=None):
        """
        Run one predict-update step for all active streams with observation shape (B, F) or (F,).

        parameters:
        - obs_t: Batched observation at current time step, either as a 1D tensor of shape (F,) for a single stream or a 2D tensor of shape (B, F) for batched streams.
        - active_mask: Optional boolean mask of shape (B,) indicating which streams are active and should be updated. If None, all streams are considered active.
        """
        obs_t = self._normalize_obs(obs_t) # Ensure obs_t is shape (B, F) and on the correct device/dtype
        B = int(obs_t.shape[0])
        if self._belief_logprob is None: # If this is the first update call, initialize belief for the batch
            self.reset(B)
        elif int(self._belief_logprob.shape[0]) != B: # else if batch size has changed, raise an error
            raise ValueError(
                f"Batch size changed from {self._belief_logprob.shape[0]} to {B}. "
                "Call reset(new_batch_size) or keep batch size fixed."
            )

        active_mask = self._normalize_active_mask(B, active_mask) # Ensure active_mask is shape (B,) and boolean
        if not bool(active_mask.any()): # If no streams are active, skip the update and just return the current belief log-probabilities
            return self._belief_logprob

        predicted_logprob = self.filter.predict(BeliefState(self._belief_logprob, t=0)).log_prob # shape (B, S, A)
        belief_pred = torch.where((self._step_count == 0)[:, None, None], self._belief_logprob, predicted_logprob)
        emission_ll = self.model.emission_loglik(obs_t) # shape (B, S, A) log-likelihood of obs_t for each stream under the model's emission distribution
        belief_post = self.filter.update(BeliefState(belief_pred, t=0), emission_ll).log_prob # shape (B, S, A) posterior belief for each stream after incorporating obs_t

        write_mask = active_mask[:, None, None] # shape (B, 1, 1) mask to select which streams to update with the new belief_post values
        self._belief_logprob = torch.where(write_mask, belief_post, self._belief_logprob) # Update belief_logprob for active streams, keep old values for inactive streams

        self._step_count = self._step_count + active_mask.to(dtype=torch.long) # Increment step count for active streams only
        self._is_warmup = self._step_count < self.warmup_steps # Update warmup status based on step count for each stream

        return self._belief_logprob

    def predict_next(self, active_mask=None, strict_ready=False):
        """
        Predict one-step-ahead latent log-distribution from current posterior for all active streams.

        parameters:
        - active_mask: Optional boolean mask of shape (B,) indicating which streams are active and should be included in the prediction output. If None, all streams are considered active.
        - strict_ready: If True, raises an error if any active stream is still in warmup. 
            If False, returns masked predictions: streams that are not both active and ready are filled with -inf in pred_logprob. Default is False.
        """
        if self._belief_logprob is None: # if no updates have been made yet and belief_logprob is still None, we cannot make predictions, so raise an error
            raise RuntimeError("No observations yet. Call update() first.")

        B = int(self._belief_logprob.shape[0])
        active_mask = self._normalize_active_mask(B, active_mask) # Ensure active_mask is shape (B,) and boolean

        if strict_ready: # If strict_ready is True, check if any active stream is still in warmup and raise an error if so
            not_ready = active_mask & self._is_warmup # Boolean mask of active streams that are still in warmup
            if bool(not_ready.any()): # If any active stream is still in warmup, raise an error with the indices of those streams
                idx = torch.nonzero(not_ready, as_tuple=False).flatten().tolist()
                raise RuntimeError(
                    f"Still in warmup for batch indices: {idx}. "
                    f"Require >= {self.warmup_steps} updates per active stream."
                )

        pred_logprob = self.filter.predict(BeliefState(self._belief_logprob, t=0)).log_prob # shape (B, S, A) predicted belief for each stream for the next time step
        valid_mask = active_mask & (~self._is_warmup) # Boolean mask of streams that are both active and past warmup (ready)
        pred_logprob = torch.where(
            valid_mask[:, None, None],
            pred_logprob,
            torch.full_like(pred_logprob, float("-inf")),
        ) # Mask out predictions for streams that are not both active and ready by filling them with -inf
        return BatchedPredictionOutput(
            belief_logprob=self._belief_logprob, # shape (B, S, A) current posterior belief log-probabilities for all streams
            pred_logprob=pred_logprob, # shape (B, S, A) predicted belief log-probabilities for the next time step, with non-ready streams masked to -inf
            ready_mask=~self._is_warmup, # shape (B,) boolean mask indicating which streams are past warmup and ready for prediction
            active_mask=active_mask, # shape (B,) boolean mask indicating which streams are active in the current batch
        )

    @property
    def is_ready(self):
        """Per-stream readiness mask; returns None before first update."""
        if self._is_warmup is None:
            return None
        return ~self._is_warmup

    @property
    def current_belief(self):
        """Current batched posterior log-belief; shape (B, S, A)."""
        return self._belief_logprob
