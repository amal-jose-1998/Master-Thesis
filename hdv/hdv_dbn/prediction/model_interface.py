"""
This module wraps HDVTrainer so the rest of the pipeline doesn’t depend on trainer internals.
So it’s like an “adapter layer” between training code and prediction code.
"""
from abc import ABC, abstractmethod
from typing import Tuple
import torch


class GenerativeModel(ABC):
    """
    Abstraction for any discrete-latent generative model suitable for filtering.
    
    A model provides:
      1. Initial latent distribution
      2. Transition dynamics
      3. Emission likelihood
    """
    
    @property
    @abstractmethod
    def num_styles(self):
        """Number of style/regime latents (S)."""
        pass
    
    @property
    @abstractmethod
    def num_actions(self):
        """Number of action latents (A)."""
        pass
    
    @property
    @abstractmethod
    def num_obs_features(self):
        """Observation feature dimension."""
        pass
    
    @property
    @abstractmethod
    def device(self):
        """PyTorch device (CPU or GPU)."""
        pass
    
    @property
    @abstractmethod
    def dtype(self):
        """PyTorch precision (float32, float64)."""
        pass
    
    @abstractmethod
    def initial_belief(self):
        """
        Return (pi_s0, pi_a0_given_s0, A_s, A_a).
        
        Returns
        pi_s0 : torch.Tensor, shape (S,)
            p(s_0)
        pi_a0_given_s0 : torch.Tensor, shape (S, A)
            p(a_0 | s_0)
        A_s : torch.Tensor, shape (S, S)
            p(s_{t+1} | s_t)
        A_a : torch.Tensor, shape (S, A, A)
            p(a_{t+1} | a_t, s_{t+1}), indexed as A_a[s_next, a_prev, a_next]
        """
        pass
    
    @abstractmethod
    def emission_loglik(self, obs, obs_names=None):
        """
        Compute log emission likelihood for an observation sequence.
        
        Parameters
        obs : torch.Tensor
            Observation sequence, shape (T, F) where T is time, F is feature dim.
        obs_names : list[str], optional
            Feature names (for validation).
        
        Returns
        torch.Tensor
            Log emission, shape (T, S, A) where entry [t, s, a] is log p(o_t | s, a).
        """
        pass


class HDVDbnModel(GenerativeModel):
    """
    Wraps HDV DBN (from trainer.hdv_dbn + trainer.emissions).
    
    Parameters
    trainer : HDVTrainer
        Trained model object with:
          - .hdv_dbn (HDVDBN)
          - .emissions (MixedEmissionModel)
          - .device, .dtype
    """
    
    def __init__(self, trainer):
        self._trainer = trainer # Stores the loaded trainer object (has transitions, priors, emissions, etc.).
        self._dbn = trainer.hdv_dbn # Keeps a handle to the DBN object to read num_style, num_action.
        self._emissions = trainer.emissions # Keeps a handle to the emission model.
        self._device = getattr(trainer, "device", torch.device("cpu"))
        self._dtype = getattr(trainer, "dtype", torch.float32)
    
    @property
    def num_styles(self):
        return int(self._dbn.num_style)
    
    @property
    def num_actions(self):
        return int(self._dbn.num_action)
    
    @property
    def num_obs_features(self):
        # Infer from emissions obs_names
        return len(self._emissions.obs_names) if hasattr(self._emissions, "obs_names") else None
    
    @property
    def device(self):
        return self._device
    
    @property
    def dtype(self):
        return self._dtype
    
    def initial_belief(self):
        """Extract from trainer attributes."""
        return (
            torch.as_tensor(self._trainer.pi_s0, device=self._device, dtype=self._dtype),
            torch.as_tensor(self._trainer.pi_a0_given_s0, device=self._device, dtype=self._dtype),
            torch.as_tensor(self._trainer.A_s, device=self._device, dtype=self._dtype),
            torch.as_tensor(self._trainer.A_a, device=self._device, dtype=self._dtype),
        )
    
    def emission_loglik(self, obs, obs_names=None):
        """
        Call the emission model's loglikelihood method.
        
        Parameters
        obs : torch.Tensor or np.ndarray
            Shape (T, F) observations.
        obs_names : list[str], optional
            Feature names (validated against self._emissions.obs_names if provided).
        
        Returns
        torch.Tensor
            Shape (T, S, A) log likelihoods.
        """
        obs_tensor = torch.as_tensor(obs, device=self._device, dtype=self._dtype)

        # Call emission model's loglikelihood method
        if not hasattr(self._emissions, "loglikelihood"):
            raise AttributeError("Emission model missing loglikelihood() method.")

        loglik = self._emissions.loglikelihood(obs_tensor)

        # If emissions returned a list, extract the first element
        if isinstance(loglik, (list, tuple)):
            return loglik[0]

        return loglik
