"""
This module wraps HDVTrainer so the rest of the pipeline doesn’t depend on trainer internals.
So it’s like an “adapter layer” between training code and prediction code.
"""
import torch
from pathlib import Path
import sys

try:
    from hdv.hdv_dbn.trainer import HDVTrainer
except ImportError:
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))
    from hdv.hdv_dbn.trainer import HDVTrainer


class HDVDbnModel():
    """
    Wraps HDV DBN (from trainer.hdv_dbn + trainer.emissions).
    
    Parameters
    trainer : HDVTrainer
        Trained model object with:
          - .hdv_dbn (HDVDBN)
          - .emissions (MixedEmissionModel)
          - .device, .dtype
    """
    
    def __init__(self, trainer:HDVTrainer, device=None, dtype=None):
        self._trainer = trainer # Stores the loaded trainer object (has transitions, priors, emissions, etc.).
        self._dbn = trainer.hdv_dbn # Keeps a handle to the DBN object to read num_style, num_action.
        self._emissions = trainer.emissions # Keeps a handle to the emission model.
        trainer_device = getattr(trainer, "device", torch.device("cpu"))
        trainer_dtype = getattr(trainer, "dtype", torch.float32)

        # Allow override; otherwise keep existing behavior
        self._device = torch.device(device) if device is not None else trainer_device
        self._dtype = dtype if dtype is not None else trainer_dtype

        if hasattr(self._emissions, "to_device"):
            self._emissions.to_device(self._device, self._dtype)
    
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
        """Returns the parameters needed to build the filter initial belief and transitions."""
        return (
            torch.as_tensor(self._trainer.pi_s0, device=self._device, dtype=self._dtype),
            torch.as_tensor(self._trainer.pi_a0_given_s0, device=self._device, dtype=self._dtype),
            torch.as_tensor(self._trainer.A_s, device=self._device, dtype=self._dtype),
            torch.as_tensor(self._trainer.A_a, device=self._device, dtype=self._dtype),
        )
    
    def emission_loglik(self, obs):
        """
        Call the emission model's loglikelihood method.
        
        Parameters
        obs : torch.Tensor or np.ndarray
            Shape (T, F) observations.
        
        Returns
        torch.Tensor
            Shape (T, S, A) log likelihoods.
        """
        obs_tensor = torch.as_tensor(obs, device=self._device, dtype=self._dtype)

        # Call emission model's loglikelihood method
        if not hasattr(self._emissions, "loglikelihood"):
            raise AttributeError("Emission model missing loglikelihood() method.")

        loglik = self._emissions.loglikelihood(obs_tensor, device=self._device, dtype=self._dtype) # Calls the emission model's loglikelihood method to compute p(o_t | z_t) for all z_t=(s_t, a_t); shape: (T, S, A).

        return loglik
