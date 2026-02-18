"""
Validation pipeline: orchestrates filtering, prediction, and metrics.

1. For each trajectory:
    a. creates an OnlinePredictor
    b. feeds the observations one timestep at a time
2. After a warmup period, at each timestep it asks:
    a. “What does the model think the next (style,action) will be?”
3. It compares that predicted (s,a) to the ground truth in two ways:
    a. Exact 1-step accuracy: did we predict the exact next state?
    b. Hit@H: does the predicted state occur anytime in the next H steps?
    c. TTE: if it occurs, how long until it happens?
"""

from dataclasses import dataclass
import numpy as np
import torch

from .online_predictor import OnlinePredictor
from .metrics import MetricsAccumulator, ExactAccuracy, HitAtHorizon, TimeToEvent


@dataclass
class TrajectoryData:
    """Single trajectory for evaluation."""
    obs: np.ndarray                        # (T, F) observations
    latents_gt: np.ndarray                 # (T,) or (T, 2) ground-truth latent indices
    trajectory_id: str = None              # For logging
    
    def __post_init__(self):
        self.obs = np.asarray(self.obs, dtype=np.float32)
        self.latents_gt = np.asarray(self.latents_gt, dtype=np.int32)
        
        # Ensure latents are (T, 2) with (s, a)
        if self.latents_gt.ndim == 1:
            raise ValueError("latents_gt should be (T, 2) with (style, action) pairs or a single joint index.")
        if self.latents_gt.ndim == 2 and self.latents_gt.shape[1] != 2:
            raise ValueError(f"latents_gt expected shape (T, 2), got {self.latents_gt.shape}")
        
        if len(self.obs) != len(self.latents_gt):
            raise ValueError(
                f"Observation/latent length mismatch: obs={len(self.obs)}, latents={len(self.latents_gt)}"
            )
    
    @property
    def T(self):
        """Sequence length."""
        return len(self.obs)


@dataclass
class ValidationConfig:
    """Configuration for validation pipeline."""
    warmup_steps: int = 10
    horizon: int = 10                  # H for Hit@H metric
    fps: float = 25.0                  # For TTE conversion
    stride_frames: int = 10            # For TTE conversion
    skip_partial_horizons: bool = True # If True, skip t > T - H - 1


class ValidationStep:
    """Single-shot evaluation of a batch of trajectories."""
    def __init__(self, model, config=None):
        """
        Parameters
        model : GenerativeModel
            Model with initial_belief(), emission_loglik(), num_styles, num_actions.
        config : ValidationConfig, optional
            Defaults to ValidationConfig().
        """
        self.model = model
        self.config = config or ValidationConfig()
        self.S = model.num_styles
        self.A = model.num_actions
    
    def predict_one_trajectory(self,  traj):
        """
        Evaluate predictions on a single trajectory. Applies filtering recursion and computes metrics.
        
        Parameters
        traj : TrajectoryData
            Single trajectory with obs and latents_gt.
        
        Yields
        (pred_z, true_z, hit_h, tte_steps) tuples for each valid prediction time.
        
        Where:
          - pred_z, true_z : (s, a) tuples
          - hit_h : bool, whether predicted z appeared in horizon
          - tte_steps : int or None, steps to event (1-indexed) or None if miss
        """
        
        # Initialize predictor
        predictor = OnlinePredictor(
            self.model,
            warmup_steps=self.config.warmup_steps,
            device=self.model.device,
            dtype=self.model.dtype,
        )
        
        # Convert obs to tensor
        obs_tensor = torch.as_tensor(traj.obs, device=self.model.device, dtype=self.model.dtype)
        
        # Filtering loop (§ 10, lines 1-3)
        predictions = []
        for t in range(traj.T):
            # 1. Compute O_t (already provided in traj.obs)
            obs_t = obs_tensor[t:t+1]  # (1, F)
            
            # 2. Update belief (predict + update)
            belief = predictor.update(obs_t)
            
            # 3. If past warmup and within horizon bounds, predict
            if not predictor.is_ready:
                continue
            
            # Check horizon bound: t <= T - H - 1
            # (can't evaluate if not enough future data)
            if self.config.skip_partial_horizons and t + self.config.horizon >= traj.T:
                continue
            
            # 4. Predict z_{t+1}
            try:
                pred_out = predictor.predict_next()
            except RuntimeError:
                # Still in warmup, skip
                continue
            
            # pred_out now contains joint logits `pred_logprob` of shape (S, A).
            # Compute MAP over the joint latent (style, action).
            logprob = pred_out.pred_logprob  # (S, A)
            if isinstance(logprob, torch.Tensor):
                idx = int(torch.argmax(logprob.view(-1)).item())
            else:
                idx = int(np.argmax(np.asarray(logprob).reshape(-1)))
            s_hat = idx // self.A # marginal style
            a_hat = idx % self.A  # marginal action
            pred_z = (s_hat, a_hat)
            true_z = tuple(traj.latents_gt[t + 1])  # Ground truth at t+1
            
            # 5. Compute Hit@H and TTE
            # Look for match in horizon [t+1, ..., t+H]
            horizon_latents = traj.latents_gt[t+1:t+1+self.config.horizon]  # (≤H, 2)
            
            hit_h = False
            tte_steps = None
            for h, lat_h in enumerate(horizon_latents, start=1):  # 1-indexed
                if tuple(lat_h) == pred_z:
                    hit_h = True
                    tte_steps = h
                    break
            
            predictions.append((pred_z, true_z, hit_h, tte_steps))
        
        return predictions
    
    def evaluate(self, trajectories):
        """
        Full evaluation on a batch of trajectories.
        
        Parameters
        trajectories : list[TrajectoryData]
            List of trajectories to evaluate.
        
        Returns
        MetricsAccumulator
            Aggregated metrics over all trajectories.
        """
        metrics = MetricsAccumulator(
            exact=ExactAccuracy(), # stores lists of (s,a) pairs and builds a confusion matrix later.
            hit_h=HitAtHorizon(), # stores booleans.
            tte=TimeToEvent(fps=self.config.fps, stride_frames=self.config.stride_frames), # stores per-sample times in seconds.
        )
        
        total_predictions = 0
        
        for i, traj in enumerate(trajectories): # Loop over trajectories
            traj_id = traj.trajectory_id or f"traj_{i}"
            print(f"[validate] Evaluating {traj_id} (T={traj.T})...", flush=True)
            
            predictions = self.predict_one_trajectory(traj) # online filtering loop
            
            for pred_z, true_z, hit_h, tte_steps in predictions:
                # Skip predictions with UNKNOWN ground truth latents (-1, -1)
                # to avoid index errors in confusion matrix computation
                if true_z == (-1, -1):
                    continue
                # Accumulate metrics
                metrics.exact.add(pred_z, true_z)
                metrics.hit_h.add(hit_h)
                metrics.tte.add(tte_steps)
                total_predictions += 1
            
            kept = [(p,t,h,tte) for (p,t,h,tte) in predictions if t != (-1,-1)]
            print(
                f"  → {len(kept)} scored predictions | "
                f"hits={sum(1 for _, _, h, _ in kept if h)}/{len(kept) if kept else 0}"
            )
        
        print(f"\n[validate] Total predictions: {total_predictions}", flush=True)
        
        n_exact = len(metrics.exact.pred_labels)
        n_hit = len(metrics.hit_h.hits)
        n_tte = len(metrics.tte.times)

        if not (n_exact == n_hit == n_tte):
            raise RuntimeError(
                f"Metric count mismatch: exact={n_exact}, hit={n_hit}, tte={n_tte}. "
                "This means metrics are being updated under different conditions."
            )
        
        return metrics