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
from .metrics import MetricsAccumulator, JointStateMetrics, HitAtHorizon, TimeToEvent
from .model_interface import HDVDbnModel


@dataclass
class TrajectoryData:
    """Single trajectory for evaluation."""
    obs: np.ndarray                        # (T, F) observations
    latents_gt: np.ndarray                 # (T, 2) ground-truth latent indices
    trajectory_id: str = None              # For logging
    
    def __post_init__(self):
        # Runs automatically after the dataclass is constructed.
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
    warmup_steps: int = 5
    horizon: int = 10                  # H for Hit@H metric
    fps: float = 25.0                  # For TTE conversion
    stride_frames: int = 10            # For TTE conversion
    skip_partial_horizons: bool = True # If True, skip t > T - H - 1
    prediction_target: str = "joint" # "action" or "joint"; whether to predict just action or the full (style, action) pair.


class ValidationStep:
    """Single-shot evaluation of a batch of trajectories."""
    def __init__(self, model: HDVDbnModel, config=None):
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
    
    def predict_one_trajectory(self,  traj: TrajectoryData):
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
        
        # Create a new predictor fresh for this trajectory.
        predictor = OnlinePredictor(
            self.model,
            warmup_steps=self.config.warmup_steps,
            device=self.model.device,
            dtype=self.model.dtype,
        )
        
        # Convert obs to tensor
        obs_tensor = torch.as_tensor(traj.obs, device=self.model.device, dtype=self.model.dtype) # (T,F)
        
        predictions = [] # collects records (pred_z, true_z, hit_h, tte_steps) for each scored timestep.

        for t in range(traj.T): # Loop over timesteps t = 0..T-1.
            # 1. Compute O_t (already provided in traj.obs)
            obs_t = obs_tensor[t:t+1]  # Slice one timestep but keep batch dimension (1,F) because emissions expect a sequence/batch shape.
            
            # 2. Update belief (predict + update)
            predictor.update(obs_t)
            
            # 3. If past warmup and within horizon bounds, predict
            if not predictor.is_ready:
                continue # skip until warmup done
            
            # Check horizon bound: t <= T - H - 1
            # (can't evaluate if not enough future data)
            if self.config.skip_partial_horizons and t + self.config.horizon >= traj.T:
                continue
            
            # 4. predict the next latent
            try:
                pred_out = predictor.predict_next() # produces a (S,A) log-prob table for t+1
            except RuntimeError:
                # Still in warmup, skip
                continue
            
            logprob = pred_out.pred_logprob  # (S, A)

            if self.config.prediction_target == "action":
                # Main mode for tied_action:
                # marginalize style and predict only the shared action.
                s_hat, a_hat = self._action_map_from_logprob(logprob)

            elif self.config.prediction_target == "joint":
                # Diagnostic mode:
                # predict the full (style, action) pair.
                s_hat, a_hat = self._joint_map_from_logprob(logprob)

            else:
                raise ValueError(
                    f"Unknown prediction_target='{self.config.prediction_target}'. "
                    "Use 'action' or 'joint'."
                )

            pred_z_joint = (s_hat, a_hat)
            true_z_joint = tuple(traj.latents_gt[t + 1])

            # Convert to the metric label space.
            # For action mode: (s,a) -> (0,a)
            # For joint mode:  (s,a) -> (s,a)
            pred_z = self._metric_pair(pred_z_joint)
            true_z = self._metric_pair(true_z_joint)

            horizon_latents = traj.latents_gt[t + 1:t + 1 + self.config.horizon]

            hit_h = False
            tte_steps = None

            for h, lat_h in enumerate(horizon_latents, start=1):
                gt_joint = tuple(lat_h)

                if gt_joint == (-1, -1):
                    continue

                gt_eval = self._metric_pair(gt_joint)

                if gt_eval == pred_z:
                    hit_h = True
                    tte_steps = h
                    break

            # First two fields are the evaluated labels.
            # Last two fields keep the original joint labels for debugging.
            predictions.append(
                (pred_z, true_z, hit_h, tte_steps, pred_z_joint, true_z_joint)
            )
        
        return predictions
    
    def evaluate(self, trajectories: list[TrajectoryData]):
        """
        Full evaluation on a batch of trajectories.
        
        Parameters
        trajectories : list[TrajectoryData]
            List of trajectories to evaluate.
        
        Returns
        (MetricsAccumulator, list)
            - Aggregated metrics over all trajectories
            - All predictions (pred_z, true_z, hit_h, tte_steps) for visualization
        """
        metrics = MetricsAccumulator(
            exact=JointStateMetrics(), # stores (pred,true) pairs for confusion matrix + accuracy.
            hit_h=HitAtHorizon(), # stores booleans.
            tte=TimeToEvent(fps=self.config.fps, stride_frames=self.config.stride_frames), # stores per-sample times in seconds.
        )
        
        all_predictions = []  # for plotting.
        total_predictions = 0 # counts scored samples (excluding unknown GT).
        
        for i, traj in enumerate(trajectories): # Loop over trajectories; each element is one vehicle sequence.
            traj_id = traj.trajectory_id
            print(f"[validate] Evaluating {traj_id} (T={traj.T})...", flush=True)
            
            predictions = self.predict_one_trajectory(traj) # Runs the full online filtering loop
            all_predictions.extend(predictions)  # Collect for visualization
            
            for pred in predictions:
                pred_z, true_z, hit_h, tte_steps = pred[:4]
                # Skip predictions with UNKNOWN ground truth latents (-1, -1) to avoid index errors in confusion matrix computation
                if true_z == (-1, -1):
                    continue
                # Accumulate metrics
                metrics.exact.add(pred_z, true_z)
                metrics.hit_h.add(hit_h)
                metrics.tte.add(tte_steps)
                total_predictions += 1 # Increment scored count.
            
            kept = [p for p in predictions if p[1] != (-1, -1)] # Keeps only those with known ground truth for printing.
            print(
                f"  -> {len(kept)} scored predictions | " # number of scored samples for this trajectory
                f"hits={sum(1 for p in kept if p[2])}/{len(kept) if kept else 0}"
            )
        
        print(f"\n[validate] Total predictions: {total_predictions}", flush=True) # total scored samples across all trajectories. 
        
        n_exact = len(metrics.exact.pred_labels)
        n_hit = len(metrics.hit_h.hits)
        n_tte = len(metrics.tte.times)

        if not (n_exact == n_hit == n_tte):
            raise RuntimeError(
                f"Metric count mismatch: exact={n_exact}, hit={n_hit}, tte={n_tte}. "
                "This means metrics are being updated under different conditions."
            )
        
        return metrics, all_predictions

    def _joint_map_from_logprob(self, logprob):
        if isinstance(logprob, torch.Tensor):
            idx = int(torch.argmax(logprob.reshape(-1)).item())
        else:
            idx = int(np.argmax(np.asarray(logprob).reshape(-1)))

        s_hat = idx // self.A
        a_hat = idx % self.A

        return int(s_hat), int(a_hat)

    def _action_map_from_logprob(self, logprob):
        if isinstance(logprob, torch.Tensor):
            logprob_action = torch.logsumexp(logprob, dim=0)  # (A,)
            a_hat = int(torch.argmax(logprob_action).item())
            s_hat = int(torch.argmax(logprob[:, a_hat]).item())
        else:
            lp = np.asarray(logprob)
            logprob_action = np.logaddexp.reduce(lp, axis=0)
            a_hat = int(np.argmax(logprob_action))
            s_hat = int(np.argmax(lp[:, a_hat]))

        return int(s_hat), int(a_hat)

    def _metric_pair(self, z):
        s, a = int(z[0]), int(z[1])

        # Preserve unknown labels in both joint and action mode.
        if s < 0 or a < 0:
            return (-1, -1)

        if self.config.prediction_target == "action":
            return (0, a)

        if self.config.prediction_target == "joint":
            return (s, a)

        raise ValueError(
            f"Unknown prediction_target='{self.config.prediction_target}'. "
            "Use 'action' or 'joint'."
        )

    @property
    def metric_S(self):
        return 1 if self.config.prediction_target == "action" else self.S

    @property
    def metric_A(self):
        return self.A