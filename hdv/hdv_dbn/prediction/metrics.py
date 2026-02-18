"""
Prediction evaluation metrics decoupled from filtering logic (bookkeeping + reporting).

Implements:
  Exact accuracy (1-step confusion matrix)
  Hit@H (target appears within horizon)
  Time-to-Event (latency to first match, in seconds)
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class ExactAccuracy:
    """
    One-step prediction accuracy (confusion matrix).
    
    Compares predicted z_hat_{t+1} vs ground truth z_{t+1}.
    """
    
    true_labels: List[Tuple[int, int]] = field(default_factory=list)  # (s, a) pairs
    pred_labels: List[Tuple[int, int]] = field(default_factory=list)  # (s, a) pairs
    
    def add(self, pred_z, true_z):
        """Record one prediction."""
        self.pred_labels.append(pred_z)
        self.true_labels.append(true_z)
    
    def confusion_matrix(self, S, A):
        """
        Compute confusion matrix for joint (S, A) states.
        
        Returns
        np.ndarray
            Shape (SA, SA) where entry [i, j] is count of 
            true=i, pred=j.
        """
        SA = S * A
        cm = np.zeros((SA, SA), dtype=np.int64)
        
        for (s_pred, a_pred), (s_true, a_true) in zip(self.pred_labels, self.true_labels):
            idx_pred = s_pred * A + a_pred
            idx_true = s_true * A + a_true
            cm[idx_true, idx_pred] += 1
        
        return cm
    
    def accuracy(self):
        """Overall 1-step accuracy."""
        if not self.pred_labels:
            return np.nan
        correct = sum(p == t for p, t in zip(self.pred_labels, self.true_labels))
        return float(correct) / len(self.pred_labels)


@dataclass
class HitAtHorizon:
    """
    Tracks whether predicted z_hat_{t+1} appears in the horizon [t+1, t+H].
      Hit@H(t) = 1[ ∃h ∈ {1, ..., H} : z_{t+h}^{GT} == z_hat_{t+1} ]
    """
    
    hits: List[bool] = field(default_factory=list)
    
    def add(self, hit):
        """Record hit/miss for one trajectory."""
        self.hits.append(bool(hit))
    
    def hit_rate(self):
        """Fraction of predictions where target appeared in horizon."""
        if not self.hits:
            return np.nan
        return float(np.mean(self.hits))
    
    def hit_count(self):
        """(num_hits, total) for manual aggregation."""
        if not self.hits:
            return (0, 0)
        return (int(np.sum(self.hits)), len(self.hits))


@dataclass
class TimeToEvent:
    """
    Time (in seconds) until predicted event appears.
      h*(t) = min{ h ∈ {1, ..., H} : z_{t+h}^{GT} == z_hat_{t+1} }
      TTE(t) = h*(t) if hit else ∞
      TTE_sec(t) = TTE(t) * Δt where Δt = r / fps
    
    Parameters
    fps : float
        Frame rate (Hz). Default 25 for highD.
    stride_frames : int
        Stride in frames. Default 10 for highD.
    """
    
    times: List[float] = field(default_factory=list)  # seconds, or inf for miss
    fps: float = 25.0
    stride_frames: int = 10
    
    @property
    def delta_t(self):
        """Window step duration in seconds."""
        return self.stride_frames / self.fps
    
    def add(self, time_to_hit_steps):
        """
        Record time-to-event in steps (or None for miss).
        
        Parameters
        time_to_hit_steps : int or None
            Number of steps until hit (1-indexed), or None if no hit.
        """
        if time_to_hit_steps is None:
            self.times.append(np.inf)
        else:
            # Convert steps to seconds
            self.times.append(float(time_to_hit_steps) * self.delta_t)
    
    def mean_tte_hits_only(self):
        """Mean TTE over hits only (excludes infinity's)."""
        hits = [t for t in self.times if np.isfinite(t)]
        if not hits:
            return np.nan
        return np.mean(hits)
    
    def median_tte_hits_only(self) :
        """Median TTE over hits only."""
        hits = [t for t in self.times if np.isfinite(t)]
        if not hits:
            return np.nan
        return np.median(hits)
    
    def percentile_tte(self, q):
        """Percentile TTE over hits only (q in [0, 100])."""
        hits = [t for t in self.times if np.isfinite(t)]
        if not hits:
            return np.nan
        return np.percentile(hits, q)
    
    def hit_count(self):
        """(num_hits, total)."""
        total = len(self.times)
        hits = sum(np.isfinite(t) for t in self.times)
        return (hits, total)


@dataclass
class MetricsAccumulator:
    """Combines all three metrics for batch evaluation."""
    exact: ExactAccuracy = field(default_factory=ExactAccuracy)
    hit_h: HitAtHorizon = field(default_factory=HitAtHorizon)
    tte: TimeToEvent = field(default_factory=TimeToEvent)
    
    def summary(self, S, A):
        n_exact = len(self.exact.pred_labels)
        n_hit = len(self.hit_h.hits)
        n_tte = len(self.tte.times)

        if not (n_exact == n_hit == n_tte):
            raise RuntimeError(
                f"Metric count mismatch: exact={n_exact}, hit={n_hit}, tte={n_tte}"
            )

        num_hits, num_total_hit = self.hit_h.hit_count()

        return {
        "exact_accuracy": float(self.exact.accuracy()),
        "hit_rate": float(self.hit_h.hit_rate()),
        "mean_tte_sec": float(self.tte.mean_tte_hits_only()),
        "median_tte_sec": float(self.tte.median_tte_hits_only()),
        "p25_tte_sec": float(self.tte.percentile_tte(25)),
        "p75_tte_sec": float(self.tte.percentile_tte(75)),
        "num_hits": int(num_hits),
        "num_total": int(n_exact),         
        "num_total_hit": int(num_total_hit),
        "num_total_exact": int(n_exact),
        "num_total_tte": int(n_tte),
    }
