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
class JointStateMetrics:
    """
    One-step prediction accuracy (confusion matrix). Compares predicted z_hat_{t+1} vs ground truth z_{t+1}.
    """
    true_labels: List[Tuple[int, int]] = field(default_factory=list)  # (s, a) pairs
    pred_labels: List[Tuple[int, int]] = field(default_factory=list)  # (s, a) pairs
    
    def add(self, pred_z, true_z):
        """Record one prediction."""
        self.pred_labels.append(pred_z)
        self.true_labels.append(true_z)
    
    def confusion_matrix(self, S, A):
        """
        Builds a joint-state confusion matrix of size (S*A, S*A).
        
        Returns
        np.ndarray
            Shape (SA, SA) where entry [i, j] is count of 
            true=i, pred=j.
        """
        SA = S * A
        cm = np.zeros((SA, SA), dtype=np.int64)
        # cm[i,j] counts how many times true joint state i was predicted as j.
        
        for (s_pred, a_pred), (s_true, a_true) in zip(self.pred_labels, self.true_labels): # Iterates aligned predicted/true pairs.
            # Converts (s,a) pairs to joint indices in [0, S*A-1].
            idx_pred = s_pred * A + a_pred
            idx_true = s_true * A + a_true
            # Increment the confusion matrix entry for this true/predicted pair.
            cm[idx_true, idx_pred] += 1
        
        return cm
    
    def accuracy(self):
        """Overall 1-step accuracy."""
        if not self.pred_labels:
            return np.nan
        correct = sum(p == t for p, t in zip(self.pred_labels, self.true_labels))
        return float(correct) / len(self.pred_labels)

    def precision_recall_f1(self, S, A, average="macro"):
        """
        Compute precision, recall, and F1 score for each joint class, and macro/micro averages.
        Args:
            S: Number of s classes
            A: Number of a classes
            average: 'macro', 'micro', or None (per-class)
        Returns:
            precision, recall, f1: np.ndarray of shape (SA,) if average=None, else float
        """
        cm = self.confusion_matrix(S, A)
        SA = S * A
        # True Positives for each class: diagonal
        tp = np.diag(cm)
        # Predicted Positives for each class: sum over columns
        pred_pos = np.sum(cm, axis=0)
        # Actual Positives for each class: sum over rows
        actual_pos = np.sum(cm, axis=1)
        # Avoid division by zero
        precision = np.divide(tp, pred_pos, out=np.zeros_like(tp, dtype=float), where=pred_pos!=0) # for each class, precision = TP / (TP + FP) = TP / predicted positives
        recall = np.divide(tp, actual_pos, out=np.zeros_like(tp, dtype=float), where=actual_pos!=0) # for each class, recall = TP / (TP + FN) = TP / actual positives
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp, dtype=float), where=(precision+recall)!=0) # for each class, F1 = 2 * (precision * recall) / (precision + recall)

        if average is None:
            return precision, recall, f1
        elif average == "macro":
            # Macro: mean over classes
            return np.mean(precision), np.mean(recall), np.mean(f1)
        elif average == "micro":
            # Micro: sum over all classes
            total_tp = np.sum(tp)
            total_pred_pos = np.sum(pred_pos)
            total_actual_pos = np.sum(actual_pos)
            micro_precision = total_tp / total_pred_pos if total_pred_pos > 0 else 0.0
            micro_recall = total_tp / total_actual_pos if total_actual_pos > 0 else 0.0
            if micro_precision + micro_recall == 0:
                micro_f1 = 0.0
            else:
                micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            return micro_precision, micro_recall, micro_f1
        else:
            raise ValueError("average must be one of None, 'macro', or 'micro'")


@dataclass
class HitAtHorizon:
    """
    Tracks whether predicted z_hat_{t+1} appears in the horizon [t+1, t+H].
      Hit@H(t) = 1[ ∃h ∈ {1, ..., H} : z_{t+h}^{GT} == z_hat_{t+1} ]
    """
    
    hits: List[bool] = field(default_factory=list) # Stores a boolean per scored timestep.
    
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
      First match index: h*(t) = min{ h ∈ {1, ..., H} : z_{t+h}^{GT} == z_hat_{t+1} }
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
        return self.stride_frames / self.fps # With 25 fps and stride 10 frames: 0.4 s per step.
    
    def add(self, time_to_hit_steps):
        """
        Record time-to-event in steps (or None for miss).
        
        Parameters
        time_to_hit_steps : int or None
            Number of steps until hit (1-indexed), or None if no hit.
        """
        if time_to_hit_steps is None: # No hit within horizon, record as infinity.
            self.times.append(np.inf)
        else:
            # Convert steps to seconds
            self.times.append(float(time_to_hit_steps) * self.delta_t)
    
    def mean_tte_hits_only(self):
        """Mean TTE over hits only (excludes infinity's)."""
        hits = [t for t in self.times if np.isfinite(t)] # Filters out misses (infinity) to compute mean TTE over hits only.
        if not hits:
            return np.nan
        return np.mean(hits)
    
    def median_tte_hits_only(self) :
        """Median TTE over hits only."""
        hits = [t for t in self.times if np.isfinite(t)] # Filters out misses (infinity) to compute median TTE over hits only.
        if not hits:
            return np.nan
        return np.median(hits)
    
    def percentile_tte(self, q):
        """Percentile TTE over hits only (q in [0, 100])."""
        hits = [t for t in self.times if np.isfinite(t)] # Filters out misses (infinity) to compute percentile TTE over hits only.
        if not hits:
            return np.nan
        return np.percentile(hits, q)
    
    def hit_count(self):
        """(num_hits, total)."""
        total = len(self.times)
        hits = sum(np.isfinite(t) for t in self.times) # Count only finite times as hits
        return (hits, total)


@dataclass
class MetricsAccumulator:
    """Bundles the three metrics into a single object."""
    exact: JointStateMetrics = field(default_factory=JointStateMetrics)
    hit_h: HitAtHorizon = field(default_factory=HitAtHorizon)
    tte: TimeToEvent = field(default_factory=TimeToEvent)
    
    def summary(self, S, A):
        n_exact = len(self.exact.pred_labels) # Total number of scored predictions (excluding unknown GT).
        n_hit = len(self.hit_h.hits) # Should be the same as n_exact, since we add one hit/miss per scored prediction.
        n_tte = len(self.tte.times) # Should also be the same as n_exact, since we add one TTE per scored prediction.

        if not (n_exact == n_hit == n_tte): # Ensures metrics were updated under identical conditions.
            raise RuntimeError(
                f"Metric count mismatch: exact={n_exact}, hit={n_hit}, tte={n_tte}"
            )

        num_hits, _ = self.hit_h.hit_count()

        # Compute macro-averaged precision, recall, F1
        macro_precision, macro_recall, macro_f1 = self.exact.precision_recall_f1(S, A, average="macro")

        return {
            "exact_accuracy": float(self.exact.accuracy()),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "hit_rate": float(self.hit_h.hit_rate()),
            "mean_tte_sec": float(self.tte.mean_tte_hits_only()),
            "median_tte_sec": float(self.tte.median_tte_hits_only()),
            "p25_tte_sec": float(self.tte.percentile_tte(25)),
            "p75_tte_sec": float(self.tte.percentile_tte(75)),
            "num_hits": int(num_hits),
            "num_total": int(n_exact),
        }
