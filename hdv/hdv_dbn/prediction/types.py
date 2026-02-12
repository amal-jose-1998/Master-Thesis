from dataclasses import dataclass
from typing import Dict
import numpy as np

@dataclass
class BeliefState:
    # posterior over joint (s,a) at time t
    gamma_sa: np.ndarray # shape: (S, A), sums to 1
    t: int

@dataclass
class PredictionTrace:
    # per time t, store belief + horizon predictions
    # action_pred[h] shape (A,)
    belief: BeliefState
    action_pred: Dict[int, np.ndarray] # h -> p(a_{t+h} | o_{1:t})
    style_pred: Dict[int, np.ndarray]  # h -> p(s_{t+h} | o_{1:t})
