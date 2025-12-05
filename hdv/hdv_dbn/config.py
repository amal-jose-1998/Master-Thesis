from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class DBNStates:
    """Dataclass to hold the different state categories for the DBN model."""
    driving_style: Tuple[str, ...]
    action: Tuple[str, ...]
   

# Define the DBN states
DBN_STATES = DBNStates(
    driving_style=("style_0", "style_1", "style_2"), #("cautious", "normal", "aggressive")
    action=("action_0", "action_1", "action_2", "action_3", "action_4", "action_5") #("maintain_speed", "accelerate", "decelerate", "hard_brake" "lane_change_left", "lane_change_right") 
    )
