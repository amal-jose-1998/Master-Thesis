import numpy as np
from pgmpy.models import DynamicBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

from .config import DBN_STATES


class HDVDBN:
    """
    Dynamic Bayesian Network for HDV behavior.
    
    Nodes per time slice:
        Style_t - discrete driver style
        Intent_t - discrete intent
        Maneuver_t - discrete maneuver
        
    Structure:
        Style_t  --> Style_{t+1}
        Intent_t --> Intent_{t+1}
        Maneuver_t --> Maneuver_{t+1}

        Style_t, Intent_t --> Maneuver_t
        Style_{t+1}, Intent_{t+1} --> Maneuver_{t+1}
    """

    def __init__(self):
        self.style_states = DBN_STATES.driving_style_states
        self.intent_states = DBN_STATES.intent_states
        self.maneuver_states = DBN_STATES.maneuver_states

        self.num_style = len(self.style_states)
        self.num_intent = len(self.intent_states)
        self.num_maneuver = len(self.maneuver_states)

        self.model = DynamicBayesianNetwork()
        self._build_structure()
        self._init_cpds()

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------
    def _build_structure(self):
        """Define the nodes and edges of the 2-slice DBN."""
        # Add nodes for time slices 0 and 1
        self.model.add_nodes_from([
            ('Style', 0), ('Intent', 0), ('Maneuver', 0),
            ('Style', 1), ('Intent', 1), ('Maneuver', 1),
        ])
        # Intra-slice edges
        self.model.add_edge(('Style', 0), ('Maneuver', 0))
        self.model.add_edge(('Intent', 0), ('Maneuver', 0))
        self.model.add_edge(('Style', 1), ('Maneuver', 1))
        self.model.add_edge(('Intent', 1), ('Maneuver', 1))
        # Inter-slice edges
        self.model.add_edge(('Style', 0), ('Style', 1))
        self.model.add_edge(('Intent', 0), ('Intent', 1))
        self.model.add_edge(('Maneuver', 0), ('Maneuver', 1))

    # ------------------------------------------------------------------
    # Slice-0 priors
    # ------------------------------------------------------------------
    def _cpd_prior_style(self):
        """P(Style_0): uniform over driving styles."""
        probs = np.full((self.num_style, 1), 1.0 / self.num_style) # rows=style, cols=prior
        return TabularCPD(
            variable=('Style', 0),
            variable_card=self.num_style,
            values=probs, 
            state_names={('Style', 0): self.style_states}
        )

    def _cpd_prior_intent(self):
        """P(Intent_0): slight bias to 'keep_lane'."""
        base = np.ones(self.num_intent)
        kl_idx = self.intent_states.index('keep_lane')
        base[kl_idx] *= 2.0 # double weight for keep_lane
        base = base / base.sum()
        probs = base.reshape(-1, 1) # rows=intent, cols=prior
        return TabularCPD(
            variable=('Intent', 0),
            variable_card=self.num_intent,
            values=probs,
            state_names={('Intent', 0): self.intent_states}
        )
    
    def _cpd_prior_maneuver(self):
        """ 
        P(Maneuver_0 | Style_0, Intent_0)

        Heuristic: intent determines the maneuvers that align with it, style modulates aggressiveness / caution.
        """
        cols = []
        for style in self.style_states:
            for intent in self.intent_states:
                col = self._maneuver_prior_given(style, intent)
                cols.append(col)
        values = np.array(cols).T # rows=maneuver, cols=(style,intent)prior
        return TabularCPD(
            variable=('Maneuver', 0),   
            variable_card=self.num_maneuver,
            values=values,
            evidence=[('Style', 0), ('Intent', 0)],
            evidence_card=[self.num_style, self.num_intent],
            state_names={
                ('Maneuver', 0): self.maneuver_states,
                ('Style', 0): self.style_states,
                ('Intent', 0): self.intent_states
            }
        )

    # ------------------------------------------------------------------
    # Temporal CPDs (t -> t+1)
    # ------------------------------------------------------------------
    def _cpd_style_1(self):
        """P(Style_1 | Style_0): high chance to remain in same style."""
        stay = 0.85
        off = (1.0 - stay) / (self.num_style - 1)
        mat = np.full((self.num_style, self.num_style), off) 
        np.fill_diagonal(mat, stay)

        return TabularCPD(
            variable=('Style', 1),
            variable_card=self.num_style,
            values=mat, # rows=new style, cols=old style
            evidence=[('Style', 0)],
            evidence_card=[self.num_style],
            state_names={
                ('Style', 1): self.style_states,
                ('Style', 0): self.style_states
            }
        )

    def _cpd_intent_1(self):
        """P(Intent_1 | Intent_0): persistent but more flexible."""
        stay = 0.7
        off = (1.0 - stay) / (self.num_intent - 1)
        mat = np.full((self.num_intent, self.num_intent), off)
        np.fill_diagonal(mat, stay)

        return TabularCPD(
            variable=('Intent', 1),
            variable_card=self.num_intent,
            values=mat, # rows=new intent, cols=old intent
            evidence=[('Intent', 0)],
            evidence_card=[self.num_intent],
            state_names={
                ('Intent', 1): self.intent_states,
                ('Intent', 0): self.intent_states
            }
        )

    def _cpd_maneuver_1(self):
        """ 
        P(Maneuver_1 | Style_1, Intent_1, Maneuver_0)

        Heuristic: 
        - inertia: M_1 tends to equal M_0
        - influence of intent on maneuver
        - modulation by style (aggressive vs cautious)
        """
        cols = []
        for style in self.style_states:
            for intent in self.intent_states:
                for prev_m in self.maneuver_states:
                    col = self._maneuver_transition_column(style=style, intent=intent, prev_maneuver=prev_m)
                    cols.append(col)
        
        values = np.array(cols).T # rows=maneuver, cols=(style,intent,prev_maneuver)
        return TabularCPD(
            variable=('Maneuver', 1),
            variable_card=self.num_maneuver,
            values=values,
            evidence=[('Maneuver', 0), ('Style', 1), ('Intent', 1)],
            evidence_card=[self.num_maneuver, self.num_style, self.num_intent],
            state_names={
                ('Maneuver', 1): self.maneuver_states,
                ('Maneuver', 0): self.maneuver_states,
                ('Style', 1): self.style_states,
                ('Intent', 1): self.intent_states
            }
        )

    # ------------------------------------------------------------------
    # Helper heuristics for maneuver CPDs
    # ------------------------------------------------------------------
    def _maneuver_prior_given(self, style, intent):
        """
        Column for P(Maneuver_0 | Style_0=style, Intent_0=intent).
        """
        scores = {m: 1.0 for m in self.maneuver_states}

        # Intent-driven preferences
        if intent == "keep_lane":
            if "maintain_speed" in scores:
                scores["maintain_speed"] += 2.0
        elif intent == "lane_change_left":
            if "prepare_lc_left" in scores:
                scores["prepare_lc_left"] += 2.0
            if "perform_lc_left" in scores:
                scores["perform_lc_left"] += 1.0
        elif intent == "lane_change_right":
            if "prepare_lc_right" in scores:
                scores["prepare_lc_right"] += 2.0
            if "perform_lc_right" in scores:
                scores["perform_lc_right"] += 1.0
        elif intent == "speed_up":
            if "accelerate" in scores:
                scores["accelerate"] += 2.0
        elif intent == "slow_down":
            if "decelerate" in scores:
                scores["decelerate"] += 1.5
            if "hard_brake" in scores:
                scores["hard_brake"] += 0.5

        # Style modulation
        if style == "aggressive":
            for m in self.maneuver_states:
                if any(k in m for k in ["accelerate", "prepare_lc_left", "prepare_lc_right", "perform_lc_left", "perform_lc_right"]):
                    scores[m] *= 1.4
        elif style == "cautious":
            for m in self.maneuver_states:
                if m in ["maintain_speed", "decelerate", "hard_brake"]:
                    scores[m] *= 1.4
        elif style == "normal":
            pass  # no change
        
        # Normalize to get probabilities
        total = sum(scores.values())
        return [scores[m] / total for m in self.maneuver_states]
    
    def _maneuver_transition_column(self, style, intent, prev_maneuver):
        """Column for P(Maneuver_1 | Style_1=style, Intent_1=intent, Maneuver_0=prev_maneuver)."""
        persist = 0.6 # probability to persist in same maneuver
        base_other = (1.0 - persist) / (self.num_maneuver - 1) # distribute among others

        scores = {m: base_other for m in self.maneuver_states}
        scores[prev_maneuver] = persist

        # Intent pushes
        if intent == "keep_lane":
            if "maintain_speed" in scores:
                scores["maintain_speed"] += 0.15
        elif intent == "lane_change_left":
            if "prepare_lc_left" in scores:
                scores["prepare_lc_left"] += 0.15
            if "perform_lc_left" in scores:
                scores["perform_lc_left"] += 0.10
        elif intent == "lane_change_right":
            if "prepare_lc_right" in scores:
                scores["prepare_lc_right"] += 0.15
            if "perform_lc_right" in scores:
                scores["perform_lc_right"] += 0.10
        elif intent == "speed_up":
            if "accelerate" in scores:
                scores["accelerate"] += 0.20
        elif intent == "slow_down":
            if "decelerate" in scores:
                scores["decelerate"] += 0.15
            if "hard_brake" in scores:
                scores["hard_brake"] += 0.15

        # Style modulation
        if style == "aggressive":
            for m in self.maneuver_states:
                if any(k in m for k in ["accelerate", "prepare_lc_left", "prepare_lc_right", "perform_lc_left", "perform_lc_right"]):
                    scores[m] *= 1.3
        elif style == "cautious":
            for m in self.maneuver_states:
                if m in ["maintain_speed", "decelerate", "hard_brake"]:
                    scores[m] *= 1.3

        total = sum(scores.values())
        return [scores[m] / total for m in self.maneuver_states]
        
    # ------------------------------------------------------------------
    # CPD assembly
    # ------------------------------------------------------------------
    def _init_cpds(self):
        """Initialize CPDs. These will be learned/updated during training."""
        prior_style     = self._cpd_prior_style()
        prior_intent    = self._cpd_prior_intent()
        prior_maneuver  = self._cpd_prior_maneuver()
        cpd_style_1     = self._cpd_style_1()
        cpd_intent_1    = self._cpd_intent_1()
        cpd_maneuver_1  = self._cpd_maneuver_1()

        # Add CPDs to model
        self.model.add_cpds(
            prior_style, prior_intent, prior_maneuver,
            cpd_style_1, cpd_intent_1, cpd_maneuver_1
        )

        self.model.initialize_initial_state() # initialize the 0th slice
        self.model.check_model() # validate the model
