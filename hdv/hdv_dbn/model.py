import numpy as np
from pgmpy.models import DynamicBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

from .config import DBN_STATES


class HDVDBN:
    """
    Dynamic Bayesian Network for HDV behavior with:
    
    Latent variables:
        Style_t  - driver style (constant over a trajectory)
        Action_t - discrete action mode at time t

    Nodes per time slice:
        ('Style', t), ('Action', t)

    Structure (2-slice template):
        Intra-slice:
            Style_t  --> Action_t

        Inter-slice:
            Style_t  --> Style_{t+1}  
            Action_t --> Action_{t+1}

    Notes:
        - Observations O_t (x, y, vx, vy, ax, ay) are NOT in this graph; they will enter through an external emission model during training/inference.
    """

    def __init__(self):
        # State names from config.py
        self.style_states = DBN_STATES.driving_style      
        self.action_states = DBN_STATES.action            

        self.num_style = len(self.style_states)
        self.num_action = len(self.action_states)

        self.model = DynamicBayesianNetwork()
        self._build_structure()
        self._init_cpds()

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------
    def _build_structure(self):
        """Define the nodes and edges of the 2-slice DBN."""
        # Nodes for slices 0 and 1
        self.model.add_nodes_from([
            ('Style', 0), ('Action', 0),
            ('Style', 1), ('Action', 1),
        ])

        # Intra-slice edges
        self.model.add_edge(('Style', 0), ('Action', 0))
        self.model.add_edge(('Style', 1), ('Action', 1))

        # Inter-slice edges
        # Style is modeled as constant via self-transition
        self.model.add_edge(('Style', 0), ('Style', 1))
        # Actions follow a first-order Markov chain
        self.model.add_edge(('Action', 0), ('Action', 1))

    # ------------------------------------------------------------------
    # Slice-0 priors
    # ------------------------------------------------------------------
    def _cpd_prior_style(self):
        """P(Style_0): uniform over driving styles."""
        probs = np.full((self.num_style, 1), 1.0 / self.num_style)  # rows=style, cols=prior
        return TabularCPD(
            variable=('Style', 0),
            variable_card=self.num_style,
            values=probs,
            state_names={('Style', 0): list(self.style_states)},
        )

    def _cpd_prior_action(self):
        """
        P(Action_0 | Style_0)

        For every style, the prior over actions is uniform.
        """
        # values shape: (num_action, num_style)
        col = np.full(self.num_action, 1.0 / self.num_action)
        values = np.tile(col[:, None], (1, self.num_style))

        return TabularCPD(
            variable=('Action', 0),
            variable_card=self.num_action,
            values=values,
            evidence=[('Style', 0)],
            evidence_card=[self.num_style],
            state_names={
                ('Action', 0): list(self.action_states),
                ('Style', 0): list(self.style_states),
            },
        )

    # ------------------------------------------------------------------
    # Temporal CPDs (t -> t+1)
    # ------------------------------------------------------------------
    def _cpd_style_1(self):
        """
        P(Style_1 | Style_0)

        Encode style as effectively constant.
        """
        mat = np.eye(self.num_style)

        return TabularCPD(
            variable=('Style', 1),
            variable_card=self.num_style,
            values=mat,  # rows=new style, cols=old style
            evidence=[('Style', 0)],
            evidence_card=[self.num_style],
            state_names={
                ('Style', 1): list(self.style_states),
                ('Style', 0): list(self.style_states),
            },
        )

    def _cpd_action_1(self, stay=0.7):
        """
        P(Action_1 | Action_0, Style_1)
            - strong inertia to stay in the same action
            - otherwise uniform over other actions
        """
        off = (1.0 - stay) / max(self.num_action - 1, 1)

        cols = []
        # evidence order: [ ('Action', 0), ('Style', 1) ]
        for prev_action_idx in range(self.num_action):
            for _style_idx in range(self.num_style):
                col = np.full(self.num_action, off)
                col[prev_action_idx] = stay
                cols.append(col)

        values = np.array(cols).T  # rows=new action, cols=(prev_action, style)

        return TabularCPD(
            variable=('Action', 1),
            variable_card=self.num_action,
            values=values,
            evidence=[('Action', 0), ('Style', 1)],
            evidence_card=[self.num_action, self.num_style],
            state_names={
                ('Action', 1): list(self.action_states),
                ('Action', 0): list(self.action_states),
                ('Style', 1): list(self.style_states),
            },
        )

    # ------------------------------------------------------------------
    # CPD assembly
    # ------------------------------------------------------------------
    def _init_cpds(self):
        """Initialize CPDs. These are neutral priors to be refined during learning."""
        prior_style = self._cpd_prior_style()
        prior_action = self._cpd_prior_action()
        cpd_style_1 = self._cpd_style_1()
        cpd_action_1 = self._cpd_action_1()

        self.model.add_cpds(
            prior_style,
            prior_action,
            cpd_style_1,
            cpd_action_1,
        )

        # Initialize slice-0 structure internally for pgmpy
        self.model.initialize_initial_state()
        # Validate model consistency
        self.model.check_model()
