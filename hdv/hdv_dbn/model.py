import numpy as np
from pgmpy.models import DynamicBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

from .config import DBN_STATES


class HDVDBN:
    """
    Dynamic Bayesian Network for HDV behavior with:
    
    Latent variables:
        Style_t  - driver style (assumed approximately constant over a trajectory)
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
        - Continuous Observations O_t (x, y, vx, vy, ax, ay) are NOT in this graph. They are handled by an external emission 
          model that is trained along with this DBN.
    """

    def __init__(self):
        """ Initialize the HDVDBN structure and neutral CPDs."""
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
        """
        Construct the prior CPD
        P(Style_0): uniform over driving styles.

        Returns
        TabularCPD: 
            A pgmpy :class:`TabularCPD` object whose:
            - ``variable`` is ``('Style', 0)``
            - shape is ``(num_style, 1)``, where each row corresponds to a style and the single column encodes the prior 
              probability.
        """
        probs = np.full((self.num_style, 1), 1.0 / self.num_style)  # rows=style, cols=prior
        return TabularCPD(
            variable=('Style', 0),
            variable_card=self.num_style,
            values=probs,
            state_names={('Style', 0): list(self.style_states)},
        )

    def _cpd_prior_action(self):
        """
        Construct the prior CPD
        P(Action_0 | Style_0)
        For every style, the prior over actions is initialized to be uniform.
        
        Returns
        TabularCPD
            A pgmpy :class:`TabularCPD` object whose:
            - ``variable`` is ``('Action', 0)``
            - ``evidence`` is ``[('Style', 0)]``
            - ``values`` has shape ``(num_action, num_style)``, where each column corresponds to a style state and each row 
              corresponds to an action state.
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
        Construct the temporal CPD
        P(Style_1 | Style_0)
        Encode style as effectively constant.

        Returns
        TabularCPD
            A pgmpy :class:`TabularCPD` object whose:
            - ``variable`` is ``('Style', 1)``
            - ``evidence`` is ``[('Style', 0)]``
            - ``values`` is a ``(num_style, num_style)`` identity matrix, where columns index the previous style and rows index 
              the next style.
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
        Construct the temporal CPD
        P(Action_1 | Action_0, Style_1)
            - strong inertia to stay in the same action
            - otherwise uniform over other actions

        Parameters
        stay : float, optional
            Probability of remaining in the same action from one time step to the next. Must be in the interval ``[0, 1]``. Default is ``0.7``.

        Returns
        TabularCPD
            A pgmpy :class:`TabularCPD` object whose:
            - ``variable`` is ``('Action', 1)``
            - ``evidence`` is ``[('Action', 0), ('Style', 1)]``
            - ``values`` has shape ``(num_action, num_action * num_style)``, where each block of ``num_action`` columns corresponds to a particular 
                combination of previous action and style.
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
        """Create and attach all CPDs to the underlying DBN."""
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
