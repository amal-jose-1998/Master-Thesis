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
            Style_t  --> Action_{t+1}

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
        # Temporal influence of previous style on next action
        self.model.add_edge(('Style', 0), ('Action', 1))

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

    def _cpd_action_1(self, stay=0.7, style_change_scale=0.5):
        """
        Construct the temporal CPD
        P(Action_1 | Action_0, Style_0, Style_1)
            - strong inertia to stay in the same action, otherwise uniform over other actions
            - additionally modulated by whether the style changed between t and t+1:
              if Style_1 != Style_0, the stay probability is reduced by style_change_scale.


        Parameters
        stay : float, optional
            Probability of remaining in the same action from one time step to the next, when the style does not change. 
            Must be in the interval ``[0, 1]``. Default is ``0.7``.
        style_change_scale : float, optional
            Multiplicative factor applied to `stay_same_action` when Style_1 != Style_0.
            For example, 0.5 halves the stay probability when the style changes. Must be in (0, 1]. Default is 0.5.

        Returns
        TabularCPD
            A pgmpy :class:`TabularCPD` object whose:
            - ``variable`` is ``('Action', 1)``
            - ``evidence`` is ``[('Action', 0), ('Style', 0), ('Style', 1)]``
            - ``values`` has shape ``(num_action, num_action * num_style * num_style)``, where each column corresponds to a particular combination of
              (Action_0, Style_0, Style_1) in that evidence order.
        """
        A = self.num_action
        S = self.num_style

        # Clamp to a safe range
        stay_same_action = float(np.clip(stay, 1e-6, 1.0 - 1e-6))
        style_change_scale = float(np.clip(style_change_scale, 1e-6, 1.0))

        #off = (1.0 - stay) / max(self.num_action - 1, 1)

        cols = []
        # Evidence order: [('Action', 0), ('Style', 0), ('Style', 1)]
        for a_prev in range(A):
            for s_prev in range(S):
                for s_curr in range(S):
                    # If style stays the same, use the baseline stay prob; otherwise reduce it.
                    if s_curr == s_prev:
                        stay = stay_same_action
                    else:
                        stay = stay_same_action * style_change_scale
                    if A > 1:
                        off = (1.0 - stay) / (A - 1)
                        col = np.full(A, off, dtype=float)
                        col[a_prev] = stay
                    else:
                        # only one action, must have prob=1.
                        col = np.array([1.0], dtype=float)

                    cols.append(col)

        values = np.array(cols).T  # rows=new action, cols=(prev_action, prev_style, curr_style)

        return TabularCPD(
            variable=('Action', 1),
            variable_card=self.num_action,
            values=values,
            evidence=[('Action', 0), ('Style', 0), ('Style', 1)],
            evidence_card=[A, S, S],
            state_names={
                ('Action', 1): list(self.action_states),
                ('Action', 0): list(self.action_states),
                ('Style', 0): list(self.style_states),
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
