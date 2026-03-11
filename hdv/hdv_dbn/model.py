import numpy as np
from pgmpy.models import DynamicBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

from .config import DBN_STATES, TRAINING_CONFIG


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
            style_t+1 --> Action_t+1

        Inter-slice:
            Style_t  --> Style_{t+1}  
            Action_t --> Action_{t+1}

    Notes:
        - Continuous Observations O_t (x, y, vx, vy, ax, ay) are NOT in this graph. They are handled by an external emission 
          model that is trained along with this DBN.
    """

    def __init__(self):
        """Initialise the DBN structure and attach default (neutral) CPDs for slices 0 and 1."""
        # State names from config.py
        self.style_states = DBN_STATES.driving_style      
        self.action_states = DBN_STATES.action            

        self.num_style = len(self.style_states)
        self.num_action = len(self.action_states)

        self.model = DynamicBayesianNetwork()
        self._build_structure()
        self._init_cpds()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _dirichlet_cols(self, n_rows, n_cols, alpha, seed=None):
        """
        Sample a (n_rows, n_cols) array where each column is a probability vector
        drawn i.i.d. from Dirichlet(alpha,...,alpha).

        Parameters
        n_rows : int    
            Number of rows.
        n_cols : int
            Number of columns.
        alpha : float
            Concentration parameter for the Dirichlet distribution.
        seed : int | None
            Random seed for reproducibility. If None, uses default RNG state.

        Returns
        np.ndarray
            Shape (n_rows, n_cols). Each column sums to 1.
        """
        rng = np.random.default_rng(seed)
        X = rng.gamma(shape=alpha, scale=1.0, size=(n_cols, n_rows))  # (n_cols, n_rows)
        X = X / X.sum(axis=1, keepdims=True)
        return X.T

    def _floor_and_renorm_cols(self, values, eps=1e-6):
        """
        Ensure strictly-positive probabilities per column (avoid exact zeros),
        then renormalize columns to sum to 1.

        Parameters
        values : np.ndarray
            Array of shape (n_rows, n_cols) representing conditional probabilities.
        eps : float
            Minimum value for any entry.

        Returns
        np.ndarray
            Adjusted array of same shape as input, with no entries < eps
            and columns summing to 1.
        """
        values = np.maximum(values, eps)
        values = values / values.sum(axis=0, keepdims=True)
        return values

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------
    def _build_structure(self):
        """Define the 2-slice DBN template: nodes for t=0,1 and the intra-/inter-slice edges."""
        # Nodes for slices 0 and 1
        self.model.add_nodes_from([
            ('Style', 0), ('Action', 0),
            ('Style', 1), ('Action', 1),
        ])

        # Intra-slice edges
        self.model.add_edge(('Style', 0), ('Action', 0))
        self.model.add_edge(('Style', 1), ('Action', 1))

        # Inter-slice edges
        self.model.add_edge(('Style', 0), ('Style', 1))
        self.model.add_edge(('Action', 0), ('Action', 1))

    # ------------------------------------------------------------------
    # Slice-0 priors
    # ------------------------------------------------------------------
    def _cpd_prior_style(self, init="uniform", alpha=1.0, seed=None):
        """
        Construct the prior CPD P(Style_0)

        Parameters
        init : {"uniform","random"}
            Initialisation mode.
        alpha : float
            Dirichlet concentration if init="random".
        seed : int | None
            RNG seed for reproducibility.

        Returns
        TabularCPD: 
            CPD for ('Style', 0) with shape (num_style, 1).
        """ 
        if init == "random":
            probs = self._dirichlet_cols(self.num_style, 1, alpha=alpha, seed=seed)
            probs = self._floor_and_renorm_cols(probs, eps=1e-6)
        else:
            probs = np.full((self.num_style, 1), 1.0 / self.num_style, dtype=float)

        return TabularCPD(
            variable=('Style', 0),
            variable_card=self.num_style,
            values=probs,
            state_names={('Style', 0): list(self.style_states)},
        )

    def _cpd_prior_action(self, init="uniform", alpha=1.0, seed=None):
        """
        Build P(Action_0 | Style_0).

        Parameters
        init : {"uniform","random"}
            Initialisation mode.
        alpha : float
            Dirichlet concentration if init="random" (per Style_0 column).
        seed : int | None
            RNG seed.

        Returns
        TabularCPD
            CPD for ('Action', 0) with evidence [('Style', 0)] and shape (num_action, num_style).
        """
        if init == "random":
            values = self._dirichlet_cols(self.num_action, self.num_style, alpha=alpha, seed=seed)
            values = self._floor_and_renorm_cols(values, eps=1e-6)
        else:
            values = np.full((self.num_action, self.num_style), 1.0 / self.num_action, dtype=float)

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
    def _cpd_style_1(self, init="sticky", stay_style=0.8, alpha=1.0, seed=None):
        """
        Build P(Style_1 | Style_0).

        Parameters
        init : {"sticky","uniform","random"}
            - "sticky": diagonal-biased transitions with stay_style.
            - "uniform": each column uniform.
            - "random": Dirichlet-random columns.
        stay_style : float
            Used when init="sticky". Probability of Style_1 == Style_0.
        alpha : float
            Dirichlet concentration if init="random".
        seed : int | None
            RNG seed.

        Returns
        TabularCPD
            CPD for ('Style', 1) with evidence [('Style', 0)] and shape (num_style, num_style).
        """
        S = self.num_style

        if init == "random":
            mat = self._dirichlet_cols(S, S, alpha=alpha, seed=seed)
            mat = self._floor_and_renorm_cols(mat, eps=1e-6)

        elif init == "uniform":
            mat = np.full((S, S), 1.0 / S, dtype=float)

        else:  # sticky
            stay_style = float(np.clip(stay_style, 1e-6, 1.0 - 1e-6))
            if S == 1:
                mat = np.array([[1.0]], dtype=float)
            else:
                off = (1.0 - stay_style) / (S - 1)
                mat = np.full((S, S), off, dtype=float)
                np.fill_diagonal(mat, stay_style)

        return TabularCPD(
            variable=('Style', 1),
            variable_card=S,
            values=mat,  # rows=new style, cols=old style
            evidence=[('Style', 0)],
            evidence_card=[S],
            state_names={
                ('Style', 1): list(self.style_states),
                ('Style', 0): list(self.style_states),
            },
        )

    def _cpd_action_1(self, init="random", alpha=1.0, seed=None):
        """
        Build P(Action_1 | Action_0, Style_1).

        Parameters
        init : {"uniform","random"}
            Initialisation mode per evidence configuration.
        alpha : float
            Dirichlet concentration if init="random".
        seed : int | None
            RNG seed.

        Returns
        TabularCPD
            CPD for ('Action', 1) with evidence [('Action', 0), ('Style', 1)].
            Values shape: (num_action, num_action * num_style).
        """
        A = self.num_action
        S = self.num_style
        n_cols = A * S 

        if init == "random":
            values = self._dirichlet_cols(A, n_cols, alpha=alpha, seed=seed)
            values = self._floor_and_renorm_cols(values, eps=1e-6)
        else:
            values = np.full((A, n_cols), 1.0 / A, dtype=float)

        return TabularCPD(
            variable=('Action', 1),
            variable_card=A,
            values=values,
            evidence=[('Action', 0), ('Style', 1)],
            evidence_card=[A, S],
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
        """Create the default CPDs for slice 0 and the 0->1 transition and attach them to the pgmpy DBN."""
        init_mode = getattr(TRAINING_CONFIG, "cpd_init", "random") 
        alpha = float(getattr(TRAINING_CONFIG, "cpd_alpha", 1.0))
        stay_style = float(getattr(TRAINING_CONFIG, "cpd_stay_style", 0.8))

        seed = getattr(TRAINING_CONFIG, "cpd_seed", None)
        if seed is None:
            seed = getattr(TRAINING_CONFIG, "seed", None)

        if init_mode == "random":
            prior_style = self._cpd_prior_style(init="random", alpha=alpha, seed=seed)
            prior_action = self._cpd_prior_action(init="random", alpha=alpha, seed=None if seed is None else seed + 1)
            cpd_style_1 = self._cpd_style_1(init="random", alpha=alpha, seed=None if seed is None else seed + 2)
            cpd_action_1 = self._cpd_action_1(init="random", alpha=alpha, seed=None if seed is None else seed + 3)

        elif init_mode == "uniform":
            prior_style = self._cpd_prior_style(init="uniform")
            prior_action = self._cpd_prior_action(init="uniform")
            cpd_style_1 = self._cpd_style_1(init="uniform")
            cpd_action_1 = self._cpd_action_1(init="uniform")

        else:  # "sticky" default
            prior_style = self._cpd_prior_style(init="uniform")
            prior_action = self._cpd_prior_action(init="uniform")
            cpd_style_1 = self._cpd_style_1(init="sticky", stay_style=stay_style)
            cpd_action_1 = self._cpd_action_1(init="uniform")

        self.model.add_cpds(prior_style, prior_action, cpd_style_1, cpd_action_1)

        self.model.initialize_initial_state()
        self.model.check_model()
