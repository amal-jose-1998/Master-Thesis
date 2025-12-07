import numpy as np
from dataclasses import dataclass

from .config import DBN_STATES


@dataclass
class GaussianEmissionParams:
    """
    Container for the parameters of a multivariate Gaussian emission. This class represents a single Gaussian 
    distribution used as an emission model for a specific (style, action) latent-state pair.

    Attributes
    mean : np.ndarray
        Mean vector of the Gaussian distribution with shape ``(obs_dim,)``.
    cov : np.ndarray
        Covariance matrix of the Gaussian distribution with shape ``(obs_dim, obs_dim)``.
    """
    mean: np.ndarray      # shape (obs_dim,)
    cov: np.ndarray       # shape (obs_dim, obs_dim)


class GaussianEmissionModel:
    """
    Continuous emission model p(o_t | style, action) with multivariate Gaussians.
    o ~ N(μ_(s,a),Σ_(s,a))
    A separate Gaussian distribution is maintained for every (style, action) combination.
    
    Notes
    - The emission parameters are learned using an EM-style update.
    - This class does **not** perform inference over latent variables itself; it only evaluates and updates emission likelihoods.
    """
    def __init__(self, obs_dim):
        """
        Initialize the Gaussian emission model.

        Parameters
        obs_dim : int
            Dimensionality of the observation vector ``o_t``.
            For example, obs_dim = 6 for (x, y, vx, vy, ax, ay).
        """
        self.obs_dim = obs_dim # dimension of input feature vector
        self.style_states = DBN_STATES.driving_style
        self.action_states = DBN_STATES.action

        self.num_style = len(self.style_states)
        self.num_action = len(self.action_states)

        # Parameter table: one Gaussian per (style, action) pair
        # Shape: (num_style, num_action)
        self.params = np.empty((self.num_style, self.num_action), dtype=object) 

        # Initialisation: zero mean, identity covariance
        for s in range(self.num_style):
            for a in range(self.num_action):
                self.params[s, a] = GaussianEmissionParams(
                    mean=np.zeros(self.obs_dim),
                    cov=np.eye(self.obs_dim),
                )

    def log_likelihood(self, obs, style_idx, action_idx):
        """
        Compute the log-likelihood of a single observation vector.
        Specifically, this evaluates: 
            log p(obs | Style=style_idx, Action=action_idx) 
        using the corresponding multivariate Gaussian parameters.
        
        Parameters
        obs : np.ndarray
            Observation vector with shape ``(obs_dim,)``.
        style_idx : int
            Index of the style state.
        action_idx : int
            Index of the action state.

        Returns
        float
            Log-probability density of the observation under the selected Gaussian emission model.

        Raises
        ValueError
            If the covariance matrix is not positive definite.
        """
        p = self.params[style_idx, action_idx] # Gaussian parameters for that state.
        x = obs - p.mean # deviation from mean

        sign, logdet = np.linalg.slogdet(p.cov)  # Computes log-determinant of covariance
        if sign <= 0:
            raise ValueError("Covariance matrix not positive definite.")

        #TODO check whether both logics are same or not
        #inv_cov = np.linalg.inv(p.cov)
        #quad = float(x.T @ inv_cov @ x) # Mahalanobis distance squared. 
        sol = np.linalg.solve(p.cov, x) # for better numerical stability
        quad = np.dot(x, sol)
        d = self.obs_dim

        return -0.5 * (d * np.log(2 * np.pi) + logdet + quad) # log-density of a multivariate Gaussian 

    def update_from_posteriors(self, obs_seqs, gamma_seqs):
        """
        Update Gaussian emission parameters using posterior state probabilities.
        This method performs the M-step for the emission model, given posterior responsibilities over the joint latent state
        ``z = (Style, Action)`` at each time step.
        For each (style, action) pair, the mean and covariance are updated as:
            μ = (∑_t γ_t(z) o_t) / (∑_t γ_t(z))
            Σ = (∑_t γ_t(z) o_t o_tᵀ) / (∑_t γ_t(z)) − μ μᵀ

        Parameters
        obs_seqs : list of np.ndarray
            List of observation sequences. Each element has shape
            ``(T_n, obs_dim)``, where ``T_n`` is the length of trajectory ``n``.
        gamma_seqs : list of np.ndarray
            List of posterior responsibility arrays. Each element has shape
            ``(T_n, num_style * num_action)``, where each column corresponds
            to a joint (style, action) state.

        """
        obs_dim = self.obs_dim
        num_states = self.num_style * self.num_action # total number of joint states

        # init accumulators
        # Accumulates total responsibility mass for each (style, action) pair:
        # weights[s, a] = sum over all vehicles n and time steps t of γ_{n,t}(style=s, action=a)
        weights = np.zeros((self.num_style, self.num_action)) 
        # Accumulates responsibility-weighted sum of observations for each (style, action) pair:
        # sum_x[s, a] = sum over n,t of γ_{n,t}(s,a) * o_{n,t}
        # Used later to compute the Gaussian mean μ_{s,a}
        sum_x = np.zeros((self.num_style, self.num_action, obs_dim))
        # Accumulates responsibility-weighted second moments for each (style, action) pair:
        # sum_xxT[s, a] = sum over n,t of γ_{n,t}(s,a) * o_{n,t} o_{n,t}^T
        # Used later to compute the Gaussian covariance Σ_{s,a}        
        sum_xxT = np.zeros((self.num_style, self.num_action, obs_dim, obs_dim))

        for obs, gamma in zip(obs_seqs, gamma_seqs): # loop iterates over vehicles (index n)
            T_n = obs.shape[0] # number of time steps in trajectory n, or, the no of observations in that trajectory
            assert gamma.shape == (T_n, num_states)

            for z in range(num_states): # loop iterates over latent states
                s = z // self.num_action
                a = z % self.num_action
                gamma_z = gamma[:, z][:, None] # (T_n, 1)
                weights[s, a] += gamma_z.sum() # sum over time for this particular trajectory 'n' and state z
                sum_x[s, a] += (gamma_z * obs).sum(axis=0) # Broadcast multiplication gives shape (T_n, obs_dim). Sums over time t
                
                for t in range(T_n):
                    sum_xxT[s, a] += gamma[t, z] * np.outer(obs[t], obs[t]) # sum over t of gamma_t(z) * (o_t o_t^T)

        # update parameters
        eps = 1e-6 # Small threshold to avoid division by 0
        # Loop over all style indices s and action indices a.
        for s in range(self.num_style):
            for a in range(self.num_action):
                w = weights[s, a] # total responsibility mass for state (s,a)
                if w < eps:
                    # no data for this (s,a), keep old params
                    continue
                mean = sum_x[s, a] / w
                cov = sum_xxT[s, a] / w - np.outer(mean, mean)
                # add small regularization to ensure it’s positive definite and invertible
                cov += 1e-6 * np.eye(obs_dim)
                self.params[s, a] = GaussianEmissionParams(mean=mean, cov=cov)
