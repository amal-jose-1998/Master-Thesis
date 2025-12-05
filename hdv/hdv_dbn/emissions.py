import numpy as np
from dataclasses import dataclass

from .config import DBN_STATES


@dataclass
class GaussianEmissionParams:
    # container for one Gaussian 
    mean: np.ndarray      # shape (obs_dim,)
    cov: np.ndarray       # shape (obs_dim, obs_dim)


class GaussianEmissionModel:
    """
    Continuous emission model p(o_t | style, action) with multivariate Gaussians.
    o ~ N(μ(s,a),Σ(s,a))
    """

    def __init__(self, obs_dim):
        self.obs_dim = obs_dim # dimension of input feature vector
        self.style_states = DBN_STATES.driving_style
        self.action_states = DBN_STATES.action

        self.num_style = len(self.style_states)
        self.num_action = len(self.action_states)

        # a 2D array which will hold GaussianEmissionParams objects. Shape = (num_style, num_action).
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
        Compute log p(obs | Style=style_idx, Action=action_idx) for a single observation vector `obs` of shape (obs_dim,).
        """
        p = self.params[style_idx, action_idx] # Gaussian parameters for that state.
        x = obs - p.mean # deviation from mean

        sign, logdet = np.linalg.slogdet(p.cov)  # Computes log-determinant of covariance
        if sign <= 0:
            raise ValueError("Covariance matrix not positive definite.")

        inv_cov = np.linalg.inv(p.cov)
        quad = float(x.T @ inv_cov @ x) # Mahalanobis distance squared.
        d = self.obs_dim

        return -0.5 * (d * np.log(2 * np.pi) + logdet + quad) # log-density of a multivariate Gaussian 

    def update_from_posteriors(self, obs_seqs, gamma_seqs):
        """
        M-step for emissions: update mean/cov for each (style, action) using responsibilities gamma_t(z = (s,a)).

        obs_seqs: list of arrays with shape (T_n, obs_dim)
        gamma_seqs: list of arrays with shape (T_n, num_style * num_action)
        """
        obs_dim = self.obs_dim
        num_states = self.num_style * self.num_action # total number of joint states

        # init accumulators
        weights = np.zeros((self.num_style, self.num_action))
        sum_x = np.zeros((self.num_style, self.num_action, obs_dim))
        sum_xxT = np.zeros((self.num_style, self.num_action, obs_dim, obs_dim))

        for obs, gamma in zip(obs_seqs, gamma_seqs):
            T_n = obs.shape[0] # number of time steps in trajectory n, or, the no of observations in that trajectory
            assert gamma.shape == (T_n, num_states)

            for z in range(num_states):
                s = z // self.num_action
                a = z % self.num_action
                gamma_z = gamma[:, z][:, None]       # (T_n, 1)
                weights[s, a] += gamma_z.sum()
                sum_x[s, a] += (gamma_z * obs).sum(axis=0)
                # sum over t of gamma_t(z) * (o_t o_t^T)
                for t in range(T_n):
                    sum_xxT[s, a] += gamma[t, z] * np.outer(obs[t], obs[t])

        # update parameters
        eps = 1e-6
        for s in range(self.num_style):
            for a in range(self.num_action):
                w = weights[s, a]
                if w < eps:
                    # no data for this (s,a), keep old params
                    continue
                mean = sum_x[s, a] / w
                cov = sum_xxT[s, a] / w - np.outer(mean, mean)
                # add small regularization
                cov += 1e-6 * np.eye(obs_dim)
                self.params[s, a] = GaussianEmissionParams(mean=mean, cov=cov)
