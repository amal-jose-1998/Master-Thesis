import numpy as np
from dataclasses import dataclass
from tqdm.auto import tqdm

from .config import DBN_STATES

# Numerical stability constants
EPSILON = 1e-6
MAX_JITTER_ATTEMPTS = 4


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
            If numerical issues occur, the covariance is regularised with diagonal jitter; in extreme cases a large negative value is returned.

        
        """
        p = self.params[style_idx, action_idx] # Gaussian parameters for that state.
        x = obs - p.mean # deviation from mean
        d = self.obs_dim

        base_cov = p.cov
        cov = base_cov
        success = False

        for attempt in range(MAX_JITTER_ATTEMPTS):
            try:
                sign, logdet = np.linalg.slogdet(cov)
                if sign <= 0:
                    raise np.linalg.LinAlgError("Covariance matrix not positive definite.")
                sol = np.linalg.solve(cov, x)
                success = True
                # If we had to add jitter (attempt > 0), store the stabilised cov
                if attempt > 0:
                    p.cov = cov
                break
            except np.linalg.LinAlgError:
                # Increase diagonal jitter: 1e-6, 1e-5, 1e-4, 1e-3
                jitter = 10.0 ** (-6 + attempt)
                cov = base_cov + jitter * np.eye(d)
        if not success:
            # Extreme fallback: use identity covariance
            cov = np.eye(d)
            p.cov = cov
            sign, logdet = 1.0, 0.0  # det(I) = 1 => logdet = 0
            sol = x  # solving I * sol = x gives sol = x

        quad = np.dot(x, sol)
        log_likelihood = -0.5 * (d * np.log(2 * np.pi) + logdet + quad)
        
        if np.isnan(log_likelihood):
            print(
                "[GaussianEmissionModel] WARNING: NaN log-likelihood encountered, "
                "falling back to large negative value."
            )
            log_likelihood = -1e10
        
        return log_likelihood 

    def update_from_posteriors(self, obs_seqs, gamma_seqs, use_progress, verbose):
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
        verbose : int
            0 = no prints,
            1 = per-iteration summary,
            2 = detailed (more debug prints).
        use_progress : bool
            If True, show progress bars for the emission M-step.

        Returns
        weights_flat : np.ndarray
            1D array of length (num_style * num_action) with total responsibility mass per joint state z = (style, action).
        """
        obs_dim = self.obs_dim
        num_states = self.num_style * self.num_action # total number of joint states

        # init accumulators
        # Accumulates total responsibility mass for each (style, action) pair:
        # weights[s, a] = sum over all vehicles n and time steps t of γ_{n,t}(style=s, action=a)
        #weights = np.zeros((self.num_style, self.num_action)) 
        # Accumulates responsibility-weighted sum of observations for each (style, action) pair:
        # sum_x[s, a] = sum over n,t of γ_{n,t}(s,a) * o_{n,t}
        # Used later to compute the Gaussian mean μ_{s,a}
        #sum_x = np.zeros((self.num_style, self.num_action, obs_dim))
        # Accumulates responsibility-weighted second moments for each (style, action) pair:
        # sum_xxT[s, a] = sum over n,t of γ_{n,t}(s,a) * o_{n,t} o_{n,t}^T
        # Used later to compute the Gaussian covariance Σ_{s,a}        
        #sum_xxT = np.zeros((self.num_style, self.num_action, obs_dim, obs_dim))

        # Flatten (style, action) -> z index
        weights_flat = np.zeros(num_states)
        sum_x_flat = np.zeros((num_states, obs_dim))
        sum_xxT_flat = np.zeros((num_states, obs_dim, obs_dim))

        if use_progress:
            iterator = tqdm(
                zip(obs_seqs, gamma_seqs),
                total=len(obs_seqs),
                desc="M-step emissions (accumulate)",
                leave=False,
            )
        else:
            iterator = zip(obs_seqs, gamma_seqs)

        num_traj = 0
        for obs, gamma in iterator: 
            num_traj += 1
            T_n = obs.shape[0] # number of time steps in trajectory n, or, the no of observations in that trajectory
            assert gamma.shape == (T_n, num_states)

            #for z in range(num_states): # loop iterates over latent states
            #    s = z // self.num_action
            #    a = z % self.num_action
            #    gamma_z = gamma[:, z][:, None] # (T_n, 1)
            #    weights[s, a] += gamma_z.sum() # sum over time for this particular trajectory 'n' and state z
            #    sum_x[s, a] += (gamma_z * obs).sum(axis=0) # Broadcast multiplication gives shape (T_n, obs_dim). Sums over time t
            #    
            #    for t in range(T_n):
            #        sum_xxT[s, a] += gamma[t, z] * np.outer(obs[t], obs[t]) # sum over t of gamma_t(z) * (o_t o_t^T)
            #----------------------------------------------------
            # Vectorised version
            #----------------------------------------------------
            # weights[z] = sum_t gamma[t, z]
            weights_flat += gamma.sum(axis=0)
            # sum_x[z, :] = sum_t gamma[t, z] * o_t
            # gamma.T: (num_states, T_n), obs: (T_n, obs_dim)
            sum_x_flat += gamma.T @ obs
            # For second moments:
            # outer_t[t, i, j] = obs[t, i] * obs[t, j]
            outer_t = np.einsum("ti,tj->tij", obs, obs)        # (T_n, D, D)
            # sum_xxT_flat[z, :, :] = sum_t gamma[t, z] * outer_t[t, :, :]
            sum_xxT_flat += np.einsum("tz,tij->zij", gamma, outer_t)
        
        # Reshape flat accumulators to (style, action, ...)
        weights = weights_flat.reshape(self.num_style, self.num_action)
        sum_x = sum_x_flat.reshape(self.num_style, self.num_action, obs_dim)
        sum_xxT = sum_xxT_flat.reshape(self.num_style, self.num_action, obs_dim, obs_dim)

        # update parameters
        total_states = self.num_style * self.num_action
        if use_progress:
            bar = tqdm(
                total=total_states,
                desc="M-step emissions (update)",
                leave=False,
            )
        else:
            bar = None
        for s in range(self.num_style):
            for a in range(self.num_action):
                w = weights[s, a]
                if w < EPSILON:
                    # no data for this (s,a), keep old params
                    if bar is not None:
                        bar.update(1)
                    continue
                mean = sum_x[s, a] / w
                cov = sum_xxT[s, a] / w - np.outer(mean, mean)
                cov += EPSILON * np.eye(obs_dim)
                self.params[s, a] = GaussianEmissionParams(mean=mean, cov=cov)
                if bar is not None:
                    bar.update(1)

        if bar is not None:
            bar.close()
        total_weight = float(weights_flat.sum())
        if verbose >= 1:
            print(
                f"  [GaussianEmissionModel] Emission update done. "
                f"Total responsibility mass = {total_weight:.3e}"
            )
            frac = weights_flat / total_weight
            print("     Responsibility mass per joint state:")
            for z, (w, f) in enumerate(zip(weights_flat, frac)):
                s = z // self.num_action
                a = z % self.num_action
                print(
                    f"      z={z:02d} (s={s}, a={a}) "
                    f"mass={w:.0f}  frac={f:.4f}"
                )
        if verbose >= 2 and total_weight > 0.0:
            print("     Example means for first few states:")
            shown = 0
            for s in range(self.num_style):
                for a in range(self.num_action):
                    mean = self.params[s, a].mean
                    print(f"        (s={s}, a={a}) mean[:3] = {mean[:3]}")
                    shown += 1
                    if shown >= 3:
                        break
                if shown >= 3:
                    break
        
        return weights_flat

    def to_arrays(self):
        """
        Export all Gaussian parameters as dense NumPy arrays.

        Returns
        means : np.ndarray
            Array of shape (num_style, num_action, obs_dim) with the mean vector
            for each (style, action) pair.
        covs : np.ndarray
            Array of shape (num_style, num_action, obs_dim, obs_dim) with the
            covariance matrices for each (style, action) pair.
        """
        means = np.zeros((self.num_style, self.num_action, self.obs_dim))
        covs = np.zeros((self.num_style, self.num_action, self.obs_dim, self.obs_dim))
        for s in range(self.num_style):
            for a in range(self.num_action):
                p = self.params[s, a]
                means[s, a] = p.mean
                covs[s, a] = p.cov
        return means, covs

    def from_arrays(self, means, covs):
        """
        Load Gaussian parameters from dense arrays.

        Parameters
        means : np.ndarray
            Shape (num_style, num_action, obs_dim).
        covs : np.ndarray
            Shape (num_style, num_action, obs_dim, obs_dim).
        """
        assert means.shape == (self.num_style, self.num_action, self.obs_dim)
        assert covs.shape == (self.num_style, self.num_action, self.obs_dim, self.obs_dim)

        for s in range(self.num_style):
            for a in range(self.num_action):
                self.params[s, a] = GaussianEmissionParams(
                    mean=means[s, a],
                    cov=covs[s, a],
                )