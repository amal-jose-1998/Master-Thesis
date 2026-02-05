import torch
import math

from ..config import TRAINING_CONFIG

EPSILON = TRAINING_CONFIG.EPSILON if hasattr(TRAINING_CONFIG, "EPSILON") else 1e-6

def get_device_dtype():
    dtype_str = getattr(TRAINING_CONFIG, "dtype", "float32")
    dtype = torch.float32 if dtype_str == "float32" else torch.float64
    device = torch.device(getattr(TRAINING_CONFIG, "device", "cpu"))
    return device, dtype


def as_torch(x, device, dtype):
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)

def inv_softplus(y, min_y=1e-6):
    """
    Approx inverse of softplus for initialization.

    If softplus(rho) = y, then rho ~ log(exp(y) - 1).
    Clamp y to avoid log(0) when y is tiny.
    """
    return torch.log(torch.expm1(torch.clamp(y, min=min_y)))

def build_unconstrained_emission_params_and_opt(*,style_gauss_mean, action_gauss_mean, style_gauss_var, action_gauss_var, device, dtype,
                                                 disable_discrete_obs, bin_dim, weight_decay, lr, style_bern_p=None, action_bern_p=None):
    """
    Convert constrained emission params (var>0, p in (0,1)) into unconstrained torch Parameters,
    and build an AdamW optimizer.

    Returns a dict with:
      mu_s, mu_a, rho_s, rho_a, logit_s, logit_a, params, opt, vmin, jitter
    """
    # ----------------------------
    # Unconstrained parameterization
    # ----------------------------
    # This section converts the constrained emission parameters (variance > 0, probability in (0,1)) 
    # into unconstrained variables that an optimizer can safely update.
    # Start from current numpy params for continuity
    mu_s0 = torch.as_tensor(style_gauss_mean, device=device, dtype=dtype)      # (S,Dc)
    mu_a0 = torch.as_tensor(action_gauss_mean, device=device, dtype=dtype)    # (A,Dc)

    # Use softplus for variances to keep > 0
    vmin = float(getattr(TRAINING_CONFIG, "min_cov_diag", 1e-5)) # minimum allowed variance (floor).
    jitter = float(getattr(TRAINING_CONFIG, "emission_jitter", 1e-6)) # tiny extra constant for numerical stability.
    # Loads variances and clamps them to be at least vmin
    var_s0 = torch.as_tensor(style_gauss_var, device=device, dtype=dtype).clamp_min(vmin)
    var_a0 = torch.as_tensor(action_gauss_var, device=device, dtype=dtype).clamp_min(vmin)

    # unconstrained variance parameters the optimizer will directly update.
    rho_s = torch.nn.Parameter(inv_softplus(var_s0 - vmin)) # subtract vmin here because softplus(rho) should represent “variance above the floor”.
    rho_a = torch.nn.Parameter(inv_softplus(var_a0 - vmin))
    # Means have no constraints, so no special transform is needed.
    mu_s = torch.nn.Parameter(mu_s0)
    mu_a = torch.nn.Parameter(mu_a0)

    if (not disable_discrete_obs) and (bin_dim > 0):
        if style_bern_p is None or action_bern_p is None:
            raise ValueError("Bernoulli enabled (bin_dim>0 and disable_discrete_obs=False) but style_bern_p/action_bern_p is None.")
        # Load Bernoulli probabilities and clamp away from 0/1
        p_s0 = torch.as_tensor(style_bern_p, device=device, dtype=dtype).clamp(EPSILON, 1.0 - EPSILON)
        p_a0 = torch.as_tensor(action_bern_p, device=device, dtype=dtype).clamp(EPSILON, 1.0 - EPSILON)
        # Converts p to logit: logit = log(p / (1 - p))
        logit_s = torch.nn.Parameter(torch.log(p_s0) - torch.log1p(-p_s0))  # (S,B)
        logit_a = torch.nn.Parameter(torch.log(p_a0) - torch.log1p(-p_a0))  # (A,B)  
        # Now the optimizer can update logit_s, logit_a freely
        params = [mu_s, mu_a, rho_s, rho_a, logit_s, logit_a] 
    else:
        logit_s = logit_a = None
        params = [mu_s, mu_a, rho_s, rho_a]

    opt = torch.optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay)) # Create optimizer
    return {
        "mu_s": mu_s,
        "mu_a": mu_a,
        "rho_s": rho_s,
        "rho_a": rho_a,
        "logit_s": logit_s,
        "logit_a": logit_a,
        "params": params,
        "opt": opt,
        "vmin": vmin,
        "jitter": jitter,
    }

def chunk_objective(x, gamma, ctx):
    device, dtype = ctx["device"], ctx["dtype"]
    S, A = ctx["S"], ctx["A"]
    Dc = ctx["Dc"]
    cont_idx, bin_idx = ctx["cont_idx"], ctx["bin_idx"]
    vmin, jitter = ctx["vmin"], ctx["jitter"]
    EPSILON = ctx["EPSILON"]

    mu_s, mu_a = ctx["mu_s"], ctx["mu_a"]
    rho_s, rho_a = ctx["rho_s"], ctx["rho_a"]
    logit_s, logit_a = ctx["logit_s"], ctx["logit_a"]

    # x: (T,D), gamma: (T,S,A)
    T = int(x.shape[0]) # Number of time steps in the chunk.

    # Continuous part
    if Dc > 0:
        xc_raw = x[:, cont_idx]               # Extract continuous features from x (T,Dc)
        finite_c = torch.isfinite(xc_raw)     # Mask of which entries are not NaN/inf.      
        m_c = finite_c.to(dtype=dtype)        # Convert boolean mask to float mask (1.0 where valid, 0.0 where invalid). (T,Dc)
        xc = torch.where(finite_c, xc_raw, torch.zeros_like(xc_raw)) # Replace invalid values with 0 so the math doesn’t explode.

        # Converts unconstrained rho to valid variances. softplus ensures >0, plus floor and jitter.
        var_s = torch.nn.functional.softplus(rho_s) + vmin + jitter  # (S,Dc)
        var_a = torch.nn.functional.softplus(rho_a) + vmin + jitter  # (A,Dc)

        # log p_s(x|s): (T,S)
        diff_s = xc[:, None, :] - mu_s[None, :, :] # (T, S, Dc)
        logps_dim = -0.5 * (math.log(2.0 * math.pi) + torch.log(var_s)[None, :, :] + (diff_s * diff_s) / var_s[None, :, :]) # Gaussian log-likelihood per dimension; Shape: (T, S, Dc).
        # Apply mask per dimension (ignore invalid dims).
        logps = (logps_dim * m_c[:, None, :]).sum(dim=2) # Sum over dimensions; shape (T, S).

        # log p_a(x|a): (T,A)
        diff_a = xc[:, None, :] - mu_a[None, :, :] # (T, A, Dc)
        logpa_dim = -0.5 * (math.log(2.0 * math.pi) + torch.log(var_a)[None, :, :] + (diff_a * diff_a) / var_a[None, :, :])
        logpa = (logpa_dim * m_c[:, None, :]).sum(dim=2) # Sum over dimensions; shape (T, A).

        # Continuous PoE normalization term: log Z_c(t,s,a): (T,S,A)
        dmu = mu_s[:, None, :] - mu_a[None, :, :]         # Mean difference between style mean and action mean for each pair (s,a). (S,A,Dc)
        v = var_s[:, None, :] + var_a[None, :, :]         # Variances add for the product normalization in Gaussian×Gaussian case. (S,A,Dc)
        logZc_dim = -0.5 * (math.log(2.0 * math.pi) + torch.log(v) + (dmu * dmu) / v)  # per-dimension log normalization contribution for the Gaussian PoE pair. (S,A,Dc)
        logZc = torch.einsum("td,sad->tsa", m_c, logZc_dim) # Applies the time-dependent mask: (T, S, A)
    else:
        logps = torch.zeros((T, S), device=device, dtype=dtype)
        logpa = torch.zeros((T, A), device=device, dtype=dtype)
        logZc = torch.zeros((T, S, A), device=device, dtype=dtype)

    # Bernoulli part
    if (logit_s is not None) and (logit_a is not None):
        xb_raw = x[:, bin_idx]             # Extract binary feature columns. (T,B)
        finite_b = torch.isfinite(xb_raw)
        m_b = finite_b.to(dtype=dtype)     # Mask invalid entries. (T,B)
        xb = (xb_raw > 0.5).to(dtype=dtype) # Convert to 0/1 values by thresholding.

        # Convert logits to probabilities
        # sigmoid maps logits to probabilities. clamp avoids log(0).
        ps = torch.sigmoid(logit_s).clamp(EPSILON, 1.0 - EPSILON)  # (S,B)
        pa = torch.sigmoid(logit_a).clamp(EPSILON, 1.0 - EPSILON)  # (A,B)

        # log p_s(b|s): (T,S)
        xbS = xb[:, None, :]  # (T,1,B)
        logpsb_dim = xbS * torch.log(ps)[None, :, :] + (1.0 - xbS) * torch.log(1.0 - ps)[None, :, :] # Bernoulli log-likelihood per bit: Shape (T, S, B).
        logpsb = (logpsb_dim * m_b[:, None, :]).sum(dim=2) # Mask invalid bits and sum over bits: shape (T, S).

        # log p_a(b|a): (T,A)
        logpab_dim = xbS * torch.log(pa)[None, :, :] + (1.0 - xbS) * torch.log(1.0 - pa)[None, :, :]
        logpab = (logpab_dim * m_b[:, None, :]).sum(dim=2) # (T, A)

        # Bernoulli PoE normalization: log Z_b(t,s,a): (T,S,A)
        Zj = ps[:, None, :] * pa[None, :, :] + (1.0 - ps[:, None, :]) * (1.0 - pa[None, :, :])  # (S,A,B)
        # Avoid log(0), then take log.
        Zj = Zj.clamp(EPSILON, 1.0)
        logZj = torch.log(Zj)  # (S,A,B)
        logZb = torch.einsum("tb,sab->tsa", m_b, logZj) # Apply time mask across bits: (T,S,A)

    else:
        logpsb = torch.zeros((T, S), device=device, dtype=dtype)
        logpab = torch.zeros((T, A), device=device, dtype=dtype)
        logZb = torch.zeros((T, S, A), device=device, dtype=dtype)

    # Combine continuous + Bernoulli into PoE log p(x|s,a): (T,S,A)
    log_joint = (logps + logpsb)[:, :, None] + (logpa + logpab)[:, None, :] - (logZc + logZb) # PoE log emission for each (t,s,a)

    # Q contribution + responsibility mass
    Q = (gamma * log_joint).sum() # Multiply each (t,s,a) emission log-prob by its responsibility gamma[t,s,a]. Sum everything → scalar Q for this chunk.
    mass = gamma.sum() # Total gamma mass for this chunk.
    return Q, mass

def accumulate_Q_and_mass(obs_seqs, gamma_sa_seqs, ctx, chunk_size=None):
    device, dtype = ctx["device"], ctx["dtype"]
    Q_sum = torch.zeros((), device=device, dtype=dtype) # total PoE Q-objective (weighted log emission)
    mass = torch.zeros((), device=device, dtype=dtype) # total responsibility mass (sum of all gamma values)

    for obs, gam in zip(obs_seqs, gamma_sa_seqs): # Loops over observation sequences and their matching gamma sequences together.
        x = as_torch(obs, device=device, dtype=dtype) # (T,D)
        g = as_torch(gam, device=device, dtype=dtype) # (T,S,A)

        if chunk_size is None: # If no chunking is requested, process the whole sequence at once.
            Qi, mi = chunk_objective(x, g, ctx)
            Q_sum = Q_sum + Qi
            mass = mass + mi
        else: # If chunking is enabled, process the sequence in time slices.
            T = int(x.shape[0])
            cs = int(chunk_size)
            for t0 in range(0, T, cs):
                t1 = min(T, t0 + cs)
                Qi, mi = chunk_objective(x[t0:t1], g[t0:t1], ctx)
                Q_sum = Q_sum + Qi
                mass = mass + mi
    return Q_sum, mass

def regularizer(ctx):
    """computes a scalar regularization penalty, which will be added to the loss during the M-step."""
    device, dtype = ctx["device"], ctx["dtype"]
    vmin, jitter = ctx["vmin"], ctx["jitter"]
    lam_mu, lam_logvar, lam_logit = ctx["lam_mu"], ctx["lam_logvar"], ctx["lam_logit"]

    # Gaussian means:
    mu_s, mu_a = ctx["mu_s"], ctx["mu_a"]
    # Unconstrained variance parameters: These are not variances yet. They will be transformed using softplus.
    rho_s, rho_a = ctx["rho_s"], ctx["rho_a"]
    # Bernoulli logits (may be None if discrete obs are disabled).
    logit_s, logit_a = ctx["logit_s"], ctx["logit_a"]

    # Recompute constrained params once per step 
    var_s = torch.nn.functional.softplus(rho_s) + vmin + jitter # Converts unconstrained rho_s to valid variance; Shape: (S, Dc)
    var_a = torch.nn.functional.softplus(rho_a) + vmin + jitter # shape: (A, Dc).

    reg = torch.zeros((), device=device, dtype=dtype) # Initialize regularization accumulator

    if lam_mu > 0: # Mean regularization
        reg = reg + lam_mu * (mu_s.pow(2).mean() + mu_a.pow(2).mean()) # L2 regularization on the means.
    if lam_logvar > 0: # Log-variance regularization
        reg = reg + lam_logvar * (torch.log(var_s).pow(2).mean() + torch.log(var_a).pow(2).mean()) # Penalizes squared log-variances, not raw variances.
    if (logit_s is not None) and (logit_a is not None) and lam_logit > 0: # Bernoulli logit regularization
        reg = reg + lam_logit * (logit_s.pow(2).mean() + logit_a.pow(2).mean()) # Penalizes large logits. Equivalent to an L2 prior on logits.
    return reg

def snapshot_params(ctx):
    mu_s, mu_a = ctx["mu_s"], ctx["mu_a"]
    rho_s, rho_a = ctx["rho_s"], ctx["rho_a"]
    logit_s, logit_a = ctx["logit_s"], ctx["logit_a"]
    return {
        "mu_s": mu_s.detach().clone(),
        "mu_a": mu_a.detach().clone(),
        "rho_s": rho_s.detach().clone(),
        "rho_a": rho_a.detach().clone(),
        "logit_s": None if logit_s is None else logit_s.detach().clone(),
        "logit_a": None if logit_a is None else logit_a.detach().clone(),
    }

def restore_params(ctx, snap):
    mu_s, mu_a = ctx["mu_s"], ctx["mu_a"]
    rho_s, rho_a = ctx["rho_s"], ctx["rho_a"]
    logit_s, logit_a = ctx["logit_s"], ctx["logit_a"]
    with torch.no_grad():
        mu_s.copy_(snap["mu_s"])
        mu_a.copy_(snap["mu_a"])
        rho_s.copy_(snap["rho_s"])
        rho_a.copy_(snap["rho_a"])
        if logit_s is not None and snap["logit_s"] is not None:
            logit_s.copy_(snap["logit_s"])
        if logit_a is not None and snap["logit_a"] is not None:
            logit_a.copy_(snap["logit_a"])