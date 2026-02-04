import numpy as np
from sklearn.cluster import MiniBatchKMeans

try:
    # When imported as a package module
    from ..config import TRAINING_CONFIG
except ImportError:
    # When run as a standalone script (debug)
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))
    from hdv.hdv_dbn.config import TRAINING_CONFIG


def init_emissions(train_obs_seqs, emission_obj, device, dtype):
    """
    Initializer for emission models.

    Supports:
      (1) PoE MixedEmissionModel interface:
            - style_gauss, action_gauss, style_bern, action_bern
      (2) Hierarchical MixedEmissionModel interface:
            - gauss (joint (S,A)), bern (joint (S,A))

    Uses MiniBatchKMeans on continuous dims (finite rows preferred),
    and per-cluster Bernoulli initialization for binary dims.
   
    Parameters
    train_obs_seqs : list[np.ndarray]
        List of sequences, each shape (T, obs_dim)
    emission_obj : MixedEmissionModel
        Must expose:
          - cont_idx, bin_idx, cont_dim, bin_dim
          - num_style (S), num_action (A)
          - style_gauss, action_gauss (with set_params or writable mean/var)
          - style_bern, action_bern (with set_p or writable p)
    """
    max_samples = int(getattr(TRAINING_CONFIG, "max_kmeans_samples", 200000))
    seed = int(getattr(TRAINING_CONFIG, "seed", 0))
    rng = np.random.default_rng(seed)

    S = int(getattr(emission_obj, "num_style"))
    A = int(getattr(emission_obj, "num_action"))

    cont_idx = np.asarray(getattr(emission_obj, "cont_idx", []), dtype=np.int64)
    bin_idx = np.asarray(getattr(emission_obj, "bin_idx", []), dtype=np.int64)

    Dc = int(getattr(emission_obj, "cont_dim", 0))
    B = int(getattr(emission_obj, "bin_dim", 0))

    print("[Init] Initialising emissions with (subsampled) k-means...")
    
    if Dc <= 0 or cont_idx.size == 0:
        raise RuntimeError("MixedEmissionModel has no continuous dimensions to initialise (Dc=0).")
    
    # -----------------------------
    # Continuous part (Gaussian)
    # -----------------------------
    pool = [] # to store a small sample of continuous rows for clustering (size ≤ max_samples)
    pool_n = 0 # counter of total rows seen so far

    def try_add_rows(Xc_rows):
        """
        Add rows to reservoir pool up to max_samples (uniform reservoir sampling).
        The rows are added only with a probabiity of max_samples / total_rows_seen_so_far,
        ensuring that at the end, each row has equal probability of being included in the pool. 
        
        parameters
        Xc_rows : np.ndarray
            Continuous data rows to consider adding, shape (M, cont_dim).
        """
        nonlocal pool, pool_n
        for row in Xc_rows:
            pool_n += 1
            if len(pool) < max_samples: # pool not full yet
                pool.append(row) # The first max_samples rows are always kept
            else: # pool is full => probabilistic replacement
                j = rng.integers(0, pool_n)
                if j < max_samples:
                    pool[j] = row

    for seq in train_obs_seqs:
        if seq is None or len(seq) == 0: # Skip empty sequences.
            continue
        X = np.asarray(seq)
        Xc = X[:, cont_idx] # continuous part; (T, Dc)
        # Prefer fully-finite rows
        finite_rows = np.isfinite(Xc).all(axis=1) # (no NaNs/inf).
        Xc_f = Xc[finite_rows]
        if Xc_f.size: # Only add if there is at least one valid row.
            try_add_rows(Xc_f)

    # If pool is empty (all rows had NaNs), fall back to partially-finite rows
    if len(pool) == 0:
        # Simple safe fallback: replace non-finite with 0.0 *per row* 
        for seq in train_obs_seqs:
            if seq is None or len(seq) == 0:
                continue
            X = np.asarray(seq)
            Xc = X[:, cont_idx].copy()
            bad = ~np.isfinite(Xc)
            if bad.any():
                Xc[bad] = 0.0
            try_add_rows(Xc)

    Xc_pool = np.asarray(pool, dtype=np.float64)  # (n_pool, Dc)
    if Xc_pool.ndim != 2 or Xc_pool.shape[1] != Dc:
        raise ValueError(f"Continuous pool has shape {Xc_pool.shape}, expected (_, {Dc})")
    
    # global fallback stats
    mu_global = np.nanmean(Xc_pool, axis=0) if Xc_pool.size else np.zeros((Dc,), dtype=np.float64)
    var_global = np.nanvar(Xc_pool, axis=0) + 1e-6 if Xc_pool.size else np.ones((Dc,), dtype=np.float64)
    # Ensures variance is finite and not too small.
    var_global = np.where(np.isfinite(var_global), var_global, 1.0)
    var_global = np.maximum(var_global, 1e-6)

    # If extremely small support, just copy global params to all clusters
    min_needed = max(10 * max(S, A), Dc + 1)
    tiny_support = (Xc_pool.shape[0] < min_needed)

    # ------------------------------------------------------------------
    # KMeans initialisation helpers
    # ------------------------------------------------------------------
    def fit_kmeans(K, random_state):
        # Runs KMeans with K clusters on Xc_pool.
        if tiny_support:
            # skip clustering and just use global mean.
            labels = np.zeros((Xc_pool.shape[0],), dtype=np.int64)
            centers = np.tile(mu_global[None, :], (K, 1))
            return labels, centers

        mbk = MiniBatchKMeans(
            n_clusters=K,
            batch_size=2048,
            max_iter=100,
            n_init=5,
            random_state=random_state,
        )
        labels = mbk.fit_predict(Xc_pool)
        centers = mbk.cluster_centers_
        return labels, centers

    def cluster_mean_var(labels, centers, K):
        # Convert labels/centers into per-cluster mean/var
        mean = np.zeros((K, Dc), dtype=np.float64)
        var = np.zeros((K, Dc), dtype=np.float64)
        for k in range(K):
            mask = (labels == k)
            n = int(mask.sum())
            if n < Dc + 1: # If too few points
                mean[k] = centers[k] # Use center for mean.
                var[k] = var_global  # Use global variance because sample variance would be degenerate.
            else:
                Xk = Xc_pool[mask]
                mean[k] = Xk.mean(axis=0)
                vk = np.var(Xk, axis=0) + 1e-6
                vk = np.where(np.isfinite(vk), vk, var_global)
                var[k] = np.maximum(vk, 1e-6)
        return mean, var

    # =====================================================================
    # Branch 1: PoE emissions 
    # =====================================================================
    if hasattr(emission_obj, "style_gauss"):
        # ------------------------------------------------------------------
        # Style expert init (S clusters)
        # ------------------------------------------------------------------
        lab_s, cen_s = fit_kmeans(S, seed)
        style_mean, style_var = cluster_mean_var(lab_s, cen_s, S)

        # ------------------------------------------------------------------
        # Action expert init (A clusters)
        # ------------------------------------------------------------------
        lab_a, cen_a = fit_kmeans(A, seed + 1)
        action_mean, action_var = cluster_mean_var(lab_a, cen_a, A)

        # Write Gaussian params
        emission_obj.style_gauss.mean = np.asarray(style_mean, dtype=np.float64)
        emission_obj.style_gauss.var  = np.asarray(style_var, dtype=np.float64)
        emission_obj.action_gauss.mean = np.asarray(action_mean, dtype=np.float64)
        emission_obj.action_gauss.var  = np.asarray(action_var, dtype=np.float64)

        # ------------------------------------------------------------------
        # Bernoulli init (per-cluster), optional
        # ------------------------------------------------------------------
        if (not emission_obj.disable_discrete_obs) and (B > 0) and (bin_idx.size > 0):
            # Convert centers for numeric use
            cen_s_f = np.asarray(cen_s, dtype=np.float64)
            cen_a_f = np.asarray(cen_a, dtype=np.float64)
            # Initialize accumulators
            sum_s = np.zeros((S, B), dtype=np.float64) # sum_s[s,b] = weighted count of ones in Bernoulli dim b for style cluster s
            cnt_s = np.zeros((S, B), dtype=np.float64) # cnt_s[s,b] = count of finite Bernoulli entries contributing
            sum_a = np.zeros((A, B), dtype=np.float64)
            cnt_a = np.zeros((A, B), dtype=np.float64)

            for seq in train_obs_seqs:
                if seq is None or len(seq) == 0:
                    continue
                X = np.asarray(seq)

                Xc = X[:, cont_idx]
                xb = X[:, bin_idx]

                finite_rows = np.isfinite(Xc).all(axis=1)
                if not finite_rows.any():
                    continue

                Xc_f = Xc[finite_rows].astype(np.float64, copy=False)

                # Assign each row to nearest style center and nearest action center
                ds = ((Xc_f[:, None, :] - cen_s_f[None, :, :]) ** 2).sum(axis=2) # shape (Tf, S)
                da = ((Xc_f[:, None, :] - cen_a_f[None, :, :]) ** 2).sum(axis=2) # shape (Tf, A)
                as_s = ds.argmin(axis=1)  # (Tf,); as_s[t] is closest style cluster for row t.
                as_a = da.argmin(axis=1)  # (Tf,); as_a[t] is closest action cluster for row t.

                # bin features on the same rows:
                xb_f = xb[finite_rows]
                finite_b = np.isfinite(xb_f)
                xb_bin = (xb_f > 0.5)

                # accumulate per row 
                for t in range(xb_bin.shape[0]):
                    s = int(as_s[t])
                    a = int(as_a[t])
                    fb = finite_b[t]
                    ones = (xb_bin[t] & fb)

                    sum_s[s] += ones.astype(np.float64)
                    cnt_s[s] += fb.astype(np.float64)

                    sum_a[a] += ones.astype(np.float64)
                    cnt_a[a] += fb.astype(np.float64)

            p_s = sum_s / np.maximum(cnt_s, 1.0)
            p_a = sum_a / np.maximum(cnt_a, 1.0)
            p_s = np.clip(np.where(np.isfinite(p_s), p_s, 0.5), 1e-3, 1.0 - 1e-3)
            p_a = np.clip(np.where(np.isfinite(p_a), p_a, 0.5), 1e-3, 1.0 - 1e-3)

            emission_obj.style_bern.p = np.asarray(p_s, dtype=np.float64)
            emission_obj.action_bern.p = np.asarray(p_a, dtype=np.float64)
        else:
            # leave as default 0.5
            pass

        emission_obj.invalidate_cache()
        emission_obj.to_device(device=device, dtype=dtype)

        print("[Init] PoE k-means initialisation done.")
        return
    
    # =====================================================================
    # Branch 2: Hierarchical emissions (joint (s,a))
    # =====================================================================
    if not hasattr(emission_obj, "gauss"):
        raise TypeError("init_emissions: emission_obj is neither PoE nor hierarchical (missing style_gauss and gauss).")

    K = S * A
    lab_k, cen_k = fit_kmeans(K, seed)
    mean_k, var_k = cluster_mean_var(lab_k, cen_k, K)

    # reshape K -> (S,A)
    mean_sa = mean_k.reshape(S, A, Dc)
    var_sa = var_k.reshape(S, A, Dc)

    emission_obj.gauss.mean = np.asarray(mean_sa, dtype=np.float64)
    emission_obj.gauss.var = np.asarray(var_sa, dtype=np.float64)

    # Bernoulli init for joint clusters
    if (not emission_obj.disable_discrete_obs) and (B > 0) and (bin_idx.size > 0):
        sum_sa = np.zeros((S, A, B), dtype=np.float64)
        cnt_sa = np.zeros((S, A, B), dtype=np.float64)

        # Use the same pool labels for Bernoulli init *approximately* by re-assigning per-row.
        # We re-run nearest-center assignment on each sequence's finite continuous rows.
        cen_k_f = np.asarray(cen_k, dtype=np.float64)

        for seq in train_obs_seqs:
            if seq is None or len(seq) == 0:
                continue
            X = np.asarray(seq)
            Xc = X[:, cont_idx]
            xb = X[:, bin_idx]

            finite_rows = np.isfinite(Xc).all(axis=1)
            if not finite_rows.any():
                continue

            Xc_f = Xc[finite_rows].astype(np.float64, copy=False)
            # nearest cluster in K
            d = ((Xc_f[:, None, :] - cen_k_f[None, :, :]) ** 2).sum(axis=2)  # (Tf,K)
            as_k = d.argmin(axis=1)  # (Tf,)

            xb_f = xb[finite_rows]
            finite_b = np.isfinite(xb_f)
            xb_bin = (xb_f > 0.5)

            for t in range(xb_bin.shape[0]):
                k = int(as_k[t])
                s = k // A
                a = k % A
                fb = finite_b[t]
                ones = (xb_bin[t] & fb)

                sum_sa[s, a] += ones.astype(np.float64)
                cnt_sa[s, a] += fb.astype(np.float64)

        p_sa = sum_sa / np.maximum(cnt_sa, 1.0)
        p_sa = np.clip(np.where(np.isfinite(p_sa), p_sa, 0.5), 1e-3, 1.0 - 1e-3)
        emission_obj.bern.p = np.asarray(p_sa, dtype=np.float64)

    emission_obj.invalidate_cache()
    emission_obj.to_device(device=device, dtype=dtype)
    print("[Init] Hierarchical (S*A) k-means initialisation done.")