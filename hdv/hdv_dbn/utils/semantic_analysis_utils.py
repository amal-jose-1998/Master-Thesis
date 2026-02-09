from pathlib import Path
import json
import numpy as np
import torch

from ..config import WINDOW_CONFIG, TRAINING_CONFIG, WINDOW_FEATURE_COLS, CONTINUOUS_FEATURES,DBN_STATES
from ..datasets import load_highd_folder, load_or_build_windowized
from .eval_core import scale_obs_masked, seq_key
from ..inference import infer_posterior
from .trainer_diagnostics import posterior_weighted_feature_stats

def make_semantic_out_dir(*, exp_dir, split_name, semantic_cfg):
    """
    Build a stable output directory for semantic analysis runs so different
    feature-set / rmse settings do not overwrite each other.

    Returns:
        out_dir: Path to write artifacts into
        run_tag: folder tag used (for logging / summary.json)
    """
    split_name = str(split_name).lower().strip()

    # Auto-tag output folder from config (prevents overwrites across semantic settings)
    set_name = str(getattr(semantic_cfg, "semantic_feature_set", "all")).lower()
    rmse_mode = str(getattr(semantic_cfg, "rmse_mode", "raw")).lower()
    risk = bool(getattr(semantic_cfg, "include_risk_block", False))

    # filesystem-safe tag
    set_name = set_name.replace("+", "_")

    tag_parts = [set_name]
    if risk:
        tag_parts.append("with_risk_features")
    
    tag_parts.append(rmse_mode)

    run_tag = "_".join(tag_parts)

    out_dir = Path(exp_dir) / "semantic_analysis" / split_name / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, run_tag


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path, header, rows):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def _load_split_json(exp_dir):
    p = exp_dir / "split.json"
    print(f"[semantic_analysis] Loading split -> {p}")
    if not p.exists():
        raise FileNotFoundError(f"Missing split.json: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if "keys" not in payload:
        raise ValueError(f"split.json format unexpected (missing 'keys'): {p}")
    for k in ("train", "val", "test"):
        if k not in payload["keys"]:
            raise ValueError(f"split.json format unexpected (missing keys['{k}']): {p}")
    return payload

def _weighted_stats_1d(obs_seqs_raw, w_seqs, obs_names, feat_names):
    """
    Posterior-weighted mean/std for a 1D latent (style or action).

    Returns:
      feat_names_used
      mass:  (K,)
      means: (K,F)
      stds:  (K,F)
    """
    name_to_idx = {n: i for i, n in enumerate(obs_names)}
    idxs = [name_to_idx.get(n, None) for n in feat_names]

    feat_pairs = [(n, j) for (n, j) in zip(feat_names, idxs) if j is not None]
    feat_used = [n for (n, _) in feat_pairs]
    j_used = [j for (_, j) in feat_pairs]

    if len(feat_used) == 0:
        return [], np.zeros((0,)), np.zeros((0, 0)), np.zeros((0, 0))

    K = None
    for w in w_seqs:
        if w is not None and w.size > 0:
            K = int(w.shape[1])
            break
    if K is None:
        return feat_used, np.zeros((0,)), np.zeros((0, len(feat_used))), np.zeros((0, len(feat_used)))

    F = len(feat_used)
    mass = np.zeros((K,), dtype=np.float64)
    s1 = np.zeros((K, F), dtype=np.float64)
    s2 = np.zeros((K, F), dtype=np.float64)

    for X, W in zip(obs_seqs_raw, w_seqs):
        if X is None or W is None or X.shape[0] == 0:
            continue
        T = int(X.shape[0])
        if W.shape[0] != T:
            raise ValueError(f"weights T mismatch: obs T={T}, w T={W.shape[0]}")
        if W.shape[1] != K:
            raise ValueError(f"weights K mismatch: expected K={K}, got {W.shape[1]}")

        Xf = X[:, j_used]  # (T,F)
        finite = np.isfinite(Xf)
        Xf0 = np.where(finite, Xf, 0.0)

        for k in range(K):
            wk = W[:, k].astype(np.float64, copy=False)  # (T,)
            mass[k] += float(np.sum(wk))
            wkm = wk[:, None] * finite.astype(np.float64, copy=False)
            s1[k] += np.sum(wkm * Xf0, axis=0)
            s2[k] += np.sum(wkm * (Xf0 * Xf0), axis=0)

    means = np.full((K, F), np.nan, dtype=np.float64)
    stds = np.full((K, F), np.nan, dtype=np.float64)

    for k in range(K):
        m = mass[k]
        if m <= 1e-12:
            continue
        mu = s1[k] / m
        var = np.maximum(s2[k] / m - mu * mu, 0.0)
        means[k] = mu
        stds[k] = np.sqrt(var)

    return feat_used, mass, means, stds

def load_sequences_from_experiment_split(exp_dir, data_root, split_name):
    """
    Loads windowized sequences matching the experiment split.json and returns:
      seqs, feature_cols, split_payload, split_keys
    """
    exp_dir = Path(exp_dir)
    data_root = Path(data_root)
    split_name = str(split_name).lower().strip()
    if split_name not in ("train", "val", "test"):
        raise ValueError(f"split_name must be one of ['train','val','test'], got '{split_name}'")

    split_payload = _load_split_json(exp_dir)
    split_keys = set(split_payload["keys"][split_name])

    # Use W/stride saved with experiment (single source of truth)
    W = int(split_payload.get("W", int(WINDOW_CONFIG.W)))
    stride = int(split_payload.get("stride", int(WINDOW_CONFIG.stride)))

    print(f"[semantic_analysis] split='{split_name}' vehicles={len(split_keys)}  W={W} stride={stride}")
    print(f"[semantic_analysis] Loading highD df from: {data_root}")

    df = load_highd_folder(data_root, cache_path=None, force_rebuild=False, max_recordings=getattr(TRAINING_CONFIG, "max_highd_recordings", None))

    cache_dir = data_root / "cache"
    print(f"[semantic_analysis] Building/loading window cache: cache_dir={cache_dir}  W={W} stride={stride}")
    all_seqs = load_or_build_windowized( df, cache_dir=cache_dir, W=W, stride=stride, force_rebuild=False)

    print(f"[semantic_analysis] Total windowized sequences in cache: {len(all_seqs)}")
    seqs = [s for s in all_seqs if seq_key(s) in split_keys]
    print(f"[semantic_analysis] Matched split sequences: {len(seqs)}/{len(split_keys)}")

    found = {seq_key(s) for s in seqs}
    missing = split_keys - found
    if missing:
        raise RuntimeError(
            f"[semantic_analysis] Missing {len(missing)} vehicles from cache for split='{split_name}'. "
            f"Likely W/stride/max_recordings mismatch.\n"
            f"Example missing keys: {list(sorted(missing))[:10]}"
        )

    return seqs, list(WINDOW_FEATURE_COLS), split_payload, split_keys

def compute_scale_idx(feature_cols):
    cont = set(CONTINUOUS_FEATURES)
    return np.asarray([i for i, n in enumerate(feature_cols) if n in cont], dtype=np.int64)

@torch.no_grad()
def infer_gamma_on_split(*, trainer, seqs, feature_cols, scale_idx, split_name, progress_every=50):
    """
    Runs posterior inference on the split.

    Returns:
      obs_raw_seqs: list[np.ndarray]
      gamma_sa_seqs: list[torch.Tensor] on CPU, each (T,S,A)
    """
    emissions = trainer.emissions

    obs_raw_seqs = []
    gamma_sa_seqs = []

    for i, seq in enumerate(seqs):
        mean, std = _select_scaler_for_seq(trainer, seq)

        obs_raw = seq.obs
        obs_scaled = scale_obs_masked(obs_raw, mean, std, scale_idx)

        gamma_sa, gamma_s, gamma_a, loglik = infer_posterior(obs_scaled, trainer.pi_s0, trainer.pi_a0_given_s0, trainer.A_s, trainer.A_a, emissions)

        obs_raw_seqs.append(obs_raw)
        gamma_sa_seqs.append(gamma_sa.detach().cpu())

        if progress_every and ((i + 1) % progress_every == 0 or (i + 1) == len(seqs)):
            print(f"[semantic_analysis:{split_name}] processed {i+1}/{len(seqs)}", flush=True)

    return obs_raw_seqs, gamma_sa_seqs

def _select_scaler_for_seq(trainer, seq):
    mean = trainer.scaler_mean
    std = trainer.scaler_std
    if mean is None or std is None:
        raise RuntimeError("Trainer missing scaler_mean/std.")

    if isinstance(mean, dict) and isinstance(std, dict):
        meta = getattr(seq, "meta", None) or {}
        cls = str(meta.get("meta_class", "NA"))
        if cls not in mean or cls not in std:
            raise KeyError(f"Classwise scaler missing key '{cls}'. Available: {sorted(mean.keys())[:10]} ...")
        return mean[cls], std[cls]

    return mean, std

def compute_joint_semantics(*, obs_raw_seqs, gamma_sa_seqs, feature_cols, semantic_feature_cols, S, A):
    """
    Returns:
      feat_names: list[str]
      means_sa: (S,A,F)
      stds_sa:  (S,A,F)
      mass_sa:  (S,A)
      frac_sa:  (S,A)
    """
    feat_names, means_sa, stds_sa = posterior_weighted_feature_stats(obs_names=feature_cols, obs_seqs=obs_raw_seqs, gamma_sa_seqs=gamma_sa_seqs, 
                                                                     semantic_feature_names=semantic_feature_cols, include_derived=False, S=S, A=A)

    mass_sa = np.zeros((S, A), dtype=np.float64)
    for g in gamma_sa_seqs:
        if g is None or g.numel() == 0:
            continue
        mass_sa += g.sum(dim=0).numpy().astype(np.float64)

    total_mass = float(np.sum(mass_sa))
    frac_sa = (mass_sa / total_mass) if total_mass > 0 else np.zeros_like(mass_sa)

    return feat_names, means_sa, stds_sa, mass_sa, frac_sa

def write_joint_csv(*, out_dir, feat_names, means_sa, stds_sa, mass_sa, frac_sa, S, A):
    rows = []
    for s in range(S):
        for a in range(A):
            row = [
                s, a,
                str(DBN_STATES.driving_style[s]) if s < len(DBN_STATES.driving_style) else f"style_{s}",
                str(DBN_STATES.action[a]) if a < len(DBN_STATES.action) else f"action_{a}",
                float(mass_sa[s, a]),
                float(frac_sa[s, a]),
            ]
            for j in range(len(feat_names)):
                row.append(float(means_sa[s, a, j]) if np.isfinite(means_sa[s, a, j]) else np.nan)
                row.append(float(stds_sa[s, a, j]) if np.isfinite(stds_sa[s, a, j]) else np.nan)
            rows.append(row)

    header = ["s", "a", "style_name", "action_name", "mass", "mass_frac"]
    for fn in feat_names:
        header += [f"{fn}_mean", f"{fn}_std"]
    _write_csv(out_dir / "joint_semantics.csv", header, rows)


def write_style_csv(*, out_dir, feat_used_s, mass_s, means_s, stds_s, S):
    rows = []
    total_ms = float(np.sum(mass_s))
    for s in range(S):
        row = [
            s,
            str(DBN_STATES.driving_style[s]) if s < len(DBN_STATES.driving_style) else f"style_{s}",
            float(mass_s[s]),
            float(mass_s[s] / total_ms) if total_ms > 0 else 0.0,
        ]
        for j in range(len(feat_used_s)):
            row.append(float(means_s[s, j]) if np.isfinite(means_s[s, j]) else np.nan)
            row.append(float(stds_s[s, j]) if np.isfinite(stds_s[s, j]) else np.nan)
        rows.append(row)

    header = ["s", "style_name", "mass", "mass_frac"]
    for fn in feat_used_s:
        header += [f"{fn}_mean", f"{fn}_std"]
    _write_csv(out_dir / "style_semantics_mixture_over_action.csv", header, rows)


def write_action_csv(*, out_dir, feat_used_a, mass_a, means_a, stds_a, A):
    rows = []
    total_ma = float(np.sum(mass_a))
    for a in range(A):
        row = [
            a,
            str(DBN_STATES.action[a]) if a < len(DBN_STATES.action) else f"action_{a}",
            float(mass_a[a]),
            float(mass_a[a] / total_ma) if total_ma > 0 else 0.0,
        ]
        for j in range(len(feat_used_a)):
            row.append(float(means_a[a, j]) if np.isfinite(means_a[a, j]) else np.nan)
            row.append(float(stds_a[a, j]) if np.isfinite(stds_a[a, j]) else np.nan)
        rows.append(row)

    header = ["a", "action_name", "mass", "mass_frac"]
    for fn in feat_used_a:
        header += [f"{fn}_mean", f"{fn}_std"]
    _write_csv(out_dir / "action_semantics_mixture_over_style.csv", header, rows)


def compute_marginal_semantics(*, obs_raw_seqs, gamma_sa_seqs, feature_cols, feat_names):
    """
    Computes marginal (style) and (action) semantics using weights derived from gamma_sa.

    Returns dict with:
      feat_used_s, mass_s, means_s, stds_s
      feat_used_a, mass_a, means_a, stds_a
    """
    style_w_seqs = []
    action_w_seqs = []

    for g in gamma_sa_seqs:
        gg = g.numpy().astype(np.float64)
        style_w_seqs.append(np.sum(gg, axis=2))   # (T,S)
        action_w_seqs.append(np.sum(gg, axis=1))  # (T,A)

    feat_used_s, mass_s, means_s, stds_s = _weighted_stats_1d(
        obs_seqs_raw=obs_raw_seqs,
        w_seqs=style_w_seqs,
        obs_names=feature_cols,
        feat_names=feat_names,
    )
    feat_used_a, mass_a, means_a, stds_a = _weighted_stats_1d(
        obs_seqs_raw=obs_raw_seqs,
        w_seqs=action_w_seqs,
        obs_names=feature_cols,
        feat_names=feat_names,
    )

    return dict(
        feat_used_s=feat_used_s, mass_s=mass_s, means_s=means_s, stds_s=stds_s,
        feat_used_a=feat_used_a, mass_a=mass_a, means_a=means_a, stds_a=stds_a,
    )


def compute_action_style_consistency(*, means_sa, feat_names, S, A, rmse_mode="raw", stds_sa=None, frac_sa=None):
    """
    Returns rows: [a, action_name, rmse_between_style0_and_style1]

    rmse_mode:
      - "raw":    RMSE in raw feature units
      - "zscore": normalize each feature by a pooled posterior-weighted std to avoid domination by large-range features
    """
    if S < 2:
        return []

    if len(feat_names) == 0:
        return []
    
    rmse_mode = str(rmse_mode).lower().strip()

    # feature indices (all used)
    js = np.arange(len(feat_names), dtype=np.int64)

    # Compute feature scales for zscore mode
    if rmse_mode == "zscore":
        if stds_sa is None or frac_sa is None:
            raise ValueError("rmse_mode='zscore' requires stds_sa and frac_sa.")
        eps = 1e-6
        F = len(feat_names)
        scale = np.zeros((F,), dtype=np.float64)
        wsum = 0.0
        for s in range(S):
            for a in range(A):
                w = float(frac_sa[s, a])
                if w <= 0:
                    continue
                ss = np.asarray(stds_sa[s, a, :], dtype=np.float64)
                ss = np.where(np.isfinite(ss), ss, 0.0)
                scale += w * ss
                wsum += w
        scale = scale / max(wsum, eps)
        scale = np.maximum(scale, eps)
    
    else:
        scale = np.ones((len(feat_names),), dtype=np.float64)

    rows = []
    for a in range(A):
        v0 = np.asarray(means_sa[0, a, js], dtype=np.float64)
        v1 = np.asarray(means_sa[1, a, js], dtype=np.float64)
        mask = np.isfinite(v0) & np.isfinite(v1) & np.isfinite(scale)
        if not np.any(mask):
            dist = np.nan
        else:
            d = (v0[mask] - v1[mask]) / scale[mask]
            dist = float(np.sqrt(np.mean(d * d)))
        rows.append([
            a,
            str(DBN_STATES.action[a]) if a < len(DBN_STATES.action) else f"action_{a}",
            dist
        ])
    return rows