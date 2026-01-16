"""
Evaluation entry point for the HDV DBN on the highD test set.
Loads:
  - highD sequences (same feature columns as training)
  - trained model from models/<config-based-name>.npz (including scaler)
Evaluates:
  - average per-timestep log-likelihood 
  - basic per-trajectory stats 
"""
from pathlib import Path
import sys
import numpy as np

from .datasets import load_highd_folder, df_to_sequences, train_val_test_split
from .trainer import HDVTrainer
from .config import (
    TRAINING_CONFIG,
    FRAME_FEATURE_COLS,
    META_COLS,
    CONTINUOUS_FEATURES,
)
from .inference import infer_posterior_structured

def _slug(s):
    s = str(s).strip().lower()
    out = []
    for ch in s:
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        elif ch.isspace():
            out.append("-")
    s = "".join(out)
    while "--" in s:
        s = s.replace("--", "-")
    return s

def scale_obs_masked(obs, mean, std, scale_idx):
    """
    Scale only selected feature indices (continuous dims) using mean/std.
    Discrete dims (lane_pos, lc, *_exists) remain unchanged.
    """
    x = np.asarray(obs, dtype=np.float64).copy()
    m = np.asarray(mean, dtype=np.float64)
    s = np.asarray(std, dtype=np.float64)
    denom = s[scale_idx] + 1e-12
    x[:, scale_idx] = (x[:, scale_idx] - m[scale_idx]) / denom
    return x

def build_model_filename(cfg):
    """
    Mirror the training-side naming so evaluation picks the correct file by default.
    (Matches train_highd_dbn.build_model_filename logic.)
    """
    # In your current config, DBN_STATES can vary; but S/A are encoded in the checkpoint too.
    rec = getattr(cfg, "max_highd_recordings", None)
    rec_str = "all" if (rec is None) else str(int(rec))
    parts = [
        "dbn_highd",
        f"S{1}",      # DBN_STATES.driving_style is currently dummy in config; checkpoint is authoritative.
        f"A{3}",      # same note; used only to keep filename stable with your current codebase defaults.
        f"init-{_slug(getattr(cfg, 'cpd_init', 'unknown'))}",
        f"rec{rec_str}",
        f"seed{int(getattr(cfg, 'seed', 0))}",
    ]
    if hasattr(cfg, "bern_weight"):
        parts.append(f"bw-{_slug(getattr(cfg, 'bern_weight'))}")
    return "_".join(parts) + ".npz"


def main():
    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "highd"
    cache_path = project_root / "data" / "highd_all_with_meta.feather"

    model_dir = project_root / "models"
    # Default: same naming convention as training
    model_path = model_dir / build_model_filename(TRAINING_CONFIG)

    print(f"[evaluate_highd_dbn] Loading highD data from: {data_root}")
    print(f"[evaluate_highd_dbn] Loading model from: {model_path}")

    if not model_path.exists():
        print(f"[evaluate_highd_dbn] ERROR: model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load data and build sequences (same as in training)
    # ------------------------------------------------------------------
    try:
        df = load_highd_folder(
            data_root,
            cache_path=cache_path,
            force_rebuild=False,
            max_recordings=TRAINING_CONFIG.max_highd_recordings,
        )
    except Exception as e:
        print(f"[evaluate_highd_dbn] ERROR loading highD: {e}", file=sys.stderr)
        sys.exit(1)

    feature_cols = list(FRAME_FEATURE_COLS)
    meta_cols = list(META_COLS)

    # - lane_pos invalid/unknown -> -1
    # - other features NaN -> 0.0
    if "lane_pos" in feature_cols:
        df["lane_pos"] = df["lane_pos"].fillna(-1)
    fill_cols = [c for c in feature_cols if c != "lane_pos"]
    df[fill_cols] = df[fill_cols].fillna(0.0)

    sequences = df_to_sequences(df, feature_cols=feature_cols, meta_cols=meta_cols)

    train_seqs, val_seqs, test_seqs = train_val_test_split(
        sequences,
        train_frac=0.7,
        val_frac=0.1,
    )

    print(
        f"[evaluate_highd_dbn] Split sizes -> "
        f"Train: {len(train_seqs)}  "
        f"Val: {len(val_seqs)}  "
        f"Test: {len(test_seqs)}"
    )

    # ------------------------------------------------------------------
    # Load trained model
    # ------------------------------------------------------------------
    trainer = HDVTrainer.load(model_path)
    emissions = trainer.emissions
    scaler_mean = trainer.scaler_mean
    scaler_std = trainer.scaler_std
    if scaler_mean is None or scaler_std is None:
        raise RuntimeError(
            f"Loaded model from '{model_path}' missing scaler_mean/std. "
            f"Ensure the model was trained with feature scaling enabled."
        )
    # Indices of continuous dims (scale those only)
    scale_idx = np.array([i for i, n in enumerate(feature_cols) if n in CONTINUOUS_FEATURES], dtype=np.int64)
 
    

    # Build scaled test obs
    test_obs_seqs = []
    for seq in test_seqs:
        if isinstance(scaler_mean, dict) and isinstance(scaler_std, dict):
            if seq.meta is None or "meta_class" not in seq.meta:
                raise RuntimeError("Class-wise scaler loaded but seq.meta['meta_class'] is missing.")
            cls = str(seq.meta["meta_class"])
            if cls not in scaler_mean:
                raise RuntimeError(f"Class '{cls}' not found in saved class-wise scalers.")
            mean = scaler_mean[cls]
            std = scaler_std[cls]
        else:
            mean = scaler_mean
            std = scaler_std

        test_obs_seqs.append(scale_obs_masked(seq.obs, mean, std, scale_idx))

    # ------------------------------------------------------------------
    # Evaluate log-likelihood on the test set
    # ------------------------------------------------------------------
    total_test_loglik = 0.0
    per_traj_loglik = []
    total_test_T = 0

    gamma_all = []
    for i, obs in enumerate(test_obs_seqs):
        gamma, _, _, loglik = infer_posterior_structured(
            obs=obs,
            pi_s0=trainer.pi_s0,
            pi_a0_given_s0=trainer.pi_a0_given_s0,
            A_s=trainer.A_s,
            A_a=trainer.A_a,
            emissions=emissions,
        )
        gamma_all.append(gamma)
        total_test_loglik += loglik
        per_traj_loglik.append(loglik)
        total_test_T += int(obs.shape[0])

        if TRAINING_CONFIG.verbose >= 2:
            print(f"[evaluate_highd_dbn] traj {i:05d}: loglik={loglik:.3f}, T={obs.shape[0]}")

    avg_loglik_per_traj = total_test_loglik / max(len(test_obs_seqs), 1)
    avg_loglik_per_timestep = total_test_loglik / max(total_test_T, 1)

    print(f"[evaluate_highd_dbn] Total test log-likelihood: {total_test_loglik:.3f}")
    print(f"[evaluate_highd_dbn] Average per-trajectory log-likelihood: {avg_loglik_per_traj:.3f}")
    print(f"[evaluate_highd_dbn] Average per-timestep log-likelihood: {avg_loglik_per_timestep:.6f}")

    # --------------------------------------------------------------
    # Posterior diagnostics (test-time only)
    # --------------------------------------------------------------
    run_lengths, runlen_median = HDVTrainer._run_lengths_from_gamma_argmax(trainer, gamma_all)
    ent_all, ent_mean = HDVTrainer._posterior_entropy_from_gamma(trainer, gamma_all)

    print(
        f"[evaluate_highd_dbn] Posterior entropy (mean over traj): "
        f"{np.nanmean(ent_mean):.3f}"
    )
    print(
        f"[evaluate_highd_dbn] Median run-length (timesteps): "
        f"{np.nanmedian(runlen_median):.1f}"
    )
 
    if len(per_traj_loglik) > 0:
        p = np.percentile(per_traj_loglik, [0, 5, 25, 50, 75, 95, 100])
        print(
            "[evaluate_highd_dbn] Per-trajectory loglik percentiles "
            f"(min,5,25,50,75,95,max): {p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}, "
            f"{p[3]:.1f}, {p[4]:.1f}, {p[5]:.1f}, {p[6]:.1f}"
        )

    print("[evaluate_highd_dbn] Evaluation finished.")
    print(
        "[evaluate_highd_dbn] Outputs available for comparison:\n"
        "  - avg_loglik_per_timestep\n"
        "  - per_traj_loglik distribution\n"
        "  - posterior entropy statistics\n"
        "  - run-length statistics\n"
    )


if __name__ == "__main__":
    main()
