"""
Evaluation entry point for the HDV DBN on the highD test set.
Loads:
  - highD sequences (same feature columns as training)
  - trained model from models/dbn_highd.npz (including scaler)
Evaluates:
  - total test log-likelihood
  - average per-trajectory log-likelihood
"""
from pathlib import Path
import sys
import numpy as np

from .datasets import load_highd_folder, df_to_sequences, train_val_test_split
from .trainer import HDVTrainer
from .config import TRAINING_CONFIG
from .inference import infer_posterior

def scale_obs(obs, mean, std):
    """Standardise observations feature-wise using training-set mean/std."""
    return (obs - mean) / (std + 1e-12)

def main():
    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "highd"
    cache_path = project_root / "data" / "highd_all.feather"

    model_dir = project_root / "models"
    model_path = model_dir / "dbn_highd.npz"

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

    feature_cols = [
        "y", "vx", "vy", "ax", "ay", "lane_id",
        "front_exists", "front_dx", "front_dvx",
        "rear_exists",  "rear_dx",  "rear_dvx",
    ]

    meta_cols = [
        "meta_class",
        "meta_drivingDirection"
    ]

    df[feature_cols] = df[feature_cols].fillna(0.0) # Fill NaNs in features with 0.0

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
    pi_z = trainer.pi_z
    A_zz = trainer.A_zz
    emissions = trainer.emissions
    scaler_mean = trainer.scaler_mean
    scaler_std = trainer.scaler_std
    if scaler_mean is None or scaler_std is None:
        raise RuntimeError(
            f"Loaded model from '{model_path}' missing scaler_mean/std. "
            f"Ensure the model was trained with feature scaling enabled."
        )
    
    def scale_obs(obs, mean, std):
        return (obs - mean) / (std + 1e-12)

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

        test_obs_seqs.append(scale_obs(seq.obs, mean, std))

    # ------------------------------------------------------------------
    # Evaluate log-likelihood on the test set
    # ------------------------------------------------------------------
    total_test_loglik = 0.0
    per_traj_loglik = []

    for i, obs in enumerate(test_obs_seqs):
        _, _, _, loglik = infer_posterior(
            obs=obs,
            pi_z=pi_z,
            A_zz=A_zz,
            emissions=emissions,
        )
        total_test_loglik += loglik
        per_traj_loglik.append(loglik)

        if TRAINING_CONFIG.verbose >= 2:
            print(f"[evaluate_highd_dbn] traj {i:05d}: loglik={loglik:.3f}, T={obs.shape[0]}")

    avg_loglik = total_test_loglik / max(len(test_obs_seqs), 1)

    print(f"[evaluate_highd_dbn] Total test log-likelihood: {total_test_loglik:.3f}")
    print(f"[evaluate_highd_dbn] Average per-trajectory log-likelihood: {avg_loglik:.3f}")

    if len(per_traj_loglik) > 0:
        p = np.percentile(per_traj_loglik, [0, 5, 25, 50, 75, 95, 100])
        print(
            "[evaluate_highd_dbn] Per-trajectory loglik percentiles "
            f"(min,5,25,50,75,95,max): {p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}, "
            f"{p[3]:.1f}, {p[4]:.1f}, {p[5]:.1f}, {p[6]:.1f}"
        )

    print("[evaluate_highd_dbn] Evaluation finished.")


if __name__ == "__main__":
    main()
