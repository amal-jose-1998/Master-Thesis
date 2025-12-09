"""Evaluation entry point for the HDV DBN on the highD test set."""
from pathlib import Path

from .datasets import load_highd_folder, df_to_sequences, train_val_test_split
from .trainer import HDVTrainer
from .inference import infer_posterior, infer_viterbi_paths

def scale_obs(obs, mean, std):
    return (obs - mean) / std

def main():
    # ------------------------------------------------------------------
    # 1) Paths
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "highd"
    model_dir = project_root / "models"
    model_path = model_dir / "dbn_highd.npz"

    print(f"[evaluate_highd_dbn] Loading highD data from: {data_root}")
    print(f"[evaluate_highd_dbn] Loading model from: {model_path}")

    # ------------------------------------------------------------------
    # 2) Load data and build sequences (same as in training)
    # ------------------------------------------------------------------
    df = load_highd_folder(data_root)

    feature_cols = ["x", "y", "vx", "vy", "ax", "ay"]
    sequences = df_to_sequences(df, feature_cols)

    train_seqs, val_seqs, test_seqs = train_val_test_split(sequences, train_frac=0.7, val_frac=0.1, seed=123)

    print(
        f"[evaluate_highd_dbn] Split sizes -> "
        f"Train: {len(train_seqs)}  "
        f"Val: {len(val_seqs)}  "
        f"Test: {len(test_seqs)}"
    )

    # ------------------------------------------------------------------
    # 3) Load trained model
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
    
    test_obs_seqs = [scale_obs(seq.obs, scaler_mean, scaler_std) for seq in test_seqs]

    # ------------------------------------------------------------------
    # 4) Evaluate log-likelihood on the test set
    # ------------------------------------------------------------------
    total_test_loglik = 0.0
    per_traj_loglik = []

    for i, obs in enumerate(test_obs_seqs):
        gamma, gamma_style, gamma_action, loglik = infer_posterior(
            obs=obs,
            pi_z=pi_z,
            A_zz=A_zz,
            emissions=emissions,
        )
        total_test_loglik += loglik
        per_traj_loglik.append(loglik)

    avg_loglik = total_test_loglik / len(test_obs_seqs)
    print(f"[evaluate_highd_dbn] Total test log-likelihood: {total_test_loglik:.3f}")
    print(f"[evaluate_highd_dbn] Average per-trajectory log-likelihood: {avg_loglik:.3f}")

    # ------------------------------------------------------------------
    # 5) Optional: Viterbi decoding on the test set
    # ------------------------------------------------------------------
    print("[evaluate_highd_dbn] Running Viterbi decoding on test trajectories...")
    for i, (seq, obs) in enumerate(zip(test_seqs, test_obs_seqs)):
        z_star, style_star, action_star, log_p_star = infer_viterbi_paths(
            obs=obs,
            pi_z=pi_z,
            A_zz=A_zz,
            emissions=emissions,
        )
        print(
            f"  Traj {i:03d} | rec={seq.recording_id} | vehicle_id={seq.vehicle_id} | "
            f"T={seq.T} | log p*(z, o)={log_p_star:.3f}"
        )
        # Here you could also store style_star/action_star somewhere
        # for plotting or further analysis.

    print("[evaluate_highd_dbn] Evaluation finished.")


if __name__ == "__main__":
    main()
