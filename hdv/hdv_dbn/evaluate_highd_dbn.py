"""Evaluation entry point for the HDV DBN on the highD test set."""

from pathlib import Path

from datasets import load_highd_folder, df_to_sequences, train_val_test_split
from trainer import HDVTrainer
from inference import infer_posterior, infer_viterbi_paths


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

    feature_cols = ["x", "y", "vx", "vy", "ax", "ay", "lane_id"]
    sequences = df_to_sequences(df, feature_cols)

    # IMPORTANT: use the same split fractions and seed as in training
    train_seqs, val_seqs, test_seqs = train_val_test_split(
        sequences, train_frac=0.8, val_frac=0.1, seed=0
    )

    print(
        f"[evaluate_highd_dbn] Split sizes -> "
        f"Train: {len(train_seqs)}  "
        f"Val: {len(val_seqs)}  "
        f"Test: {len(test_seqs)}"
    )

    test_obs_seqs = [seq.obs for seq in test_seqs]

    # ------------------------------------------------------------------
    # 3) Load trained model
    # ------------------------------------------------------------------
    trainer = HDVTrainer.load(model_path)
    pi_z = trainer.pi_z
    A_zz = trainer.A_zz
    emissions = trainer.emissions

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
            f"  Traj {i:03d} | vehicle_id={seq.vehicle_id} | "
            f"T={seq.T} | log p*(z, o)={log_p_star:.3f}"
        )
        # Here you could also store style_star/action_star somewhere
        # for plotting or further analysis.

    print("[evaluate_highd_dbn] Evaluation finished.")


if __name__ == "__main__":
    main()
