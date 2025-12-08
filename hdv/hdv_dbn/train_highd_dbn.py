"""Training entry point for the HDV DBN on the highD dataset."""

from pathlib import Path

from datasets import load_highd_folder, df_to_sequences, train_val_test_split
from trainer import HDVTrainer


def main(): 
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "highd"
    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train_highd_dbn] Loading highD data from: {data_root}")
    df = load_highd_folder(data_root)

    # Observation features used by the emission model
    feature_cols = ["x", "y", "vx", "vy", "ax", "ay", "lane_id"]

    sequences = df_to_sequences(df, feature_cols) # Convert the dataframe into per–vehicle sequences
    print(f"[train_highd_dbn] Total sequences (vehicles) loaded: {len(sequences)}")

    train_seqs, val_seqs, test_seqs = train_val_test_split(
        sequences, train_frac=0.7, val_frac=0.1, seed=0
    )

    print(
        f"[train_highd_dbn] Split sizes -> "
        f"Train: {len(train_seqs)}  "
        f"Val: {len(val_seqs)}  "
        f"Test: {len(test_seqs)}"
    )

    obs_dim = len(feature_cols)
    trainer = HDVTrainer(obs_dim=obs_dim)

    # Convert TrajectorySequence objects -> raw numpy observation sequences
    train_obs_seqs = [seq.obs for seq in train_seqs]
    val_obs_seqs = [seq.obs for seq in val_seqs] if len(val_seqs) > 0 else None

    history = trainer.em_train(
            train_obs_seqs=train_obs_seqs,
            val_obs_seqs=val_obs_seqs,
            num_iters=100,
            tol=1e-3,
        )

    model_path = model_dir / "dbn_highd.npz"
    trainer.save(model_path)

    print(f"[train_highd_dbn] Training finished. Model saved to: {model_path}")


if __name__ == "__main__":
    main()
