"""Training entry point for the HDV DBN on the highD dataset."""

from pathlib import Path
import sys

from .datasets import load_highd_folder, df_to_sequences, train_val_test_split, compute_feature_scaler, scale_sequences
from .trainer import HDVTrainer
from .config import TRAINING_CONFIG

def main():
    try:
        project_root = Path(__file__).resolve().parents[1]
        data_root = project_root / "data" / "highd"
        model_dir = project_root / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"[train_highd_dbn] Loading highD data from: {data_root}")
        if not data_root.exists():
            raise FileNotFoundError(f"Data directory not found: {data_root}")
        
        cache_path = project_root / "data" / "highd_all.feather"
        df = load_highd_folder(data_root, cache_path=cache_path, force_rebuild=False, max_recordings=TRAINING_CONFIG.max_highd_recordings)
    except Exception as e:
        print(f"[train_highd_dbn] ERROR during initialization: {e}", file=sys.stderr)
        sys.exit(1)

    # Observation features used by the emission model
    feature_cols = ["x", "y", "vx", "vy", "ax", "ay"]

    sequences = df_to_sequences(df, feature_cols) # Convert the dataframe into per–vehicle sequences
    print(f"[train_highd_dbn] Total sequences (vehicles) loaded: {len(sequences)}")

    train_seqs, val_seqs, test_seqs = train_val_test_split(sequences, train_frac=0.7, val_frac=0.1)
    train_mean, train_std = compute_feature_scaler(train_seqs) # compute scaler on training set
    # scale all splits
    train_seqs_scaled = scale_sequences(train_seqs, train_mean, train_std)
    val_seqs_scaled   = scale_sequences(val_seqs,   train_mean, train_std)
    test_seqs_scaled  = scale_sequences(test_seqs,  train_mean, train_std)
    print(
        f"[train_highd_dbn] Split sizes -> "
        f"Train: {len(train_seqs)}  "
        f"Val: {len(val_seqs)}  "
        f"Test: {len(test_seqs)}"
    )

    obs_dim = len(feature_cols)
    trainer = HDVTrainer(obs_dim=obs_dim)
    trainer.scaler_mean = train_mean
    trainer.scaler_std = train_std

    # Convert TrajectorySequence objects -> raw numpy observation sequences
    train_obs_seqs = [seq.obs for seq in train_seqs_scaled]
    val_obs_seqs = [seq.obs for seq in val_seqs_scaled] if len(val_seqs) > 0 else None

    try:
        history = trainer.em_train(
                train_obs_seqs=train_obs_seqs,
                val_obs_seqs=val_obs_seqs,
            )

        model_path = model_dir / "dbn_highd.npz"
        trainer.save(model_path)

        print(f"[train_highd_dbn] Training finished.")
    except Exception as e:
        print(f"[train_highd_dbn] ERROR during training: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
