"""
Training entry point for the HDV DBN on the highD dataset.

Run from the project root (the folder that contains 'hdv/'):
    python -m hdv.hdv_dbn.train_highd_dbn
"""

from pathlib import Path

from .datasets import load_highd_folder, df_to_sequences, train_val_test_split

# These imports assume you have something like DBNConfig, HDVDBN, train_em
# defined already. If the names differ in your files, just change them here.
from .config import DBNConfig          # adjust if your config class is named differently
from .model import HDVDBN              # adjust if your model class is named differently
from .trainer import train_em          # adjust to your EM training function


def main():
    # ------------------------------------------------------------------
    # 1) Paths
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "highd"
    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading highD data from: {data_root}")

    # ------------------------------------------------------------------
    # 2) Load and convert to sequences
    # ------------------------------------------------------------------
    df = load_highd_folder(data_root)

    # Choose the observation features for the DBN
    feature_cols = ["x", "y", "vx", "vy", "ax", "ay", "lane_id"]

    sequences = df_to_sequences(df, feature_cols)

    print(f"Total sequences (vehicles) loaded: {len(sequences)}")

    train_seqs, val_seqs, test_seqs = train_val_test_split(
        sequences, train_frac=0.8, val_frac=0.1, seed=0
    )

    print(f"Train: {len(train_seqs)}  Val: {len(val_seqs)}  Test: {len(test_seqs)}")

    # ------------------------------------------------------------------
    # 3) Build model and config
    # ------------------------------------------------------------------
    # Fill in numbers based on your DBN structure
    cfg = DBNConfig(
        num_style_states=3,
        num_action_states=6,
        obs_feature_names=feature_cols,
        # add other hyperparameters used in your config
    )

    model = HDVDBN(cfg)

    # ------------------------------------------------------------------
    # 4) Train using EM
    # ------------------------------------------------------------------
    # Assumes train_em returns some history dict with log-likelihood curves.
    # If your trainer has a different signature, adjust this call only.
    history = train_em(
        model=model,
        train_sequences=train_seqs,
        val_sequences=val_seqs,
        num_epochs=30,
        tol=1e-3,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # 5) Save trained model
    # ------------------------------------------------------------------
    # Implement `save` in your HDVDBN class (e.g. saving params to npz).
    model_path = model_dir / "dbn_highd.npz"
    model.save(model_path)

    print(f"Training finished. Model saved to: {model_path}")

    # Optionally print last log-likelihoods
    if history is not None and "train_loglik" in history:
        print("Final train log-likelihood:", history["train_loglik"][-1])
    if history is not None and "val_loglik" in history:
        print("Final val log-likelihood:", history["val_loglik"][-1])


if __name__ == "__main__":
    main()
