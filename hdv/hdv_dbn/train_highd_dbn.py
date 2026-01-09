"""
Entry point for training the HDV DBN / joint-HMM model on the highD dataset.

Pipeline
- Load highD CSVs from `data/highd/` (optionally using a cached feather file).
- Build per-vehicle observation sequences from selected feature columns.
- Split sequences into train/val/test.
- Fit a feature scaler (mean/std) on the training split only, then scale all splits.
- Train the model with EM (optionally evaluating validation log-likelihood each iteration).
- Save the trained parameters and the scaler to `models/dbn_highd.npz`.
"""

from pathlib import Path
import sys
import re
import numpy as np

from .datasets import (
    load_highd_folder,
    df_to_sequences,
    train_val_test_split,
    compute_feature_scaler_masked,  
    scale_sequences,
    compute_classwise_feature_scalers_masked,  
    scale_sequences_classwise,
)
from .trainer import HDVTrainer
from .config import TRAINING_CONFIG, BASELINE_FEATURE_COLS, META_COLS, CONTINUOUS_FEATURES, DBN_STATES

if TRAINING_CONFIG.use_wandb:
    import wandb
else:
    wandb = None

USE_CLASSWISE_SCALING = TRAINING_CONFIG.use_classwise_scaling

# -----------------------------
# Model naming helpers
# -----------------------------
def _slug(s):
    """Make a string safe for filenames."""
    s = str(s)
    s = s.strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9._-]+", "", s)
    s = re.sub(r"-{2,}", "-", s)
    return s


def build_model_filename(cfg):
    """
    Build a deterministic model filename based on TRAINING_CONFIG and DBN_STATES.

    Includes the key factors that define a training run identity so models do not overwrite.
    """
    S = len(DBN_STATES.driving_style)
    A = len(DBN_STATES.action)

    rec = cfg.max_highd_recordings
    rec_str = "all" if (rec is None) else str(int(rec))

    parts = [
        "dbn_highd",
        f"S{S}",
        f"A{A}",
        f"init-{_slug(getattr(cfg, 'cpd_init', 'unknown'))}",
        f"rec{rec_str}",
        f"seed{int(cfg.seed)}",
    ]

    # Optional: include bernoulli settings if present (keeps filename stable across these toggles)
    if hasattr(cfg, "exists_as_bernoulli"):
        parts.append(f"bern-{int(bool(cfg.exists_as_bernoulli))}")
    if hasattr(cfg, "bern_weight"):
        parts.append(f"bw-{_slug(cfg.bern_weight)}")

    return "_".join(parts) + ".npz"

def main():
    """Run the full training job."""
    try:
        project_root = Path(__file__).resolve().parents[1]
        data_root = project_root / "data" / "highd"
        model_dir = project_root / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"[train_highd_dbn] Loading highD data from: {data_root}")
        if not data_root.exists():
            raise FileNotFoundError(f"Data directory not found: {data_root}")
        
        cache_path = project_root / "data" / "highd_all_with_meta.feather"
        df = load_highd_folder(data_root, cache_path=cache_path, force_rebuild=False, max_recordings=TRAINING_CONFIG.max_highd_recordings)
    except Exception as e:
        print(f"[train_highd_dbn] ERROR during initialization: {e}", file=sys.stderr)
        sys.exit(1)

    # Observation features used by the emission model (baseline set)
    feature_cols = BASELINE_FEATURE_COLS
    # Meta features for analysis / scaling choices
    meta_cols = META_COLS

    scale_idx = [i for i, n in enumerate(feature_cols) if n in CONTINUOUS_FEATURES] # scale exactly the continuous features from config

    # Fill NaNs in features with 0.0 (e.g., missing front/rear vehicle info)
    if "lane_pos" in feature_cols:
        df["lane_pos"] = df["lane_pos"].fillna(-1)  # invalid/unknown stays -1
    fill_cols = [c for c in feature_cols if c != "lane_pos"]  
    df[fill_cols] = df[fill_cols].fillna(0.0)                 

    sequences = df_to_sequences(df, feature_cols=feature_cols, meta_cols=meta_cols)
    print(f"[train_highd_dbn] Total sequences (vehicles) loaded: {len(sequences)}")

    train_seqs, val_seqs, test_seqs = train_val_test_split(sequences, train_frac=0.7, val_frac=0.1)
    # -----------------------------
    # Choose scaling strategy
    # -----------------------------
    if USE_CLASSWISE_SCALING:
        scalers = compute_classwise_feature_scalers_masked(train_seqs, scale_idx=scale_idx, class_key="meta_class")  
        train_seqs_scaled = scale_sequences_classwise(train_seqs, scalers, class_key="meta_class")  
        val_seqs_scaled   = scale_sequences_classwise(val_seqs, scalers, class_key="meta_class")    
        test_seqs_scaled  = scale_sequences_classwise(test_seqs, scalers, class_key="meta_class")   
        scaler_to_store = scalers  # dict: {class -> (mean,std)}
    else:
        train_mean, train_std = compute_feature_scaler_masked(train_seqs, scale_idx=scale_idx)  
        train_seqs_scaled = scale_sequences(train_seqs, train_mean, train_std)
        val_seqs_scaled   = scale_sequences(val_seqs, train_mean, train_std)
        test_seqs_scaled  = scale_sequences(test_seqs, train_mean, train_std)
        scaler_to_store = (train_mean, train_std)  # tuple: (mean,std)

    print(
        f"[train_highd_dbn] Split sizes -> "
        f"Train: {len(train_seqs)}  "
        f"Val: {len(val_seqs)}  "
        f"Test: {len(test_seqs)}"
    )

    obs_dim = len(feature_cols)
    trainer = HDVTrainer(obs_names=feature_cols)
    # Store scalers for saving later (trainer.save will handle dict vs arrays after our patch below)
    if USE_CLASSWISE_SCALING:
        trainer.scaler_mean = {k: v[0] for k, v in scaler_to_store.items()}
        trainer.scaler_std  = {k: v[1] for k, v in scaler_to_store.items()}
    else:
        trainer.scaler_mean, trainer.scaler_std = scaler_to_store

    # Convert TrajectorySequence objects -> numpy observation sequences
    # Training uses SCALED sequences; we also keep RAW sequences for physical-unit semantic logging.
    train_obs_seqs = [seq.obs for seq in train_seqs_scaled]
    train_obs_seqs_raw = [seq.obs for seq in train_seqs]              # raw (unscaled)
    val_obs_seqs = [seq.obs for seq in val_seqs_scaled] if len(val_seqs_scaled) > 0 else None
    val_obs_seqs_raw = [seq.obs for seq in val_seqs] if len(val_seqs) > 0  else None

    wandb_run = None
    if TRAINING_CONFIG.use_wandb and wandb is not None:
        # Initialise a Weights & Biases run for logging training diagnostics.
        wandb_run = wandb.init(
            project=TRAINING_CONFIG.wandb_project,
            name=TRAINING_CONFIG.wandb_run_name,
            config={
                "obs_dim": obs_dim,
                "num_style": trainer.S,
                "num_action": trainer.A,
                "em_num_iters": TRAINING_CONFIG.em_num_iters,
                "seed": TRAINING_CONFIG.seed,
                "max_kmeans_samples": TRAINING_CONFIG.max_kmeans_samples,
                "max_highd_recordings": TRAINING_CONFIG.max_highd_recordings,
                "cpd_init": TRAINING_CONFIG.cpd_init,  
                "use_classwise_scaling": TRAINING_CONFIG.use_classwise_scaling, 
            },
        )

    try:
        # Run EM training. 
        history = trainer.em_train(
            train_obs_seqs=train_obs_seqs,
            val_obs_seqs=val_obs_seqs,
            wandb_run=wandb_run,
            train_obs_seqs_raw=train_obs_seqs_raw,
            val_obs_seqs_raw=val_obs_seqs_raw,
        )

        # -----------------------------
        # Save model with config-based name
        # -----------------------------
        model_filename = build_model_filename(TRAINING_CONFIG)  
        model_path = model_dir / model_filename 
        trainer.save(model_path)
        print(f"[train_highd_dbn] Model saved to: {model_path}") 
        print(f"[train_highd_dbn] Training finished.")
    except Exception as e:
        print(f"[train_highd_dbn] ERROR during training: {e}", file=sys.stderr)
        sys.exit(1)
    
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
