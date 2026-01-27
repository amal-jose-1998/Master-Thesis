"""
Entry point for training the HDV DBN / joint-HMM model on the highD dataset.

Pipeline
- Load highD CSVs from `data/highd/` (optionally using a cached feather file).
- Build per-vehicle FRAME sequences, then windowize into per-window timesteps.
- Split sequences into train/val/test.
- Fit a feature scaler (mean/std) on the training split only, then scale all splits.
- Train the model with EM (optionally evaluating validation log-likelihood each iteration).
- Save checkpoints to `models/<experiment>_S{S}_A{A}_{emission_model}/ckpt_iterXXXX.npz` and final to `final.npz`.
"""

from pathlib import Path
import sys
import re
import numpy as np
import json
from dataclasses import asdict

from .datasets import (
    load_highd_folder,
    df_to_sequences,
    train_val_test_split,
    compute_feature_scaler,  
    scale_sequences,
    compute_classwise_feature_scalers,  
    scale_sequences_classwise,
    load_or_build_windowized,
)
from .trainer import HDVTrainer
from .config import (
    TRAINING_CONFIG, FRAME_FEATURE_COLS, META_COLS,
    WINDOW_FEATURE_COLS, CONTINUOUS_FEATURES, WINDOW_CONFIG, DBN_STATES
)

if TRAINING_CONFIG.use_wandb:
    import wandb
else:
    wandb = None

USE_CLASSWISE_SCALING = TRAINING_CONFIG.use_classwise_scaling

# -----------------------------
# W&B table builders 
# -----------------------------
def _scaler_rows_global(feature_cols, scale_idx, mean_vec, std_vec):
    """Rows for W&B table: [feature, mean, std]."""
    return [[feature_cols[j], float(mean_vec[j]), float(std_vec[j])] for j in scale_idx]


def _scaler_rows_classwise(feature_cols, scale_idx, scalers):
    """Rows for W&B table: [class, feature, mean, std]."""
    rows = []
    for cls, (mean_vec, std_vec) in scalers.items():
        for j in scale_idx:
            rows.append([str(cls), feature_cols[j], float(mean_vec[j]), float(std_vec[j])])
    return rows


def _post_scale_rows(feature_cols, scale_idx, scaled_seqs, max_seqs=500):
    """Rows for W&B table: [feature, scaled_mean, scaled_std, n_finite] (finite-only)."""
    if not scaled_seqs:
        return []
    
    # Subsample sequences for speed (evenly spaced)
    if max_seqs is not None and len(scaled_seqs) > max_seqs and max_seqs > 0:
        idx = np.linspace(0, len(scaled_seqs) - 1, num=max_seqs, dtype=np.int64)
        seqs = [scaled_seqs[i] for i in idx]
    else:
        seqs = scaled_seqs

    scale_idx = np.asarray(scale_idx, dtype=np.int64)

    # Accumulate per-feature: sum, sumsq, count over finite entries only
    s = np.zeros(scale_idx.size, dtype=np.float64)
    ssq = np.zeros(scale_idx.size, dtype=np.float64)
    n = np.zeros(scale_idx.size, dtype=np.int64)

    for seq in seqs:
        X = np.asarray(seq.obs)[:, scale_idx]   # (T, K)
        X = X.astype(np.float64, copy=False)  # avoid copies if already float64
        finite = np.isfinite(X)
        s   += np.sum(X,   axis=0, where=finite)
        ssq += np.sum(X*X, axis=0, where=finite)
        n   += np.sum(finite, axis=0)

    rows = []
    for k, j in enumerate(scale_idx):
        if n[k] == 0:
            continue
        mean = s[k] / n[k]
        var = ssq[k] / n[k] - mean * mean          # E[x^2] - (E[x])^2
        var = max(float(var), 0.0)                 # guard tiny negative due to rounding
        std = float(np.sqrt(var))
        rows.append([feature_cols[int(j)], float(mean), std, int(n[k])])

    return rows


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

def _veh_key(seq):
    return (getattr(seq, "recording_id", None), getattr(seq, "vehicle_id", None))

def _seq_key(seq):
    """Stable key for a vehicle sequence (split is vehicle-level)."""
    rec_id = getattr(seq, "recording_id", None)
    veh_id = getattr(seq, "vehicle_id", None)
    return f"{rec_id}:{veh_id}"


def save_split_json(path, train_seqs, val_seqs, test_seqs, seed, train_frac, val_frac):
    payload = {
        "seed": int(seed),
        "train_frac": float(train_frac),
        "val_frac": float(val_frac),
        "test_frac": float(1.0 - train_frac - val_frac),

        "W": int(WINDOW_CONFIG.W),
        "stride": int(WINDOW_CONFIG.stride),
        "max_highd_recordings": int(getattr(TRAINING_CONFIG, "max_highd_recordings", -1)),
        "split_level": "vehicle",  

        "keys": {
            "train": [_seq_key(s) for s in train_seqs],
            "val":   [_seq_key(s) for s in val_seqs],
            "test":  [_seq_key(s) for s in test_seqs],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def build_model_filename(cfg, wandb_run=None):
    """
    Robust model filename:
      <experiment-name>_S{S}_A{A}.npz
    """
    # experiment identity
    if wandb_run is not None and getattr(wandb_run, "name", None):
        exp_name = wandb_run.name
    elif getattr(cfg, "wandb_run_name", None):
        exp_name = cfg.wandb_run_name
    else:
        exp_name = "unnamed_experiment"

    # model structure identity
    S = len(DBN_STATES.driving_style)
    A = len(DBN_STATES.action)

    return f"{_slug(exp_name)}_S{S}_A{A}.npz"

def main():
    """Run the full training job."""
    # -----------------------------
    # Initialization: paths, data loading
    # -----------------------------
    try:
        project_root = Path(__file__).resolve().parents[1]
        data_root = project_root / "data" / "highd"
        model_dir = project_root / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"[train_highd_dbn] Loading highD data from: {data_root}")
        if not data_root.exists():
            raise FileNotFoundError(f"Data directory not found: {data_root}")
        
        df = load_highd_folder(data_root, cache_path=None, force_rebuild=False, max_recordings=TRAINING_CONFIG.max_highd_recordings)
    except Exception as e:
        print(f"[train_highd_dbn] ERROR during initialization: {e}", file=sys.stderr)
        sys.exit(1)

    # -----------------------------
    # Frame -> window pipeline -> train/val/test split
    # -----------------------------
    cache_dir = data_root / "cache"
    # trajectory windowing with caching. returns a list of TrajectorySequence objects
    win_sequences = load_or_build_windowized(df, cache_dir=cache_dir, W=int(WINDOW_CONFIG.W), 
                                             stride=int(WINDOW_CONFIG.stride), force_rebuild=False) 
    print(f"[train_highd_dbn] Total WINDOW sequences produced: {len(win_sequences)}")

    # ensure the windowized list is vehicle-unique (i.e., not one element per window)
    keys_all = [_veh_key(s) for s in win_sequences]
    assert len(keys_all) == len(set(keys_all)), "windowized output has duplicate (recording_id,vehicle_id) => likely window-level list!"
    assert all(k[0] is not None and k[1] is not None for k in keys_all), "Missing recording_id/vehicle_id in sequences"

    # Split by sequence (vehicle) to avoid leakage
    train_seqs, val_seqs, test_seqs = train_val_test_split(win_sequences, train_frac=0.7, val_frac=0.1, seed=TRAINING_CONFIG.seed)

    # ensure no leakage across splits
    train_keys = set(_veh_key(s) for s in train_seqs)
    val_keys   = set(_veh_key(s) for s in val_seqs)
    test_keys  = set(_veh_key(s) for s in test_seqs)

    assert train_keys.isdisjoint(val_keys)
    assert train_keys.isdisjoint(test_keys)
    assert val_keys.isdisjoint(test_keys)
    
    # -----------------------------
    # Choose scaling strategy
    # -----------------------------
    scaler_table_rows = None
    scalecheck_table_rows = None
    feature_cols = list(WINDOW_FEATURE_COLS)
    scale_idx = [i for i, n in enumerate(feature_cols) if n in CONTINUOUS_FEATURES] # indices of features to scale (only continuous ones)
    if USE_CLASSWISE_SCALING:
        # Compute class-wise scalers on training split. ie., one (mean,std) per vehicle class for each continuous feature.
        scalers = compute_classwise_feature_scalers(train_seqs, scale_idx=scale_idx, class_key="meta_class")  # dict: {class -> (mean_vec,std_vec)}
        train_seqs_scaled = scale_sequences_classwise(train_seqs, scalers, class_key="meta_class")  # scaled training sequences (z scorings)
        val_seqs_scaled   = scale_sequences_classwise(val_seqs, scalers, class_key="meta_class")    # scaled validation sequences
        test_seqs_scaled  = scale_sequences_classwise(test_seqs, scalers, class_key="meta_class")   # scaled test sequences
        scaler_to_store = scalers    
        
        scaler_table_rows = _scaler_rows_classwise(feature_cols, scale_idx, scalers)
        scalecheck_table_rows = _post_scale_rows(feature_cols, scale_idx, train_seqs_scaled)
    else:
        train_mean, train_std = compute_feature_scaler(train_seqs, scale_idx=scale_idx)  
        train_seqs_scaled = scale_sequences(train_seqs, train_mean, train_std)
        val_seqs_scaled   = scale_sequences(val_seqs, train_mean, train_std)
        test_seqs_scaled  = scale_sequences(test_seqs, train_mean, train_std)
        scaler_to_store = (train_mean, train_std)  # tuple: (mean,std)
        
        scaler_table_rows = _scaler_rows_global(feature_cols, scale_idx, train_mean, train_std)
        scalecheck_table_rows = _post_scale_rows(feature_cols, scale_idx, train_seqs_scaled)
    print(
        f"[train_highd_dbn] Split sizes -> "
        f"Train: {len(train_seqs)}  "
        f"Val: {len(val_seqs)}  "
        f"Test: {len(test_seqs)}"
    )

    obs_dim = len(feature_cols)
    trainer = HDVTrainer(obs_names=feature_cols)

    # Set scaler in trainer for saving alongside model. This is used during inference.
    if USE_CLASSWISE_SCALING:
        trainer.scaler_mean = {k: v[0] for k, v in scaler_to_store.items()}
        trainer.scaler_std  = {k: v[1] for k, v in scaler_to_store.items()}
    else:
        trainer.scaler_mean, trainer.scaler_std = scaler_to_store

    # Convert TrajectorySequence objects -> numpy observation sequences
    # Training uses SCALED sequences; keep RAW for physical-unit semantics.
    train_obs_seqs = [seq.obs for seq in train_seqs_scaled]
    train_obs_seqs_raw = [seq.obs for seq in train_seqs]              # raw (unscaled)
    val_obs_seqs = [seq.obs for seq in val_seqs_scaled] if len(val_seqs_scaled) > 0 else None

    # -----------------------------
    # W&B init + log scaler tables
    # -----------------------------
    wandb_run = None
    if TRAINING_CONFIG.use_wandb and wandb is not None:
        # Initialise a Weights & Biases run for logging training diagnostics.
        def _wandb_safe(obj):
            """Convert objects to W&B-friendly JSON-ish values."""
            if obj is None:
                return None
            if isinstance(obj, (int, float, bool, str)):
                return obj
            if isinstance(obj, (list, tuple)):
                return [_wandb_safe(x) for x in obj] 
            if isinstance(obj, dict):
                return {str(k): _wandb_safe(v) for k, v in obj.items()}
            # fallback: string representation
            return str(obj)

        wandb_run = wandb.init(
            project=TRAINING_CONFIG.wandb_project,
            name=TRAINING_CONFIG.wandb_run_name,
            config=_wandb_safe({
                # full training config (dataclass)
                **asdict(TRAINING_CONFIG),

                # model identity / structure
                "num_style": int(trainer.S),
                "num_action": int(trainer.A),
                "style_names": list(DBN_STATES.driving_style),
                "action_names": list(DBN_STATES.action),

                # data / features used
                "obs_dim": int(obs_dim),
                "frame_feature_cols": list(FRAME_FEATURE_COLS),
                "obs_feature_cols": list(WINDOW_FEATURE_COLS),
                "meta_cols": list(META_COLS),
            }),
        )

        # Log scaler stats to W&B once 
        #try:
        #    if scaler_table_rows is not None:
        #        if USE_CLASSWISE_SCALING:
        #            t = wandb.Table(columns=["class", "feature", "mean", "std"], data=scaler_table_rows)
        #            wandb_run.log({"scaler/classwise_mean_std": t})
        #        else:
        #            t = wandb.Table(columns=["feature", "mean", "std"], data=scaler_table_rows)
        #            wandb_run.log({"scaler/global_mean_std": t})

        #    if scalecheck_table_rows is not None:
        #        t2 = wandb.Table(columns=["feature", "scaled_mean", "scaled_std", "n_finite"], data=scalecheck_table_rows)
        #        wandb_run.log({"scaler/train_scaled_sanitycheck": t2})
        #except Exception as e:
        #    print(f"[train_highd_dbn] WARNING: failed to log scaler tables to W&B: {e}", file=sys.stderr)

    # -----------------------------
    # Train + save
    # -----------------------------
    try:
        # -----------------------------
        # Experiment folder for checkpoints
        # -----------------------------
        S = len(DBN_STATES.driving_style)
        A = len(DBN_STATES.action)
        em_mode = str(getattr(TRAINING_CONFIG, "emission_model", "poe")).lower().strip()

        if wandb_run is not None and getattr(wandb_run, "name", None):
            exp_name = wandb_run.name
        elif getattr(TRAINING_CONFIG, "wandb_run_name", None):
            exp_name = TRAINING_CONFIG.wandb_run_name
        else:
            exp_name = "unnamed_experiment"

        exp_folder = model_dir / f"{_slug(exp_name)}_S{S}_A{A}_{em_mode}"
        exp_folder.mkdir(parents=True, exist_ok=True)
        print(f"[train_highd_dbn] Experiment folder: {exp_folder}")

        # Save split (vehicle-level keys) for reproducible evaluation
        split_path = exp_folder / "split.json"
        save_split_json(split_path, train_seqs=train_seqs, val_seqs=val_seqs, test_seqs=test_seqs, seed=TRAINING_CONFIG.seed, train_frac=0.7, val_frac=0.1)
        print(f"[train_highd_dbn] Saved split to: {split_path}")

        # Run EM training. 
        history = trainer.em_train(
            train_obs_seqs=train_obs_seqs,
            val_obs_seqs=val_obs_seqs,
            wandb_run=wandb_run,
            train_obs_seqs_raw=train_obs_seqs_raw,
            checkpoint_dir=exp_folder,
            checkpoint_every=5,
        )

        # -----------------------------
        # Save FINAL model (in the experiment folder)
        # -----------------------------
        final_path = exp_folder / "final.npz"
        trainer.save(final_path)
        print(f"[train_highd_dbn] Model saved to: {final_path}")
        print(f"[train_highd_dbn] Training finished.")
    except Exception as e:
        print(f"[train_highd_dbn] ERROR during training: {e}", file=sys.stderr)
        sys.exit(1)
    
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
