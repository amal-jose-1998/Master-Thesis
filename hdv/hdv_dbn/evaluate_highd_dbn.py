"""Evaluate ONE experiment folder on its saved TEST split (vehicle-level) and SAVE results"""

import json
import sys
from pathlib import Path

from .datasets import load_highd_folder, load_or_build_windowized
from .config import TRAINING_CONFIG, WINDOW_FEATURE_COLS, WINDOW_CONFIG, DBN_STATES
from .utils.eval_core import evaluate_checkpoint, evaluate_online_predictive_ll, evaluate_anticipatory_predictive_ll, seq_key, evaluate_iid_baseline, evaluate_frozen_belief_online_ll


# -----------------------------
# IO helpers 
# -----------------------------
def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

# -----------------------------
# split + test set loading
# -----------------------------
def load_split_json(exp_dir):
    exp_dir = Path(exp_dir)
    p = exp_dir / "split.json"
    print(f"[evaluate_highd_dbn] Loading split -> {p}")
    if not p.exists():
        raise FileNotFoundError(f"Missing split.json: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if "keys" not in payload or "test" not in payload["keys"] or "train" not in payload["keys"]:
        raise ValueError(f"split.json format unexpected: {p}")
    
    keys = payload.get("keys", {})
    test_keys = set(keys["test"])
    train_keys = set(keys["train"])

    print(
        f"[evaluate_highd_dbn] split.json loaded: "
        f"train={len(train_keys)} vehicles, test={len(test_keys)} vehicles, "
        f"W={payload.get('W')}, stride={payload.get('stride')}"
    )
    return payload, train_keys, test_keys


def load_test_sequences_from_experiment_split(exp_dir, data_root):
    """
    Loads windowized sequences matching the experiment split.json and returns:
       test_seqs, feature_cols, split_payload, test_keys
    """
    exp_dir = Path(exp_dir)
    data_root = Path(data_root)

    split_payload, train_keys, test_keys = load_split_json(exp_dir)

    # Use W/stride saved with experiment (single source of truth)
    W = int(split_payload.get("W", int(WINDOW_CONFIG.W)))
    stride = int(split_payload.get("stride", int(WINDOW_CONFIG.stride)))

    print(f"[evaluate_highd_dbn] Loading highD df from: {data_root}")
    df = load_highd_folder(
        data_root,
        cache_path=None,
        force_rebuild=False,
        max_recordings=getattr(TRAINING_CONFIG, "max_highd_recordings", None),
    )

    cache_dir = data_root / "cache"
    print(f"[evaluate_highd_dbn] Building/loading window cache: cache_dir={cache_dir}  W={W} stride={stride}")
    all_seqs = load_or_build_windowized(
        df,
        cache_dir=cache_dir,
        W=W,
        stride=stride,
        force_rebuild=False,
    )

    print(f"[evaluate_highd_dbn] Total windowized sequences in cache: {len(all_seqs)}")
    train_seqs = [s for s in all_seqs if seq_key(s) in train_keys]
    test_seqs  = [s for s in all_seqs if seq_key(s) in test_keys]
    print(f"[evaluate_highd_dbn] Matched train sequences: {len(train_seqs)}/{len(train_keys)}")
    print(f"[evaluate_highd_dbn] Matched test  sequences: {len(test_seqs)}/{len(test_keys)}")

    found_train = {seq_key(s) for s in train_seqs}
    found_test  = {seq_key(s) for s in test_seqs}

    missing_train = train_keys - found_train
    missing_test  = test_keys - found_test

    if missing_train or missing_test:
        msg = []
        if missing_train:
            msg.append(f"Missing {len(missing_train)} train vehicles. Example: {list(sorted(missing_train))[:10]}")
        if missing_test:
            msg.append(f"Missing {len(missing_test)} test vehicles. Example: {list(sorted(missing_test))[:10]}")
        raise RuntimeError(
            "[evaluate_highd_dbn] Split vehicles missing from cache. "
            "Likely W/stride/max_recordings mismatch.\n" + "\n".join(msg)
        )

    return train_seqs, test_seqs, list(WINDOW_FEATURE_COLS), split_payload, train_keys, test_keys

# -----------------------------
# main evaluation wrapper
# -----------------------------
def evaluate_experiment(exp_dir, checkpoint_name="final.npz", data_root=None):
    exp_dir = Path(exp_dir)
    project_root = Path(__file__).resolve().parents[1]
    data_root = Path(data_root) if data_root is not None else (project_root / "data" / "highd")
    ckpt_path = exp_dir / checkpoint_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"[evaluate_highd_dbn] Using exp_dir={exp_dir}")
    print(f"[evaluate_highd_dbn] Using checkpoint={ckpt_path}")

    train_seqs, test_seqs, feature_cols, split_payload, train_keys, test_keys = load_test_sequences_from_experiment_split(
        exp_dir=exp_dir,
        data_root=data_root,
    )

    print(f"[evaluate_highd_dbn] Starting checkpoint evaluation...")
    metrics = evaluate_checkpoint(
        model_path=ckpt_path,
        test_seqs=test_seqs,
        feature_cols=feature_cols,
        out_dir=exp_dir,
        save_heatmaps=True,
    )

    print(f"[evaluate_highd_dbn] Starting ONLINE-LL (filtering-only) evaluation...")
    online_ll = evaluate_online_predictive_ll(
        model_path=ckpt_path,
        test_seqs=test_seqs,
        feature_cols=feature_cols,
        out_dir=exp_dir,
        save_plot=True,
    )
    print(f"[evaluate_highd_dbn] Finished ONLINE-LL evaluation.")

    print(f"[evaluate_highd_dbn] Starting ANTICIPATORY-LL evaluation...")
    anticipatory_ll = evaluate_anticipatory_predictive_ll(
        model_path=ckpt_path,
        test_seqs=test_seqs,
        feature_cols=feature_cols,
        H=10, # ≈ 4.0 s forecast horizon
        t_warmup=5, # ≈ 2.0 s of observed past
        out_dir=exp_dir,
        save_plot=True,
    )
    print(f"[evaluate_highd_dbn] Finished ANTICIPATORY-LL evaluation.")

    print(f"[evaluate_highd_dbn] Starting IID baseline evaluation (train-fit, test-score)...")
    iid_baseline = evaluate_iid_baseline(
        model_path=ckpt_path,
        train_seqs=train_seqs,
        test_seqs=test_seqs,
        feature_cols=feature_cols,
        out_dir=exp_dir / "eval_baselines" / "iid",
        save_plot=True,
    )

    print(f"[evaluate_highd_dbn] Starting frozen-belief baseline evaluation (no dynamics)...")
    frozen_baseline = evaluate_frozen_belief_online_ll(
        model_path=ckpt_path,
        test_seqs=test_seqs,
        feature_cols=feature_cols,
        out_dir=exp_dir / "eval_baselines" / "frozen_belief",
        save_plot=True,
    )


    print(f"[evaluate_highd_dbn] Finished checkpoint evaluation.")

    meta = dict(
        experiment_dir=str(exp_dir),
        checkpoint_name=str(checkpoint_name),
        checkpoint_path=str(ckpt_path),
        n_test_vehicles=int(len(test_keys)),
        W=int(split_payload.get("W", int(WINDOW_CONFIG.W))),
        stride=int(split_payload.get("stride", int(WINDOW_CONFIG.stride))),
        S=int(len(DBN_STATES.driving_style)),
        A=int(len(DBN_STATES.action)),
        emission_model=str(getattr(TRAINING_CONFIG, "emission_model", "unknown")),
        lc_weight_mode=str(getattr(TRAINING_CONFIG, "lc_weight_mode", "unknown")),
        learn_pi0=bool(getattr(TRAINING_CONFIG, "learn_pi0", False)),
        cpd_init=str(getattr(TRAINING_CONFIG, "cpd_init", "unknown")),
        bernoulli_ll_enabled=not bool(getattr(TRAINING_CONFIG, "disable_discrete_obs", False)),
        bern_weight=float(getattr(TRAINING_CONFIG, "bern_weight", 1.0)),
    )

    return dict(
        meta=meta,
        metrics=metrics,
        online_ll=online_ll,
        anticipatory_ll=anticipatory_ll,
        baselines=dict(
            iid=iid_baseline,
            frozen_belief=frozen_baseline,
        )
    )


def main():
    project_root = Path(__file__).resolve().parents[1]
    model_root = project_root / "models"
    data_root = project_root / "data" / "highd"

    exp_name = getattr(TRAINING_CONFIG, "wandb_run_name", "unnamed_experiment")
    S = len(DBN_STATES.driving_style)
    A = len(DBN_STATES.action)
    em_mode = str(getattr(TRAINING_CONFIG, "emission_model", "poe")).lower().strip()
    exp_dir = model_root / f"{exp_name.lower()}_S{S}_A{A}_{em_mode}"

    if not exp_dir.exists():
        print(f"[evaluate_highd_dbn] ERROR: exp folder not found: {exp_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[evaluate_highd_dbn] exp_dir resolved -> {exp_dir}")
    print("[evaluate_highd_dbn] Running evaluate_experiment()")

    checkpoint_name = "final.npz"
    result = evaluate_experiment(exp_dir=exp_dir, checkpoint_name=checkpoint_name, data_root=data_root)
    print("[evaluate_highd_dbn] Writing eval_metrics.json")

    metrics = result["metrics"]

    # Output paths (saved inside the experiment folder)
    out_json = exp_dir / "eval_metrics.json"
    write_json(out_json, result)

    # Console summary
    print(f"[evaluate_highd_dbn] exp={exp_dir.name} ckpt={checkpoint_name}")
    print(f"[evaluate_highd_dbn] saved JSON -> {out_json}")
    print(f"[evaluate_highd_dbn] ll/t={metrics['ll_per_timestep']:.6f}  BIC={metrics['BIC']:.2f}  k={metrics['k_params']}")
    print(f"[evaluate_highd_dbn] ent_joint_vehicle_median={metrics['ent_joint_vehicle_median']:.3f}")
    print("[evaluate_highd_dbn] Done.")


if __name__ == "__main__":
    main()