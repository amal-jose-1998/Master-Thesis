"""Evaluate ONE experiment folder on its saved TEST split (vehicle-level) and SAVE results"""

import json
import sys
from pathlib import Path

from .datasets import load_highd_folder, load_or_build_windowized
from .config import TRAINING_CONFIG, WINDOW_FEATURE_COLS, WINDOW_CONFIG, DBN_STATES
from .utils.eval_core import evaluate_checkpoint, evaluate_online_predictive_ll, evaluate_anticipatory_predictive_ll, seq_key


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
    if "keys" not in payload or "test" not in payload["keys"]:
        raise ValueError(f"split.json format unexpected: {p}")
    print(f"[evaluate_highd_dbn] split.json loaded: test={len(payload['keys']['test'])} vehicles, W={payload.get('W')}, stride={payload.get('stride')}")
    return payload, set(payload["keys"]["test"])


def load_test_sequences_from_experiment_split(exp_dir, data_root):
    """
    Loads windowized sequences matching the experiment split.json and returns:
       test_seqs, feature_cols, split_payload, test_keys
    """
    exp_dir = Path(exp_dir)
    data_root = Path(data_root)

    split_payload, test_keys = load_split_json(exp_dir)

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
    test_seqs = [s for s in all_seqs if seq_key(s) in test_keys]
    print(f"[evaluate_highd_dbn] Matched test sequences: {len(test_seqs)}/{len(test_keys)}")

    found = {seq_key(s) for s in test_seqs}
    missing = test_keys - found
    if missing:
        raise RuntimeError(
            f"[evaluate_highd_dbn] Missing {len(missing)} test vehicles from cache. "
            f"Likely W/stride/max_recordings mismatch.\n"
            f"Example missing keys: {list(sorted(missing))[:10]}"
        )

    return test_seqs, list(WINDOW_FEATURE_COLS), split_payload, test_keys

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

    test_seqs, feature_cols, split_payload, test_keys = load_test_sequences_from_experiment_split(
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