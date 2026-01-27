"""
Evaluate ONE experiment folder on its saved TEST split (vehicle-level).

Single source of truth for metrics:
  uses eval_core.evaluate_checkpoint() which:
    - loads trainer (+ scaler)
    - scales test obs
    - runs canonical infer_posterior
    - computes LL, entropy, run-lengths, occupancy, approx BIC, etc.
"""

import json
import sys
from pathlib import Path

from .datasets import load_highd_folder, load_or_build_windowized
from .config import TRAINING_CONFIG, WINDOW_FEATURE_COLS, WINDOW_CONFIG
from .utils.eval_core import evaluate_checkpoint


def seq_key(seq):
    rid = getattr(seq, "recording_id", None)
    vid = getattr(seq, "vehicle_id", None)
    return f"{rid}:{vid}"


def load_split_json(exp_dir):
    p = exp_dir / "split.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing split.json: {p}")
    payload = json.loads(p.read_text())
    if "keys" not in payload or "test" not in payload["keys"]:
        raise ValueError(f"split.json format unexpected: {p}")
    return payload, set(payload["keys"]["test"])


def load_test_sequences_from_experiment_split(exp_dir, data_root):
    """
    Loads windowized sequences matching the experiment split.json and returns:
      test_seqs, feature_cols
    """
    split_payload, test_keys = load_split_json(exp_dir)

    W = int(split_payload.get("W", int(WINDOW_CONFIG.W)))
    stride = int(split_payload.get("stride", int(WINDOW_CONFIG.stride)))

    df = load_highd_folder(data_root, cache_path=None, force_rebuild=False, max_recordings=getattr(TRAINING_CONFIG, "max_highd_recordings", None))

    cache_dir = data_root / "cache"
    all_seqs = load_or_build_windowized(df, cache_dir=cache_dir, W=W, stride=stride, force_rebuild=False)

    test_seqs = [s for s in all_seqs if seq_key(s) in test_keys]

    found = {seq_key(s) for s in test_seqs}
    missing = test_keys - found
    if missing:
        raise RuntimeError(
            f"[evaluate_highd_dbn] Missing {len(missing)} test vehicles from cache. "
            f"Likely W/stride/max_recordings mismatch.\n"
            f"Example missing keys: {list(sorted(missing))[:10]}"
        )

    return test_seqs, list(WINDOW_FEATURE_COLS)


def evaluate_experiment(exp_dir, checkpoint_name="final.npz", data_root=None):
    project_root = Path(__file__).resolve().parents[1]
    data_root = data_root or (project_root / "data" / "highd")

    ckpt_path = exp_dir / checkpoint_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    test_seqs, feature_cols = load_test_sequences_from_experiment_split(exp_dir=exp_dir, data_root=data_root)

    metrics = evaluate_checkpoint(model_path=ckpt_path, test_seqs=test_seqs, feature_cols=feature_cols)
    return metrics


def main():
    project_root = Path(__file__).resolve().parents[1]
    model_root = project_root / "models"
    data_root = project_root / "data" / "highd"

    exp_name = getattr(TRAINING_CONFIG, "wandb_run_name", "unnamed_experiment")
    S = len(getattr(__import__("hdv.hdv_dbn.config", fromlist=["DBN_STATES"]).DBN_STATES.driving_style))
    A = len(getattr(__import__("hdv.hdv_dbn.config", fromlist=["DBN_STATES"]).DBN_STATES.action))
    em_mode = str(getattr(TRAINING_CONFIG, "emission_model", "poe")).lower().strip()
    exp_dir = model_root / f"{exp_name.lower()}_S{S}_A{A}_{em_mode}"

    if not exp_dir.exists():
        print(f"[evaluate_highd_dbn] ERROR: exp folder not found: {exp_dir}", file=sys.stderr)
        sys.exit(1)

    metrics = evaluate_experiment(exp_dir=exp_dir, checkpoint_name="final.npz", data_root=data_root)

    print(f"[evaluate_highd_dbn] exp={exp_dir.name} ckpt=final.npz")
    print(f"[evaluate_highd_dbn] ll/t={metrics['ll_per_timestep']:.6f}  BIC={metrics['BIC_approx']:.2f}  k={metrics['k_params_approx']}")
    print(f"[evaluate_highd_dbn] ent_joint_mean={metrics['ent_joint_mean']:.3f}  runlen_joint_median={metrics['runlen_joint_median']:.1f}")
    print("[evaluate_highd_dbn] Done.")


if __name__ == "__main__":
    main()
