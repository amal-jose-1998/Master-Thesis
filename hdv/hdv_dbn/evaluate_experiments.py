"""Evaluate many experiment folders on ONE fixed TEST vehicle set, taken from split.json."""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

from .datasets import load_highd_folder, load_or_build_windowized
from .config import TRAINING_CONFIG, WINDOW_FEATURE_COLS
from .utils.eval_core import evaluate_checkpoint
from .utils.trainer_diagnostics import (
    posterior_entropy_from_gamma_sa,
    run_lengths_from_gamma_sa,
)


def _seq_key(seq):
    rid = getattr(seq, "recording_id", None)
    vid = getattr(seq, "vehicle_id", None)
    if rid is None or vid is None:
        meta = getattr(seq, "meta", None) or {}
        rid = meta.get("recording_id", meta.get("recordingId", "NA"))
        vid = meta.get("vehicle_id", meta.get("vehicleId", "NA"))
    return f"{rid}:{vid}"


def _load_split(exp_dir: Path) -> Tuple[dict, Set[str]]:
    split_path = exp_dir / "split.json"
    if not split_path.exists():
        raise FileNotFoundError(split_path)
    payload = json.loads(split_path.read_text())

    if "keys" in payload and isinstance(payload["keys"], dict) and "test" in payload["keys"]:
        return payload, set(payload["keys"]["test"])

    if "test" in payload:
        return payload, set(payload["test"])

    raise ValueError(f"Unrecognized split.json format: {split_path}")


def list_checkpoints(exp_dir: Path) -> List[Path]:
    ckpts = sorted(exp_dir.glob("*.npz"))

    def key(p: Path):
        name = p.stem.lower()
        if name == "final":
            return (10**9, name)
        m = re.search(r"iter(\d+)", name)
        if m:
            return (int(m.group(1)), name)
        m = re.search(r"(\d+)", name)
        if m:
            return (int(m.group(1)), name)
        return (0, name)

    return sorted(ckpts, key=key)


def write_csv(path: Path, rows: List[Dict[str, object]]):
    if not rows:
        return
    preferred_cols = [
        "experiment_dir", "checkpoint", "checkpoint_path",
        "ll_per_timestep", "BIC_approx", "k_params_approx",
        "ent_joint_mean", "ent_style_mean", "ent_action_mean",
        "runlen_joint_median", "runlen_style_median", "runlen_action_median",
        "occ_joint_max", "occ_style_max", "occ_action_max",
        "keff_joint", "keff_style", "keff_action",
        "total_ll", "total_T", "ll_per_traj",
        "traj_ll_p0", "traj_ll_p5", "traj_ll_p25", "traj_ll_p50", "traj_ll_p75", "traj_ll_p95", "traj_ll_p100",
    ]
    all_cols = list(dict.fromkeys(preferred_cols + sorted({k for r in rows for k in r.keys()})))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(all_cols) + "\n")
        for r in rows:
            vals = []
            for c in all_cols:
                v = r.get(c, "")
                if isinstance(v, str):
                    v = '"' + v.replace('"', '""') + '"'
                vals.append(str(v))
            f.write(",".join(vals) + "\n")


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "highd"
    models_root = project_root / "models"
    out_csv = models_root / "evaluation_summary.csv"

    exp_dirs = sorted([p for p in models_root.iterdir() if p.is_dir()])

    # Pick reference split
    ref_exp = None
    ref_payload = None
    ref_test_keys: Set[str] = set()
    for d in exp_dirs:
        if (d / "split.json").exists():
            ref_exp = d
            ref_payload, ref_test_keys = _load_split(d)
            break

    if ref_exp is None:
        print("[evaluate_experiments] ERROR: no experiment folder contains split.json.", file=sys.stderr)
        sys.exit(1)

    W = int(ref_payload.get("W", 0))
    stride = int(ref_payload.get("stride", 0))
    if W <= 0 or stride <= 0:
        print(
            f"[evaluate_experiments] ERROR: reference split.json missing valid W/stride in {ref_exp}/split.json",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[evaluate_experiments] Reference split: {ref_exp.name}  test_vehicles={len(ref_test_keys)}  W={W} stride={stride}")

    # Load dataset once -> windowized cache once
    df = load_highd_folder(
        data_root,
        cache_path=None,
        force_rebuild=False,
        max_recordings=getattr(TRAINING_CONFIG, "max_highd_recordings", None),
    )

    cache_dir = data_root / "cache"
    sequences = load_or_build_windowized(
        df,
        cache_dir=cache_dir,
        W=W,
        stride=stride,
        force_rebuild=False,
    )

    # Filter to reference TEST vehicles only
    test_seqs = [s for s in sequences if _seq_key(s) in ref_test_keys]
    found = {_seq_key(s) for s in test_seqs}
    missing = ref_test_keys - found
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} reference test vehicles in windowized cache. "
            f"Example missing: {list(sorted(missing))[:10]}"
        )

    feature_cols = list(WINDOW_FEATURE_COLS)

    rows: List[Dict[str, object]] = []

    for exp_dir in exp_dirs:
        ckpts = list_checkpoints(exp_dir)
        if not ckpts:
            continue

        # If exp has its own split.json, check for mismatch (warn only)
        if (exp_dir / "split.json").exists():
            try:
                _, exp_test_keys = _load_split(exp_dir)
                if exp_test_keys != ref_test_keys:
                    print(
                        f"[evaluate_experiments] WARNING: {exp_dir.name} test split differs from reference. "
                        f"Evaluating on REFERENCE test split anyway for fairness."
                    )
            except Exception:
                print(f"[evaluate_experiments] WARNING: failed to read split.json for {exp_dir.name}, continuing.")

        for ckpt in ckpts:
            try:
                metrics = evaluate_checkpoint(
                    model_path=ckpt,
                    test_seqs=test_seqs,
                    feature_cols=feature_cols,
                )
            except Exception as e:
                print(f"[evaluate_experiments] ERROR {exp_dir.name}/{ckpt.name}: {e}", file=sys.stderr)
                continue

            row = dict(
                experiment_dir=exp_dir.name,
                checkpoint=ckpt.name,
                checkpoint_path=str(ckpt),
            )
            row.update(metrics)
            rows.append(row)

            print(f"[evaluate_experiments] OK {exp_dir.name}/{ckpt.name}  ll/t={metrics['ll_per_timestep']:.6f}  BIC={metrics['BIC_approx']:.2f}")

    if not rows:
        print("[evaluate_experiments] No results. Check models/ contains experiment folders with .npz checkpoints.")
        return

    write_csv(out_csv, rows)
    print(f"[evaluate_experiments] Wrote {len(rows)} rows -> {out_csv}")
    print("[evaluate_experiments] Done.")


if __name__ == "__main__":
    main()
