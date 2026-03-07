from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

try:
    from ..config import CONTINUOUS_FEATURES, WINDOW_FEATURE_COLS
    from ..evaluate_highd_dbn import load_test_sequences_from_experiment_split
    from ..trainer import HDVTrainer
    from ..utils.eval_common import scale_obs_masked, seq_key
    from .explainability import explain_prediction_at_t
except ImportError:
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))
    from hdv.hdv_dbn.config import CONTINUOUS_FEATURES, WINDOW_FEATURE_COLS
    from hdv.hdv_dbn.evaluate_highd_dbn import load_test_sequences_from_experiment_split
    from hdv.hdv_dbn.trainer import HDVTrainer
    from hdv.hdv_dbn.utils.eval_common import scale_obs_masked, seq_key
    from hdv.hdv_dbn.prediction.explainability import explain_prediction_at_t


def _resolve_seq_index(test_seqs, trajectory_id: str | None, trajectory_index: int | None) -> int:
    if trajectory_id is not None:
        keys = [seq_key(s) for s in test_seqs]
        if trajectory_id not in keys:
            preview = ", ".join(keys[:10])
            raise KeyError(f"trajectory_id='{trajectory_id}' not found in test split. Example ids: {preview}")
        return int(keys.index(trajectory_id))

    if trajectory_index is None:
        return 0

    idx = int(trajectory_index)
    if idx < 0 or idx >= len(test_seqs):
        raise IndexError(f"trajectory_index={idx} out of range [0, {len(test_seqs)-1}]")
    return idx


def _select_scaler_for_seq(trainer: HDVTrainer, seq):
    mean = trainer.scaler_mean
    std = trainer.scaler_std

    if mean is None or std is None:
        raise RuntimeError("Model checkpoint is missing scaler_mean/scaler_std")

    if isinstance(mean, dict) and isinstance(std, dict):
        meta = getattr(seq, "meta", None) or {}
        cls = str(meta.get("meta_class", "NA"))
        if cls not in mean or cls not in std:
            raise KeyError(f"Classwise scaler missing key '{cls}'")
        return np.asarray(mean[cls], dtype=np.float64), np.asarray(std[cls], dtype=np.float64)

    return np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Explain DBN prediction at a chosen timestep")
    p.add_argument("--exp-dir", type=str, required=True, help="Experiment directory containing final.npz and split.json")
    p.add_argument("--data-root", type=str, required=True, help="highD data root")
    p.add_argument("--checkpoint", type=str, default="final.npz", help="Checkpoint file name in exp-dir")
    p.add_argument("--trajectory-id", type=str, default=None, help="Canonical trajectory key 'recording_id:vehicle_id'")
    p.add_argument("--trajectory-index", type=int, default=None, help="Index within test split (used if trajectory-id is not provided)")
    p.add_argument("--t", type=int, required=True, help="Window timestep index")
    p.add_argument("--top-k", type=int, default=3, help="Number of top states to report")
    p.add_argument("--out-json", type=str, default=None, help="Optional output path for full JSON explanation")
    return p


def main():
    parser = _build_parser()
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    data_root = Path(args.data_root)
    ckpt = exp_dir / args.checkpoint

    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    trainer = HDVTrainer.load(ckpt)

    test_seqs, feature_cols, _, _ = load_test_sequences_from_experiment_split(exp_dir=exp_dir, data_root=data_root)
    if list(feature_cols) != list(WINDOW_FEATURE_COLS):
        raise RuntimeError("Feature mismatch with WINDOW_FEATURE_COLS")

    seq_idx = _resolve_seq_index(test_seqs, args.trajectory_id, args.trajectory_index)
    seq = test_seqs[seq_idx]

    mean, std = _select_scaler_for_seq(trainer, seq)
    scale_idx = np.asarray([i for i, n in enumerate(feature_cols) if n in set(CONTINUOUS_FEATURES)], dtype=np.int64)

    obs_raw = np.asarray(seq.obs, dtype=np.float64)
    obs_scaled = scale_obs_masked(obs_raw, mean, std, scale_idx)
    frames = np.asarray(seq.frames)

    result = explain_prediction_at_t(
        obs_scaled=obs_scaled,
        obs_raw=obs_raw,
        frames=frames,
        emissions=trainer.emissions,
        pi_s0=trainer.pi_s0,
        pi_a0_given_s0=trainer.pi_a0_given_s0,
        A_s=trainer.A_s,
        A_a=trainer.A_a,
        t=int(args.t),
        top_k=int(args.top_k),
        feature_cols=list(feature_cols),
    )

    traj_key = seq_key(seq)
    post_map = result["posterior"]["map"]
    next_map = result["forecast_t_plus_1"]["map"]
    diag = result["diagnostics"]
    top_group = None
    ranked_groups = result.get("feature_group_contributions", {}).get("ranked_groups", [])
    if ranked_groups:
        top_group = ranked_groups[0]["group"]

    print(f"trajectory={traj_key}  t={result['t']}  frame={result.get('frame')}")
    print(
        "posterior_map="
        f"({post_map['style_name']}, {post_map['action_name']}) p={post_map['prob']:.4f}  "
        "next_map="
        f"({next_map['style_name']}, {next_map['action_name']}) p={next_map['prob']:.4f}"
    )
    print(
        f"driver={diag['driver']}  tv(prior,posterior)={diag['tv_distance_prior_to_posterior']:.4f}  "
        f"top_feature_group={top_group}"
    )

    if args.out_json is not None:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"saved_json={out_path}")


if __name__ == "__main__":
    main()