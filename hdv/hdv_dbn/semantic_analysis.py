"""Semantic analysis for a trained HDV DBN model."""

import numpy as np
import torch
from pathlib import Path

from .trainer import HDVTrainer
from .config import (
    SEMANTIC_CONFIG,
    WINDOW_FEATURE_COLS,
)
from .utils.semantic_analysis_utils import (load_sequences_from_experiment_split, compute_scale_idx, infer_gamma_on_split, compute_joint_semantics, _write_csv, _write_json, 
                                            write_joint_csv, write_action_csv, write_style_csv, compute_marginal_semantics, compute_action_style_consistency)


# -----------------------------
# main analysis
# -----------------------------
@torch.no_grad()
def run_semantic_analysis(model_path, data_root, split_name="train", semantic_feature_cols=None, max_sequences=None):
    ckpt_path = Path(str(model_path))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    exp_dir = ckpt_path.parent
    split_name = str(split_name).lower().strip()

    out_dir = exp_dir / "semantic_analysis" / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load sequences
    seqs, feature_cols, split_payload, split_keys = load_sequences_from_experiment_split(exp_dir=exp_dir, data_root=Path(data_root), split_name=split_name)
    if max_sequences is not None:
        seqs = seqs[: int(max_sequences)]

    # 2) Load model + setup
    trainer = HDVTrainer.load(ckpt_path)
    if trainer.scaler_mean is None or trainer.scaler_std is None:
        raise RuntimeError(f"Model '{ckpt_path}' missing scaler_mean/std.")

    scale_idx = compute_scale_idx(feature_cols)

    S = int(trainer.S)
    A = int(trainer.A)

    # 3) Inference (gamma_sa)
    obs_raw_seqs, gamma_sa_seqs = infer_gamma_on_split(trainer=trainer, seqs=seqs, feature_cols=feature_cols, scale_idx=scale_idx, split_name=split_name)

    # 4) Joint semantics
    feat_names, means_sa, stds_sa, mass_sa, frac_sa = compute_joint_semantics(obs_raw_seqs=obs_raw_seqs, gamma_sa_seqs=gamma_sa_seqs, 
                                                                               feature_cols=feature_cols, semantic_feature_cols=semantic_feature_cols, S=S,  A=A)
    write_joint_csv(out_dir=out_dir, feat_names=feat_names, means_sa=means_sa, stds_sa=stds_sa, mass_sa=mass_sa, frac_sa=frac_sa, S=S, A=A)

    # 5) Marginals
    marg = compute_marginal_semantics(obs_raw_seqs=obs_raw_seqs, gamma_sa_seqs=gamma_sa_seqs, feature_cols=feature_cols, feat_names=feat_names)

    write_style_csv(out_dir=out_dir, S=S, **{k: marg[k] for k in ("feat_used_s", "mass_s", "means_s", "stds_s")})
    write_action_csv(out_dir=out_dir, A=A, **{k: marg[k] for k in ("feat_used_a", "mass_a", "means_a", "stds_a")})

    # 6) Consistency check
    consistency_rows = compute_action_style_consistency(means_sa=means_sa, feat_names=feat_names, S=S, A=A)
    _write_csv(
        out_dir / "action_style_consistency.csv",
        ["a", "action_name", "rmse_core_features_between_style0_and_style1"],
        consistency_rows,
    )

    # 7) Summary JSON
    total_mass = float(np.sum(mass_sa))
    summary = {
        "model_path": str(ckpt_path),
        "exp_dir": str(exp_dir),
        "data_root": str(data_root),
        "split_name": split_name,
        "n_sequences": int(len(seqs)),
        "n_split_vehicles_expected": int(len(split_keys)),
        "S": int(S),
        "A": int(A),
        "semantic_feature_cols_requested": list(semantic_feature_cols),
        "semantic_feature_cols_used": list(feat_names),
        "total_mass": float(total_mass),
        "inactive_joint_cells_mass_frac_lt_0p01": [
            {"s": int(s), "a": int(a), "mass_frac": float(frac_sa[s, a])}
            for s in range(S) for a in range(A)
            if float(frac_sa[s, a]) < 0.01
        ],
        "artifacts": {
            "joint_semantics_csv": str(out_dir / "joint_semantics.csv"),
            "style_semantics_csv": str(out_dir / "style_semantics.csv"),
            "action_semantics_csv": str(out_dir / "action_semantics.csv"),
            "action_style_consistency_csv": str(out_dir / "action_style_consistency.csv"),
        }
    }
    _write_json(out_dir / "summary.json", summary)
    print(f"[semantic_analysis:{split_name}] wrote results to: {out_dir}", flush=True)
    return summary


def main():
    model_path = SEMANTIC_CONFIG.model_path
    data_root = SEMANTIC_CONFIG.data_root
    max_sequences = SEMANTIC_CONFIG.max_sequences

    # Default = TRAIN semantics; optionally run TEST later as validation by changing this
    split_name = getattr(SEMANTIC_CONFIG, "split_name", "train")

    semantic_cols = getattr(SEMANTIC_CONFIG, "semantic_feature_cols", None)
    if semantic_cols is None or len(semantic_cols) == 0:
        semantic_cols = list(WINDOW_FEATURE_COLS)

    run_semantic_analysis(
        model_path=model_path,
        data_root=data_root,
        split_name=split_name,
        semantic_feature_cols=list(semantic_cols),
        max_sequences=max_sequences,
    )


if __name__ == "__main__":
    main()