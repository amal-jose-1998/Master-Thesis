from __future__ import annotations

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


def _load_semantic_map_yaml(path: Path):
    if path is None or not Path(path).exists():
        return None
    try:
        import yaml
    except Exception:
        return None
    try:
        with Path(path).open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _semantic_names_for_sa(sem_map, s: int, a: int, default_style: str, default_action: str) -> tuple[str, str]:
    if sem_map is None:
        return default_style, default_action
    try:
        s_key = f"s{int(s)}"
        a_key = f"a{int(a)}"
        style_name = sem_map.get("styles", {}).get(s_key, {}).get("name")
        action_name = sem_map.get("actions_by_style", {}).get(s_key, {}).get(a_key, {}).get("name")
        return str(style_name or default_style), str(action_name or default_action)
    except Exception:
        return default_style, default_action


def _rewrite_result_with_semantics(result: dict, sem_map) -> dict:
    if sem_map is None:
        return result

    def _rewrite_row(row: dict) -> None:
        style_name, action_name = _semantic_names_for_sa(
            sem_map,
            row.get("s", -1),
            row.get("a", -1),
            str(row.get("style_name", "")),
            str(row.get("action_name", "")),
        )
        row["style_name"] = style_name
        row["action_name"] = action_name

    for block in ("emission", "prior", "posterior", "forecast_t_plus_1"):
        if block in result and isinstance(result[block], dict):
            if "map" in result[block] and isinstance(result[block]["map"], dict):
                _rewrite_row(result[block]["map"])
            for row in result[block].get("topk", []):
                if isinstance(row, dict):
                    _rewrite_row(row)

    diag = result.get("diagnostics", {})
    winning = diag.get("winning_state", {}) if isinstance(diag, dict) else {}
    if isinstance(winning, dict):
        _rewrite_row(winning)

    return result


def resolve_seq_index(test_seqs, trajectory_id=None, trajectory_index=None) -> int:
    """
    Resolve which test trajectory to explain.

    Priority:
    1. trajectory_id, if given
    2. trajectory_index, if given
    3. default to first test sequence
    """
    if trajectory_id is not None:
        keys = [seq_key(s) for s in test_seqs]
        if trajectory_id not in keys:
            preview = ", ".join(keys[:10])
            raise KeyError(
                f"trajectory_id='{trajectory_id}' not found in test split. "
                f"Example ids: {preview}"
            )
        return int(keys.index(trajectory_id))

    if trajectory_index is None:
        return 0

    idx = int(trajectory_index)
    if idx < 0 or idx >= len(test_seqs):
        raise IndexError(f"trajectory_index={idx} out of range [0, {len(test_seqs) - 1}]")
    return idx


def select_scaler_for_seq(trainer: HDVTrainer, seq):
    """
    Get the correct scaler for this sequence.

    Supports:
    - global scaler
    - classwise scaler
    """
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


def frame_to_window_index(
    *,
    frame_no: int,
    window_start_frame: int,
    W: int,
    stride: int,
    allow_nearest: bool = False,
) -> tuple[int, int | None]:
    """
    Convert absolute frame number to window-local index using (start, W, stride).

    Returns:
    - (index, None) if frame aligns exactly to the window grid
    - (index, snapped_frame) if allow_nearest=True and nearest grid point is used
    """
    frame_no = int(frame_no)
    window_start_frame = int(window_start_frame)
    W = int(W)
    stride = int(stride)

    if W <= 0:
        raise ValueError(f"W must be > 0, got {W}")
    if stride <= 0:
        raise ValueError(f"stride must be > 0, got {stride}")

    max_frame = window_start_frame + (W - 1) * stride
    if frame_no < window_start_frame or frame_no > max_frame:
        raise IndexError(
            f"frame_no={frame_no} out of window range [{window_start_frame}, {max_frame}]"
        )

    offset = frame_no - window_start_frame
    q, r = divmod(offset, stride)
    if r == 0:
        return int(q), None

    if not allow_nearest:
        raise ValueError(
            f"frame_no={frame_no} is not aligned with stride={stride} from start={window_start_frame}"
        )

    nearest_idx = int(np.rint(offset / float(stride)))
    nearest_idx = max(0, min(W - 1, nearest_idx))
    nearest_frame = window_start_frame + nearest_idx * stride
    return nearest_idx, int(nearest_frame)


def resolve_local_t(
    frames: np.ndarray,
    requested_t: int,
    *,
    window_W: int | None = None,
    window_stride: int | None = None,
) -> tuple[int, int | None]:
    """
    Resolve absolute frame number to a local timestep index.

    Notes:
    - `requested_t` is treated strictly as a frame number.
    - If frame is within window range but off-stride, nearest stride frame is used.
    """
    T = int(frames.shape[0])
    frame_no = int(requested_t)

    if window_W is not None and window_stride is not None and frames.size > 0:
        try:
            local_W = min(int(window_W), T)
            idx, snapped_frame = frame_to_window_index(
                frame_no=frame_no,
                window_start_frame=int(frames[0]),
                W=local_W,
                stride=int(window_stride),
                allow_nearest=True,
            )
            if 0 <= idx < T:
                return idx, snapped_frame
        except (ValueError, IndexError):
            pass

    idx = np.where(frames == frame_no)[0]
    if idx.size > 0:
        return int(idx[0]), None

    frame_min = int(np.min(frames))
    frame_max = int(np.max(frames))
    if frame_min <= frame_no <= frame_max:
        nearest_idx = int(np.argmin(np.abs(frames.astype(np.float64) - float(frame_no))))
        nearest_frame = int(frames[nearest_idx])
        return nearest_idx, nearest_frame

    raise IndexError(
        f"Requested frame={frame_no} is outside this sequence window "
        f"(frame range: {frame_min}..{frame_max})."
    )


def main():
    # ======================================================================
    # SETTINGS: edit these only
    # ======================================================================
    EXP_DIR = Path("/home/RUS_CIP/st184634/implementation/hdv/models/main-model-sticky_S2_A4_hierarchical")
    DATA_ROOT = Path("/home/RUS_CIP/st184634/implementation/hdv/data/highd")
    CHECKPOINT = EXP_DIR / "final.npz"
    SEMANTIC_MAP_YAML = EXP_DIR / "semantic_map.yaml"

    # Choose ONE of the below:
    TRAJECTORY_ID = "4.0:19.0"                 # example: "4.0:123.0"
    TRAJECTORY_INDEX = 0                # used only if TRAJECTORY_ID is None

    T = 325                              # absolute frame number to explain (324)
    TOP_K = 3                   # how many of the top posterior states and top emission states to include in the printed output and the output JSON
    OUT_JSON = EXP_DIR / "explanations" / f"explain_t{T}.json"                     
    # ======================================================================

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    trainer = HDVTrainer.load(CHECKPOINT)

    test_seqs, feature_cols, split_payload, _ = load_test_sequences_from_experiment_split(
        exp_dir=EXP_DIR,
        data_root=DATA_ROOT,
    )

    if list(feature_cols) != list(WINDOW_FEATURE_COLS):
        raise RuntimeError("Feature mismatch with WINDOW_FEATURE_COLS")

    seq_idx = resolve_seq_index(
        test_seqs=test_seqs,
        trajectory_id=TRAJECTORY_ID,
        trajectory_index=TRAJECTORY_INDEX,
    )
    seq = test_seqs[seq_idx]

    mean, std = select_scaler_for_seq(trainer, seq)

    scale_idx = np.asarray(
        [i for i, name in enumerate(feature_cols) if name in set(CONTINUOUS_FEATURES)],
        dtype=np.int64,
    )

    obs_raw = np.asarray(seq.obs, dtype=np.float64)
    obs_scaled = scale_obs_masked(obs_raw, mean, std, scale_idx)
    frames = np.asarray(seq.frames)
    local_t, nearest_frame = resolve_local_t(
        frames=frames,
        requested_t=int(T),
        window_W=int(split_payload.get("W", int(frames.shape[0]))),
        window_stride=int(split_payload.get("stride", 1)),
    )

    result = explain_prediction_at_t(
        obs_scaled=obs_scaled,
        obs_raw=obs_raw,
        frames=frames,
        emissions=trainer.emissions,
        pi_s0=trainer.pi_s0,
        pi_a0_given_s0=trainer.pi_a0_given_s0,
        A_s=trainer.A_s,
        A_a=trainer.A_a,
        t=local_t,
        top_k=int(TOP_K),
        feature_cols=list(feature_cols),
    )
    sem_map = _load_semantic_map_yaml(SEMANTIC_MAP_YAML)
    result = _rewrite_result_with_semantics(result, sem_map)

    traj_key = seq_key(seq)

    post_map = result["posterior"]["map"]
    next_map = result["forecast_t_plus_1"]["map"]
    diag = result["diagnostics"]

    ranked_groups = result.get("feature_group_contributions", {}).get("ranked_groups", [])
    top_group = ranked_groups[0]["group"] if ranked_groups else None

    print("=" * 80)
    print(f"trajectory = {traj_key}")
    print(f"requested_frame = {T}")
    if nearest_frame is None:
        print(f"resolved_t  = {local_t}")
    else:
        print(
            "requested frame is within this window but not aligned to stride; "
            f"snapping to nearest stride frame {nearest_frame}"
        )
        print(f"resolved_t  = {local_t}")
    print(f"t          = {result['t']}")
    print(f"frame      = {result.get('frame')}")
    print("=" * 80)

    print("\nPOSTERIOR MAP STATE AT t")
    print(
        f"style={post_map['style_name']}  "
        f"action={post_map['action_name']}  "
        f"prob={post_map['prob']:.4f}"
    )

    print("\nFORECAST MAP STATE AT t+1")
    print(
        f"style={next_map['style_name']}  "
        f"action={next_map['action_name']}  "
        f"prob={next_map['prob']:.4f}"
    )

    print("\nDIAGNOSTICS")
    print(f"driver                         = {diag['driver']}")
    print(f"tv(prior -> posterior)         = {diag['tv_distance_prior_to_posterior']:.4f}")
    print(f"prior/posterior MAP changed    = {diag['prior_to_posterior_map_changed']}")
    print(f"top feature group contribution = {top_group}")

    print("\nTOP POSTERIOR STATES")
    for i, row in enumerate(result["posterior"]["topk"], start=1):
        print(
            f"{i}. style={row['style_name']}, "
            f"action={row['action_name']}, "
            f"prob={row['prob']:.4f}"
        )

    print("\nTOP EMISSION STATES")
    for i, row in enumerate(result["emission"]["topk"], start=1):
        print(
            f"{i}. style={row['style_name']}, "
            f"action={row['action_name']}, "
            f"prob={row['prob']:.4f}"
        )

    print("\nFEATURE GROUP CONTRIBUTIONS")
    for row in ranked_groups:
        print(f"{row['group']}: {row['loglik']:.4f}")

    if OUT_JSON is not None:
        out_path = Path(OUT_JSON)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nSaved JSON to: {out_path}")


if __name__ == "__main__":
    main()