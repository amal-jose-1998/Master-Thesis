import json
from pathlib import Path
import sys
import numpy as np



try:
    from ..config import CONTINUOUS_FEATURES, WINDOW_FEATURE_COLS, WindowConfig
    from ..evaluate_highd_dbn import load_test_sequences_from_experiment_split
    from ..trainer import HDVTrainer
    from ..utils.eval_common import scale_obs_masked, seq_key
    from .explainability import explain_prediction_at_t
    from ..datasets.dataset import TrajectorySequence
except ImportError:
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))
    from hdv.hdv_dbn.config import CONTINUOUS_FEATURES, WINDOW_FEATURE_COLS, WindowConfig
    from hdv.hdv_dbn.evaluate_highd_dbn import load_test_sequences_from_experiment_split
    from hdv.hdv_dbn.trainer import HDVTrainer
    from hdv.hdv_dbn.utils.eval_common import scale_obs_masked, seq_key
    from hdv.hdv_dbn.prediction.explainability import explain_prediction_at_t
    from hdv.hdv_dbn.datasets.dataset import TrajectorySequence


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


def resolve_seq_index(test_seqs, trajectory_id=None, trajectory_index=None):
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


def frame_to_window_index(frame_no, window_start_frame, n_steps, stride):
    """
    Convert absolute frame number to window-local index using
    (window grid start frame, number of windowed steps, stride).

    Returns:
    - (index, None) if frame aligns exactly to the window grid
    - (index, snapped_frame) if nearest grid point is used
    """
    frame_no = int(frame_no) # absolute frame number to resolve
    window_start_frame = int(window_start_frame)
    n_steps = int(n_steps)
    stride = int(stride)

    if n_steps <= 0:
        raise ValueError(f"n_steps must be > 0, got {n_steps}")
    if stride <= 0:
        raise ValueError(f"stride must be > 0, got {stride}")

    max_frame = window_start_frame + (n_steps - 1) * stride
    if frame_no < window_start_frame or frame_no > max_frame:
        raise IndexError(
            f"frame_no={frame_no} out of window range [{window_start_frame}, {max_frame}]"
        )

    offset = frame_no - window_start_frame # how many frames from the start of the window
    q, r = divmod(offset, stride) # q is how many strides fit in the offset, r is the remainder
    if r == 0: # exact match to the window grid
        return int(q), None

    nearest_idx = int(np.rint(offset / float(stride))) # round to nearest stride index
    nearest_idx = max(0, min(n_steps - 1, nearest_idx)) # clamp to valid range
    nearest_frame = window_start_frame + nearest_idx * stride # compute the actual frame number of the snapped index
    return nearest_idx, int(nearest_frame)


def resolve_local_t(frames, requested_t, window_stride=None):
    """
    Resolve absolute frame number to a local timestep index.

    Notes:
    - `requested_t` is treated strictly as a frame number.
    - If frame is within window range but off-stride, nearest stride frame is used.
    """
    T = int(frames.shape[0]) # total number of frames in this windowed sequence
    frame_no = int(requested_t) # absolute frame number to resolve

    if window_stride is not None and frames.size > 0:
        try:
            idx, snapped_frame = frame_to_window_index(frame_no=frame_no, window_start_frame=int(frames[0]), n_steps=T, stride=int(window_stride))
            if 0 <= idx < T: # if it is between the first and last frame of this window, return it
                return idx, snapped_frame
        except (ValueError, IndexError):
            pass

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

    T = 214                              # absolute frame number to explain 
    TOP_K = 3                   # how many of the top posterior states and top emission states to include in the printed output and the output JSON
    OUT_JSON = EXP_DIR / "explanations" / f"explain_t{T}.json"                     
    # ======================================================================

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    trainer = HDVTrainer.load(CHECKPOINT, device="cpu")

    test_seqs, feature_cols, _, _ = load_test_sequences_from_experiment_split(
        exp_dir=EXP_DIR,
        data_root=DATA_ROOT,
    )

    if list(feature_cols) != list(WINDOW_FEATURE_COLS):
        raise RuntimeError("Feature mismatch with WINDOW_FEATURE_COLS")

    seq_idx = resolve_seq_index(test_seqs=test_seqs, trajectory_id=TRAJECTORY_ID, trajectory_index=TRAJECTORY_INDEX)
    seq: TrajectorySequence = test_seqs[seq_idx] # windowed sequence, not the full original trajectory

    mean, std = select_scaler_for_seq(trainer, seq)

    scale_idx = np.asarray( # integer index array of all continuous feature positions.
        [i for i, name in enumerate(feature_cols) if name in set(CONTINUOUS_FEATURES)],
        dtype=np.int64,
    )

    obs_raw = np.asarray(seq.obs, dtype=np.float64)
    obs_scaled = scale_obs_masked(obs_raw, mean, std, scale_idx) # actual observation matrix that will go into emission likelihood evaluation. (T_seq, F)
    frames = np.asarray(seq.frames)
    local_t, nearest_frame = resolve_local_t(frames=frames, requested_t=int(T), window_stride=WindowConfig.stride)
    sem_map = _load_semantic_map_yaml(SEMANTIC_MAP_YAML)

    result = explain_prediction_at_t(obs_scaled=obs_scaled, obs_raw=obs_raw, frames=frames, emissions=trainer.emissions, pi_s0=trainer.pi_s0,
                                     pi_a0_given_s0=trainer.pi_a0_given_s0, A_s=trainer.A_s, A_a=trainer.A_a, t=local_t, top_k=int(TOP_K), feature_cols=list(feature_cols), semantic_map=sem_map)

    traj_key = seq_key(seq) # trajectory identifier string, e.g. "4.0:19.0"

    post_map = result["posterior"]["map"] # dict for posterior MAP state
    next_map = result["forecast_t_plus_1"]["map"] # dict for one-step forecast MAP state
    diag = result["diagnostics"] # dict of summary diagnostics

    group_scores = result.get("feature_group_contributions", {}).get("group_scores_loglik", {})

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
    print(f"window_index   = {result.get('window_index', local_t)}")
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
    driver_name = diag.get("driver")
    if driver_name is None:
        seq_meta = getattr(seq, "meta", None) or {}
        driver_name = seq_meta.get("driver") or seq_meta.get("meta_class")
    print(f"driver                         = {driver_name if driver_name is not None else 'n/a'}")
    print(f"tv(prior -> posterior)         = {diag['tv_distance_prior_to_posterior']:.4f}")
    print(f"prior/posterior MAP changed    = {diag['prior_to_posterior_map_changed']}")

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
    for group, loglik in group_scores.items():
        print(f"{group}: {float(loglik):.4f}")

    if OUT_JSON is not None:
        out_path = Path(OUT_JSON)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nSaved JSON to: {out_path}")


if __name__ == "__main__":
    main()