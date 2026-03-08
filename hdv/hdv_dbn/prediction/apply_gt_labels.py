"""
Rule-based ground-truth latent labeling for the S=1, A=5 driver model.

This module converts window-level feature vectors into discrete latent labels
z = (s, a) using hand-crafted thresholds loaded from semantic_map.yaml.
For this model there is a single dummy style (s=0), so z == a.
"""

from dataclasses import dataclass, fields
from pathlib import Path
import sys
import numpy as np
from tabulate import tabulate

UNKNOWN_Z = -1

EXP_DIR = r"/home/RUS_CIP/st184634/implementation/hdv/models/ablation_test_no_style_5_actions_S1_A5_hierarchical"
DATA_ROOT = r"/home/RUS_CIP/st184634/implementation/hdv/data/highd"  
CHECKPOINT_NAME = "final.npz"
SEMANTIC_MAP = r"/home/RUS_CIP/st184634/implementation/hdv/models/ablation_test_no_style_5_actions_S1_A5_hierarchical/semantic_map.yaml"

NUM_SEQS_SUMMARY = 10        # number of sequences to summarize in the final table output
DETAIL_FIRST_N = 10          # number of sequences to show detailed per-sequence results for 

# -----------------------------
# (s,a) <-> z
# -----------------------------
def sa_to_z(s, a, A):
    """
    Convert a (style, action) pair into a single joint latent index z.
         z = s*A + a

    Parameters
    s : int or int-like
        Style index.
    a : int or int-like
        Action index.
    A : int
        Number of actions per style.

    Returns
    int
        Joint latent index z = s*A + a.
    """
    return int(s) * int(A) + int(a)

def z_to_sa(z, A):
    """
    Convert a joint latent index z back into (style, action).

    Parameters
    z : int or int-like
        Joint latent index. Special case: z=UNKNOWN_Z (-1) returns (-1, -1).
    A : int
        Number of actions per style.

    Returns
    (int, int)
        Tuple (s, a) where:
            s = z // A
            a = z % A
        Special case: if z == UNKNOWN_Z, returns (-1, -1) to preserve the sentinel.
    """
    z = int(z)
    if z == UNKNOWN_Z:
        return (-1, -1)
    return z // int(A), z % int(A)

# -----------------------------
# Feature access
# -----------------------------
def _build_index(feature_cols):
    """
    Build a mapping from feature name to column index.

    Parameters
    feature_cols : Sequence[str]
        Feature names aligned with columns of the observation vector.

    Returns
    dict[str, int]
        Dictionary mapping feature name -> column index.
    """
    return {c: i for i, c in enumerate(feature_cols)}

def _require(idx, name):
    """
    Ensure a required feature exists in the index mapping.

    Parameters
    idx : dict[str, int]
        Feature index mapping.
    name : str
        Feature name that must exist.

    Returns
    int
        Column index corresponding to `name`.

    Raises
    KeyError
        If `name` is not present in `idx`.
    """
    if name not in idx:
        raise KeyError(f"Required feature '{name}' not found in feature_cols.")
    return idx[name]

def _get(obs_t, idx, name):
    """
    Retrieve one scalar feature value from a single observation vector.

    Parameters
    obs_t : array-like of shape (D,)
        One window feature vector at time t.
    idx : dict[str, int]
        Feature name -> column index mapping.
    name : str
        Name of the feature to retrieve.

    Returns
    float
        The feature value cast to float.

    Raises
    KeyError
        If the requested feature is missing.
    """
    j = _require(idx, name)
    return float(obs_t[j])

# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class RuleThresholds:
    # Lane change
    lc_flag_thresh: float = 0.5
    lateral_velocity_threshold: float = 0.10
    lateral_acceleration_threshold: float = 0.10
    lateral_velocity_slope_threshold: float = 0.003
    ay_zero_frac_max: float = 0.10

    # Sustained braking
    brake_ax_neg_frac: float = 0.90
    brake_ax_last: float = -0.15
    brake_front_exists_min: float = 0.90

    # Sustained acceleration
    accel_ax_pos_frac: float = 0.90
    accel_ax_last: float = 0.15
    accel_front_exists_min: float = 0.90

    # Stable following
    stable_follow_front_exists_min: float = 0.65
    stable_follow_vx_slope_abs_max: float = 0.008
    stable_follow_ax_last_abs_max: float = 0.10

    # Low-speed regulation
    low_speed_vx_last_max: float = 21.0
    low_speed_front_exists_max: float = 0.85
    low_speed_side_exists_min: float = 0.20
    low_speed_ax_neg_frac_min: float = 0.45
    low_speed_jerk_x_min: float = 0.18

    @classmethod
    def from_dict(cls, d):
        if not d:
            return cls()
        params = {}
        for f in fields(cls):
            if f.name in d:
                try:
                    params[f.name] = float(d[f.name])
                except Exception:
                    params[f.name] = d[f.name]
        return cls(**params)

def _label_debug_values(obs_t, idx):
    def safe(name, default=float("nan")):
        return float(obs_t[idx[name]]) if name in idx else float(default)

    return {
        "vx_last": safe("vx_last"),
        "vx_slope": safe("vx_slope"),
        "ax_last": safe("ax_last"),
        "ax_neg_frac": safe("ax_neg_frac"),
        "ax_pos_frac": safe("ax_pos_frac"),
        "vy_last": safe("vy_last"),
        "vy_slope": safe("vy_slope"),
        "ay_last": safe("ay_last"),
        "ay_zero_frac": safe("ay_zero_frac"),
        "jerk_x_p95": safe("jerk_x_p95"),
        "front_exists_frac": safe("front_exists_frac"),
        "left_side_exists_frac": safe("left_side_exists_frac"),
        "right_side_exists_frac": safe("right_side_exists_frac"),
        "lc_left_present": safe("lc_left_present"),
        "lc_right_present": safe("lc_right_present"),
    }

# -----------------------------
# Label one window
# -----------------------------
def label_one_window_z(obs_t, feature_cols, thr: RuleThresholds, A=5, debug=False):
    
    idx = _build_index(feature_cols) # Creates mapping {feature_name: column_index}

    # Common features
    vx_last = _get(obs_t, idx, "vx_last")
    vx_slope = _get(obs_t, idx, "vx_slope")
    ax_last = _get(obs_t, idx, "ax_last")
    ax_neg_frac = _get(obs_t, idx, "ax_neg_frac")
    ax_pos_frac = _get(obs_t, idx, "ax_pos_frac")
    vy_last = _get(obs_t, idx, "vy_last")
    vy_slope = _get(obs_t, idx, "vy_slope")
    ay_last = _get(obs_t, idx, "ay_last")
    ay_zero_frac = _get(obs_t, idx, "ay_zero_frac")
    jerk_x_p95 = _get(obs_t, idx, "jerk_x_p95")
    front_exists_frac = _get(obs_t, idx, "front_exists_frac")
    left_side_exists_frac = _get(obs_t, idx, "left_side_exists_frac")
    right_side_exists_frac = _get(obs_t, idx, "right_side_exists_frac")
    lc_left_present = _get(obs_t, idx, "lc_left_present")
    lc_right_present = _get(obs_t, idx, "lc_right_present")

    values = _label_debug_values(obs_t, idx) if debug else None

    # Priority 1: lane change (a1)
    is_lane_change = (
        (lc_left_present > thr.lc_flag_thresh)
        or (lc_right_present > thr.lc_flag_thresh)
        or (
            (abs(vy_last) >= thr.lateral_velocity_threshold)
            and (abs(ay_last) >= thr.lateral_acceleration_threshold)
            and (abs(vy_slope) >= thr.lateral_velocity_slope_threshold)
            and (ay_zero_frac <= thr.ay_zero_frac_max)
        )
    )
    if is_lane_change:
        z = sa_to_z(0, 1, A)
        return (z, values) if debug else z

    # Priority 2: sustained braking (a2)
    is_brake = (
        (ax_neg_frac >= thr.brake_ax_neg_frac)
        and (ax_last <= thr.brake_ax_last)
        and (front_exists_frac >= thr.brake_front_exists_min)
    )
    if is_brake:
        z = sa_to_z(0, 2, A)
        return (z, values) if debug else z

    # Priority 3: constrained acceleration (a4)
    is_accel = (
        (ax_pos_frac >= thr.accel_ax_pos_frac)
        and (ax_last >= thr.accel_ax_last)
        and (front_exists_frac >= thr.accel_front_exists_min)
    )
    if is_accel:
        z = sa_to_z(0, 4, A)
        return (z, values) if debug else z

    # Priority 4: low-speed regulation (a0)
    is_low_speed_reg = (
        (vx_last <= thr.low_speed_vx_last_max)
        and (ax_neg_frac >= thr.low_speed_ax_neg_frac_min)
        and (jerk_x_p95 >= thr.low_speed_jerk_x_min)
        and (
            (left_side_exists_frac >= thr.low_speed_side_exists_min)
            or (right_side_exists_frac >= thr.low_speed_side_exists_min)
            or (front_exists_frac <= thr.low_speed_front_exists_max)
        )
    )
    if is_low_speed_reg:
        z = sa_to_z(0, 0, A)
        return (z, values) if debug else z

    # Priority 5: stable following (a3)
    is_stable_follow = (
    (front_exists_frac >= thr.stable_follow_front_exists_min)
    and (abs(vx_slope) <= thr.stable_follow_vx_slope_abs_max)
    and (abs(ax_last) <= thr.stable_follow_ax_last_abs_max)
)
    if is_stable_follow:
        z = sa_to_z(0, 3, A)
        return (z, values) if debug else z

    return (UNKNOWN_Z, values) if debug else UNKNOWN_Z


def fill_unknown_nearest(z, unknown=UNKNOWN_Z, max_gap=5, tie_break="future"):
    """
    Fill UNKNOWN blocks between known labels by nearest label (split at midpoint).
    If max_gap is set, only fill gaps with length <= max_gap (in timesteps).
    """
    z = np.asarray(z, dtype=int).copy()
    T = len(z)

    known = np.where(z != unknown)[0]
    if known.size == 0:
        return z

    # fill prefix
    first = known[0]
    if first > 0:
        z[:first] = z[first]

    # fill gaps between known labels
    for i in range(len(known) - 1):
        L = known[i]
        R = known[i + 1]
        gap_len = (R - L - 1)
        if gap_len <= 0:
            continue
        if (max_gap is not None) and (gap_len > max_gap):
            continue
        mid = (L + R) // 2
        if tie_break == "future":
            z[L+1:mid] = z[L]
            z[mid:R] = z[R]
        else:
            z[L+1:mid+1] = z[L]
            z[mid+1:R] = z[R]

    # fill suffix
    last = known[-1]
    if last < T - 1:
        z[last+1:] = z[last]

    return z


# -----------------------------
# Main API
# -----------------------------
def compute_gt_latents(obs_seq, feature_cols, thr=None, A=5, debug=False, fill_unknown="none", semantic_map_path=None):
    if thr is None:
        sem_map_local = None
        if semantic_map_path is None:
            semantic_map_path = SEMANTIC_MAP
        if semantic_map_path:
            try:
                sem_map_local = _load_semantic_map_yaml(semantic_map_path)
            except Exception:
                sem_map_local = None
        thr = RuleThresholds.from_dict((sem_map_local or {}).get("label_rules", {}).get("thresholds", {}))

    obs_seq = np.asarray(obs_seq)
    if obs_seq.ndim != 2: # Validate obs shape
        raise ValueError(f"obs_seq must be (T,D), got {obs_seq.shape}")

    T = obs_seq.shape[0] # Number of windows (time steps) in the sequence.
    out = np.full((T,), UNKNOWN_Z, dtype=int) # output label array initialized to UNKNOWN_Z

    if not debug:
        for t in range(T): # Loop over each window index
            out[t] = label_one_window_z(obs_seq[t], feature_cols, thr, A=A) # assigns a label for that timestep/window

        if fill_unknown == "nearest":
            out = fill_unknown_nearest(out, tie_break="future")
        return out

    values = [{} for _ in range(T)]
    for t in range(T):
        z, v = label_one_window_z(obs_seq[t], feature_cols, thr, A=A, debug=True)
        out[t] = z # the chosen z before persistence
        values[t] = v # feature snapshot at that timestep

    if fill_unknown == "nearest":
        out = fill_unknown_nearest(out, tie_break="future")
    return out, values


def _load_semantic_map_yaml(path):
    try:
        import yaml
    except Exception:
        print("[gt_labeler] PyYAML not installed; semantic names disabled.")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _sa_semantic_name(sem_map, s, a):
    if sem_map is None:
        return "-"
    try:
        s_key = f"s{s}"
        a_key = f"a{a}"
        #style = sem_map.get("styles", {}).get(s_key, {})
        action = sem_map.get("actions_by_style", {}).get(s_key, {}).get(a_key, {})
        #style_name = style.get("name", s_key) if isinstance(style, dict) else str(style)
        action_name = action.get("name", a_key) if isinstance(action, dict) else str(action)
        return f"{action_name}"
    except Exception:
        return "-"


def main():
    try:
        # When executed/imported as part of the package
        from .data_loader import load_test_data_for_prediction
    except ImportError:
        # When run directly: python path/to/apply_gt_labels.py
        project_root = Path(__file__).resolve().parents[3]  # .../implementation
        sys.path.insert(0, str(project_root))

        from hdv.hdv_dbn.prediction.data_loader import load_test_data_for_prediction

    trainer, test = load_test_data_for_prediction(
        exp_dir=Path(EXP_DIR),
        data_root=Path(DATA_ROOT),
        checkpoint_name=CHECKPOINT_NAME,
    )  

    A = int(getattr(trainer, "A", 5))
    sem_map = _load_semantic_map_yaml(SEMANTIC_MAP) if SEMANTIC_MAP else None
    thr = RuleThresholds.from_dict((sem_map or {}).get("label_rules", {}).get("thresholds", {}))

    # 1) quick summary counts for first N sequences
    n = min(NUM_SEQS_SUMMARY, len(test.raw_obs))
    print(f"[gt_labeler] Loaded {len(test.raw_obs)} test sequences. Showing summary for first {n}.")
    for i in range(n):
        z = compute_gt_latents(test.raw_obs[i], test.feature_cols, thr=thr, A=A, debug=False, fill_unknown="none")
        uniq, cnt = np.unique(z, return_counts=True)
        pairs = sorted(zip(uniq.tolist(), cnt.tolist()), key=lambda x: -x[1])
        print(f"  seq[{i}] T={len(z)} label_counts: {pairs}")

    # 2) detailed print for the first N trajectories (with reasons)
    m = min(int(DETAIL_FIRST_N), len(test.raw_obs))
    for i in range(m):
        out_post, values = compute_gt_latents(test.raw_obs[i], test.feature_cols, thr=thr, A=A, debug=True, fill_unknown="none")
        rows = []
        for t in range(len(out_post)):
            z = int(out_post[t])
            s, a = z_to_sa(z, A)
            sem = _sa_semantic_name(sem_map, s, a) if z != UNKNOWN_Z else "-"
            v = values[t] if isinstance(values[t], dict) else {}
            rows.append([
                t, z, s, a, sem,
                round(v.get("vx_last", np.nan), 2),
                round(v.get("vx_slope", np.nan), 4),
                round(v.get("ax_last", np.nan), 2),
                round(v.get("ax_neg_frac", np.nan), 2),
                round(v.get("ax_pos_frac", np.nan), 2),
                round(v.get("vy_last", np.nan), 2),
                round(v.get("vy_slope", np.nan), 4),
                round(v.get("ay_last", np.nan), 2),
                round(v.get("ay_zero_frac", np.nan), 2),
                round(v.get("front_exists_frac", np.nan), 2),
                round(v.get("left_side_exists_frac", np.nan), 2),
                round(v.get("right_side_exists_frac", np.nan), 2),
                round(v.get("jerk_x_p95", np.nan), 2),
                round(v.get("lc_left_present", np.nan), 2),
                round(v.get("lc_right_present", np.nan), 2),
            ])
        headers = [
            "t", "z", "s", "a", "semantic",
            "vx_last", "vx_slope", "ax_last", "ax_neg", "ax_pos",
            "vy_last", "vy_slope", "ay_last", "ay_zero",
            "front", "left_side", "right_side", "jerk_x", "lc_L", "lc_R",
        ]
        print("\n" + "=" * 140)
        print(f"DETAIL seq[{i}] T={len(out_post)}")
        print("=" * 140)
        print(tabulate(rows, headers=headers, tablefmt="simple", stralign="left", numalign="right", disable_numparse=True))



if __name__ == "__main__":
    main()
