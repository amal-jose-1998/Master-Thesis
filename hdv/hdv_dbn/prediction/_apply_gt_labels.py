"""
Rule-based ground-truth latent labeling for a 2-style / A-action driver model.

This module converts window-level feature vectors into discrete joint latent labels
z = (s, a) using hand-crafted thresholds, then optionally enforces temporal
persistence by removing (or merging) very short runs.

Latent encoding:
    - s: style index (e.g., 0=free-flow/low interaction, 1=tactical/high interaction)
    - a: action index (0..A-1)
    - z: joint index in [0, S*A-1] computed as z = s*A + a
    - UNKNOWN_Z: special label (-1) meaning "no rule fired / unknown"

Expected input:
    obs_seq: array of shape (T, D) where each row corresponds to a window (time step)
             and each column corresponds to a named window feature in `feature_cols`.
"""

from dataclasses import dataclass, fields
from pathlib import Path
import sys
import numpy as np
from tabulate import tabulate

UNKNOWN_Z = -1

EXP_DIR = r"/home/RUS_CIP/st184634/implementation/hdv/models/paper-run_S2_A4_tied_action"
DATA_ROOT = r"/home/RUS_CIP/st184634/implementation/hdv/data/highd"  
CHECKPOINT_NAME = "final.npz"
SEMANTIC_MAP = r"/home/RUS_CIP/st184634/implementation/hdv/models/paper-run_S2_A4_tied_action/semantic_map.yaml"

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

def _get_optional(obs_t, idx, name, default=np.nan):
    if name not in idx:
        return float(default)

    value = float(obs_t[idx[name]])
    return value if np.isfinite(value) else float(default)

# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class RuleThresholds:
    """
    Thresholds for tied_action_lc_none semantic labeling.

    Style:
      s0 = low_interaction
      s1 = high_interaction

    Action:
      a0 = cruising
      a1 = braking
      a2 = acceleration_negative_lateral_tendency
      a3 = acceleration_positive_lateral_tendency
    """
    # Style / interaction context
    high_interaction_front_exists: float = 0.80
    low_interaction_front_exists_max: float = 0.60

    high_interaction_front_dx_max: float = 70.0
    low_interaction_front_dx_min: float = 70.0

    high_interaction_thw_max: float = 3.0

    # Braking action: a1
    brake_ax_neg_frac: float = 0.90
    brake_ax_last: float = -0.15
    brake_vx_slope: float = -0.006

    # Acceleration actions: a2/a3
    accel_ax_pos_frac: float = 0.90
    accel_ax_last: float = 0.15
    accel_vx_slope: float = 0.005

    # Negative lateral tendency: a2
    negative_lateral_vy: float = -0.02
    negative_lateral_ay: float = -0.015
    negative_lateral_ay_neg_frac: float = 0.60

    # Positive lateral tendency: a3
    positive_lateral_vy: float = 0.03
    positive_lateral_ay: float = 0.02
    positive_lateral_ay_pos_frac: float = 0.75

    # Cruising: a0
    cruise_vx_slope_abs_max: float = 0.005
    cruise_ax_last_abs_max: float = 0.12
    cruise_ax_neg_frac_min: float = 0.45
    cruise_ax_neg_frac_max: float = 0.75
    cruise_ax_pos_frac_min: float = 0.20
    cruise_ax_pos_frac_max: float = 0.50

    @classmethod
    def from_dict(cls, d):
        """
        Create a RuleThresholds instance from a dict (parsed from YAML).
        Unknown/missing keys fall back to dataclass defaults.
        """
        if not d: # If dict is empty/None, return defaults.
            return cls()
        params = {}
        for f in fields(cls): # Iterates over all dataclass fields defined in RuleThresholds
            name = f.name # threshold key
            if name in d: # If the dict provides a value:
                try: # try converting to float
                    params[name] = float(d[name])
                except Exception: # else keep as-is
                    params[name] = d[name]
        return cls(**params) # Construct a RuleThresholds object overriding only provided keys. Any missing keys stay at default values.

# -----------------------------
# Label one window
# -----------------------------
def label_one_window_z(obs_t, feature_cols, thr: RuleThresholds, A=4, debug=False):
    """
    Assign rule-based GT label for tied_action_lc_none.

    Style:
        s0 = low_interaction
        s1 = high_interaction

    Action:
        a0 = cruising
        a1 = braking
        a2 = acceleration_negative_lateral_tendency
        a3 = acceleration_positive_lateral_tendency
    """
    idx = _build_index(feature_cols) # Creates mapping {feature_name: column_index}

    # small helper: only build these if debug=True
    values = None
    def record(extra=None):
        if not debug:
            return None
        
        nonlocal values
        if values is None:
            def safe(name, default=float("nan")):
                return _get_optional(obs_t, idx, name, default=default)
            
            values = {
                "ax_last": safe("ax_last"),
                "vx_last": safe("vx_last"),
                "vx_slope": safe("vx_slope"),
                "ax_neg_frac": safe("ax_neg_frac"),
                "ax_pos_frac": safe("ax_pos_frac"),
                "front_exists_frac": safe("front_exists_frac"),
                "front_dx_min": safe("front_dx_min"),
                "front_thw_last": safe("front_thw_last"),
                "front_ttc_min": safe("front_ttc_min"),
                "lc_left_present": safe("lc_left_present"),
                "lc_right_present": safe("lc_right_present"),
                "vy_last": safe("vy_last"),
                "vy_slope": safe("vy_slope"),
                "ay_last": safe("ay_last"),
                "ay_neg_frac": safe("ay_neg_frac"),
                "ay_pos_frac": safe("ay_pos_frac"),
                "ay_zero_frac": safe("ay_zero_frac"),
            }
        if extra:
            values.update(extra)

        return values

    # ---------------------------------------------------------
    # Read required features
    # ---------------------------------------------------------
    ax_last = _get(obs_t, idx, "ax_last")
    vx_slope = _get(obs_t, idx, "vx_slope")

    ax_neg_frac = _get(obs_t, idx, "ax_neg_frac")
    ax_pos_frac = _get(obs_t, idx, "ax_pos_frac")

    front_exists_frac = _get(obs_t, idx, "front_exists_frac")

    # Optional but useful context features
    front_dx_min = _get_optional(obs_t, idx, "front_dx_min")
    front_thw_last = _get_optional(obs_t, idx, "front_thw_last")

    # Optional lateral features
    vy_last = _get_optional(obs_t, idx, "vy_last")
    ay_last = _get_optional(obs_t, idx, "ay_last")
    ay_neg_frac = _get_optional(obs_t, idx, "ay_neg_frac")
    ay_pos_frac = _get_optional(obs_t, idx, "ay_pos_frac")

    # ---------------------------------------------------------
    # 1. Style decision
    # ---------------------------------------------------------
    s = None

    high_by_front = front_exists_frac >= thr.high_interaction_front_exists
    high_by_gap = (
        np.isfinite(front_dx_min)
        and np.isfinite(front_thw_last)
        and front_dx_min <= thr.high_interaction_front_dx_max
        and front_thw_last <= thr.high_interaction_thw_max
    )

    low_by_front = front_exists_frac < thr.low_interaction_front_exists_max
    low_by_gap = (
        np.isfinite(front_dx_min)
        and front_dx_min > thr.low_interaction_front_dx_min
    )

    # Give high-interaction priority when both cues conflict.
    if high_by_front or high_by_gap:
        s = 1
        style_reason = "high_interaction"
    elif low_by_front or low_by_gap:
        s = 0
        style_reason = "low_interaction"
    else:
        style_reason = "unknown_style"

    # ---------------------------------------------------------
    # 2. Action decision
    # ---------------------------------------------------------
    a = None
    action_reason = "unknown_action"

    # a1: braking
    is_braking = (
        ax_neg_frac >= thr.brake_ax_neg_frac
        and ax_last <= thr.brake_ax_last
    )

    if is_braking:
        a = 1
        action_reason = "braking"

    else:
        # a2/a3: acceleration with lateral sign
        is_accel = (
            ax_pos_frac >= thr.accel_ax_pos_frac
            and ax_last >= thr.accel_ax_last
            and vx_slope >= thr.accel_vx_slope
        )

        if is_accel:
            neg_lateral = (
                (np.isfinite(vy_last) and vy_last <= thr.negative_lateral_vy)
                or (np.isfinite(ay_last) and ay_last <= thr.negative_lateral_ay)
                or (np.isfinite(ay_neg_frac) and ay_neg_frac >= thr.negative_lateral_ay_neg_frac)
            )

            pos_lateral = (
                (np.isfinite(vy_last) and vy_last >= thr.positive_lateral_vy)
                or (np.isfinite(ay_last) and ay_last >= thr.positive_lateral_ay)
                or (np.isfinite(ay_pos_frac) and ay_pos_frac >= thr.positive_lateral_ay_pos_frac)
            )

            if neg_lateral and not pos_lateral:
                a = 2
                action_reason = "acceleration_negative_lateral_tendency"

            elif pos_lateral and not neg_lateral:
                a = 3
                action_reason = "acceleration_positive_lateral_tendency"

            elif neg_lateral and pos_lateral:
                # Rare conflict: choose the stronger lateral direction.
                neg_score = 0.0
                pos_score = 0.0

                if np.isfinite(vy_last):
                    neg_score += max(0.0, thr.negative_lateral_vy - vy_last)
                    pos_score += max(0.0, vy_last - thr.positive_lateral_vy)

                if np.isfinite(ay_last):
                    neg_score += max(0.0, thr.negative_lateral_ay - ay_last)
                    pos_score += max(0.0, ay_last - thr.positive_lateral_ay)

                if np.isfinite(ay_neg_frac):
                    neg_score += max(0.0, ay_neg_frac - thr.negative_lateral_ay_neg_frac)

                if np.isfinite(ay_pos_frac):
                    pos_score += max(0.0, ay_pos_frac - thr.positive_lateral_ay_pos_frac)

                if neg_score >= pos_score:
                    a = 2
                    action_reason = "acceleration_negative_lateral_tendency"
                else:
                    a = 3
                    action_reason = "acceleration_positive_lateral_tendency"

        # a0: cruising / mild deceleration
        if a is None:
            is_cruising = (
                abs(vx_slope) <= thr.cruise_vx_slope_abs_max
                and abs(ax_last) <= thr.cruise_ax_last_abs_max
                and ax_neg_frac >= thr.cruise_ax_neg_frac_min
                and ax_neg_frac <= thr.cruise_ax_neg_frac_max
                and ax_pos_frac >= thr.cruise_ax_pos_frac_min
                and ax_pos_frac <= thr.cruise_ax_pos_frac_max
            )

            if is_cruising:
                a = 0
                action_reason = "cruising"

    # ---------------------------------------------------------
    # 3. Final label
    # ---------------------------------------------------------
    if s is None or a is None:
        if debug:
            vals = record({
                "style_reason": style_reason,
                "action_reason": action_reason,
            })
            return UNKNOWN_Z, vals

        return UNKNOWN_Z

    z = sa_to_z(s, a, A)

    if debug:
        vals = record({
            "style_reason": style_reason,
            "action_reason": action_reason,
        })
        return z, vals

    return z

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
def compute_gt_latents(obs_seq, feature_cols, thr=None, A=4, debug=False, fill_unknown="none"):
    """
    Compute rule-based "ground-truth" joint latents for a sequence of window features.

    Parameters
    obs_seq : array-like of shape (T, D)
        Window-feature matrix. Each row corresponds to one window/time index.
        Prefer raw/unscaled features when thresholds are defined in physical units.
    feature_cols : Sequence[str]
        Names of the D window features (must match the column order of obs_seq).
    thr : RuleThresholds, optional
        Threshold configuration. If None, defaults to RuleThresholds().
    persist : PersistenceConfig, optional
        Persistence configuration. If None, defaults to PersistenceConfig().
    A : int, default=4
        Number of actions per style for encoding z = s*A + a.

    Returns
    np.ndarray of shape (T,)
        Integer label sequence where each element is:
            - z in [0, 2*A - 1] if labeled by rules
            - UNKNOWN_Z (-1) otherwise
        After labeling, persistence is applied (minimum run-length post-processing).

    Raises
    ValueError
        If obs_seq is not 2D (T, D).
    KeyError
        If any rule-required feature name is missing from feature_cols.
    """
    if thr is None:
        # Try to use thresholds from the semantic map YAML if available
        sem_map_local = None
        if SEMANTIC_MAP:
            try:
                sem_map_local = _load_semantic_map_yaml(SEMANTIC_MAP)
            except Exception:
                sem_map_local = None

        if sem_map_local is not None:
            thr_cfg = sem_map_local.get("label_rules", {}).get("thresholds", {})
            thr = RuleThresholds.from_dict(thr_cfg)
        else:
            thr = RuleThresholds()
    
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
        style_name = sem_map.get("styles", {}).get(s_key, {}).get("name", None)
        action_name = sem_map.get("actions_by_style", {}).get(s_key, {}).get(a_key, {}).get("name", None)
        if style_name or action_name:
            return f"{style_name or s_key} / {action_name or a_key}"
    except Exception:
        return "-"
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

    A = int(getattr(trainer, "A", 4))

    sem_map = None
    if SEMANTIC_MAP:
        sem_map = _load_semantic_map_yaml(SEMANTIC_MAP)

    # Prefer thresholds defined in the semantic map YAML when available
    if sem_map is not None:
        thr_cfg = sem_map.get("label_rules", {}).get("thresholds", {})
        thr = RuleThresholds.from_dict(thr_cfg)
    else:
        thr = RuleThresholds()

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

            rows.append([t, z, s, a, sem,
                round(v.get("ax_last", np.nan), 2),
                round(v.get("vx_last", np.nan), 2),
                round(v.get("vx_slope", np.nan), 4),
                round(v.get("ax_neg_frac", np.nan), 2),
                round(v.get("ax_pos_frac", np.nan), 2),
                round(v.get("front_exists_frac", np.nan), 2),
                round(v.get("jerk_x_p95", np.nan), 2),
                round(v.get("front_thw_last", np.nan), 2),
                round(v.get("front_ttc_min", np.nan), 2),
                round(v.get("lc_left_present", np.nan), 2),
                round(v.get("lc_right_present", np.nan), 2),
                round(v.get("ay_zero_frac", np.nan), 2),
                round(v.get("vy_last", np.nan), 2),
                round(v.get("ay_last", np.nan), 2),
                round(v.get("vy_slope", np.nan), 4),    
            ])

        headers = [
            "t", "z", "s", "a", "semantic",
            "ax_last", "vx_last", "vx_slope",
            "ax_neg", "ax_pos", "front", "jerk_p95",
            "THW", "TTC", "lc_L", "lc_R", "ay_zero_frac",
            "vy_last", "ay_last", "vy_slope",
        ]

        print("\n" + "=" * 120)
        print(f"DETAIL seq[{i}] T={len(out_post)}")
        print("=" * 120)

        print(tabulate(
            rows,
            headers=headers,
            tablefmt="simple",
            floatfmt=("", "", "", "", "", ".2f", ".2f", ".4f", ".2f", ".2f", ".2f", ".2f", ".2f", ".2f",".2f", ".2f", ".2f", ".2f", ".2f", ".4f"),
            stralign="left",
            numalign="right",
            maxcolwidths=[None, None, None, None, 45, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
            disable_numparse=True,
        ))


if __name__ == "__main__":
    main()
