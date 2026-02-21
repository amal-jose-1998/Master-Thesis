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

EXP_DIR = r"/home/RUS_CIP/st184634/implementation/hdv/models/5-actions-hierarchical_S2_A5_hierarchical"
DATA_ROOT = r"/home/RUS_CIP/st184634/implementation/hdv/data/highd"  
CHECKPOINT_NAME = "final.npz"
SEMANTIC_MAP = r"/home/RUS_CIP/st184634/implementation/hdv/models/5-actions-hierarchical_S2_A5_hierarchical/semantic_map.yaml"

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
    """
    Thresholds tuned for S2A5 map.

    Notes on robustness:
    - THW/TTC are only used when front_exists_frac is sufficiently high.
    - front_dx_min is preferred for interaction tightness when available.
    """

    # Lane change (s0-a0)
    lc_flag_thresh: float = 0.02

    # Sustained braking (generic gate)
    brake_ax_neg_frac: float = 0.90
    brake_ax_last: float = -0.15

    # Critical braking (s1-a3) vs non-critical tight braking (s0-a3)
    critical_brake_ax_last: float = -0.35
    critical_brake_vx_slope: float = -0.015
    critical_brake_vx_last_max: float = 20.0

    # Interaction tightness (for s0-a3 and other “tight” checks)
    front_exists_gate: float = 0.80
    interaction_thw_tight: float = 2.0
    interaction_ttc_tight: float = 60.0
    interaction_front_dx_close: float = 55.0  # smaller = closer lead

    # Acceleration detection
    acc_ax_pos_frac: float = 0.90

    # s0 strong accel vs s0 smooth accel
    s0_strong_acc_ax_last: float = 0.18
    s0_strong_acc_vx_slope_min: float = 0.007
    s0_smooth_acc_ax_last: float = 0.06
    s0_free_space_front_exists_max: float = 0.75
    s0_free_space_front_dx_min: float = 70.0

    # s1 accel split: constrained flow vs stop-and-go pickup
    s1_constrained_acc_front_exists_min: float = 0.90
    s1_constrained_acc_ax_last_min: float = 0.04
    s1_constrained_acc_ax_pos_frac_min: float = 0.60
    s1_pickup_vx_last_max: float = 20.0
    s1_pickup_jerk_x_p95_min: float = 0.20

    # Brake-biased regulation (s1-a0) (mild braking, not sustained)
    mild_brake_ax_last_max: float = -0.02
    mild_brake_ax_neg_frac_min: float = 0.55

    # Open-gap regulation (s1-a2)
    open_gap_front_exists_max: float = 0.50
    open_gap_front_dx_min: float = 80.0

    # Stable/adaptive regulation (s0-a1)
    stable_follow_vx_slope_abs_max: float = 0.006
    stable_follow_ax_last_abs_max: float = 0.12

    @classmethod
    def from_dict(cls, d):
        if not d:
            return cls()
        params = {}
        for f in fields(cls):
            name = f.name
            if name in d:
                try:
                    params[name] = float(d[name])
                except Exception:
                    params[name] = d[name]
        return cls(**params)

# -----------------------------
# Helper predicates
# -----------------------------
def _tight_interaction(front_exists_frac, front_dx_min, thw, ttc, thr: RuleThresholds):
    if front_exists_frac < thr.front_exists_gate:
        return False

    tight = False
    if front_dx_min is not None:
        tight = tight or (front_dx_min <= thr.interaction_front_dx_close)
    if thw is not None:
        tight = tight or (thw <= thr.interaction_thw_tight)
    if ttc is not None:
        tight = tight or (ttc <= thr.interaction_ttc_tight)
    return bool(tight)

def _critical_brake(ax_last, vx_slope, vx_last, thr: RuleThresholds):
    return (
        (ax_last <= thr.critical_brake_ax_last)
        or (vx_slope <= thr.critical_brake_vx_slope)
        or (vx_last <= thr.critical_brake_vx_last_max)
    )

# -----------------------------
# Label one window
# -----------------------------
def label_one_window_z(obs_t, feature_cols, thr: RuleThresholds, A=5, debug=False):
    """
    Priority order (S2A5):
      1) s0-a0 lane change
      2) sustained braking -> split critical (s1-a3) vs tight (s0-a3)
      3) s0 acceleration -> strong (s0-a2) vs smooth/free-space (s0-a4)
      4) s1 stop-and-go pickup accel (s1-a4)
      5) s1 constrained following accel (s1-a1)
      6) s1 open-gap regulation (s1-a2)
      7) s1 brake-biased regulation (s1-a0)
      8) s0 adaptive follow regulation (s0-a1)
      9) UNKNOWN
    """
    idx = _build_index(feature_cols) # Creates mapping {feature_name: column_index}

    # small helper: only build these if debug=True
    values = None
    def record():
        if not debug:
            return None
        nonlocal values
        if values is None:
            def safe(name, default=float("nan")):
                return float(obs_t[idx[name]]) if name in idx else float(default)
            values = {
                "ax_last": safe("ax_last"),
                "vx_last": safe("vx_last"),
                "vx_slope": safe("vx_slope"),
                "ax_neg_frac": safe("ax_neg_frac"),
                "ax_pos_frac": safe("ax_pos_frac"),
                "front_exists_frac": safe("front_exists_frac"),
                "jerk_x_p95": safe("jerk_x_p95"),
                "lc_left_present": safe("lc_left_present"),
                "lc_right_present": safe("lc_right_present"),
                "front_thw_last": safe("front_thw_last"),
                "front_ttc_min": safe("front_ttc_min"),
            }
        return values

    # Required core features
    ax_last = _get(obs_t, idx, "ax_last")
    vx_last = _get(obs_t, idx, "vx_last")
    vx_slope = _get(obs_t, idx, "vx_slope")
    ax_neg_frac = _get(obs_t, idx, "ax_neg_frac")
    ax_pos_frac = _get(obs_t, idx, "ax_pos_frac")
    front_exists_frac = _get(obs_t, idx, "front_exists_frac")

    front_dx_min = _get(obs_t, idx, "front_dx_min")
    thw = _get(obs_t, idx, "front_thw_last")
    ttc = _get(obs_t, idx, "front_ttc_min")
    jerk_x_p95 = _get(obs_t, idx, "jerk_x_p95")

    # 1) Lane change -> s0-a0
    lc_l = _get(obs_t, idx, "lc_left_present")
    lc_r = _get(obs_t, idx, "lc_right_present")
    if (lc_l >= thr.lc_flag_thresh) or (lc_r >= thr.lc_flag_thresh):
        z = sa_to_z(0, 0, A)
        return (z, record()) if debug else z

    # Interaction predicate
    is_tight = _tight_interaction(front_exists_frac, front_dx_min, thw, ttc, thr)

    # 2) Sustained braking -> s1-a3 critical vs s0-a3 tight braking
    is_brake = (
        (ax_neg_frac >= thr.brake_ax_neg_frac)
        and (ax_last <= thr.brake_ax_last)
        and (front_exists_frac >= 0.70)  # allow some missing, but avoid open-gap
    )
    if is_brake:
        critical = _critical_brake(ax_last, vx_slope, vx_last, thr)
        z = sa_to_z(1, 3, A) if critical else sa_to_z(0, 3, A)
        return (z, record()) if debug else z

    # 3) s0 acceleration: strong (a2) vs smooth/free-space (a4)
    if ax_pos_frac >= thr.acc_ax_pos_frac:
        # strong accel requires higher ax_last AND positive vx trend
        if (ax_last >= thr.s0_strong_acc_ax_last) and (vx_slope >= thr.s0_strong_acc_vx_slope_min):
            z = sa_to_z(0, 2, A)
            return (z, record()) if debug else z

        # smooth/free-space accel: gentler ax_last and open-ish context
        openish = (
            (front_exists_frac <= thr.s0_free_space_front_exists_max)
            or (front_dx_min is not None and front_dx_min >= thr.s0_free_space_front_dx_min)
        )
        if (ax_last >= thr.s0_smooth_acc_ax_last) and openish:
            z = sa_to_z(0, 4, A)
            return (z, record()) if debug else z

    # 4) s1 stop-and-go pickup accel (a4): low speed + high jerk + accel-dominant + leader present
    if (
        (front_exists_frac >= thr.front_exists_gate)
        and (vx_last <= thr.s1_pickup_vx_last_max)
        and (ax_pos_frac >= thr.s1_constrained_acc_ax_pos_frac_min)
        and (ax_last >= thr.s1_constrained_acc_ax_last_min)
        and (jerk_x_p95 >= thr.s1_pickup_jerk_x_p95_min)
    ):
        z = sa_to_z(1, 4, A)
        return (z, record()) if debug else z

    # 5) s1 constrained following acceleration (a1): accel-dominant + leader present, not stop-and-go
    if (
        (front_exists_frac >= thr.s1_constrained_acc_front_exists_min)
        and (ax_pos_frac >= thr.s1_constrained_acc_ax_pos_frac_min)
        and (ax_last >= thr.s1_constrained_acc_ax_last_min)
        and (vx_last > thr.s1_pickup_vx_last_max)
    ):
        z = sa_to_z(1, 1, A)
        return (z, record()) if debug else z

    # 6) s1 open-gap regulation (a2): leader often missing AND far dx if available
    if front_exists_frac <= thr.open_gap_front_exists_max:
        if (front_dx_min is None) or (front_dx_min >= thr.open_gap_front_dx_min):
            z = sa_to_z(1, 2, A)
            return (z, record()) if debug else z

    # 7) s1 brake-biased regulation (a0): mild braking tendency but not sustained braking
    sustained_brake_gate = (ax_neg_frac >= thr.brake_ax_neg_frac) and (ax_last <= thr.brake_ax_last)
    if (
        (ax_last <= thr.mild_brake_ax_last_max)
        and (ax_neg_frac >= thr.mild_brake_ax_neg_frac_min)
        and (not sustained_brake_gate)
    ):
        z = sa_to_z(1, 0, A)
        return (z, record()) if debug else z

    # 8) s0 adaptive follow regulation (a1): leader present, near steady, not tight
    if (
        (front_exists_frac >= thr.front_exists_gate)
        and (abs(vx_slope) <= thr.stable_follow_vx_slope_abs_max)
        and (abs(ax_last) <= thr.stable_follow_ax_last_abs_max)
        and (not is_tight)
    ):
        z = sa_to_z(0, 1, A)
        return (z, record()) if debug else z

    return (UNKNOWN_Z, record()) if debug else UNKNOWN_Z

# -----------------------------
# Utilities
# -----------------------------
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
def compute_gt_latents(obs_seq, feature_cols, thr=None, A=5, debug=False, fill_unknown="none"):
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

# -----------------------------
# Semantic map helpers
# -----------------------------
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

# -----------------------------
# Standalone debug runner
# -----------------------------
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
    if A != 5:
        print(f"[gt_labeler] Warning: trainer.A={A} but this script is tuned for A=5.")

    sem_map = _load_semantic_map_yaml(SEMANTIC_MAP) if SEMANTIC_MAP else None
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
            ])

        headers = [
            "t", "z", "s", "a", "semantic",
            "ax_last", "vx_last", "vx_slope",
            "ax_neg", "ax_pos", "front", "jerk_p95",
            "THW", "TTC", "lc_L", "lc_R",
        ]

        print("\n" + "=" * 120)
        print(f"DETAIL seq[{i}] T={len(out_post)}")
        print("=" * 120)

        print(tabulate(
            rows,
            headers=headers,
            tablefmt="simple",
            floatfmt=("", "", "", "", "", ".2f", ".2f", ".4f", ".2f", ".2f", ".2f", ".2f", ".2f", ".2f", ".2f", ".2f"),
            stralign="left",
            numalign="right",
            maxcolwidths=[None, None, None, None, 45, None, None, None, None, None, None, None, None, None, None, None],
            disable_numparse=True,
        ))


if __name__ == "__main__":
    main()
