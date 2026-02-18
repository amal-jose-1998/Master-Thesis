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

from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np
from tabulate import tabulate

UNKNOWN_Z = -1
EPS = 1e-8

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
    Numeric thresholds used by the rule-based labeling logic.

    Attributes
    lc_flag_thresh : float
        Lane-change presence flag threshold (probability/indicator) for firing LC rule.
    brake_ax_neg_frac : float
        Minimum fraction of samples in the window with negative longitudinal accel
        to consider the window "sustained braking".
    brake_ax_last : float
        Maximum allowed `ax_last` (more negative is stronger braking) to be considered braking.
    tactical_brake_ax_last : float
        Threshold to classify braking as "tactical/stronger" vs "mild".
    tactical_brake_vx_slope : float
        Additional tactical braking indicator using speed trend (negative slope).
    tactical_brake_thw : float
        Optional tactical braking indicator using time headway (smaller -> tighter).
    tactical_brake_ttc : float
        Optional tactical braking indicator using TTC (smaller -> higher urgency).
    strong_acc_ax_pos_frac : float
        Minimum fraction of positive accel samples to consider a window accel-dominant.
    strong_acc_ax_last : float
        Minimum `ax_last` to consider "strong acceleration".
    constrained_acc_ax_pos_frac : float
        Minimum positive accel fraction for "constrained acceleration".
    constrained_acc_ax_last : float
        Minimum `ax_last` for "constrained acceleration".
    constrained_acc_front_exists : float
        Minimum leader presence fraction to gate constrained acceleration.
    free_flow_front_exists_max : float
        Maximum leader presence fraction to consider free-flow (low interaction).
    constrained_follow_front_exists : float
        Minimum leader presence fraction to consider constrained following.
    constrained_follow_vx_last_max : float
        Maximum speed to consider constrained following (lower speeds imply congestion).
    constrained_follow_jerk_min : float
        Minimum jerk statistic to indicate stop-go / high variability following.
    stable_follow_front_exists : float
        Minimum leader presence fraction to consider stable following.
    stable_follow_vx_slope_abs_max : float
        Maximum absolute speed slope (near zero -> steady).
    stable_follow_ax_last_abs_max : float
        Maximum absolute accel value (near zero -> steady).
    """
    # Lane change
    lc_flag_thresh: float = 0.5

    # Sustained braking
    brake_ax_neg_frac: float = 0.90 # almost the whole window is braking
    brake_ax_last: float = -0.15 # it sits between the mild braking mean (−0.154) and strong braking mean (−0.321), and safely away from near-zero following

    # Split: tactical braking (s1/a3) vs mild braking (s0/a1)
    tactical_brake_ax_last: float = -0.25  # Directly separating the two braking means
    tactical_brake_vx_slope: float = -0.01 # sits between them and is within the global negative tail (p10 = −0.0137).
    tactical_brake_thw: float = 2.2 # below it looks like the tighter tactical braking.
    tactical_brake_ttc: float = 80.0

    # Strong accel (s1/a0)
    strong_acc_ax_pos_frac: float = 0.90 # isolates “accel-dominant window”
    strong_acc_ax_last: float = 0.15 # clearly positive acceleration but not too extreme.

    # Constrained accel (s0/a2)
    constrained_acc_ax_pos_frac: float = 0.65 # above typical but not extreme tail.
    constrained_acc_ax_last: float = 0.05 # positive enough to be accelerating
    constrained_acc_front_exists: float = 0.80 # a strong “leader present” gate.

    # Free-flow modulation (s0/a0)
    free_flow_front_exists_max: float = 0.60 # <0.60 => low interaction.

    # Constrained following (s1/a2)
    constrained_follow_front_exists: float = 0.80 # a strong “leader present” gate.
    constrained_follow_vx_last_max: float = 21.0 # low speed
    constrained_follow_jerk_min: float = 0.20 # sits between the “normal” cluster and the “high-jerk” cluster

    # Stable following (s0/a3)
    stable_follow_front_exists: float = 0.80 # a strong “leader present” gate.
    stable_follow_vx_slope_abs_max: float = 0.008 # approximately 2× the within-state std
    stable_follow_ax_last_abs_max: float = 0.20 # approximately 1× the within-state std (since ax_last is noisier and overlaps more across states than vx_slope)

# -----------------------------
# Label one window
# -----------------------------
def label_one_window_z(obs_t, feature_cols, thr, A=4, debug=False):
    """
    Assign a joint latent label z to a single window feature vector using rule thresholds.

    Parameters
    obs_t : array-like of shape (D,)
        Window feature vector at time/window index t.
    feature_cols : Sequence[str]
        Names for the D features, aligned with obs_t.
    thr : RuleThresholds
        Threshold configuration used by the rule set.
    A : int, default=4
        Number of actions per style. Used for (s,a)->z encoding.

    Returns
    int
        Joint label z in [0, 2*A - 1] if a rule fires, otherwise UNKNOWN_Z (-1).

    Raises
    KeyError
        If any required feature used by a rule is missing from `feature_cols`.

    Notes
    The rules are applied in priority order:
        1) lane change
        2) braking (with split mild vs tactical)
        3) strong acceleration
        4) constrained acceleration
        5) constrained following
        6) stable following
        7) free-flow modulation
        8) fallback based on leader presence or UNKNOWN
    """
    idx = _build_index(feature_cols) # Creates mapping {feature_name: column_index}

    # small helper: only build these if debug=True
    values = None
    def record(reason):
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
        return reason, values

    # Lane change
    lc_l = _get(obs_t, idx, "lc_left_present")
    lc_r = _get(obs_t, idx, "lc_right_present")
    if (lc_l > thr.lc_flag_thresh) or (lc_r > thr.lc_flag_thresh):
        z = sa_to_z(1, 1, A)
        if debug:
            reason, vals = record("lane_change")
            return z, reason, vals
        return z

    # Common longitudinal + interaction features
    ax_last = _get(obs_t, idx, "ax_last")
    vx_slope = _get(obs_t, idx, "vx_slope")
    vx_last = _get(obs_t, idx, "vx_last")

    ax_neg_frac = _get(obs_t, idx, "ax_neg_frac")
    ax_pos_frac = _get(obs_t, idx, "ax_pos_frac")

    front_exists_frac = _get(obs_t, idx, "front_exists_frac")
    jerk_x_p95 = _get(obs_t, idx, "jerk_x_p95")

    # Braking + tactical split
    is_brake = (
        (ax_neg_frac >= thr.brake_ax_neg_frac)
        and (ax_last <= thr.brake_ax_last)
        and (front_exists_frac >= thr.stable_follow_front_exists)  
    )
    if is_brake:
        # Split braking into s0/a1 vs s1/a3 (tactical tends to be stronger/tighter)
        tactical = (ax_last <= thr.tactical_brake_ax_last) or (vx_slope <= thr.tactical_brake_vx_slope)

        if not tactical:
            # only consult THW/TTC when primary indicators are not decisive
            if "front_thw_last" in idx:
                thw = float(obs_t[idx["front_thw_last"]])
                if thw <= thr.tactical_brake_thw:
                    tactical = True
            if "front_ttc_min" in idx:
                ttc = float(obs_t[idx["front_ttc_min"]])
                if ttc <= thr.tactical_brake_ttc:
                    tactical = True

        z = sa_to_z(1, 3, A) if tactical else sa_to_z(0, 1, A)
        if debug:
            reason, vals = record("tactical_brake" if tactical else "mild_brake")
            return z, reason, vals
        return z
    
    # Strong acceleration (s1/a0)
    if (ax_pos_frac >= thr.strong_acc_ax_pos_frac) and (ax_last >= thr.strong_acc_ax_last):
        z = sa_to_z(1, 0, A)
        if debug:
            reason, vals = record("strong_acc")
            return z, reason, vals
        return z

    # Constrained acceleration (s0/a2): accel + leader present
    if (
        (ax_pos_frac >= thr.constrained_acc_ax_pos_frac)
        and (ax_last >= thr.constrained_acc_ax_last)
        and (front_exists_frac >= thr.constrained_acc_front_exists)
    ):
        z = sa_to_z(0, 2, A)
        if debug:
            reason, vals = record("constrained_acc")
            return z, reason, vals
        return z

    # Constrained following (s1/a2): leader present + low speed + high jerk
    if (
        (front_exists_frac >= thr.constrained_follow_front_exists)
        and (vx_last <= thr.constrained_follow_vx_last_max)
        and (jerk_x_p95 >= thr.constrained_follow_jerk_min)
    ):
        z = sa_to_z(1, 2, A)
        if debug:
            reason, vals = record("constrained_follow")
            return z, reason, vals
        return z

    # Stable following (s0/a3): leader present + steady speed/acc
    if (
        (front_exists_frac >= thr.stable_follow_front_exists)
        and (abs(vx_slope) <= thr.stable_follow_vx_slope_abs_max)
        and (abs(ax_last) <= thr.stable_follow_ax_last_abs_max)
    ):
        z = sa_to_z(0, 3, A)
        if debug:
            reason, vals = record("stable_follow")
            return z, reason, vals
        return z

    # Free flow modulation (s0/a0): no strong leader evidence
    if front_exists_frac < thr.free_flow_front_exists_max:
        z = sa_to_z(0, 0, A)
        if debug:
            reason, vals = record("free_flow")
            return z, reason, vals
        return z

    # Leader exists but no rule fired -> closer to stable following
    #if front_exists_frac >= thr.stable_follow_front_exists:
    #    z = sa_to_z(0, 3, A)
    #    if debug:
    #        reason, vals = record("fallback_to_stable_follow")
    #        return z, reason, vals
    #    return z

    if debug:
        reason, vals = record("unknown")
        return UNKNOWN_Z, reason, vals
    return UNKNOWN_Z

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
        z, r, v = label_one_window_z(obs_seq[t], feature_cols, thr, A=A, debug=True)
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

EXP_DIR = r"/home/RUS_CIP/st184634/implementation/hdv/models/main-model-sticky_S2_A4_hierarchical"
DATA_ROOT = r"/home/RUS_CIP/st184634/implementation/hdv/data/highd"  
CHECKPOINT_NAME = "final.npz"
SEMANTIC_MAP = r"/home/RUS_CIP/st184634/implementation/hdv/models/main-model-sticky_S2_A4_hierarchical/semantic_map.yaml"

NUM_SEQS_SUMMARY = 20
DETAIL_FIRST_N = 20          # print detailed tables for first 5 trajectories
MAX_STEPS_DETAIL = 100

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

    thr = RuleThresholds()

    # 1) quick summary counts for first N sequences
    n = min(NUM_SEQS_SUMMARY, len(test.raw_obs))
    print(f"[gt_labeler] Loaded {len(test.raw_obs)} test sequences. Showing summary for first {n}.")
    for i in range(n):
        z = compute_gt_latents(test.raw_obs[i], test.feature_cols, thr=thr, A=A, debug=False, fill_unknown="nearest")
        uniq, cnt = np.unique(z, return_counts=True)
        pairs = sorted(zip(uniq.tolist(), cnt.tolist()), key=lambda x: -x[1])
        print(f"  seq[{i}] T={len(z)} label_counts(top10): {pairs[:10]}")

    # 2) detailed print for the first N trajectories (with reasons)
    m = min(int(DETAIL_FIRST_N), len(test.raw_obs))
    for i in range(m):
        out_post, values = compute_gt_latents(test.raw_obs[i], test.feature_cols, thr=thr, A=A, debug=True, fill_unknown="nearest")

        T_show = min(len(out_post), int(MAX_STEPS_DETAIL))

        rows = []
        for t in range(T_show):
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
            ])

        headers = [
            "t", "z", "s", "a", "semantic",
            "ax_last", "vx_last", "vx_slope",
            "ax_neg", "ax_pos", "front", "jerk_p95"
        ]

        print("\n" + "=" * 120)
        print(f"DETAIL seq[{i}] T={len(out_post)} (showing first {T_show})")
        print("=" * 120)

        print(tabulate(
            rows,
            headers=headers,
            tablefmt="simple",
            floatfmt=("", "", "", "", "", ".2f", ".2f", ".4f", ".2f", ".2f", ".2f", ".2f"),
            stralign="left",
            numalign="right",
            maxcolwidths=[None, None, None, None, 45, None, None, None, None, None, None, None],
            disable_numparse=True,
        ))


if __name__ == "__main__":
    main()
