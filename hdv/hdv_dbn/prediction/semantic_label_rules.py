from dataclasses import dataclass
import numpy as np

UNKNOWN_Z = -1

# -----------------------------
# (s,a) <-> z
# -----------------------------
def sa_to_z(s, a, A):
    return int(s) * int(A) + int(a)

def z_to_sa(z, A):
    z = int(z)
    return z // int(A), z % int(A)

# -----------------------------
# Feature access
# -----------------------------
def _build_index(feature_cols):
    return {c: i for i, c in enumerate(feature_cols)}

def _require(idx, name):
    if name not in idx:
        raise KeyError(f"Required feature '{name}' not found in feature_cols.")
    return idx[name]

def _get(obs_t, idx, name):
    j = _require(idx, name)
    return float(obs_t[j])

# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class RuleThresholds:
    # Lane change
    lc_flag_thresh: float = 0.5

    # Sustained braking
    brake_ax_neg_frac: float = 0.90 # almost the whole window is braking
    brake_ax_last: float = -0.20 # it sits between the mild braking mean (−0.154) and strong braking mean (−0.321), and safely away from near-zero following

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
    constrained_follow_vx_last_max = 21.0 # low speed
    constrained_follow_jerk_min: float = 0.20 # sits between the “normal” cluster and the “high-jerk” cluster

    # Stable following (s0/a3)
    stable_follow_front_exists: float = 0.80 # a strong “leader present” gate.
    stable_follow_vx_slope_abs_max: float = 0.007 # approximately 2× the within-state std
    stable_follow_ax_last_abs_max: float = 0.15 # approximately 1× the within-state std (since ax_last is noisier and overlaps more across states than vx_slope)

@dataclass(frozen=True)
class PersistenceConfig:
    persistence: int = 2  # min run length to accept
    short_run_to_unknown: bool = True


# -----------------------------
# Label one window
# -----------------------------
def label_one_window_z(obs_t, feature_cols, thr, A=4):
    idx = _build_index(feature_cols)

    # Lane change
    lc_l = _get(obs_t, idx, "lc_left_present")
    lc_r = _get(obs_t, idx, "lc_right_present")
    if (lc_l > thr.lc_flag_thresh) or (lc_r > thr.lc_flag_thresh):
        return sa_to_z(1, 1, A)  # s1/a1

    # Common longitudinal + interaction features
    ax_last = _get(obs_t, idx, "ax_last")
    vx_slope = _get(obs_t, idx, "vx_slope")
    vx_last = _get(obs_t, idx, "vx_last")

    ax_neg_frac = _get(obs_t, idx, "ax_neg_frac")
    ax_pos_frac = _get(obs_t, idx, "ax_pos_frac")

    front_exists_frac = _get(obs_t, idx, "front_exists_frac")
    jerk_x_p95 = _get(obs_t, idx, "jerk_x_p95")

    # Braking
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

        # these two are optional but useful if present
        # (keep as best-effort: only apply if feature exists)
        if "front_thw_last" in idx:
            thw = float(obs_t[idx["front_thw_last"]])
            if thw <= thr.tactical_brake_thw:
                tactical = True
        if "front_ttc_min" in idx:
            ttc = float(obs_t[idx["front_ttc_min"]])
            if ttc <= thr.tactical_brake_ttc:
                tactical = True

        return sa_to_z(1, 3, A) if tactical else sa_to_z(0, 1, A)

    # Strong acceleration (s1/a0)
    if (ax_pos_frac >= thr.strong_acc_ax_pos_frac) and (ax_last >= thr.strong_acc_ax_last):
        return sa_to_z(1, 0, A)

    # Constrained acceleration (s0/a2): accel + leader present
    if (
        (ax_pos_frac >= thr.constrained_acc_ax_pos_frac)
        and (ax_last >= thr.constrained_acc_ax_last)
        and (front_exists_frac >= thr.constrained_acc_front_exists)
    ):
        return sa_to_z(0, 2, A)

    # Constrained following (s1/a2): leader present + low speed + high jerk
    if (
        (front_exists_frac >= thr.constrained_follow_front_exists)
        and (vx_last <= thr.constrained_follow_vx_last_max)
        and (jerk_x_p95 >= thr.constrained_follow_jerk_min)
    ):
        return sa_to_z(1, 2, A)

    # Stable following (s0/a3): leader present + steady speed/acc
    if (
        (front_exists_frac >= thr.stable_follow_front_exists)
        and (abs(vx_slope) <= thr.stable_follow_vx_slope_abs_max)
        and (abs(ax_last) <= thr.stable_follow_ax_last_abs_max)
    ):
        return sa_to_z(0, 3, A)

    # Free flow modulation (s0/a0): no strong leader evidence
    if front_exists_frac < thr.free_flow_front_exists_max:
        return sa_to_z(0, 0, A)

    # Leader exists but no rule fired -> closer to stable following
    if front_exists_frac >= thr.stable_follow_front_exists:
        return sa_to_z(0, 3, A)

    return UNKNOWN_Z

# -----------------------------
# Persistence
# -----------------------------
def _run_lengths(x):
    runs = []
    if x.size == 0:
        return runs
    start = 0
    cur = int(x[0])
    for i in range(1, x.size):
        if int(x[i]) != cur:
            runs.append((cur, start, i - start))
            start = i
            cur = int(x[i])
    runs.append((cur, start, x.size - start))
    return runs


def apply_persistence(labels, cfg):
    labels = np.asarray(labels, dtype=int).copy()
    p = int(cfg.persistence)
    if p <= 1:
        return labels

    for val, start, length in _run_lengths(labels):
        if val == UNKNOWN_Z:
            continue
        if length < p:
            if cfg.short_run_to_unknown:
                labels[start:start + length] = UNKNOWN_Z
            else:
                prev_val = labels[start - 1] if start > 0 else UNKNOWN_Z
                next_val = labels[start + length] if (start + length) < labels.size else UNKNOWN_Z
                merge_val = prev_val if prev_val != UNKNOWN_Z else next_val
                labels[start:start + length] = merge_val
    return labels

# -----------------------------
# Main API
# -----------------------------
def compute_gt_latents(obs_seq, feature_cols, thr=None, persist=None, A=4):
    """
    obs_seq: (T,D) window features (prefer raw/unscaled for physical thresholds)
    returns: (T,) z in {0..7} or -1
    """
    if thr is None:
        thr = RuleThresholds()
    if persist is None:
        persist = PersistenceConfig()

    obs_seq = np.asarray(obs_seq)
    if obs_seq.ndim != 2:
        raise ValueError(f"obs_seq must be (T,D), got {obs_seq.shape}")

    T = obs_seq.shape[0]
    out = np.full((T,), UNKNOWN_Z, dtype=int)

    for t in range(T):
        out[t] = label_one_window_z(obs_seq[t], feature_cols, thr, A=A)

    return apply_persistence(out, persist)
