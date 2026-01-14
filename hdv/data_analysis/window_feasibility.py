"""
Window feasibility + imbalance diagnostics for fixed-length training windows.

Purpose
Given per-vehicle trajectory dicts, this module answers:
1) How many trajectories are long enough for a window length W?
2) How many sliding windows exist (given stride)?
3) How many event-centered windows are valid (do not cross boundaries)?
4) How many LC vs KL windows exist under a consistent labeling rule:
     "LC window" iff it contains at least one lane-change event index.

Lane-change event source
- Use `lc` directly, where an event is any frame index t with lc[t] != 0
- If `lc` is not present, and a lane sequence is present (lane_id / laneId),
  detect events where lane[t] != lane[t-1].
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Config (no dataclass, no type annotations)
# -----------------------------
class WindowFeasibilityConfig:
    """
    Configuration for window feasibility diagnostics.

    Parameters
    window_len : int
        Fixed window length W in frames.
    stride : int
        Sliding stride in frames.
    min_gap_frames : int
        Debounce gap between detected events (frames).
    lc_key : str
        Preferred key in each trajectory dict containing lc values.
    lane_key : str
        Fallback key for lane sequence (if lc_key not found).
    """
    def __init__(self, window_len=150, stride=10, min_gap_frames=10, lc_key="lc", lane_key="lane_id"):
        self.window_len = int(window_len)
        self.stride = int(stride)
        self.min_gap_frames = int(min_gap_frames)
        self.lc_key = str(lc_key)
        self.lane_key = str(lane_key)


# -----------------------------
# Helpers
# -----------------------------
def _infer_T(tr, fallback_key="frame"):
    """
    Infer trajectory length T from a trajectory dict.
    Priority:
      1) tr["T"]
      2) len(tr[fallback_key]) if exists
      3) len(tr[cfg.lc_key]) or lane sequence if exists
    """
    if "T" in tr:
        try:
            return int(tr["T"])
        except Exception:
            pass

    if fallback_key in tr and tr[fallback_key] is not None:
        a = np.asarray(tr[fallback_key])
        if a.ndim == 1:
            return int(a.size)

    for k in ("lc", "lane_id", "laneId"):
        if k in tr and tr[k] is not None:
            a = np.asarray(tr[k])
            if a.ndim == 1:
                return int(a.size)

    return None


def _as_1d_array(x, dtype=None):
    if x is None:
        return None
    a = np.asarray(x)
    if a.ndim != 1:
        return None
    if dtype is None:
        return a
    return a.astype(dtype, copy=False)


def _debounce_indices(idxs, min_gap_frames):
    """
    Debounce a sorted list of indices: keep first, then require >= min_gap_frames.
    """
    if not idxs:
        return []
    idxs = sorted(int(t) for t in idxs)
    out = [idxs[0]]
    last = idxs[0]
    for t in idxs[1:]:
        if t - last >= int(min_gap_frames):
            out.append(t)
            last = t
    return out


def _detect_event_indices_from_lc(lc, min_gap_frames):
    """
    Event indices where lc != 0, debounced.
    """
    lc = _as_1d_array(lc, dtype=np.float64)
    if lc is None or lc.size == 0:
        return []
    s = np.round(lc)
    s = np.where(np.isfinite(s), s, 0.0)
    idxs = np.flatnonzero(s != 0.0).tolist()
    return _debounce_indices(idxs, min_gap_frames)


def _detect_event_indices_from_lane(lane, min_gap_frames):
    """
    Fallback: event indices t where lane[t] != lane[t-1], debounced.
    """
    lane = _as_1d_array(lane, dtype=np.int64)
    if lane is None or lane.size < 2:
        return []
    raw = (np.flatnonzero(lane[1:] != lane[:-1]) + 1).tolist()
    return _debounce_indices(raw, min_gap_frames)


def _sliding_window_starts(T, W, stride):
    if T < W:
        return np.zeros((0,), dtype=np.int64)
    n = 1 + (T - W) // stride
    return (np.arange(n, dtype=np.int64) * int(stride))


def _count_lc_windows_from_events(T, W, starts, event_idxs):
    """
    Label a sliding window as LC if it contains any event index.
    Uses prefix sums over an impulse array for speed.
    """
    if T < W or len(starts) == 0 or len(event_idxs) == 0:
        return 0

    mark = np.zeros((T,), dtype=np.int32)
    for t in event_idxs:
        t = int(t)
        if 0 <= t < T:
            mark[t] = 1
    pref = np.concatenate([[0], np.cumsum(mark, dtype=np.int64)])  # length T+1

    lc = 0
    for s in starts:
        s = int(s)
        e = s + int(W)
        if pref[e] - pref[s] > 0:
            lc += 1
    return int(lc)


# -----------------------------
# Main computation
# -----------------------------
def compute_window_feasibility(trajs, cfg):
    """
    Compute feasibility and imbalance statistics.

    Parameters
    trajs : list of dict
        Per-trajectory dicts.
    cfg : WindowFeasibilityConfig
        Configuration.

    Returns
    dict
        Summary statistics suitable for JSON output.
    """
    W = int(cfg.window_len)
    stride = int(cfg.stride)

    lengths = []
    lc_pos_norm = []

    n_trajs_total = 0
    n_trajs_usable = 0
    n_trajs_short = 0

    n_events_total = 0
    n_events_valid_centered = 0
    n_events_dropped_boundary = 0

    total_sliding_windows = 0
    total_lc_windows = 0
    total_kl_windows = 0

    half_left = W // 2
    half_right = W - half_left  # total = W

    for tr in trajs:
        n_trajs_total += 1

        T = _infer_T(tr)
        if T is None or T <= 0:
            continue
        T = int(T)
        lengths.append(T)

        if T < W:
            n_trajs_short += 1
            continue

        # Prefer lc-based events
        events = []
        if cfg.lc_key in tr and tr[cfg.lc_key] is not None:
            events = _detect_event_indices_from_lc(tr[cfg.lc_key], cfg.min_gap_frames)
        else:
            lane = tr.get(cfg.lane_key, None)
            if lane is None:
                lane = tr.get("laneId", None)
            if lane is not None:
                events = _detect_event_indices_from_lane(lane, cfg.min_gap_frames)

        n_trajs_usable += 1

        starts = _sliding_window_starts(T, W, stride)
        n_win = int(starts.size)
        total_sliding_windows += n_win

        n_events_total += int(len(events))

        # centered feasibility
        for t in events:
            t = int(t)
            lo = t - half_left
            hi = t + half_right
            if lo >= 0 and hi <= T:
                n_events_valid_centered += 1
                lc_pos_norm.append(float(t) / float(T))
            else:
                n_events_dropped_boundary += 1

        # LC vs KL windows
        n_lc_win = _count_lc_windows_from_events(T, W, starts, events)
        total_lc_windows += int(n_lc_win)
        total_kl_windows += int(n_win - n_lc_win)

    lengths_arr = np.asarray(lengths, dtype=np.int64)
    pos_arr = np.asarray(lc_pos_norm, dtype=float)

    def _q(x, p):
        if x.size == 0:
            return None
        return float(np.quantile(x, p))

    summary = {
        "config": {
            "window_len": int(W),
            "stride": int(stride),
            "min_gap_frames": int(cfg.min_gap_frames),
            "lc_key": str(cfg.lc_key),
            "lane_key_fallback": str(cfg.lane_key),
        },
        "trajectories": {
            "num_total": int(n_trajs_total),
            "num_usable_T_ge_W": int(n_trajs_usable),
            "num_shorter_than_W": int(n_trajs_short),
            "length_stats": {
                "count": int(lengths_arr.size),
                "min": int(lengths_arr.min()) if lengths_arr.size else None,
                "p10": _q(lengths_arr, 0.10),
                "median": _q(lengths_arr, 0.50),
                "p90": _q(lengths_arr, 0.90),
                "max": int(lengths_arr.max()) if lengths_arr.size else None,
            },
        },
        "lane_change_events": {
            "num_events_total": int(n_events_total),
            "num_events_valid_centered": int(n_events_valid_centered),
            "num_events_dropped_boundary": int(n_events_dropped_boundary),
            "valid_centered_rate": float(n_events_valid_centered / n_events_total) if n_events_total else None,
            "event_pos_norm_stats": {
                "count": int(pos_arr.size),
                "p10": _q(pos_arr, 0.10),
                "median": _q(pos_arr, 0.50),
                "p90": _q(pos_arr, 0.90),
            },
        },
        "windows": {
            "total_sliding_windows": int(total_sliding_windows),
            "total_lc_windows": int(total_lc_windows),
            "total_kl_windows": int(total_kl_windows),
            "lc_window_fraction": float(total_lc_windows / total_sliding_windows) if total_sliding_windows else None,
        },
    }

    return summary


def save_window_feasibility_report(out_dir, trajs, cfg):
    """
    Compute and save feasibility report outputs.

    Parameters
    out_dir : str or pathlib.Path
        Output directory.
    trajs : list of dict
        Per-trajectory dicts.
    cfg : WindowFeasibilityConfig
        Configuration.

    Returns
    dict
        Summary dictionary written to JSON.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = compute_window_feasibility(trajs, cfg)
    (out_dir / "window_feasibility_summary.json").write_text(json.dumps(summary, indent=2))

    # Plot 1: trajectory length histogram
    lengths = []
    for tr in trajs:
        T = _infer_T(tr)
        if T is not None and T > 0:
            lengths.append(int(T))
    lengths = np.asarray(lengths, dtype=np.int64)

    if lengths.size:
        fig = plt.figure()
        plt.hist(lengths, bins=80, edgecolor="black", alpha=0.75)
        plt.axvline(int(cfg.window_len), linestyle="--", linewidth=2.0, label=f"W={cfg.window_len}")
        plt.title("Trajectory length distribution (frames)")
        plt.xlabel("trajectory length T [frames]")
        plt.ylabel("count")
        plt.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "trajectory_length_hist.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Plot 2: normalized LC position histogram (valid centered only)
    pos = []
    W = int(cfg.window_len)
    half_left = W // 2
    half_right = W - half_left

    for tr in trajs:
        T = _infer_T(tr)
        if T is None or T < W:
            continue
        T = int(T)

        events = []
        if cfg.lc_key in tr and tr[cfg.lc_key] is not None:
            events = _detect_event_indices_from_lc(tr[cfg.lc_key], cfg.min_gap_frames)
        else:
            lane = tr.get(cfg.lane_key, None)
            if lane is None:
                lane = tr.get("laneId", None)
            if lane is not None:
                events = _detect_event_indices_from_lane(lane, cfg.min_gap_frames)

        for t in events:
            lo = int(t) - half_left
            hi = int(t) + half_right
            if lo >= 0 and hi <= T:
                pos.append(float(int(t)) / float(T))

    pos = np.asarray(pos, dtype=float)
    if pos.size:
        fig = plt.figure()
        plt.hist(pos, bins=60, edgecolor="black", alpha=0.75)
        plt.title("Lane-change event positions (normalized), valid centered only")
        plt.xlabel("t_lc / T")
        plt.ylabel("count")
        fig.tight_layout()
        fig.savefig(out_dir / "lc_event_position_hist.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Plot 3: feasibility counts bar
    tr = summary["trajectories"]
    ev = summary["lane_change_events"]
    wn = summary["windows"]

    fig = plt.figure(figsize=(8, 4))
    labels = [
        "traj_total",
        "traj_short(<W)",
        "LC_events_total",
        "LC_events_valid_center",
        "LC_events_dropped_boundary",
    ]
    vals = [
        int(tr["num_total"]),
        int(tr["num_shorter_than_W"]),
        int(ev["num_events_total"]),
        int(ev["num_events_valid_centered"]),
        int(ev["num_events_dropped_boundary"]),
    ]
    x = np.arange(len(labels))
    plt.bar(x, vals, edgecolor="black", alpha=0.8)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.title("Feasibility counts (trajectories + lane-change events)")
    plt.ylabel("count")
    fig.tight_layout()
    fig.savefig(out_dir / "window_counts_bar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Plot 4: LC vs KL window counts
    if int(wn["total_sliding_windows"]) > 0:
        fig = plt.figure(figsize=(6, 4))
        labels = ["LC windows", "KL windows"]
        vals = [int(wn["total_lc_windows"]), int(wn["total_kl_windows"])]
        plt.bar(labels, vals, edgecolor="black", alpha=0.8)
        plt.title(f"Sliding windows labeled by LC presence (W={cfg.window_len}, stride={cfg.stride})")
        plt.ylabel("count")
        fig.tight_layout()
        fig.savefig(out_dir / "lc_vs_kl_windows_bar.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    return summary
