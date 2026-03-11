"""
Window feasibility + imbalance diagnostics for longitudinal events:
- braking
- accelerating

A window is labeled "brake" if it contains enough frames with ax < -a_brake
A window is labeled "accel" if it contains enough frames with ax > +a_accel
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class LongitudinalFeasibilityConfig:
    """
    Configuration for longitudinal window feasibility.

    Parameters
    window_len : int
        Window length W (frames).
    stride : int
        Sliding stride (frames).
    a_brake : float
        Braking threshold: ax < -a_brake.
    a_accel : float
        Acceleration threshold: ax > +a_accel.
    min_event_frames : int
        Minimum run length (consecutive frames) to register an event.
    min_gap_frames : int
        Debounce gap between detected events (frames).
    min_frames_in_window : int
        Window is labeled brake/accel if it contains >= this many frames
        satisfying the condition.
    """
    def __init__(
        self,
        window_len=100,
        stride=10,
        a_brake=0.5,
        a_accel=0.5,
        min_event_frames=5,
        min_gap_frames=10,
        min_frames_in_window=5,
    ):
        self.window_len = int(window_len)
        self.stride = int(stride)
        self.a_brake = float(a_brake)
        self.a_accel = float(a_accel)
        self.min_event_frames = int(min_event_frames)
        self.min_gap_frames = int(min_gap_frames)
        self.min_frames_in_window = int(min_frames_in_window)


def _infer_T(tr):
    if "T" in tr:
        try:
            return int(tr["T"])
        except Exception:
            return None
    ax = tr.get("ax", None)
    if ax is None:
        return None
    a = np.asarray(ax)
    if a.ndim == 1:
        return int(a.size)
    return None


def _sliding_starts(T, W, stride):
    if T < W:
        return np.zeros((0,), dtype=np.int64)
    n = 1 + (T - W) // stride
    return (np.arange(n, dtype=np.int64) * int(stride))


def _runs_to_event_centers(mask, min_len):
    """
    Given a boolean mask, find centers of contiguous True runs of length >= min_len.
    Returns integer indices (centers).
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []

    centers = []
    start = int(idx[0])
    prev = int(idx[0])

    for t in idx[1:]:
        t = int(t)
        if t == prev + 1:
            prev = t
        else:
            if (prev - start + 1) >= int(min_len):
                centers.append(int((start + prev) // 2))
            start = t
            prev = t

    if (prev - start + 1) >= int(min_len):
        centers.append(int((start + prev) // 2))

    return centers


def _debounce_events(events, min_gap):
    if not events:
        return []
    events = sorted(int(e) for e in events)
    out = [events[0]]
    last = events[0]
    for e in events[1:]:
        if e - last >= int(min_gap):
            out.append(e)
            last = e
    return out


def _count_presence_windows(T, W, starts, presence_mask, min_count):
    """
    Count windows where sum(presence_mask[s:s+W]) >= min_count.
    """
    if T < W or starts.size == 0:
        return 0
    presence = np.asarray(presence_mask, dtype=np.int32)
    pref = np.concatenate([[0], np.cumsum(presence, dtype=np.int64)])  # length T+1

    c = 0
    for s in starts:
        s = int(s)
        e = s + int(W)
        if (pref[e] - pref[s]) >= int(min_count):
            c += 1
    return int(c)


def compute_longitudinal_window_feasibility(trajs, cfg):
    """
    Compute longitudinal feasibility and imbalance statistics.

    Parameters
    trajs : list of dict
        Each trajectory dict must contain "ax" (1D).
    cfg : LongitudinalFeasibilityConfig
        Configuration.

    Returns
    dict
        Summary statistics suitable for JSON output.
    """
    W = int(cfg.window_len)
    stride = int(cfg.stride)

    n_trajs_total = 0
    n_trajs_short = 0
    n_trajs_missing_ax = 0

    brake_events_total = 0
    accel_events_total = 0
    brake_events_valid_center = 0
    accel_events_valid_center = 0
    brake_events_dropped_boundary = 0
    accel_events_dropped_boundary = 0

    total_windows = 0
    brake_windows = 0
    accel_windows = 0

    brake_pos_norm = []
    accel_pos_norm = []

    half_left = W // 2
    half_right = W - half_left

    for tr in trajs:
        n_trajs_total += 1

        T = _infer_T(tr)
        if T is None or T <= 0:
            continue
        T = int(T)

        ax = tr.get("ax", None)
        if ax is None:
            n_trajs_missing_ax += 1
            continue
        ax = np.asarray(ax, dtype=np.float64)
        if ax.ndim != 1 or ax.size != T:
            n_trajs_missing_ax += 1
            continue

        if T < W:
            n_trajs_short += 1
            continue

        starts = _sliding_starts(T, W, stride)
        total_windows += int(starts.size)

        brake_mask = ax < -float(cfg.a_brake)
        accel_mask = ax > +float(cfg.a_accel)

        brake_centers = _runs_to_event_centers(brake_mask, int(cfg.min_event_frames))
        accel_centers = _runs_to_event_centers(accel_mask, int(cfg.min_event_frames))

        brake_centers = _debounce_events(brake_centers, int(cfg.min_gap_frames))
        accel_centers = _debounce_events(accel_centers, int(cfg.min_gap_frames))

        brake_events_total += len(brake_centers)
        accel_events_total += len(accel_centers)

        for t in brake_centers:
            lo = int(t) - half_left
            hi = int(t) + half_right
            if lo >= 0 and hi <= T:
                brake_events_valid_center += 1
                brake_pos_norm.append(float(t) / float(T))
            else:
                brake_events_dropped_boundary += 1

        for t in accel_centers:
            lo = int(t) - half_left
            hi = int(t) + half_right
            if lo >= 0 and hi <= T:
                accel_events_valid_center += 1
                accel_pos_norm.append(float(t) / float(T))
            else:
                accel_events_dropped_boundary += 1

        brake_windows += _count_presence_windows(T, W, starts, brake_mask, int(cfg.min_frames_in_window))
        accel_windows += _count_presence_windows(T, W, starts, accel_mask, int(cfg.min_frames_in_window))

    def _stats(x):
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return {"count": 0, "p10": None, "median": None, "p90": None}
        return {
            "count": int(x.size),
            "p10": float(np.quantile(x, 0.10)),
            "median": float(np.quantile(x, 0.50)),
            "p90": float(np.quantile(x, 0.90)),
        }

    summary = {
        "config": {
            "window_len": int(W),
            "stride": int(stride),
            "a_brake": float(cfg.a_brake),
            "a_accel": float(cfg.a_accel),
            "min_event_frames": int(cfg.min_event_frames),
            "min_gap_frames": int(cfg.min_gap_frames),
            "min_frames_in_window": int(cfg.min_frames_in_window),
        },
        "trajectories": {
            "num_total": int(n_trajs_total),
            "num_shorter_than_W": int(n_trajs_short),
            "num_missing_or_bad_ax": int(n_trajs_missing_ax),
        },
        "brake_events": {
            "num_events_total": int(brake_events_total),
            "num_events_valid_centered": int(brake_events_valid_center),
            "num_events_dropped_boundary": int(brake_events_dropped_boundary),
            "valid_centered_rate": float(brake_events_valid_center / brake_events_total) if brake_events_total else None,
            "event_pos_norm_stats": _stats(brake_pos_norm),
        },
        "accel_events": {
            "num_events_total": int(accel_events_total),
            "num_events_valid_centered": int(accel_events_valid_center),
            "num_events_dropped_boundary": int(accel_events_dropped_boundary),
            "valid_centered_rate": float(accel_events_valid_center / accel_events_total) if accel_events_total else None,
            "event_pos_norm_stats": _stats(accel_pos_norm),
        },
        "windows": {
            "total_sliding_windows": int(total_windows),
            "brake_windows": int(brake_windows),
            "accel_windows": int(accel_windows),
            "brake_window_fraction": float(brake_windows / total_windows) if total_windows else None,
            "accel_window_fraction": float(accel_windows / total_windows) if total_windows else None,
        },
    }
    return summary


def save_longitudinal_window_feasibility_report(out_dir, trajs, cfg):
    """
    Compute and save longitudinal feasibility report outputs.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = compute_longitudinal_window_feasibility(trajs, cfg)
    (out_dir / "longitudinal_window_feasibility_summary.json").write_text(json.dumps(summary, indent=2))

    # Recompute normalized event positions for plotting (keeps JSON compact)
    brake_pos = []
    accel_pos = []
    W = int(cfg.window_len)
    half_left = W // 2
    half_right = W - half_left

    for tr in trajs:
        T = _infer_T(tr)
        if T is None or T < W:
            continue
        ax = tr.get("ax", None)
        if ax is None:
            continue
        ax = np.asarray(ax, dtype=np.float64)
        if ax.ndim != 1 or ax.size != int(T):
            continue

        brake_mask = ax < -float(cfg.a_brake)
        accel_mask = ax > +float(cfg.a_accel)

        bc = _debounce_events(_runs_to_event_centers(brake_mask, int(cfg.min_event_frames)), int(cfg.min_gap_frames))
        ac = _debounce_events(_runs_to_event_centers(accel_mask, int(cfg.min_event_frames)), int(cfg.min_gap_frames))

        for t in bc:
            lo = int(t) - half_left
            hi = int(t) + half_right
            if lo >= 0 and hi <= int(T):
                brake_pos.append(float(t) / float(T))

        for t in ac:
            lo = int(t) - half_left
            hi = int(t) + half_right
            if lo >= 0 and hi <= int(T):
                accel_pos.append(float(t) / float(T))

    if brake_pos:
        fig = plt.figure()
        plt.hist(brake_pos, bins=60, edgecolor="black", alpha=0.75)
        plt.title("Brake event positions (normalized), valid centered only")
        plt.xlabel("t_event / T")
        plt.ylabel("count")
        fig.tight_layout()
        fig.savefig(out_dir / "brake_event_positions_norm.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    if accel_pos:
        fig = plt.figure()
        plt.hist(accel_pos, bins=60, edgecolor="black", alpha=0.75)
        plt.title("Accel event positions (normalized), valid centered only")
        plt.xlabel("t_event / T")
        plt.ylabel("count")
        fig.tight_layout()
        fig.savefig(out_dir / "accel_event_positions_norm.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    wn = summary["windows"]
    if int(wn["total_sliding_windows"]) > 0:
        fig = plt.figure(figsize=(7, 4))
        labels = ["Brake windows", "Accel windows"]
        vals = [int(wn["brake_windows"]), int(wn["accel_windows"])]
        plt.bar(labels, vals, edgecolor="black", alpha=0.85)
        plt.title(f"Windows with longitudinal events (W={cfg.window_len}, stride={cfg.stride})")
        plt.ylabel("count")
        fig.tight_layout()
        fig.savefig(out_dir / "longitudinal_event_windows_counts.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    return summary
