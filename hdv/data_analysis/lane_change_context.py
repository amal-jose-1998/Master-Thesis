from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Helpers: key inference
# -----------------------------
def _pick_key(tr, candidates):
    for k in candidates:
        if k in tr and tr[k] is not None:
            return k
    return None

def _as_1d_float(x):
    if x is None:
        return None
    a = np.asarray(x)
    if a.ndim != 1:
        return None
    return a.astype(float, copy=False)

def _as_1d_int(x):
    if x is None:
        return None
    a = np.asarray(x)
    if a.ndim != 1:
        return None
    # lane ids can be floats in CSV; coerce safely
    return a.astype(np.int64, copy=False)

# -----------------------------
# Data containers
# -----------------------------
@dataclass(frozen=True)
class LaneChangeContextConfig:
    """
    Configuration for lane-change context extraction.

    Parameters
    pre_frames : int
        Number of frames before the lane-change index used as "pre" window.
    post_frames : int
        Number of frames after the lane-change index used as "post" window.
    min_gap_frames : int
        Minimum separation between detected lane changes (to avoid double-counting
        jitter). If two changes occur within this gap, keep the first.
    """
    pre_frames: int = 25
    post_frames: int = 25
    min_gap_frames: int = 10


@dataclass(frozen=True)
class LaneChangeContextStats:
    """
    Outputs of lane-change context analysis.

    Attributes
    config : LaneChangeContextConfig
        Configuration used.
    event_table : pd.DataFrame
        One row per lane-change event with pre/post means and deltas.
    summary : dict
        Aggregated summaries overall and by direction.
    """
    config: LaneChangeContextConfig
    event_table: pd.DataFrame
    summary: Dict[str, Any]

    def to_dict(self):
        return {
            "config": {
                "pre_frames": int(self.config.pre_frames),
                "post_frames": int(self.config.post_frames),
                "min_gap_frames": int(self.config.min_gap_frames),
            },
            "summary": self.summary,
            "event_table": self.event_table.to_dict(orient="records"),
        }


# -----------------------------
# Core: lane-change detection + context features
# -----------------------------
def detect_lane_change_indices(lane_seq, min_gap_frames=10):
    """
    Detect lane-change indices from a lane id sequence.

    Parameters
    lane_seq : np.ndarray, shape (T,)
        Integer lane id per frame.
    min_gap_frames : int
        Suppress events that occur too close to the previous accepted event.

    Returns
    idxs : list[int]
        Event indices in ascending order.
    """
    if lane_seq.size < 2:
        return []
    # change points
    raw = np.nonzero(lane_seq[1:] != lane_seq[:-1])[0] + 1
    if raw.size == 0:
        return []
    # gap suppression
    idxs = [int(raw[0])]
    last = int(raw[0])
    for t in raw[1:]:
        t = int(t)
        if t - last >= int(min_gap_frames):
            idxs.append(t)
            last = t
    return idxs


def _window_mean(x, lo, hi):
    if x is None:
        return None
    if hi <= lo:
        return None
    seg = x[lo:hi]
    seg = seg[np.isfinite(seg)]
    if seg.size == 0:
        return None
    return float(seg.mean())


def compute_lane_change_context(trajs, config=None, lane_key="lane_id", frame_key="frame", vx_key=None, vy_key=None, ax_key=None, ay_key=None):
    """
    Compute before/after kinematic context for each lane-change event.

    Parameters
    trajs : sequence of dict
        Per-trajectory dictionaries (one per vehicle). 
    config : LaneChangeContextConfig | None
        Window sizes and suppression settings. Defaults to LaneChangeContextConfig().
    lane_key : str
        Key name for lane id array in trajectory dict (default: 'lane_id').
    frame_key : str
        Key name for frame array (optional but used for event frame reporting).
    vx_key, vy_key, ax_key, ay_key : str | None
        If None, keys are inferred from common candidates.

    Returns
    LaneChangeContextStats
        - event_table: one row per lane-change event
        - summary: aggregated means/medians and counts, overall and by direction

    """
    if config is None:
        config = LaneChangeContextConfig()

    rows = []

    for tr in trajs:
        lane_arr = _as_1d_int(tr.get(lane_key, None))
        if lane_arr is None:
            # try highD raw key if not already aliased
            lane_arr = _as_1d_int(tr.get("laneId", None))
        if lane_arr is None:
            continue

        T = int(lane_arr.size)
        if T < 2:
            continue

        # keys (infer if not given)
        vxk = vx_key or _pick_key(tr, ("vx", "xVelocity"))
        vyk = vy_key or _pick_key(tr, ("vy", "yVelocity"))
        axk = ax_key or _pick_key(tr, ("ax", "xAcceleration"))
        ayk = ay_key or _pick_key(tr, ("ay", "yAcceleration"))
        frk = frame_key if frame_key in tr else None

        vx = _as_1d_float(tr.get(vxk)) if vxk else None
        vy = _as_1d_float(tr.get(vyk)) if vyk else None
        ax = _as_1d_float(tr.get(axk)) if axk else None
        ay = _as_1d_float(tr.get(ayk)) if ayk else None
        fr = _as_1d_int(tr.get(frk)) if frk else None

        idxs = detect_lane_change_indices(lane_arr, min_gap_frames=config.min_gap_frames)
        if not idxs:
            continue

        rec_id = tr.get("recording_id", None)
        veh_id = tr.get("vehicle_id", tr.get("id", None))

        dd_raw = tr.get("drivingDirection", None)
        try:
            dd = int(dd_raw) if dd_raw is not None and not pd.isna(dd_raw) else None
        except Exception:
            dd = None

        for t in idxs:
            lane_from = int(lane_arr[t - 1])
            lane_to = int(lane_arr[t])

            # Direction heuristic: if lane numbers increase to the right, then:
            # - lane_to > lane_from => right
            # - lane_to < lane_from => left
            dd = tr.get("drivingDirection", None)

            if dd == 1:
                # upper lanes
                if lane_to > lane_from:
                    direction = "right"
                elif lane_to < lane_from:
                    direction = "left"
                else:
                    direction = "unknown"

            elif dd == 2:
                # lower lanes (mirrored)
                if lane_to > lane_from:
                    direction = "left"
                elif lane_to < lane_from:
                    direction = "right"
                else:
                    direction = "unknown"
            else:
                direction = "unknown"


            pre_lo = max(0, t - int(config.pre_frames))
            pre_hi = t
            post_lo = t
            post_hi = min(T, t + int(config.post_frames))

            row: Dict[str, Any] = {
                "recording_id": rec_id,
                "vehicle_id": veh_id,
                "drivingDirection": dd,
                "event_index": int(t),
                "event_frame": int(fr[t]) if fr is not None and t < fr.size else None,
                "lane_from": lane_from,
                "lane_to": lane_to,
                "direction": direction,
                "pre_len": int(pre_hi - pre_lo),
                "post_len": int(post_hi - post_lo),
            }
            # Pre/post means + deltas
            for name, sig in (("vx", vx), ("vy", vy), ("ax", ax), ("ay", ay)):
                pre_m = _window_mean(sig, pre_lo, pre_hi)
                post_m = _window_mean(sig, post_lo, post_hi)
                row[f"{name}_pre_mean"] = pre_m
                row[f"{name}_post_mean"] = post_m
                row[f"{name}_delta"] = (post_m - pre_m) if (pre_m is not None and post_m is not None) else None

            rows.append(row)

    event_table = pd.DataFrame(rows)

    # Summary aggregation
    summary: Dict[str, Any] = {
        "num_events": int(len(event_table)),
        "by_side": {},
        "by_drivingDirection": {},
        "by_dd_and_side_counts": {},
    }

    def agg_block(df: pd.DataFrame) -> Dict[str, Any]:
        out = {"count": int(len(df))}
        for name in ("vx_delta", "vy_delta", "ax_delta", "ay_delta"):
            if name in df.columns:
                s = pd.to_numeric(df[name], errors="coerce").dropna()
                out[name] = {
                    "mean": float(s.mean()) if len(s) else None,
                    "median": float(s.median()) if len(s) else None,
                    "p10": float(s.quantile(0.10)) if len(s) else None,
                    "p90": float(s.quantile(0.90)) if len(s) else None,
                }
        return out

    if not event_table.empty:
        summary["overall"] = agg_block(event_table)

        for side in ["left", "right", "unknown"]:
            summary["by_side"][side] = agg_block(event_table[event_table["direction"] == side])

        dd_series = pd.to_numeric(event_table["drivingDirection"], errors="coerce")
        for dd_val in [1, 2]:
            summary["by_drivingDirection"][str(dd_val)] = agg_block(event_table[dd_series == dd_val])

        # 2x2 counts: traffic flow (dd) x lane-change side
        for dd_val in [1, 2]:
            dd_key = str(dd_val)
            summary["by_dd_and_side_counts"][dd_key] = {}
            for side in ["left", "right", "unknown"]:
                m = (dd_series == dd_val) & (event_table["direction"] == side)
                summary["by_dd_and_side_counts"][dd_key][side] = int(m.sum())

    return LaneChangeContextStats(config=config, event_table=event_table, summary=summary)

# -----------------------------
# Plotting
# -----------------------------
def save_lane_change_context_plots(out_dir, stats):
    """
    Save plots with counts printed on the figure.

    Outputs:
      1) lanechange_counts_by_side.png
      2) lanechange_counts_by_side_and_drivingDirection.png

    Parameters
    out_dir : str | Path
        Output directory.
    stats : LaneChangeContextStats
        Lane-change context statistics.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = stats.event_table
    if df is None or len(df) == 0 or "direction" not in df.columns:
        return

    # Normalize columns
    df = df.copy()
    df["direction"] = df["direction"].astype(str)
    if "drivingDirection" in df.columns:
        df["drivingDirection"] = pd.to_numeric(df["drivingDirection"], errors="coerce")

    sides = ["left", "right", "unknown"]

    # -----------------------------
    # Plot A: overall side counts
    # -----------------------------
    counts = [int((df["direction"] == s).sum()) for s in sides]
    total = int(len(df))

    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()

    x = np.arange(len(sides))
    bars = ax.bar(x, counts)
    ax.set_xticks(x, sides)
    ax.set_ylabel("Number of lane-change events")
    ax.set_title(f"Lane-change counts by side (total={total})")

    # annotate bars
    for i, c in enumerate(counts):
        ax.text(i, c, str(c), ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_dir / "lanechange_counts_by_side.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # -----------------------------
    # Plot B: side counts split by drivingDirection
    # -----------------------------
    if "drivingDirection" not in df.columns or df["drivingDirection"].isna().all():
        return

    dd1 = df["drivingDirection"] == 1
    dd2 = df["drivingDirection"] == 2

    c1 = [int(((df["direction"] == s) & dd1).sum()) for s in sides]
    c2 = [int(((df["direction"] == s) & dd2).sum()) for s in sides]

    fig = plt.figure(figsize=(7, 4))
    ax = plt.gca()

    width = 0.35
    x = np.arange(len(sides))
    ax.bar(x - width/2, c1, width=width, label="drivingDirection=1")
    ax.bar(x + width/2, c2, width=width, label="drivingDirection=2")

    ax.set_xticks(x, sides)
    ax.set_ylabel("Number of lane-change events")
    ax.set_title("Lane-change counts by side split by traffic direction")
    ax.legend()

    # annotate grouped bars
    for i in range(len(sides)):
        ax.text(x[i] - width/2, c1[i], str(c1[i]), ha="center", va="bottom", fontsize=9)
        ax.text(x[i] + width/2, c2[i], str(c2[i]), ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / "lanechange_counts_by_side_and_drivingDirection.png", dpi=200, bbox_inches="tight")
    plt.close(fig)