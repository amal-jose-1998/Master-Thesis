"""
lane_pose_analysis.py

Lane-pose reporting and debug plotting utilities for highD.

This module provides:
1) Lane-pose report tables (CSV) and stacked-fraction plots (PNG).
2) A debug plot to visualize lane markings + one vehicle trajectory, with a
   road-like background:
     - Drivable road lanes: grey
     - Median strip: green

Lane pose convention (vehicle-centric)
--------------------------------------
lane_pos in {-1, 0, 1, 2, 3, 4}

  -1 = unknown / cannot determine (missing markings, invalid direction, NaN y, etc.)
   0 = left violation   (beyond left drivable boundary, vehicle-centric)
   1 = leftmost lane
   2 = middle lane(s)
   3 = rightmost lane
   4 = right violation  (beyond right drivable boundary, vehicle-centric)

Notes
-----
- For visualization consistency with aerial screenshots, the debug plot can invert
  the y-axis (invert_y=True).
- Debug highlighting is driven by the lane_pos values already computed in your
  dataframe. No geometric fallback is used.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# Report config
# =========================================================

class LanePoseReportConfig:
    """
    Configuration for lane-pose reporting.

    Parameters
    ----------
    lane_pos_col
        Column name holding lane pose values.
    class_col
        Column name holding vehicle class labels.
    direction_col
        Column name holding driving direction labels.
    recording_col
        Column name holding recording identifier.
    vehicle_col
        Column name holding vehicle identifier.
    include_lane_pos
        Lane pose categories to keep for reporting.
    count_unique_vehicles
        If True: aggregate per vehicle (mode lane pose across frames).
        If False: count frames/rows.
    dpi
        DPI for plots.

    Notes
    -----
    - If you count unique vehicles, each vehicle contributes exactly one label,
      computed as the mode lane_pos across frames (ties -> smallest lane_pos).
    """
    def __init__(
        self,
        lane_pos_col="lane_pos",
        class_col="meta_class",
        direction_col="meta_drivingDirection",
        recording_col="recording_id",
        vehicle_col="vehicle_id",
        include_lane_pos=(-1, 0, 1, 2, 3, 4),  
        count_unique_vehicles=False,
        dpi=160,
    ):
        self.lane_pos_col = lane_pos_col
        self.class_col = class_col
        self.direction_col = direction_col
        self.recording_col = recording_col
        self.vehicle_col = vehicle_col
        self.include_lane_pos = tuple(include_lane_pos)
        self.count_unique_vehicles = bool(count_unique_vehicles)
        self.dpi = int(dpi)


# =========================================================
# Small helpers
# =========================================================

def _ensure_cols(df, cols):
    """
    Validate required columns exist.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _base_filtered_df(df, cfg):
    """
    Filter to included lane_pos values and drop NA in key columns.
    """
    _ensure_cols(df, [cfg.lane_pos_col, cfg.class_col, cfg.direction_col])

    out = df[df[cfg.lane_pos_col].isin(cfg.include_lane_pos)].copy()
    out = out.dropna(subset=[cfg.lane_pos_col, cfg.class_col, cfg.direction_col])
    return out


def _vehicle_key(df, cfg):
    """
    Compute a unique vehicle key per row.

    Prefer (recording_id, vehicle_id) if both exist, else fall back to vehicle_id.
    """
    if cfg.recording_col in df.columns and cfg.vehicle_col in df.columns:
        return df[cfg.recording_col].astype(str) + "::" + df[cfg.vehicle_col].astype(str)
    if cfg.vehicle_col in df.columns:
        return df[cfg.vehicle_col].astype(str)
    raise ValueError("Need either (recording_id and vehicle_id) or vehicle_id to count unique vehicles.")


def _vehicle_lane_pose_mode(df, cfg):
    """
    Reduce to one row per vehicle by assigning lane_pos = mode over frames.

    Tie-break: choose the smallest lane_pos among tied modes.
    """
    vk = _vehicle_key(df, cfg)
    tmp = df[[cfg.lane_pos_col, cfg.class_col, cfg.direction_col]].copy()
    tmp["vehicle_key"] = vk.values

    def mode_tie_lowest(x):
        vals = x.to_numpy()
        uniq, cnt = np.unique(vals, return_counts=True)
        maxc = cnt.max()
        return int(np.min(uniq[cnt == maxc]))

    agg = tmp.groupby("vehicle_key", sort=False).agg(
        lane_pos=(cfg.lane_pos_col, mode_tie_lowest),
        meta_class=(cfg.class_col, "first"),
        meta_drivingDirection=(cfg.direction_col, "first"),
    )
    return agg.reset_index(drop=False)


def _count_table(df, cfg, group_cols):
    """
    Build counts and fractions table for lane_pos categories.
    """
    if cfg.count_unique_vehicles:
        base = _vehicle_lane_pose_mode(df, cfg)
        pivot = base.pivot_table(
            index=list(group_cols),
            columns="lane_pos",
            values="vehicle_key",
            aggfunc="count",
            fill_value=0,
        )
    else:
        base = df.copy()
        base["__rows__"] = 1
        pivot = base.pivot_table(
            index=list(group_cols),
            columns=cfg.lane_pos_col,
            values="__rows__",
            aggfunc="sum",
            fill_value=0,
        )

    pivot = pivot.reindex(columns=list(cfg.include_lane_pos), fill_value=0).sort_index()
    pivot["total"] = pivot.sum(axis=1)

    for k in cfg.include_lane_pos:
        pivot[f"frac_{k}"] = pivot[k] / pivot["total"].replace(0, np.nan)

    return pivot.reset_index()


def _lane_pos_label(k):
    """
    Human-readable label for legends and plots.
    """
    k = int(k)
    if k == -1:
        return "unknown (-1)"
    if k == 0:
        return "left violation (0)"
    if k == 1:
        return "leftmost lane (1)"
    if k == 2:
        return "middle lane(s) (2)"
    if k == 3:
        return "rightmost lane (3)"
    if k == 4:
        return "right violation (4)"
    return f"lane_pos={k}"


def _plot_stacked_fraction(table, cfg, group_cols, out_path, title):
    """
    Plot stacked fraction bars for lane_pos categories.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if group_cols:
        x_labels = table[group_cols].astype(str).agg(" | ".join, axis=1).tolist()
    else:
        x_labels = ["all"]

    x = np.arange(len(x_labels))

    plt.figure(figsize=(max(8, 0.6 * len(x_labels)), 4.5))
    bottom = np.zeros(len(x_labels), dtype=float)

    for k in cfg.include_lane_pos:
        frac = table[f"frac_{k}"].fillna(0.0).to_numpy()
        plt.bar(x, frac, bottom=bottom, label=_lane_pos_label(k))  # MOD: readable labels
        bottom += frac

    plt.xticks(x, x_labels, rotation=30, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("fraction")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.dpi)
    plt.close()


def run_lane_pose_report(df, out_dir, cfg=None):
    """
    Write lane-pose summary tables and fraction plots.
    """
    cfg = cfg or LanePoseReportConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = _base_filtered_df(df, cfg)

    overall = _count_table(base, cfg, group_cols=[])
    overall.to_csv(out_dir / "lane_pose_overall.csv", index=False)

    by_dir = _count_table(base, cfg, group_cols=[cfg.direction_col])
    by_dir.to_csv(out_dir / "lane_pose_by_direction.csv", index=False)
    _plot_stacked_fraction(
        by_dir,
        cfg,
        [cfg.direction_col],
        out_dir / "lane_pose_by_direction_fractions.png",
        "Lane pose fractions by driving direction",
    )

    by_class = _count_table(base, cfg, group_cols=[cfg.class_col])
    by_class.to_csv(out_dir / "lane_pose_by_class.csv", index=False)
    _plot_stacked_fraction(
        by_class,
        cfg,
        [cfg.class_col],
        out_dir / "lane_pose_by_class_fractions.png",
        "Lane pose fractions by vehicle class",
    )

    by_cd = _count_table(base, cfg, group_cols=[cfg.class_col, cfg.direction_col])
    by_cd.to_csv(out_dir / "lane_pose_by_class_and_direction.csv", index=False)
    _plot_stacked_fraction(
        by_cd,
        cfg,
        [cfg.class_col, cfg.direction_col],
        out_dir / "lane_pose_by_class_and_direction_fractions.png",
        "Lane pose fractions by (class × direction)",
    )

    print(f"[lane_pose_analysis] Saved lane pose report to: {out_dir}")


# =========================================================
# Debug plotting (lane markings + trajectory)
# =========================================================

class HighDLaneMarkings:
    """
    Parsed lane markings from recordingMeta.

    Parameters
    ----------
    lower
        Lower carriageway lane markings (sorted). Typically direction=2.
    upper
        Upper carriageway lane markings (sorted). Typically direction=1.
    """
    def __init__(self, lower, upper):
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)

    @property
    def all(self):
        return np.concatenate([self.lower, self.upper])


def _parse_markings_semicolon(value):
    """
    Parse highD markings strings like "0.0;3.5;7.0;...".

    Returns
    -------
    np.ndarray
        Sorted float array. Empty array if unusable.
    """
    if value is None:
        return np.array([], dtype=np.float64)
    if isinstance(value, float) and np.isnan(value):
        return np.array([], dtype=np.float64)

    s = str(value).strip()
    if not s:
        return np.array([], dtype=np.float64)

    parts = [p.strip() for p in s.split(";") if p.strip()]
    arr = np.array([float(p) for p in parts], dtype=np.float64)

    arr = arr[np.isfinite(arr)]
    arr = np.unique(arr)

    arr.sort()
    return arr


def load_highd_lane_markings(recording_meta_df, row_idx=0,
                            upper_col="upperLaneMarkings",
                            lower_col="lowerLaneMarkings"):
    """
    Load lane markings from a recordingMeta table.
    Assumes recording_meta_df contains exactly one row.
    """
    _ensure_cols(recording_meta_df, [upper_col, lower_col])
    row = recording_meta_df.iloc[0]

    upper = _parse_markings_semicolon(row[upper_col])
    lower = _parse_markings_semicolon(row[lower_col])

    if upper.size == 0 and lower.size == 0:
        raise ValueError("No lane markings found (upper and lower are both empty).")

    return HighDLaneMarkings(lower=lower, upper=upper)


def _split_drivable_spans(markings):
    """
    Build drivable lane spans for each carriageway and the median span.
    """
    lower = np.sort(np.asarray(markings.lower, dtype=np.float64))
    upper = np.sort(np.asarray(markings.upper, dtype=np.float64))

    lower_spans = [(float(lower[i]), float(lower[i + 1])) for i in range(max(0, len(lower) - 1))]
    upper_spans = [(float(upper[i]), float(upper[i + 1])) for i in range(max(0, len(upper) - 1))]

    median_span = None
    if lower.size > 0 and upper.size > 0:
        y0 = float(np.max(lower))
        y1 = float(np.min(upper))
        if y1 > y0:
            median_span = (y0, y1)

    return lower_spans, upper_spans, median_span


def _draw_road_background(ax, markings, road_alpha=0.22, median_alpha=0.22):
    """
    Draw drivable road (grey) and median (green) using lane markings.
    """
    lower_spans, upper_spans, median_span = _split_drivable_spans(markings)

    for y0, y1 in lower_spans:
        ax.axhspan(y0, y1, color="grey", alpha=road_alpha, zorder=0)
    for y0, y1 in upper_spans:
        ax.axhspan(y0, y1, color="grey", alpha=road_alpha, zorder=0)

    if median_span is not None:
        y0, y1 = median_span
        ax.axhspan(y0, y1, color="green", alpha=median_alpha, zorder=0)
        ax.axhline((y0 + y1) * 0.5, color="green", alpha=0.6, linewidth=1.2, zorder=1)


def plot_lane_pose_debug(
    tracks_df,
    recording_meta_df,
    vehicle_id,
    out_path,
    recording_id=None,
    id_col="id",
    x_col="x_center",
    y_col="y_center",
    frame_col="frame",
    upper_col="upperLaneMarkings",
    lower_col="lowerLaneMarkings",
    invert_y=True,
    draw_road=True,
    draw_markings=True,
    time_colormap=False,
    dpi=160,
    figsize=(12.0, 4.5),
    title=None,
    lane_pos_col="lane_pos",      
    highlight_values=(0, 4),     
    highlight_unknown=True,       
    highlight_size=18.0,
):
    """
    Save a lane markings + trajectory debug plot for one vehicle.

    Visual conventions
    ------------------
    - Drivable road lanes: grey
    - Median strip: green
    - Lane markings: dashed lines
    - Trajectory: blue line
    - lane_pos in highlight_values (default 0,4): red points
    - lane_pos == -1: orange points (optional)
    """
    if tracks_df is None or len(tracks_df) == 0:
        raise ValueError("tracks_df is empty.")
    if recording_meta_df is None or len(recording_meta_df) == 0:
        raise ValueError("recording_meta_df is empty.")

    veh = tracks_df[tracks_df[id_col] == vehicle_id].copy()
    if len(veh) == 0:
        raise ValueError(f"No rows found for vehicle_id={vehicle_id}.")

    if lane_pos_col not in veh.columns:
        raise ValueError(f"tracks_df must contain '{lane_pos_col}' for lane-pose debugging.")

    veh = veh.sort_values(frame_col)
    x = veh[x_col].to_numpy(dtype=np.float64)
    y = veh[y_col].to_numpy(dtype=np.float64)
    t = veh[frame_col].to_numpy()
    lp = veh[lane_pos_col].to_numpy()

    mask_left_violation = (lp == 0)
    mask_right_violation = (lp == 4)

    mask_unknown = (lp == -1) if bool(highlight_unknown) else np.zeros(len(lp), dtype=bool)

    markings = load_highd_lane_markings(
        recording_meta_df,
        row_idx=0,
        upper_col=upper_col,
        lower_col=lower_col,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    if draw_road:
        _draw_road_background(ax, markings, road_alpha=0.22, median_alpha=0.22)

    if draw_markings:
        for yy in np.sort(markings.all):
            ax.axhline(float(yy), linestyle="--", linewidth=1.0, alpha=0.55, zorder=2)

    ax.plot(x, y, linewidth=1.6, alpha=0.9, label="trajectory", zorder=3)

    if time_colormap:
        sc = ax.scatter(x, y, c=t, s=10.0, zorder=4)
        fig.colorbar(sc, ax=ax, label="frame")

    # left violation (lane_pos == 0) -> red
    if np.any(mask_left_violation):
        ax.scatter(
            x[mask_left_violation],
            y[mask_left_violation],
            s=float(highlight_size),
            color="red",
            label="left violation (lane_pos = 0)",
            zorder=5,
        )
    else:
        ax.scatter([], [], s=float(highlight_size), color="red",
                label="left violation (lane_pos = 0)", zorder=5)

    # right violation (lane_pos == 4) -> green
    if np.any(mask_right_violation):
        ax.scatter(
            x[mask_right_violation],
            y[mask_right_violation],
            s=float(highlight_size),
            color="green",
            label="right violation (lane_pos = 4)",
            zorder=5,
        )
    else:
        ax.scatter([], [], s=float(highlight_size), color="green",
                label="right violation (lane_pos = 4)", zorder=5)

    if highlight_unknown:
        if np.any(mask_unknown):
            ax.scatter(
                x[mask_unknown],
                y[mask_unknown],
                s=float(highlight_size),
                color="orange",
                label="lane_pos == -1 (unknown)",
                zorder=5,
            )
        else:
            ax.scatter([], [], s=float(highlight_size), color="orange", label="lane_pos == -1 (unknown)", zorder=5)

    if invert_y:
        ax.invert_yaxis()

    rid = f"{recording_id} " if recording_id else ""
    ax.set_title(title or f"Lane-pose debug: {rid}vehicle {vehicle_id}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return out_path
