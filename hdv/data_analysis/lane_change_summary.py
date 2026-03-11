"""
Lane-change analysis utilities for highD Feather.
What this module computes
1) Lane changes per trajectory (vehicle):
   - For each (recording_id, vehicle_id), count number of lane-change frames:
       n_lc = count of frames with lc != 0
   - Aggregate counts: how many vehicles had 0, 1, 2, ... lane changes
   - Also compute percentages of vehicles for each count

2) Frame-level lane-change distribution:
   - Count frames with lc in {-1, 0, +1}
   - Also compute percentages over all frames

3) Lane-change side context (events = frames where lc != 0):
   - Lane-change event counts by side (left/right/unknown)
   - Same counts split by traffic direction (1/2), if available
   - Same counts split by vehicle class (car/truck), if available
   - Optionally split by (direction, class) combined, if both available
"""

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kinematic_summary import assert_cols, class_bucket_series

# =========================================================
# Internal helpers
# =========================================================
def _ensure_int_lc(lc_series):
    """
    Convert lane-change series to integer values in {-1,0,+1} where possible.

    Parameters
    lc_series : pandas.Series
        Lane-change values (may be float/object).

    Returns
    pandas.Series
        Nullable integer series (Int64) with values -1, 0, +1 (NaN preserved).
    """
    s = pd.to_numeric(lc_series, errors="coerce")
    s = s.round()
    return s.astype("Int64")


def _with_percent(df_counts, count_col, pct_col="pct"):
    """
    Add percentage column to a count table.
    """
    total = float(df_counts[count_col].sum()) if len(df_counts) else 0.0
    if total <= 0.0:
        df_counts[pct_col] = 0.0
    else:
        df_counts[pct_col] = 100.0 * df_counts[count_col] / total
    return df_counts


def _save_bar_counts(df_counts, x_col, y_col, title, xlabel, ylabel, out_path, legend=None):
    """
    Save a simple bar chart with count annotations (like your screenshots).

    Parameters
    df_counts : pandas.DataFrame
        Must contain x_col, y_col.
    legend : str or None
        If not None, adds legend label (useful for grouped bars).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if df_counts.empty:
        return

    fig, ax = plt.subplots()

    x = df_counts[x_col].astype(str).tolist()
    y = df_counts[y_col].astype(int).tolist()

    bars = ax.bar(x, y, alpha=0.9)

    ax.set_title(title, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)

    ymax = max(y) if len(y) else 1
    for b, val in zip(bars, y):
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            val + 0.01 * ymax,
            f"{val}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    if legend is not None:
        ax.legend([legend], loc="upper right", fontsize=12, framealpha=0.95)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_grouped_side_plot(counts, sides, groups, group_label_prefix, title, out_path):
    """
    Grouped bar chart: x-axis is side, bars are groups.

    Parameters
    counts : np.ndarray
        Shape (G, S) counts for each group and side.
    sides : list[str]
        Side categories, e.g. ['left','right','unknown'].
    groups : list
        Group values to show in legend.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if counts.size == 0:
        return

    counts = np.asarray(counts, dtype=int)
    G, S = counts.shape

    x = np.arange(S)
    width = 0.8 / max(1, G)

    fig, ax = plt.subplots()

    for i in range(G):
        offset = (i - (G - 1) / 2.0) * width
        bars = ax.bar(x + offset, counts[i], width=width, label=f"{group_label_prefix}{groups[i]}")

        ymax = int(max(1, counts.max()))
        for j in range(S):
            val = int(counts[i, j])
            ax.text(
                (x[j] + offset),
                val + 0.01 * ymax,
                f"{val}",
                ha="center",
                va="bottom",
                fontsize=11,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(sides)
    ax.set_title(title, fontsize=18)
    ax.set_xlabel("side", fontsize=14)
    ax.set_ylabel("Number of lane-change events", fontsize=14)
    ax.legend(loc="upper right", fontsize=12, framealpha=0.95)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# Core computations
# =========================================================
def compute_lane_changes_per_trajectory(df, recording_col="recording_id", vehicle_col="vehicle_id", lc_col="lc"):
    """
    Compute lane-change counts per trajectory (vehicle).

    Definition:
      n_lc(vehicle) = number of frames where lc != 0.

    Returns
    per_vehicle : pandas.DataFrame
        One row per vehicle with:
          recording_id, vehicle_id, n_lc, n_left, n_right
    agg_counts : pandas.DataFrame
        Aggregated counts by n_lc with:
          n_lc, vehicle_count, vehicle_pct
    """
    assert_cols(df, [recording_col, vehicle_col, lc_col], context="compute_lane_changes_per_trajectory")

    tmp = df[[recording_col, vehicle_col, lc_col]].copy()
    tmp[lc_col] = _ensure_int_lc(tmp[lc_col])

    is_nonzero = (tmp[lc_col] != 0) & tmp[lc_col].notna()
    is_left = (tmp[lc_col] == -1)
    is_right = (tmp[lc_col] == 1)

    tmp["_is_nonzero"] = is_nonzero.astype(np.int64)
    tmp["_is_left"] = is_left.astype(np.int64)
    tmp["_is_right"] = is_right.astype(np.int64)

    per_vehicle = (
        tmp.groupby([recording_col, vehicle_col], sort=False)
           .agg(
               n_lc=("_is_nonzero", "sum"),
               n_left=("_is_left", "sum"),
               n_right=("_is_right", "sum"),
           )
           .reset_index()
    )

    vc = per_vehicle["n_lc"].value_counts().sort_index()
    agg_counts = vc.rename_axis("n_lc").reset_index(name="vehicle_count")
    agg_counts = _with_percent(agg_counts, "vehicle_count", "vehicle_pct")

    return per_vehicle, agg_counts


def compute_lane_change_frame_distribution(df, lc_col="lc"):
    """
    Compute frame-level distribution of lane-change indicator lc ∈ {-1, 0, +1}.

    Returns
    pandas.DataFrame
        Columns: lc_value, frame_count, frame_pct
    """
    assert_cols(df, [lc_col], context="compute_lane_change_frame_distribution")

    s = _ensure_int_lc(df[lc_col]).dropna()

    counts = {
        -1: int((s == -1).sum()),
         0: int((s == 0).sum()),
         1: int((s == 1).sum()),
    }
    out = pd.DataFrame({
        "lc_value": [-1, 0, 1],
        "frame_count": [counts[-1], counts[0], counts[1]],
    })
    out = _with_percent(out, "frame_count", "frame_pct")
    return out


# =========================================================
# Side context (count plots like your screenshots)
# =========================================================
def extract_lane_change_events(df, lc_col="lc"):
    """
    Extract lane-change event rows (frames where lc != 0) and add lc_side.

    Returns
    pandas.DataFrame
        Event rows with lc_side ∈ {'left','right','unknown'}.
    """
    assert_cols(df, [lc_col], context="extract_lane_change_events")

    ev = df.copy()
    ev[lc_col] = _ensure_int_lc(ev[lc_col])
    ev = ev.loc[ev[lc_col].notna() & (ev[lc_col] != 0)].copy()

    ev["lc_side"] = np.where(
        ev[lc_col] == -1, "left",
        np.where(ev[lc_col] == 1, "right", "unknown")
    )
    return ev


def compute_lane_change_side_counts(ev, side_col="lc_side"):
    """
    Count lane-change events by side.

    Returns
    pandas.DataFrame
        Columns: side, count, pct
    """
    order = ["left", "right", "unknown"]
    vc = ev[side_col].value_counts()
    out = pd.DataFrame({"side": order, "count": [int(vc.get(k, 0)) for k in order]})
    out = _with_percent(out, "count", "pct")
    return out


def compute_lane_change_side_counts_split(ev, split_col, side_col="lc_side"):
    """
    Count lane-change events by side, split by a category (direction or class).

    Returns
    table : pandas.DataFrame
        Columns: split_col, side, count, pct_within_split
    """
    assert_cols(ev, [split_col, side_col], context="compute_lane_change_side_counts_split")

    sides = ["left", "right", "unknown"]
    rows = []

    for key, sub in ev.groupby(split_col, dropna=False, sort=True):
        vc = sub[side_col].value_counts()
        total = float(len(sub)) if len(sub) else 0.0
        for s in sides:
            cnt = int(vc.get(s, 0))
            pct = 100.0 * cnt / total if total > 0 else 0.0
            rows.append({split_col: key, "side": s, "count": cnt, "pct_within_split": pct})

    return pd.DataFrame(rows)


def run_lane_change_side_context(
    df,
    out_dir,
    lc_col="lc",
    direction_col="meta_drivingDirection",
    class_col="meta_class",
):
    """
    Produce side-count plots matching your earlier context outputs:

      - lane_change_counts_by_side.png
      - lane_change_counts_by_side__split_by_direction.png (if direction exists)
      - lane_change_counts_by_side__split_by_class.png (if class exists)
      - lane_change_counts_by_side__split_by_direction_and_class.png (if both exist)

    Also writes CSV tables with counts and percentages.

    Parameters
    df : pandas.DataFrame
        Input Feather dataframe.
    out_dir : pathlib.Path or str
        Output directory.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ev = extract_lane_change_events(df, lc_col=lc_col)

    # Add direction/class if present
    if direction_col in df.columns:
        ev[direction_col] = df.loc[ev.index, direction_col]
    else:
        ev[direction_col] = np.nan

    if class_col in df.columns:
        ev["class_bucket"] = class_bucket_series(df.loc[ev.index, class_col])
    else:
        ev["class_bucket"] = np.nan

    # --- Total by side ---
    side_counts = compute_lane_change_side_counts(ev, side_col="lc_side")
    side_counts.to_csv(out_dir / "lane_change_counts_by_side.csv", index=False)

    total = int(side_counts["count"].sum())
    _save_bar_counts(
        side_counts,
        x_col="side",
        y_col="count",
        title=f"Lane-change counts by side (total={total})",
        xlabel="side",
        ylabel="Number of lane-change events",
        out_path=out_dir / "lane_change_counts_by_side.png",
    )

    # --- Split by direction ---
    if ev[direction_col].notna().any():
        tab = compute_lane_change_side_counts_split(ev, split_col=direction_col, side_col="lc_side")
        tab.to_csv(out_dir / "lane_change_counts_by_side__split_by_direction.csv", index=False)

        sides = ["left", "right", "unknown"]
        groups = [g for g in sorted(ev[direction_col].dropna().unique().tolist())]
        counts = []
        for g in groups:
            sub = tab.loc[tab[direction_col] == g].set_index("side")["count"]
            counts.append([int(sub.get(s, 0)) for s in sides])
        counts = np.asarray(counts, dtype=int)

        _save_grouped_side_plot(
            counts=counts,
            sides=sides,
            groups=groups,
            group_label_prefix="drivingDirection=",
            title="Lane-change counts by side split by traffic direction",
            out_path=out_dir / "lane_change_counts_by_side__split_by_direction.png",
        )

    # --- Split by class ---
    if ev["class_bucket"].notna().any():
        tab = compute_lane_change_side_counts_split(ev, split_col="class_bucket", side_col="lc_side")
        tab.to_csv(out_dir / "lane_change_counts_by_side__split_by_class.csv", index=False)

        sides = ["left", "right", "unknown"]
        groups = [g for g in ["car", "truck"] if (ev["class_bucket"] == g).any()]
        counts = []
        for g in groups:
            sub = tab.loc[tab["class_bucket"] == g].set_index("side")["count"]
            counts.append([int(sub.get(s, 0)) for s in sides])
        counts = np.asarray(counts, dtype=int)

        _save_grouped_side_plot(
            counts=counts,
            sides=sides,
            groups=groups,
            group_label_prefix="class=",
            title="Lane-change counts by side split by vehicle class",
            out_path=out_dir / "lane_change_counts_by_side__split_by_class.png",
        )

    # --- Split by (direction, class) combined ---
    if ev[direction_col].notna().any() and ev["class_bucket"].notna().any():
        ev2 = ev.copy()
        ev2["dir_class"] = ev2[direction_col].astype(str) + "__" + ev2["class_bucket"].astype(str)

        tab = compute_lane_change_side_counts_split(ev2, split_col="dir_class", side_col="lc_side")
        tab.to_csv(out_dir / "lane_change_counts_by_side__split_by_direction_and_class.csv", index=False)

        sides = ["left", "right", "unknown"]
        groups = [g for g in sorted(ev2["dir_class"].dropna().unique().tolist())]
        counts = []
        for g in groups:
            sub = tab.loc[tab["dir_class"] == g].set_index("side")["count"]
            counts.append([int(sub.get(s, 0)) for s in sides])
        counts = np.asarray(counts, dtype=int)

        _save_grouped_side_plot(
            counts=counts,
            sides=sides,
            groups=groups,
            group_label_prefix="",
            title="Lane-change counts by side split by (direction, class)",
            out_path=out_dir / "lane_change_counts_by_side__split_by_direction_and_class.png",
        )

    print(f"[lane_change_side_context] Outputs written to: {out_dir}")


# =========================================================
# Public entry point
# =========================================================

def run_lane_change_analysis(
    df,
    out_dir,
    recording_col="recording_id",
    vehicle_col="vehicle_id",
    frame_col="frame",
    lc_col="lc",
    direction_col="meta_drivingDirection",
    class_col="meta_class",
    run_side_context=True,
):
    """
    Run lane-change analysis and write outputs.

    Parameters
    df : pandas.DataFrame
        Input Feather dataframe.
    out_dir : pathlib.Path or str
        Root output directory.
    recording_col, vehicle_col, frame_col, lc_col : str
        Column names.
    direction_col, class_col : str
        Optional meta columns used for side context plots.
    run_side_context : bool
        If True, also produces side-count context plots (like your screenshots).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assert_cols(df, [recording_col, vehicle_col, frame_col, lc_col], context="run_lane_change_analysis")

    # --- Per-trajectory analysis ---
    per_vehicle, agg_counts = compute_lane_changes_per_trajectory(
        df,
        recording_col=recording_col,
        vehicle_col=vehicle_col,
        lc_col=lc_col,
    )

    traj_dir = out_dir / "per_trajectory"
    traj_dir.mkdir(parents=True, exist_ok=True)

    agg_counts.to_csv(traj_dir / "lane_changes_per_trajectory.csv", index=False)
    _save_bar_counts(
        df_counts=agg_counts.rename(columns={"n_lc": "count"}),
        x_col="count",
        y_col="vehicle_count",
        title="Lane changes per trajectory",
        xlabel="Number of lane changes",
        ylabel="Vehicle count",
        out_path=traj_dir / "lane_changes_per_trajectory.png",
    )

    # --- Frame-level distribution ---
    frame_dist = compute_lane_change_frame_distribution(df, lc_col=lc_col)

    frame_dir = out_dir / "per_frame"
    frame_dir.mkdir(parents=True, exist_ok=True)

    frame_dist.to_csv(frame_dir / "lane_change_frame_distribution.csv", index=False)

    frame_plot = frame_dist.copy()
    label_map = {-1: "left (-1)", 0: "none (0)", 1: "right (+1)"}
    frame_plot["lc_label"] = frame_plot["lc_value"].map(label_map)

    _save_bar_counts(
        df_counts=frame_plot.rename(columns={"frame_count": "count"}),
        x_col="lc_label",
        y_col="count",
        title="Lane-change indicator distribution (per frame)",
        xlabel="Lane change (lc)",
        ylabel="Frame count",
        out_path=frame_dir / "lane_change_frame_distribution.png",
    )

    # --- Side context plots (counts by side + splits) ---
    if run_side_context:
        run_lane_change_side_context(
            df=df,
            out_dir=out_dir / "side_context",
            lc_col=lc_col,
            direction_col=direction_col,
            class_col=class_col,
        )

    print(f"[lane_change] Outputs written to: {out_dir}")
