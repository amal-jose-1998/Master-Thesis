"""
Kinematic summary utilities for the highD dataset (vehicle-centric).

Generated outputs
For each kinematic signal (vx, ax, vy, ay):
1) Histograms:
   - Global (all vehicles, no split)
   - By driving direction (1 / 2)
   - By vehicle class (car / truck)
   - By (driving direction, vehicle class)
2) Variation (per-vehicle standard deviation):
   - Same splits as histograms
3) Per-recording statistics (CSV):
   - mean, median
   - p01, p10, p90, p99
   - min, max
"""

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# Basic validation utilities
# =========================================================

def assert_cols(df, cols, context="kinematics"):
    """
    Assert that required columns exist in the dataframe.

    Parameters
    df : pandas.DataFrame
        Input dataframe.
    cols : sequence of str
        Column names that must be present.
    context : str
        Label used in the error message for easier debugging.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"[{context}] Missing required columns: {missing}")


# =========================================================
# Vehicle class handling
# =========================================================
def class_bucket_series(class_col):
    """
    Convert highD vehicle class labels into canonical buckets.

    Parameters
    class_col : pandas.Series
        Series containing class labels.

    Returns
    pandas.Series
        Series with values 'car', 'truck', or None.
    """
    def _to_bucket(x):
        if pd.isna(x):
            return None

        if isinstance(x, str):
            s = x.strip().lower()
            if "car" in s:
                return "car"
            if "truck" in s:
                return "truck"
            try:
                x = int(float(s))
            except Exception:
                return None

        try:
            xi = int(x)
        except Exception:
            return None

        if xi == 1:
            return "car"
        if xi == 2:
            return "truck"
        return None

    return class_col.map(_to_bucket)

# =========================================================
# Plotting helpers
# =========================================================
def _summary_stats(x):
    """
    Compute summary stats used for plot annotations.

    Parameters
    x : pandas.Series
        Numeric series (may contain NaNs).

    Returns
    dict
        Keys: min, max, mean, median, p10, p90
        Values are floats, or None if x is empty.
    """
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return {
            "min": None, "max": None, "mean": None, "median": None, "p10": None, "p90": None
        }

    return {
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "p10": float(x.quantile(0.10)),
        "p90": float(x.quantile(0.90)),
    }


def _robust_xlim(x, q_low=0.01, q_high=0.99):
    """
    Compute robust x-axis limits using quantiles to avoid extreme outliers.

    Parameters
    x : pandas.Series
        Numeric series.
    q_low, q_high : float
        Lower and upper quantiles.

    Returns
    tuple or None
        (low, high) limits if valid, else None.
    """
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return None
    lo = float(x.quantile(q_low))
    hi = float(x.quantile(q_high))
    return (lo, hi) if lo < hi else None


def save_hist_1d(
    x,
    title,
    xlabel,
    out_path,
    bins=200,
    xlim_quantiles=(0.01, 0.99),
    show_markers=True,
    show_stats_box=True,
):
    """
    Save a 1D histogram plot with summary markers and an annotation box.

    This matches the style:
      - histogram as probability density
      - vertical lines: mean, median, p10, p90
      - textbox: min, max, mean, median, p10, p90

    Parameters
    x : array-like
        Data to plot.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    out_path : pathlib.Path or str
        Output PNG path.
    bins : int
        Number of bins.
    xlim_quantiles : tuple or None
        Quantiles used to set x-limits (to avoid outliers). Use None for no clipping.
    show_markers : bool
        If True, draws mean/median/p10/p90 vertical lines and adds legend.
    show_stats_box : bool
        If True, shows a stats textbox in the upper-right corner.
    """
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    if x.empty:
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = _summary_stats(x)

    fig, ax = plt.subplots()
    ax.hist(
        x.values,
        bins=bins,
        density=True,
        alpha=0.80,
        edgecolor="black",
        linewidth=0.3,
    )

    # x-limits
    if xlim_quantiles is not None:
        xlim = _robust_xlim(x, xlim_quantiles[0], xlim_quantiles[1])
        if xlim is not None:
            ax.set_xlim(*xlim)

    # vertical markers (match your example style)
    if show_markers:
        # Use classic, readable linestyles
        ax.axvline(stats["mean"],   color="tab:orange", linestyle="-.", linewidth=2.0, label="mean")
        ax.axvline(stats["median"], color="tab:blue",   linestyle="-",  linewidth=2.2, label="median")
        ax.axvline(stats["p10"],    color="tab:green", linestyle="--", linewidth=2.0, label="p10")
        ax.axvline(stats["p90"],    color="tab:red",  linestyle="--", linewidth=2.0, label="p90")
        ax.legend(loc="upper left", fontsize=11, framealpha=0.95)

    # stats textbox
    if show_stats_box:
        def _fmt(v):
            return "nan" if v is None else f"{v:.3g}"

        text = (
            f"min   = {_fmt(stats['min'])}\n"
            f"max   = {_fmt(stats['max'])}\n"
            f"mean  = {_fmt(stats['mean'])}\n"
            f"median= {_fmt(stats['median'])}\n"
            f"p10   = {_fmt(stats['p10'])}\n"
            f"p90   = {_fmt(stats['p90'])}"
        )

        ax.text(
            0.98, 0.98,
            text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="black", alpha=0.95),
        )

    ax.set_title(title, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("Probability density", fontsize=14)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# Variation: per-vehicle standard deviation
# =========================================================

def per_vehicle_std(df, signal, id_cols, min_T=10):
    """
    Compute per-vehicle standard deviation of a signal.
    Vehicles are identified by a combination of recording_id and vehicle_id.

    Parameters
    df : pandas.DataFrame
        Input dataframe.
    signal : str
        Column name of the signal (e.g., vx, ax).
    id_cols : list of str
        Grouping columns (typically ['recording_id', 'vehicle_id']).
    min_T : int
        Minimum number of frames required for a trajectory.

    Returns
    numpy.ndarray
        Array of standard deviation values.
    """
    if signal not in df.columns:
        return np.asarray([])

    values = []
    for _, g in df.groupby(id_cols, sort=False):
        if len(g) < min_T:
            continue
        x = pd.to_numeric(g[signal], errors="coerce").dropna()
        if len(x) >= min_T:
            values.append(float(x.std(ddof=0)))

    return np.asarray(values)


# =========================================================
# Per-recording statistics
# =========================================================

def save_per_recording_stats(df, signals, out_csv, recording_col="recording_id"):
    """
    Compute and save per-recording statistics for kinematic signals.
    Statistics computed per (recording, signal):
      - mean
      - median
      - p01, p10, p90, p99
      - min, max

    Parameters
    df : pandas.DataFrame
        Input dataframe.
    signals : sequence of str
        Signal column names.
    out_csv : pathlib.Path
        Output CSV path.
    recording_col : str
        Recording identifier column.

    Returns
    pandas.DataFrame
        Summary dataframe.
    """
    rows = []

    for rid, g in df.groupby(recording_col, sort=True):
        for s in signals:
            if s not in g.columns:
                continue

            x = pd.to_numeric(g[s], errors="coerce").dropna()
            if x.empty:
                continue

            rows.append({
                "recording_id": rid,
                "signal": s,
                "count": int(x.shape[0]),
                "mean": float(x.mean()),
                "median": float(x.median()),
                "p01": float(x.quantile(0.01)),
                "p10": float(x.quantile(0.10)),
                "p90": float(x.quantile(0.90)),
                "p99": float(x.quantile(0.99)),
                "min": float(x.min()),
                "max": float(x.max()),
            })

    out = pd.DataFrame(rows)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


# =========================================================
# Public entry point
# =========================================================

def run_kinematics_analysis(
    df,
    out_dir,
    signals=("vx", "ax", "vy", "ay"),
    direction_col="meta_drivingDirection",
    class_col="meta_class",
    recording_col="recording_id",
    vehicle_col="vehicle_id",
    frame_col="frame",
    min_T_std=10,
):
    """
    Run the complete kinematics analysis.
    This function generates:
      - Histogram plots
      - Per-vehicle variability plots
      - Per-recording statistics CSV

    Parameters
    df : pandas.DataFrame
        Vehicle-centric Feather dataframe.
    out_dir : pathlib.Path
        Root output directory for kinematics.
    signals : sequence of str
        Kinematic signals to analyze.
    direction_col : str
        Driving direction column (1 or 2).
    class_col : str
        Vehicle class column.
    recording_col : str
        Recording identifier column.
    vehicle_col : str
        Vehicle identifier column.
    frame_col : str
        Frame index column.
    min_T_std : int
        Minimum trajectory length for std computation.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hist_dir = out_dir / "hist"
    var_dir = out_dir / "variability"
    stats_dir = out_dir / "per_recording"
    for p in (hist_dir, var_dir, stats_dir):
        p.mkdir(parents=True, exist_ok=True)

    # Validate schema
    assert_cols(
        df,
        [recording_col, vehicle_col, frame_col, direction_col, class_col, *signals],
        context="run_kinematics_analysis",
    )

    dfk = df.copy()
    dfk["class_bucket"] = class_bucket_series(dfk[class_col])

    # Per-recording statistics
    save_per_recording_stats(
        dfk,
        signals,
        stats_dir / "per_recording_vehicle_centric.csv",
        recording_col,
    )

    # Subset helpers
    def subset_dir(dd):
        return dfk.loc[dfk[direction_col] == dd]

    def subset_class(cls):
        return dfk.loc[dfk["class_bucket"] == cls]

    def subset_dir_class(dd, cls):
        return dfk.loc[(dfk[direction_col] == dd) & (dfk["class_bucket"] == cls)]

    # ---- Histograms ----
    for sig in signals:
        save_hist_1d(
            dfk[sig],
            f"{sig} (vehicle-centric) - global",
            sig,
            hist_dir / f"{sig}__global.png",
        )

        for dd in (1, 2):
            save_hist_1d(
                subset_dir(dd)[sig],
                f"{sig} (vehicle-centric) - direction={dd}",
                sig,
                hist_dir / f"{sig}__dir{dd}.png",
            )

        for cls in ("car", "truck"):
            save_hist_1d(
                subset_class(cls)[sig],
                f"{sig} (vehicle-centric) - class={cls}",
                sig,
                hist_dir / f"{sig}__class_{cls}.png",
            )

        for dd in (1, 2):
            for cls in ("car", "truck"):
                save_hist_1d(
                    subset_dir_class(dd, cls)[sig],
                    f"{sig} (vehicle-centric) - dir={dd}, class={cls}",
                    sig,
                    hist_dir / f"{sig}__dir{dd}__class_{cls}.png",
                )

    # ---- Variability (per-vehicle std) ----
    id_cols = [recording_col, vehicle_col]

    def save_std_hist(df_sub, sig, tag):
        vals = per_vehicle_std(df_sub, sig, id_cols, min_T_std)
        if vals.size == 0:
            return
        save_hist_1d(
            vals,
            f"std({sig}) (vehicle-centric) - {tag}",
            f"std({sig})",
            var_dir / f"std__{sig}__{tag}.png",
        )

    for sig in signals:
        save_std_hist(dfk, sig, "global")

        for dd in (1, 2):
            save_std_hist(subset_dir(dd), sig, f"dir{dd}")

        for cls in ("car", "truck"):
            save_std_hist(subset_class(cls), sig, f"class_{cls}")

        for dd in (1, 2):
            for cls in ("car", "truck"):
                save_std_hist(subset_dir_class(dd, cls), sig, f"dir{dd}__class_{cls}")

    print(f"[kinematics] Outputs written to: {out_dir}")
