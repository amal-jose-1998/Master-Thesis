"""
Kinematic summary utilities for highD.

This module provides:
- compact descriptive summaries (per-recording, per-vehicle std),
- plots for global distributions,
- per-vehicle variability distributions,
- per-recording summary plots,
- distributions split by driving direction or vehicle class.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# Plot style helpers (colors for reference lines)
# =========================================================

# MOD: Distinct colors for summary/reference lines (consistent across plots)
_LINE_STYLE = {
    "mean":   dict(color="tab:orange", linestyle="-.", linewidth=1.8),
    "median": dict(color="tab:blue",   linestyle="-",  linewidth=2.0),
    "p10":    dict(color="tab:purple", linestyle="--", linewidth=1.6),
    "p90":    dict(color="tab:brown",  linestyle="--", linewidth=1.6),
    "min":    dict(color="tab:green",  linestyle=":",  linewidth=1.4),
    "max":    dict(color="tab:red",    linestyle=":",  linewidth=1.4),
}


# =========================================================
# Helpers
# =========================================================

def _available_cols(df, cols):
    """
    Filter a list of candidate columns to those present in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    cols : sequence[str]

    Returns
    -------
    list[str]
    """
    return [c for c in cols if c in df.columns]


def _robust_xlim(x, q_low=0.01, q_high=0.99):
    """
    Robust axis limits using quantiles.

    Parameters
    ----------
    x : pd.Series
        Numeric series.
    q_low, q_high : float
        Quantile bounds in [0, 1].

    Returns
    -------
    tuple[float, float] | None
    """
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return None
    lo, hi = x.quantile([q_low, q_high]).values
    lo = float(lo)
    hi = float(hi)
    if lo == hi:
        return None
    return lo, hi


def _summary_1d(x):
    """
    Summarize a 1D numeric array/series into descriptive stats.

    Returns
    -------
    dict
        count, nan_count, min, mean, std, median, max, quantiles
    """
    s = pd.to_numeric(x, errors="coerce")
    n = int(s.shape[0])
    nan = int(s.isna().sum())
    s = s.dropna()

    if s.empty:
        return {
            "count": n,
            "nan_count": nan,
            "min": None,
            "mean": None,
            "std": None,
            "median": None,
            "max": None,
            "quantiles": {},
        }

    q = {
        "p1": float(s.quantile(0.01)),
        "p10": float(s.quantile(0.10)),
        "p50": float(s.quantile(0.50)),
        "p90": float(s.quantile(0.90)),
        "p99": float(s.quantile(0.99)),
    }

    return {
        "count": n,
        "nan_count": nan,
        "min": float(s.min()),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "median": float(s.median()),
        "max": float(s.max()),
        "quantiles": q,
    }


# =========================================================
# Compute API
# =========================================================

def compute_per_recording_kinematics(df_all, cols=("xVelocity", "yVelocity", "xAcceleration", "yAcceleration"),
                                     recording_col="recording_id", max_recordings=None, show_progress=False):
    """
    Compute per-recording kinematic summaries.

    Parameters
    ----------
    df_all : pd.DataFrame
        Long-table tracks data.
    cols : sequence[str]
        Kinematic columns to summarize.
    recording_col : str
        Recording id column.
    max_recordings : int | None
        Optional cap (debug convenience).
    show_progress : bool
        If True, show progress over recordings.

    Returns
    -------
    dict
        per_rec[recording_id][col] = summary dict
    """
    if recording_col not in df_all.columns:
        raise KeyError(f"Missing required column: {recording_col}")

    use_cols = _available_cols(df_all, cols)
    rec_ids = df_all[recording_col].dropna().unique().tolist()
    if max_recordings is not None:
        rec_ids = rec_ids[: int(max_recordings)]

    iterator = rec_ids
    if show_progress:
        from tqdm.auto import tqdm
        iterator = tqdm(rec_ids, desc="Per-recording kinematics")

    per_rec = {}
    for rid in iterator:
        sub = df_all[df_all[recording_col] == rid]
        per_rec[str(rid)] = {c: _summary_1d(sub[c]) for c in use_cols}

    return per_rec


def compute_per_vehicle_std(df_all, cols, group_cols=("recording_id", "id"), min_T=10, show_progress=False):
    """
    Compute per-vehicle (trajectory) standard deviation for selected signals.

    Parameters
    ----------
    df_all : pd.DataFrame
    cols : sequence[str]
    group_cols : tuple[str, str]
        Columns identifying a trajectory.
    min_T : int
        Minimum number of valid rows required.
    show_progress : bool

    Returns
    -------
    dict
        std_rows[col] = list of per-vehicle std values
    """
    for gc in group_cols:
        if gc not in df_all.columns:
            raise KeyError(f"Missing required group column: {gc}")

    use_cols = _available_cols(df_all, cols)
    std_rows = {c: [] for c in use_cols}

    grouped = df_all.groupby(list(group_cols), sort=False)
    iterator = grouped
    if show_progress:
        from tqdm.auto import tqdm
        iterator = tqdm(grouped, total=grouped.ngroups, desc="Compute per-vehicle std")

    skipped = 0
    for _, g in iterator:
        if len(g) < int(min_T):
            skipped += 1
            continue
        for c in use_cols:
            x = pd.to_numeric(g[c], errors="coerce").dropna()
            if len(x) >= int(min_T):
                std_rows[c].append(float(x.std(ddof=0)))

    if skipped > 0:
        print(f"[kinematics] Skipped {skipped} trajectories with less than {min_T} rows.")
    return std_rows


# =========================================================
# Plot API
# =========================================================

def plot_global_kinematics(df_all, cols, out_dir, bins=200, xlim_quantiles=(0.001, 0.999), fixed_xlim=None, annotate=True):
    """
    Plot global distributions of selected kinematic variables.

    Parameters
    ----------
    df_all : pd.DataFrame
    cols : sequence[str]
    out_dir : str | Path
    bins : int
    xlim_quantiles : (float, float) | None
    fixed_xlim : dict | None
    annotate : bool
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for c in cols:
        if c not in df_all.columns:
            continue

        x = pd.to_numeric(df_all[c], errors="coerce").dropna()
        if x.empty:
            continue

        s = _summary_1d(x)
        q = s.get("quantiles", {})
        mean = s["mean"]
        median = s["median"]
        p10 = q.get("p10", float(x.quantile(0.10)))
        p90 = q.get("p90", float(x.quantile(0.90)))

        if isinstance(fixed_xlim, dict) and c in fixed_xlim:
            xlim = fixed_xlim[c]
        else:
            xlim = _robust_xlim(x, q_low=xlim_quantiles[0], q_high=xlim_quantiles[1]) if xlim_quantiles else None

        fig = plt.figure()
        plt.hist(x.values, bins=bins, density=True, alpha=0.75, edgecolor="black", linewidth=0.3)
        if xlim is not None:
            plt.xlim(*xlim)

        # MOD: colored reference lines
        if mean is not None:
            plt.axvline(mean, label="mean", **_LINE_STYLE["mean"])      # MOD
        if median is not None:
            plt.axvline(median, label="median", **_LINE_STYLE["median"])  # MOD
        plt.axvline(p10, label="p10", **_LINE_STYLE["p10"])            # MOD
        plt.axvline(p90, label="p90", **_LINE_STYLE["p90"])            # MOD

        if annotate:
            lines = [
                f"min   = {s['min']:.4g}",
                f"max   = {s['max']:.4g}",
                f"mean  = {mean:.4g}",
                f"median= {median:.4g}",
                f"p10   = {p10:.4g}",
                f"p90   = {p90:.4g}",
            ]
            ax = plt.gca()
            ax.text(
                0.98, 0.98, "\n".join(lines),
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="black"),
            )

        plt.legend(loc="upper left", fontsize=9)
        plt.title(f"Global distribution: {c}")
        plt.xlabel(c)
        plt.ylabel("Probability density")
        fig.savefig(out_dir / f"global_{c}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_per_vehicle_variability(df_all, cols, out_dir, min_T=5, show_progress=True, bins=200,
                                xlim_quantiles=(0.001, 0.999), fixed_xlim=None, annotate=True):
    """
    Plot distributions of per-vehicle variability (std per trajectory).

    Parameters
    ----------
    df_all : pd.DataFrame
    cols : sequence[str]
    out_dir : str | Path
    min_T : int
    show_progress : bool
    bins : int
    xlim_quantiles : (float, float) | None
    fixed_xlim : dict | None
    annotate : bool
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    std_rows = compute_per_vehicle_std(df_all, cols=cols, min_T=min_T, show_progress=show_progress)

    for c, vals in std_rows.items():
        if not vals:
            continue

        vals_s = pd.Series(vals)
        s = _summary_1d(vals_s)
        q = s.get("quantiles", {})
        mean = s["mean"]
        median = s["median"]
        p10 = q.get("p10", float(vals_s.quantile(0.10)))
        p90 = q.get("p90", float(vals_s.quantile(0.90)))

        if isinstance(fixed_xlim, dict) and c in fixed_xlim:
            xlim = fixed_xlim[c]
        else:
            xlim = _robust_xlim(vals_s, q_low=xlim_quantiles[0], q_high=xlim_quantiles[1]) if xlim_quantiles else None

        fig = plt.figure()
        plt.hist(vals, bins=bins, alpha=0.75, edgecolor="black", linewidth=0.3)
        if xlim is not None:
            plt.xlim(*xlim)

        # MOD: colored reference lines
        if mean is not None:
            plt.axvline(mean, label="mean", **_LINE_STYLE["mean"])        # MOD
        if median is not None:
            plt.axvline(median, label="median", **_LINE_STYLE["median"])  # MOD
        plt.axvline(p10, label="p10", **_LINE_STYLE["p10"])              # MOD
        plt.axvline(p90, label="p90", **_LINE_STYLE["p90"])              # MOD

        if annotate:
            lines = [
                f"min   = {s['min']:.4g}",
                f"max   = {s['max']:.4g}",
                f"mean  = {mean:.4g}",
                f"median= {median:.4g}",
                f"p10   = {p10:.4g}",
                f"p90   = {p90:.4g}",
            ]
            ax = plt.gca()
            ax.text(
                0.98, 0.98, "\n".join(lines),
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="black"),
            )

        plt.legend(loc="upper left", fontsize=9)
        plt.title(f"Per-vehicle variability: std({c})")
        plt.xlabel(f"std({c})")
        plt.ylabel("Number of vehicles")
        fig.savefig(out_dir / f"per_vehicle_std_{c}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_per_recording_kinematics(df_all, cols, out_dir, recording_col="recording_id",
                                  max_recordings=None, show_progress=True, use_minmax_whiskers=False, annotate=True):
    """
    Plot per-recording summaries as median with p10–p90 error bars.

    Notes
    -----
    This plot shows:
    - median with p10–p90 error bars,
    - (optional) min–max whiskers,
    - mean overlay markers (distinct color/marker).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_rec = compute_per_recording_kinematics(df_all, cols=cols, recording_col=recording_col,
                                               max_recordings=max_recordings, show_progress=show_progress)
    if not per_rec:
        return

    rec_ids = list(per_rec.keys())
    for c in cols:
        rows = []
        for rid in rec_ids:
            s = per_rec[rid].get(c, {})
            q = s.get("quantiles", {})
            if s.get("median") is None or "p10" not in q or "p90" not in q:
                continue
            rows.append({
                "rid": rid,
                "median": float(s["median"]),
                "p10": float(q["p10"]),
                "p90": float(q["p90"]),
                "min": float(s["min"]) if s.get("min") is not None else np.nan,
                "max": float(s["max"]) if s.get("max") is not None else np.nan,
                "mean": float(s["mean"]) if s.get("mean") is not None else np.nan,
            })

        if not rows:
            continue

        labels = [r["rid"] for r in rows]
        med = np.array([r["median"] for r in rows], dtype=float)
        p10 = np.array([r["p10"] for r in rows], dtype=float)
        p90 = np.array([r["p90"] for r in rows], dtype=float)
        vmin = np.array([r["min"] for r in rows], dtype=float)
        vmax = np.array([r["max"] for r in rows], dtype=float)
        vmean = np.array([r["mean"] for r in rows], dtype=float)

        x = np.arange(len(rows))
        yerr = np.vstack([med - p10, p90 - med])

        fig = plt.figure(figsize=(max(8, 0.25 * len(med)), 4))

        # MOD: colored median errorbar
        plt.errorbar(
            x,
            med,
            yerr=yerr,
            fmt="o",
            capsize=3,
            label="median ± (p10–p90)",
            color=_LINE_STYLE["median"]["color"],     # MOD
            ecolor=_LINE_STYLE["p10"]["color"],       # MOD (error bars in p10/p90 family)
        )

        # MOD: overlay mean as distinct marker/color
        if np.any(np.isfinite(vmean)):
            plt.scatter(
                x[np.isfinite(vmean)],
                vmean[np.isfinite(vmean)],
                marker="x",
                s=35,
                color=_LINE_STYLE["mean"]["color"],    # MOD
                label="mean",                          # MOD
                zorder=4,
            )

        if use_minmax_whiskers:
            for i in range(len(med)):
                if np.isfinite(vmin[i]) and np.isfinite(vmax[i]):
                    # MOD: colored min/max whiskers
                    plt.vlines(i, vmin[i], vmax[i], alpha=0.35, color="tab:gray")  # MOD

        plt.xticks(x, labels, rotation=90)
        plt.title(f"Per-recording: {c}")
        plt.xlabel("recording_id")
        plt.ylabel(c)

        # MOD: show legend (now includes mean)
        plt.legend(loc="best", fontsize=9)  # MOD

        if annotate:
            lines = []
            for i, rid in enumerate(labels):
                lines.append(
                    f"{rid}: min={vmin[i]:.3g}, mean={vmean[i]:.3g}, med={med[i]:.3g}, "
                    f"max={vmax[i]:.3g}, p10={p10[i]:.3g}, p90={p90[i]:.3g}"
                )
            ax = plt.gca()
            ax.text(
                1.01, 0.5, "\n".join(lines),
                transform=ax.transAxes,
                ha="left", va="center",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.9),
            )

        fig.savefig(out_dir / f"per_recording_{c}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_signal_by_direction(df_tracks, df_tracks_meta, cols, out_dir, bins=200, xlim_quantiles=(0.001, 0.999), fixed_xlim=None):
    """
    Plot signal histograms split by driving direction.

    Parameters
    ----------
    df_tracks : pd.DataFrame
        Must contain recording_id, id, and each signal column.
    df_tracks_meta : pd.DataFrame
        Must contain recording_id, id, drivingDirection.
    cols : sequence[str]
        Signals to plot.
    out_dir : str | Path
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if df_tracks_meta is None or "drivingDirection" not in df_tracks_meta.columns:
        return

    right = df_tracks_meta[["recording_id", "id", "drivingDirection"]].copy()

    for signal_col in cols:
        if signal_col not in df_tracks.columns:
            continue

        left = df_tracks[["recording_id", "id", signal_col]].copy()
        df = left.merge(right, on=["recording_id", "id"], how="left", validate="many_to_one")

        df[signal_col] = pd.to_numeric(df[signal_col], errors="coerce")
        df["drivingDirection"] = pd.to_numeric(df["drivingDirection"], errors="coerce")

        mask = df[signal_col].notna() & df["drivingDirection"].notna()
        df = df.loc[mask]
        if df.empty:
            continue

        s_all = df[signal_col]
        s1 = df.loc[df["drivingDirection"] == 1, signal_col]
        s2 = df.loc[df["drivingDirection"] == 2, signal_col]

        fig = plt.figure()
        plt.hist(s_all.values, bins=bins, density=True, alpha=0.35, label="all")
        if len(s1):
            plt.hist(s1.values, bins=bins, density=True, alpha=0.35, label="dir=1")
        if len(s2):
            plt.hist(s2.values, bins=bins, density=True, alpha=0.35, label="dir=2")

        plt.legend()
        plt.title(f"{signal_col} split by drivingDirection")
        plt.xlabel(signal_col)
        plt.ylabel("Probability density")

        if isinstance(fixed_xlim, dict) and signal_col in fixed_xlim:
            xlim = fixed_xlim[signal_col]
        else:
            xlim = _robust_xlim(s_all, q_low=xlim_quantiles[0], q_high=xlim_quantiles[1])
        if xlim is not None:
            plt.xlim(*xlim)

        fig.savefig(out_dir / f"{signal_col}_by_drivingDirection.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_signal_by_class(df_tracks, df_tracks_meta, cols, out_dir, bins=200, xlim_quantiles=(0.001, 0.999), fixed_xlim=None):
    """
    Plot signal histograms split by vehicle class (car/truck).

    Parameters
    ----------
    df_tracks : pd.DataFrame
        Must contain recording_id, id, and each signal column.
    df_tracks_meta : pd.DataFrame
        Must contain recording_id, id, class.
    cols : sequence[str]
    out_dir : str | Path
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if df_tracks_meta is None or "class" not in df_tracks_meta.columns:
        return

    right = df_tracks_meta[["recording_id", "id", "class"]].copy()

    for signal_col in cols:
        if signal_col not in df_tracks.columns:
            continue

        left = df_tracks[["recording_id", "id", signal_col]].copy()
        df = left.merge(right, on=["recording_id", "id"], how="left", validate="many_to_one")

        df[signal_col] = pd.to_numeric(df[signal_col], errors="coerce")
        df = df.loc[df[signal_col].notna()].copy()
        if df.empty:
            continue

        cls = df["class"].astype(str).str.lower()
        cls_counts = cls.value_counts()

        car_keys = [k for k in cls_counts.index if "car" in k]
        truck_keys = [k for k in cls_counts.index if "truck" in k]

        if car_keys or truck_keys:
            cars = df.loc[cls.isin(car_keys), signal_col]
            trucks = df.loc[cls.isin(truck_keys), signal_col]
        else:
            cls_num = pd.to_numeric(df["class"], errors="coerce")
            cars = df.loc[cls_num == 1, signal_col]
            trucks = df.loc[cls_num == 2, signal_col]

        s_all = df[signal_col]

        fig = plt.figure()
        plt.hist(s_all.values, bins=bins, density=True, alpha=0.35, label="all")
        if len(cars):
            plt.hist(cars.values, bins=bins, density=True, alpha=0.35, label="cars")
        if len(trucks):
            plt.hist(trucks.values, bins=bins, density=True, alpha=0.35, label="trucks")

        plt.legend()
        plt.title(f"{signal_col} split by vehicle class")
        plt.xlabel(signal_col)
        plt.ylabel("Probability density")

        if isinstance(fixed_xlim, dict) and signal_col in fixed_xlim:
            xlim = fixed_xlim[signal_col]
        else:
            xlim = _robust_xlim(s_all, q_low=xlim_quantiles[0], q_high=xlim_quantiles[1])
        if xlim is not None:
            plt.xlim(*xlim)

        fig.savefig(out_dir / f"{signal_col}_by_class.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
