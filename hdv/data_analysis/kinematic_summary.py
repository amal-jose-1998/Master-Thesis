import numpy as np
import pandas as pd

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------
# Helpers
# -----------------------------
def _robust_xlim(x, q_low=0.01, q_high=0.99):
    """
    Compute robust x-axis limits based on quantiles.

    Parameters
    x: pd.Series
        Input series (ideally numeric; non-numeric will be coerced before calling).
    q_low: float
        Lower quantile level in [0, 1].
    q_high: float
        Upper quantile level in [0, 1].

    Returns
    tuple[float, float] | None
        (low, high) quantile values, or None if not computable.
    """
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return None
    lo, hi = x.quantile([q_low, q_high]).values
    if lo == hi:
        return None
    return float(lo), float(hi)

def _available_cols(df, cols):
    """
    Filter a list of candidate column names to those present in a DataFrame.

    Parameters
    df:
        Input DataFrame.
    cols:
        Candidate column names.

    Returns
    list[str]
        Only the column names that exist in `df.columns`, preserving order.
    """
    return [c for c in cols if c in df.columns]


def _quantiles(series, qs):
    """
    Compute quantiles for a numeric series (NaNs ignored).

    Parameters
    series:
        Input series (ideally numeric; non-numeric will be coerced before calling).
    qs:
        Quantile levels in [0, 1], e.g. (0.1, 0.9).

    Returns
    dict
        Mapping like {"p10": ..., "p90": ...}. Empty if `series` has no data.
    """
    out = {}
    if series.empty:
        return out
    qvals = series.quantile(list(qs), interpolation="linear")
    for q, v in zip(qs, qvals.values):
        out[f"p{int(round(q * 100))}"] = float(v)
    return out


def _summary_1d(x):
    """
    Summarize a 1D column/array-like into plot/report-friendly descriptive statistics.

    Parameters
    x:
        Column-like input (Series, array, list).

    Returns
    dict
        Dictionary with:
        - count: total entries (including NaNs)
        - nan_count: NaNs after coercion
        - min, mean, std, median, max: floats or None if empty after NaN removal
        - quantiles: dict with p1, p5, p10, p25, p75, p90, p95, p99 (if any data)
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

    return {
        "count": n,
        "nan_count": nan,
        "min": float(s.min()),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "median": float(s.median()),
        "max": float(s.max()),
        "quantiles": _quantiles(s, qs=[0.01, 0.10, 0.50, 0.90, 0.99]),
    }

# -----------------------------
# Compute API (compact summaries)
# -----------------------------
def compute_per_recording_kinematics(df_all, cols=("xVelocity", "yVelocity", "xAcceleration", "yAcceleration"), recording_col="recording_id", max_recordings=None, show_progress=False):
    """
    Compute per-recording kinematic summaries.

    Parameters
    df_all : pd.DataFrame
        Long-table data.
    cols : sequence of str
        Kinematic columns to summarize.
    recording_col : str
        Column identifying which file/recording a row comes from.
    max_recordings : int | None
        Optional cap to limit the number of recordings summarized (debug).
    show_progress:
        If True, shows a progress bar over recordings.

    Returns
    per_rec : dict
        per_rec[recording_id][col] = summary dict
    """
    if recording_col not in df_all.columns:
        raise KeyError(f"Missing required column: {recording_col}")

    use_cols = _available_cols(df_all, cols)
    per_rec = {}

    rec_ids = df_all[recording_col].dropna().unique().tolist()
    if max_recordings is not None:
        rec_ids = rec_ids[: int(max_recordings)]
    
    iterator = rec_ids
    if show_progress:
        from tqdm.auto import tqdm  
        iterator = tqdm(rec_ids, desc="Per-recording kinematics")

    for rid in iterator:
        sub = df_all[df_all[recording_col] == rid]
        per_rec[str(rid)] = {c: _summary_1d(sub[c]) for c in use_cols}

    return per_rec

def compute_per_vehicle_std(df_all, cols, group_cols=("recording_id", "id"), min_T=10, show_progress=False):
    """
    Compute per-vehicle (per-trajectory) standard deviation for selected signals.

    Parameters
    df_all : pd.DataFrame
        Long-table tracks data.
    cols : Sequence[str]
        Columns for which to compute per-vehicle standard deviation.
    group_cols : (str, str)
        Columns identifying a unique trajectory, typically ("recording_id", "id").
    min_T : int
        Minimum number of rows required for a trajectory to be included.
    show_progress : bool
        If True, show progress over trajectories (requires `tqdm`).

    Returns
    dict
        Mapping `col -> list_of_std_values` (one entry per included trajectory).
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
    skip = 0
    for _, g in iterator:
        if len(g) < int(min_T):
            skip+=1
            continue
        for c in use_cols:
            x = pd.to_numeric(g[c], errors="coerce").dropna()
            if len(x) >= int(min_T):
                std_rows[c].append(float(x.std(ddof=0)))
    if skip > 0:
        print(f"Skipped {skip} trajectories with less than {min_T} valid rows.")
    return std_rows


# -----------------------------
# Plot API 
# -----------------------------
def plot_global_kinematics(df_all, cols, out_dir, bins=200, xlim_quantiles=(0.001, 0.999), fixed_xlim=None, annotate=True):
    """
    Plot global distributions of selected kinematic variables. For each variable, this function generates a histogram over all rows
    (all recordings, vehicles, and time steps). It overlays reference lines for p10, median, and p90 to provide scale and tail information.

    Parameters
    df_all : pd.DataFrame
        Long-table tracks data.
    cols : Sequence[str]
        Columns to plot. Missing columns are skipped.
    out_dir : str | Path
        Output directory. One PNG is written per variable.
    bins : int
        Number of histogram bins.
    xlim_quantiles : (float, float)
        Quantiles to use for robust x-axis limits (set to None to disable).
    fixed_xlim : (float, float) | None
        If given, use these fixed x-axis limits instead of data-driven limits.
    annotate : bool
        If True, annotate p10, median, p90 values on the plot.
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

        xmin = s["min"]
        xmax = s["max"]
        mean = s["mean"]
        median = s["median"]
        p10 = q.get("p10", float(x.quantile(0.10)))
        p90 = q.get("p90", float(x.quantile(0.90)))

        qlo_val, qhi_val = None, None
        if xlim_quantiles is not None:
            qlo, qhi = xlim_quantiles
            qlo_val = float(x.quantile(qlo))
            qhi_val = float(x.quantile(qhi))

        if isinstance(fixed_xlim, dict) and c in fixed_xlim:
            xlim = fixed_xlim[c]
        else:
            xlim = _robust_xlim(x, q_low=xlim_quantiles[0], q_high=xlim_quantiles[1]) if xlim_quantiles is not None else None


        fig = plt.figure()
        plt.hist(x.values, bins=bins, density=True, alpha=0.75, edgecolor="black", linewidth=0.3)
        if xlim is not None:
            plt.xlim(*xlim)
        if mean is not None:
            plt.axvline(mean, linestyle="-.", color="green", label="mean")
        if median is not None:
            plt.axvline(median, linestyle="-",  color="red",    label="median")
        if p10 is not None:
            plt.axvline(p10, linestyle="--", color="orange", label="p10")
        if p90 is not None:
            plt.axvline(p90, linestyle="--", color="orange", label="p90")

        if annotate:
            lines = [
                f"min   = {xmin:.4g}",
                f"max   = {xmax:.4g}",
                f"mean  = {mean:.4g}",
                f"median= {median:.4g}",
                f"p10   = {p10:.4g}",
                f"p90   = {p90:.4g}",
            ]
            if qlo_val is not None and qhi_val is not None:
                lines.append(f"q{100*xlim_quantiles[0]:g}% = {qlo_val:.4g}")
                lines.append(f"q{100*xlim_quantiles[1]:g}% = {qhi_val:.4g}")

            ax = plt.gca()
            ax.text(
                0.98, 0.98,
                "\n".join(lines),
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

def plot_per_vehicle_variability(df_all, cols, out_dir, min_T=50, show_progress=True, bins=200, xlim_quantiles=(0.001, 0.999), fixed_xlim=None, annotate=True):
    """
    Plot distributions of per-vehicle kinematic variability (std per trajectory). Each trajectory is identified by (recording_id, id). For each selected signal,
    this function computes the standard deviation within each trajectory and plots a histogram of these std values across vehicles.

    Parameters
    df_all : pd.DataFrame
        Long-table tracks data.
    cols : Sequence[str]
        Kinematic columns for which to compute per-vehicle variability.
        Missing columns are skipped.
    out_dir : str | Path
        Output directory. One PNG is written per variable.
    min_T : int
        Minimum trajectory length required to include a vehicle.
    show_progress : bool
        If True, show a progress bar during computation.
    bins : int
        Number of histogram bins.
    xlim_quantiles : (float, float)
        Quantiles to use for robust x-axis limits (set to None to disable).
    fixed_xlim : dict[str, tuple[float, float]] | None
        If given, use these fixed x-axis limits instead of data-driven limits.
    annotate : bool
        If True, annotate mean and median std values on the plot.
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

        xmin = s["min"]
        xmax = s["max"]
        mean = s["mean"]
        median = s["median"]
        p10 = q.get("p10", float(vals_s.quantile(0.10)))
        p90 = q.get("p90", float(vals_s.quantile(0.90)))

        qlo_val, qhi_val = None, None
        if xlim_quantiles is not None:
            qlo, qhi = xlim_quantiles
            qlo_val = float(vals_s.quantile(qlo))
            qhi_val = float(vals_s.quantile(qhi))

        fig = plt.figure()
        plt.hist(vals, bins=bins, alpha=0.75, edgecolor="black", linewidth=0.3)
        xlim = None
        if isinstance(fixed_xlim, dict) and c in fixed_xlim:
            xlim = fixed_xlim[c]
        else:
            xlim = _robust_xlim(pd.Series(vals), q_low=xlim_quantiles[0], q_high=xlim_quantiles[1])
        if xlim is not None:
            plt.xlim(*xlim)

        if mean is not None:
            plt.axvline(mean, linestyle="-.", color="green", label="mean")
        if median is not None:
            plt.axvline(median, linestyle="-",  color="red",    label="median")
        if p10 is not None:
            plt.axvline(p10, linestyle="--", color="orange", label="p10")
        if p90 is not None:
            plt.axvline(p90, linestyle="--", color="orange", label="p90")
        
        if annotate:
            lines = [
                f"min   = {xmin:.4g}",
                f"max   = {xmax:.4g}",
                f"mean  = {mean:.4g}",
                f"median= {median:.4g}",
                f"p10   = {p10:.4g}",
                f"p90   = {p90:.4g}",
            ]
            if qlo_val is not None and qhi_val is not None:
                lines.append(f"q{100*xlim_quantiles[0]:g}% = {qlo_val:.4g}")
                lines.append(f"q{100*xlim_quantiles[1]:g}% = {qhi_val:.4g}")

            ax = plt.gca()
            ax.text(
                0.98, 0.98,
                "\n".join(lines),
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

def plot_per_recording_kinematics(df_all, cols, out_dir, recording_col="recording_id", max_recordings=None, show_progress=True, use_minmax_whiskers=False, annotate=True):
    """
    Plot per-recording kinematic summaries as median with p10–p90 error bars. This plot is useful to detect recording-level shifts (scenario effects).
    If recordings differ strongly, it supports introducing latent variables that separate scenario/context effects from driver-level variation.

    Parameters
    df_all : pd.DataFrame
        Long-table tracks data containing at least `recording_col` and the
        selected kinematic columns.
    cols : Sequence[str]
        Kinematic columns to summarize and plot.
    out_dir : str | Path
        Output directory. One PNG per variable is saved.
    recording_col : str, optional
        Column that identifies recordings (default: "recording_id").
    max_recordings : int | None, optional
        Optional cap on the number of recordings to plot (debug convenience).
    show_progress : bool, optional
        If True, shows a tqdm progress bar during computation.
    use_minmax_whiskers : bool, optional
        If True, add min/max whiskers for context without destroying readability.
    annotate : bool, optional
        If True, annotate median, p10, p90 values on the plot.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_rec = compute_per_recording_kinematics(df_all, cols=cols, recording_col=recording_col, max_recordings=max_recordings, show_progress=show_progress)
    if not per_rec:
        return

    rec_ids = list(per_rec.keys())

    for c in cols:
        rows = []
        med, p10, p90 = [], [], []
        vmin, vmax, vmean = [], [], [] # for min/max whiskers
        labels = []  

        for rid in rec_ids:
            s = per_rec[rid].get(c, {})
            q = s.get("quantiles", {})
            if s.get("median") is None or "p10" not in q or "p90" not in q:
                continue
            row = {
                "rid": rid,
                "median": float(s["median"]),
                "p10": float(q["p10"]),
                "p90": float(q["p90"]),
                "min": float(s["min"]) if s.get("min") is not None else np.nan,
                "max": float(s["max"]) if s.get("max") is not None else np.nan,
                "mean": float(s["mean"]) if s.get("mean") is not None else np.nan,
            }
            rows.append(row)

        if not rows:
            continue

        labels = [r["rid"] for r in rows]
        med    = np.array([r["median"] for r in rows], dtype=float)
        p10    = np.array([r["p10"] for r in rows], dtype=float)
        p90    = np.array([r["p90"] for r in rows], dtype=float)
        vmin   = np.array([r["min"] for r in rows], dtype=float)
        vmax   = np.array([r["max"] for r in rows], dtype=float)
        vmean  = np.array([r["mean"] for r in rows], dtype=float)

        x = np.arange(len(rows))
        yerr = np.vstack([med - p10, p90 - med])

        fig = plt.figure(figsize=(max(8, 0.25 * len(med)), 4))
        plt.errorbar(x, med, yerr=yerr, fmt="o", capsize=3, label="median ± (p10–p90)")

        if use_minmax_whiskers:
            # draw faint min/max "whiskers" to show extremes
            for i in range(len(med)):
                if np.isfinite(vmin[i]) and np.isfinite(vmax[i]):
                    plt.vlines(i, vmin[i], vmax[i], alpha=0.25)
        plt.xticks(x, labels, rotation=90)
        plt.title(f"Per-recording: {c}")
        plt.xlabel("recording_id")
        plt.ylabel(c)

        assert len(labels) == len(vmean) == len(med) == len(p10) == len(p90)

        if annotate:
            lines = []
            for i, rid in enumerate(labels):
                lines.append(
                    f"{rid}: "
                    f"min={vmin[i]:.3g}, "
                    f"mean={vmean[i]:.3g}, "
                    f"med={med[i]:.3g}, "
                    f"max={vmax[i]:.3g}, "
                    f"p10={p10[i]:.3g}, "
                    f"p90={p90[i]:.3g}"
                )

            ax = plt.gca()
            ax.text(
                1.01, 0.5,
                "\n".join(lines),
                transform=ax.transAxes,
                ha="left", va="center",
                fontsize=8,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="black",
                    alpha=0.9,
                ),
            )

        fig.savefig(out_dir / f"per_recording_{c}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

def plot_signal_by_direction(df_tracks, df_tracks_meta, cols, out_dir, bins=200, xlim_quantiles=(0.001, 0.999), fixed_xlim=None):
    """
    Plot `signal_col` histogram split by driving direction using a minimal join.
    
    Parameters
    df_tracks: pd.DataFrame
        Long-table tracks DataFrame (must contain: recording_id, id, <signal_col>)
    df_tracks_meta: pd.DataFrame
        tracksMeta DataFrame (must contain: recording_id, id, drivingDirection)
    cols: Sequence[str]
        List of signal columns to plot, e.g. ["xVelocity", "xAcceleration"]
    out_dir: Path | str
        Output directory for PNGs
    bins: int
        Number of histogram bins
    xlim_quantiles: (float, float)
        Quantiles to use for robust x-axis limits
    fixed_xlim: dict[str, tuple[float, float]] | None
        If given, use these fixed x-axis limits instead of data-driven limits.
    

    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for signal_col in cols:
        # quick checks
        if signal_col not in df_tracks.columns:
            return
        if df_tracks_meta is None or "drivingDirection" not in df_tracks_meta.columns:
            return

        left = df_tracks[["recording_id", "id", signal_col]].copy()
        right = df_tracks_meta[["recording_id", "id", "drivingDirection"]].copy()
        df = left.merge(right, on=["recording_id", "id"], how="left", validate="many_to_one")

        df[signal_col] = pd.to_numeric(df[signal_col], errors="coerce")
        df["drivingDirection"] = pd.to_numeric(df["drivingDirection"], errors="coerce")

        mask = df[signal_col].notna() & df["drivingDirection"].notna()
        df = df.loc[mask]
        if df.empty:
            return

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

        xlim = None
        if isinstance(fixed_xlim, dict) and signal_col in fixed_xlim:
            xlim = fixed_xlim[signal_col]
        else:
            xlim = _robust_xlim(s_all, q_low=xlim_quantiles[0], q_high=xlim_quantiles[1])
        if xlim is not None:
            plt.xlim(*xlim)
        out_path = out_dir / f"{signal_col}_by_drivingDirection.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_signal_by_class(df_tracks, df_tracks_meta, cols, out_dir, bins=200, xlim_quantiles=(0.001, 0.999), fixed_xlim=None):
    """
    Plot `signal_col` histogram split by vehicle class using a minimal join.

    Parameters
    df_tracks: pd.DataFrame 
        Long-table tracks DataFrame (must contain: recording_id, id, <signal_col>)
    df_tracks_meta: pd.DataFrame
        tracksMeta DataFrame (must contain: recording_id, id, class)
    cols: Sequence[str]
        List of signal columns to plot, e.g. ["xVelocity", "xAcceleration"]
    out_dir: Path | str
        Output directory for PNGs
    bins: int
        Number of histogram bins
    xlim_quantiles: (float, float)
        Quantiles to use for robust x-axis limits
    fixed_xlim: dict[str, tuple[float, float]] | None
        If given, use these fixed x-axis limits instead of data-driven limits.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for signal_col in cols:
        out_path = out_dir / f"{signal_col}_by_class.png"
        # quick checks
        if signal_col not in df_tracks.columns:
            return
        if df_tracks_meta is None or "class" not in df_tracks_meta.columns:
            return

        left = df_tracks[["recording_id", "id", signal_col]].copy()
        right = df_tracks_meta[["recording_id", "id", "class"]].copy()
        df = left.merge(right, on=["recording_id", "id"], how="left", validate="many_to_one")

        df[signal_col] = pd.to_numeric(df[signal_col], errors="coerce")
        # normalize class column to string for matching
        cls = df["class"].astype(str).str.lower()

        # mask valid signal entries
        mask = df[signal_col].notna()
        df = df.loc[mask].copy()
        if df.empty:
            return

        # detect textual class labels
        cls_counts = cls[mask].value_counts()
        car_idx = [k for k in cls_counts.index if "car" in k]
        truck_idx = [k for k in cls_counts.index if "truck" in k]

        if car_idx or truck_idx:
            cars = df.loc[cls.isin(car_idx), signal_col]
            trucks = df.loc[cls.isin(truck_idx), signal_col]
        else:
            # fallback to numeric codes (1=car, 2=truck)
            try:
                cls_num = pd.to_numeric(df["class"], errors="coerce")
                cars = df.loc[cls_num == 1, signal_col]
                trucks = df.loc[cls_num == 2, signal_col]
            except Exception:
                cars = pd.Series(dtype=float)
                trucks = pd.Series(dtype=float)

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

        xlim = None
        if isinstance(fixed_xlim, dict) and signal_col in fixed_xlim:
            xlim = fixed_xlim[signal_col]
        else:
            xlim = _robust_xlim(s_all, q_low=xlim_quantiles[0], q_high=xlim_quantiles[1])
        if xlim is not None:
            plt.xlim(*xlim)

        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)