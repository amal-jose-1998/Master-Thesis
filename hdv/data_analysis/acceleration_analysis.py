from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class AccelBinConfig:
    """Configuration for data-driven acceleration bins."""
    quantiles: Tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)


@dataclass(frozen=True)
class AccelBinStats:
    """
    Outputs of acceleration bin analysis.

    Attributes
    config : AccelBinConfig
        Binning configuration.
    edges : np.ndarray
        Bin edges (quantile values), shape (Q,).
    bin_counts : np.ndarray
        Total sample counts per bin, shape (B,).
    bin_fractions : np.ndarray
        Total sample fractions per bin, shape (B,).
    transition_counts : np.ndarray
        Bin transition counts across time, shape (B, B).
    transition_probs : np.ndarray
        Row-normalized transition probabilities, shape (B, B).
    per_lane_table : pd.DataFrame
        Lane-wise bin fractions (one row per lane).
    per_vehicle_table : pd.DataFrame
        Per-trajectory bin fractions (one row per (recording_id, id)).
    """
    config: AccelBinConfig
    edges: np.ndarray
    bin_counts: np.ndarray
    bin_fractions: np.ndarray
    transition_counts: np.ndarray
    transition_probs: np.ndarray
    per_lane_table: pd.DataFrame
    per_vehicle_table: pd.DataFrame

    def to_dict(self):
        return {
            "config": {"quantiles": list(self.config.quantiles)},
            "edges": self.edges.tolist(),
            "bin_counts": self.bin_counts.tolist(),
            "bin_fractions": self.bin_fractions.tolist(),
            "transition_counts": self.transition_counts.tolist(),
            "transition_probs": self.transition_probs.tolist(),
            "per_lane_table": self.per_lane_table.to_dict(orient="records"),
            "per_vehicle_table": self.per_vehicle_table.to_dict(orient="records"),
        }

def _coerce_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def compute_accel_bins(ax, config):
    """
    Assign each ax value to a bin index using quantile-based edges.

    Returns
    bin_id : np.ndarray, shape (N,), dtype=int
        Values in [0..B-1], where B = len(edges)+1
    """
    ax = ax[np.isfinite(ax)]
    if ax.size == 0:
        return np.array([], dtype=np.int64)

    edges = np.quantile(ax, config.quantiles)
    # digitize returns bin index in [0..len(edges)]
    return edges


def assign_bins(ax, edges):
    """Assign ax to bins given edges."""
    ax = ax.astype(float, copy=False)
    # bins: (-inf, e0], (e0, e1], ..., (e_{Q-1}, +inf)
    return np.digitize(ax, edges, right=True).astype(np.int64)


def compute_accel_bin_stats(df_all, ax_col="xAcceleration", lane_col="laneId", traj_cols=("recording_id", "id"), config=None, frame_col="frame", min_T=10):
    """
    Data-driven braking/acceleration pattern analysis via quantile bins.

    Returns
    AccelBinStats
    """
    if config is None:
        config = AccelBinConfig()

    if ax_col not in df_all.columns:
        raise KeyError(f"Missing required acceleration column: {ax_col}")
    for c in traj_cols:
        if c not in df_all.columns:
            raise KeyError(f"Missing required trajectory column: {c}")
    if frame_col not in df_all.columns:
        raise KeyError(f"Missing required frame column: {frame_col}")

    # Global edges
    ax_series = _coerce_numeric(df_all[ax_col]).dropna()
    ax_vals = ax_series.to_numpy(dtype=float)
    ax_vals = ax_vals[np.isfinite(ax_vals)]
    if ax_vals.size == 0:
        empty = np.zeros((0,), dtype=np.int64)
        return AccelBinStats(
            config=config,
            edges=np.array([], dtype=float),
            bin_counts=empty,
            bin_fractions=empty.astype(float),
            transition_counts=np.zeros((0, 0), dtype=np.int64),
            transition_probs=np.zeros((0, 0), dtype=float),
            per_lane_table=pd.DataFrame(),
            per_vehicle_table=pd.DataFrame(),
        )

    edges = np.quantile(ax_vals, config.quantiles)
    B = int(edges.size + 1)

    # Global bin usage
    global_bins = assign_bins(ax_vals, edges)
    bin_counts = np.bincount(global_bins, minlength=B).astype(np.int64)
    bin_fractions = (bin_counts / max(int(bin_counts.sum()), 1)).astype(float)

    # Transition counts across trajectories
    trans_counts = np.zeros((B, B), dtype=np.int64)
    per_vehicle_rows = []

    sub = df_all[list(traj_cols) + [frame_col, ax_col]].copy()
    sub[ax_col] = _coerce_numeric(sub[ax_col])

    for key, g in sub.groupby(list(traj_cols), sort=False):
        g = g.sort_values(frame_col)
        T = int(len(g))
        if T < int(min_T):
            continue

        v = g[ax_col].to_numpy(dtype=float)
        mask = np.isfinite(v)
        if mask.sum() < 2:
            continue

        z = assign_bins(v[mask], edges)
        # per-vehicle bin fractions
        counts = np.bincount(z, minlength=B).astype(np.int64)
        fracs = counts / max(int(counts.sum()), 1)

        row = {traj_cols[0]: key[0], traj_cols[1]: int(key[1]), "T": T}
        for b in range(B):
            row[f"bin{b}_frac"] = float(fracs[b])
        per_vehicle_rows.append(row)

        # transitions
        z0 = z[:-1]
        z1 = z[1:]
        for a, b in zip(z0, z1):
            trans_counts[int(a), int(b)] += 1

    per_vehicle_table = pd.DataFrame(per_vehicle_rows)

    # Normalize transitions row-wise
    row_sums = trans_counts.sum(axis=1, keepdims=True)
    trans_probs = np.divide(
        trans_counts.astype(float),
        np.where(row_sums == 0, 1.0, row_sums),
    )

    # Lane-wise bin fractions
    if lane_col in df_all.columns:
        lanes = df_all[[lane_col, ax_col]].copy()
        lanes[ax_col] = _coerce_numeric(lanes[ax_col])
        lanes = lanes.dropna(subset=[ax_col])

        lane_rows = []
        for lane_id, g in lanes.groupby(lane_col, sort=True):
            v = g[ax_col].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            z = assign_bins(v, edges)
            counts = np.bincount(z, minlength=B).astype(np.int64)
            fracs = counts / max(int(counts.sum()), 1)

            row = {lane_col: int(lane_id), "count": int(v.size)}
            for b in range(B):
                row[f"bin{b}_frac"] = float(fracs[b])
            lane_rows.append(row)

        per_lane_table = pd.DataFrame(lane_rows).sort_values(lane_col).reset_index(drop=True)
    else:
        per_lane_table = pd.DataFrame()

    return AccelBinStats(
        config=config,
        edges=edges.astype(float),
        bin_counts=bin_counts,
        bin_fractions=bin_fractions,
        transition_counts=trans_counts,
        transition_probs=trans_probs,
        per_lane_table=per_lane_table,
        per_vehicle_table=per_vehicle_table,
    )


def save_accel_bin_plots(out_dir, df_all, stats, ax_col="xAcceleration"):
    """
    Save plots for data-driven acceleration bins.

    Files
    - ax_hist.png
        Raw acceleration distribution (log count).
    - ax_bin_fractions.png
        Bar plot of global bin fractions.
    - ax_bin_transition_probs.png
        Heatmap of transition probabilities between bins.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Raw ax histogram
    ax = pd.to_numeric(df_all.get(ax_col, pd.Series([], dtype=float)), errors="coerce").dropna().to_numpy()
    if ax.size > 0:
        plt.figure()
        plt.hist(ax, bins=160, log=True)
        plt.title("Longitudinal acceleration distribution (ax)")
        plt.xlabel("ax [m/s^2]")
        plt.ylabel("count (log)")
        plt.tight_layout()
        plt.savefig(out_dir / "ax_hist.png", dpi=200)
        plt.close()

    # Bin fractions
    plt.figure()
    x = np.arange(stats.bin_fractions.size)
    plt.bar(x, stats.bin_fractions)
    plt.title("Acceleration bin fractions (quantile-based)")
    plt.xlabel("bin id")
    plt.ylabel("fraction of samples")
    plt.tight_layout()
    plt.savefig(out_dir / "ax_bin_fractions.png", dpi=200)
    plt.close()

    # Transition probability heatmap
    if stats.transition_probs.size > 0:
        plt.figure(figsize=(7, 6))
        plt.imshow(stats.transition_probs, aspect="auto", vmin=0.0, vmax=1.0)
        plt.title("Acceleration bin transition probabilities")
        plt.xlabel("next bin")
        plt.ylabel("current bin")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(out_dir / "ax_bin_transition_probs.png", dpi=200)
        plt.close()
