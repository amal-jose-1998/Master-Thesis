"""
Plotting helpers for W&B logging (structured Style/Action DBN).

All functions return matplotlib Figure objects and NEVER call wandb directly.

Conventions
-----------
- Style index: s in {0..S-1}
- Action index: a in {0..A-1}
- Joint grid layout: z = s*A + a  (reshape to (S, A))
"""

from __future__ import annotations

from typing import Dict, Sequence, Optional, List

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import textwrap


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _as_1d(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _as_2d(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")
    return x


def _as_3d(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {x.shape}")
    return x


def _finite_or_empty(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    return x[np.isfinite(x)]


def _wrap_header(s: str, width: int) -> str:
    s = str(s)
    if width is None or width <= 0:
        return s
    return "\n".join(
        textwrap.wrap(s, width=width, break_long_words=True, break_on_hyphens=False)
    )


def joint_grid_from_flat(values_flat, S: int, A: int) -> np.ndarray:
    v = _as_1d(values_flat)
    if v.size != S * A:
        raise ValueError(f"Expected length S*A={S*A}, got {v.size}")
    return v.reshape(S, A)


# -----------------------------------------------------------------------------
# Generic plots
# -----------------------------------------------------------------------------
def plot_bar(values, title: str, xlabel: str, ylabel: str) -> plt.Figure:
    v = _as_1d(values)
    fig, ax = plt.subplots()
    ax.bar(np.arange(v.size), v)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def plot_line(values, title: str, xlabel: str, ylabel: str, marker: str = "o") -> plt.Figure:
    v = _as_1d(values)
    fig, ax = plt.subplots()
    ax.plot(np.arange(v.size), v, marker=marker)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig

def _annotate_heatmap(ax, A, fmt="{:.3f}", fontsize=8, threshold=None):
    """
    Write numeric values into each heatmap cell.

    Parameters
    ax : matplotlib Axes
    A : 2D array
    fmt : str
        Format string for each cell value.
    fontsize : int
    threshold : float or None
        If set, use white text on cells with |val| >= threshold, else black.
        (Helps readability on strong colors.)
    """
    A = np.asarray(A, dtype=np.float64)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            v = A[i, j]
            if not np.isfinite(v):
                s = "nan"
            else:
                s = fmt.format(float(v))

            color = "black"
            if threshold is not None and np.isfinite(v):
                color = "white" if abs(float(v)) >= float(threshold) else "black"

            ax.text(j, i, s, ha="center", va="center", fontsize=fontsize, color=color)

def plot_heatmap(M, title: str, xlabel: str, ylabel: str, aspect: str = "auto", annotate: bool = True, fmt: str = "{:.3f}", fontsize: int = 8,) -> plt.Figure:
    A = _as_2d(M)
    fig, ax = plt.subplots()
    im = ax.imshow(A, aspect=aspect)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if annotate:
        # optional threshold heuristic for better contrast
        finite = A[np.isfinite(A)]
        thr = None
        if finite.size:
            thr = float(np.max(np.abs(finite))) * 0.6
        _annotate_heatmap(ax, A, fmt=fmt, fontsize=fontsize, threshold=thr)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def plot_hist(
    values,
    title: str,
    xlabel: str,
    ylabel: str = "count",
    bins: Optional[int] = None,
) -> plt.Figure:
    v = _finite_or_empty(_as_1d(values))
    fig, ax = plt.subplots()
    if v.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    if bins is None:
        vmax = float(np.max(v)) if v.size else 0.0
        if not np.isfinite(vmax):
            bins = 30
        else:
            bins = int(min(60, max(10, vmax if vmax >= 1 else 30)))

    ax.hist(v, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# DBN-specific plots
# -----------------------------------------------------------------------------
def plot_pi_s0(pi_s0) -> plt.Figure:
    return plot_bar(pi_s0, "Initial style distribution π(s0)", "style s", "probability")


def plot_pi_a0_given_s0(pi_a0_given_s0, annotate: bool = True, fmt: str = "{:.3f}", fontsize: int = 8) -> plt.Figure:
    P = _as_2d(pi_a0_given_s0)
    fig, ax = plt.subplots()
    im = ax.imshow(P, aspect="auto")
    ax.set_title("Initial action distribution π(a0 | s0)")
    ax.set_xlabel("action a")
    ax.set_ylabel("style s")
    if annotate:
        finite = P[np.isfinite(P)]
        thr = float(np.max(np.abs(finite))) * 0.6 if finite.size else None
        _annotate_heatmap(ax, P, fmt=fmt, fontsize=fontsize, threshold=thr)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def plot_A_s(A_s) -> plt.Figure:
    return plot_heatmap(A_s, "Style transition A_s: P(s_t | s_{t-1})", "s_t", "s_{t-1}")


def plot_A_s_diag(A_s) -> plt.Figure:
    A = _as_2d(A_s)
    d = np.diag(A)
    return plot_line(d, "diag(A_s): P(s_t=s | s_{t-1}=s)", "style s", "stay probability")


def plot_A_a_per_style(A_a, annotate: bool = True, fmt: str = "{:.3f}", fontsize: int = 8) -> Dict[str, plt.Figure]:
    A = _as_3d(A_a)  # (S, Aprev, Acur)
    S = int(A.shape[0])
    figs: Dict[str, plt.Figure] = {}
    for s in range(S):
        fig, ax = plt.subplots()
        im = ax.imshow(A[s], aspect="auto")
        ax.set_title(f"Action transition A_a for style s={s}\nP(a_t | a_{{t-1}}, s_t=s)")
        ax.set_xlabel("a_t")
        ax.set_ylabel("a_{t-1}")

        if annotate:
            finite = A[s][np.isfinite(A[s])]
            thr = float(np.max(np.abs(finite))) * 0.6 if finite.size else None
            _annotate_heatmap(ax, A[s], fmt=fmt, fontsize=fontsize, threshold=thr)

        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        figs[f"style_{s}"] = fig
    return figs


def plot_A_a_diag_per_style(A_a) -> Dict[str, plt.Figure]:
    A = _as_3d(A_a)
    S = int(A.shape[0])
    figs: Dict[str, plt.Figure] = {}
    for s in range(S):
        d = np.diag(A[s])
        figs[f"style_{s}"] = plot_line(d, f"diag(A_a) for style s={s}", "action a", "stay probability")
    return figs


def plot_delta_heatmap(prev, new, title: str, xlabel: str, ylabel: str) -> plt.Figure:
    P = _as_2d(prev)
    N = _as_2d(new)
    D = N - P
    vmax = float(np.max(np.abs(D))) if np.any(D != 0.0) else 1.0
    fig, ax = plt.subplots()
    im = ax.imshow(D, aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def plot_delta_A_a_per_style(prev, new) -> Dict[str, plt.Figure]:
    P = _as_3d(prev)
    N = _as_3d(new)
    if P.shape != N.shape:
        raise ValueError(f"ΔA_a requires matching shapes, got {P.shape} vs {N.shape}")
    D = N - P
    S = int(D.shape[0])
    figs: Dict[str, plt.Figure] = {}
    for s in range(S):
        vmax = float(np.max(np.abs(D[s]))) if np.any(D[s] != 0.0) else 1.0
        fig, ax = plt.subplots()
        im = ax.imshow(D[s], aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(f"ΔA_a for style s={s} = A_a(new) - A_a(prev)")
        ax.set_xlabel("a_t")
        ax.set_ylabel("a_{t-1}")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        figs[f"style_{s}"] = fig
    return figs


def plot_joint_grid(values_sa, title: str, annotate: bool = False, fmt: str = "{:.3f}") -> plt.Figure:
    G = np.asarray(values_sa, dtype=np.float64)
    if G.ndim != 2:
        raise ValueError(f"plot_joint_grid expects (S,A), got shape {G.shape}")

    S, A = int(G.shape[0]), int(G.shape[1])

    fig, ax = plt.subplots()
    im = ax.imshow(G, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("action a")
    ax.set_ylabel("style s")
    ax.set_xticks(np.arange(A))
    ax.set_yticks(np.arange(S))

    if annotate:
        for s in range(S):
            for a in range(A):
                ax.text(a, s, fmt.format(float(G[s, a])), ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def plot_joint_index_grid(S: int, A: int) -> plt.Figure:
    idx = np.arange(S * A, dtype=np.int64).reshape(S, A)
    return plot_joint_grid(idx, "Joint index z = s*A + a (layout)", annotate=True, fmt="{:d}")


# -----------------------------------------------------------------------------
# Semantics (matplotlib tables)
# -----------------------------------------------------------------------------
def _format_cell(mean, std=None, fmt="{:.2f}", pm="±") -> str:
    if std is None:
        try:
            return fmt.format(float(mean))
        except Exception:
            return str(mean)
    try:
        return f"{fmt.format(float(mean))}{pm}{fmt.format(float(std))}"
    except Exception:
        return str(mean)


def plot_semantics_tables_by_style(
    means,
    feat_names: Sequence[str],
    S: int,
    A: int,
    stds=None,
    title_prefix: str = "Semantics",
    max_cols: int = 8,
    fmt: str = "{:.2f}",
    wrap_header_at: int = 18,
    header_rotation: int = 0,
    header_fontsize: int = 8,
    cell_fontsize: int = 8,
) -> Dict[str, plt.Figure]:
    """
    For each style s:
      rows = actions (a)
      cols = features
    """
    M = np.asarray(means, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError(f"means must be 2D (K,F), got {M.shape}")
    K, F = M.shape
    if K != S * A:
        raise ValueError(f"Expected K=S*A={S*A}, got K={K}")

    SD = None
    if stds is not None:
        SD = np.asarray(stds, dtype=np.float64)
        if SD.shape != M.shape:
            SD = None

    feat_names = [str(x).split("|", 1)[0].strip() for x in feat_names]
    n_parts = int(np.ceil(F / max_cols))
    figs: Dict[str, plt.Figure] = {}

    row_labels = [f"a={a}" for a in range(A)]

    for s in range(S):
        grid = np.vstack([M[s * A + a] for a in range(A)])  # (A,F)
        grid_sd = np.vstack([SD[s * A + a] for a in range(A)]) if SD is not None else None

        for p in range(n_parts):
            j0 = p * max_cols
            j1 = min(F, (p + 1) * max_cols)

            cols_raw = feat_names[j0:j1]
            cols = [_wrap_header(c, wrap_header_at) for c in cols_raw]

            cell_text: List[List[str]] = []
            for a in range(A):
                row: List[str] = []
                for j in range(j0, j1):
                    row.append(_format_cell(grid[a, j], None if grid_sd is None else grid_sd[a, j], fmt=fmt))
                cell_text.append(row)

            fig_w = max(10.0, 0.75 * (j1 - j0))
            fig_h = max(2.8, 0.55 * A + 1.2)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            ax.axis("off")

            title = f"{title_prefix} (style s={s})"
            if n_parts > 1:
                title += f"  [features {j0}:{j1}]"
            ax.set_title(title)

            tbl = ax.table(
                cellText=cell_text,
                rowLabels=row_labels,
                colLabels=cols,
                loc="center",
                cellLoc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(cell_fontsize)
            tbl.scale(1.0, 1.35)

            for ci in range(j1 - j0):
                cell = tbl[(0, ci)]
                cell.get_text().set_rotation(int(header_rotation))
                cell.get_text().set_fontsize(header_fontsize)

            fig.tight_layout()
            figs[f"style_{s}_part_{p}"] = fig

    return figs


def plot_semantics_tables_by_action(
    means,
    feat_names: Sequence[str],
    S: int,
    A: int,
    stds=None,
    title_prefix: str = "Semantics",
    max_cols: int = 10,
    fmt: str = "{:.2f}",
    wrap_header_at: int = 18,
    header_rotation: int = 0,
    header_fontsize: int = 8,
    cell_fontsize: int = 8,
) -> Dict[str, plt.Figure]:
    """
    For each action a:
      rows = styles (s)
      cols = features
    """
    M = np.asarray(means, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError(f"means must be 2D (K,F), got {M.shape}")
    K, F = M.shape
    if K != S * A:
        raise ValueError(f"Expected K=S*A={S*A}, got K={K}")

    SD = None
    if stds is not None:
        SD = np.asarray(stds, dtype=np.float64)
        if SD.shape != M.shape:
            SD = None

    feat_names = [str(x).split("|", 1)[0].strip() for x in feat_names]
    n_parts = int(np.ceil(F / max_cols))
    figs: Dict[str, plt.Figure] = {}

    row_labels = [f"s={s}" for s in range(S)]

    M3 = M.reshape(S, A, F)
    SD3 = SD.reshape(S, A, F) if SD is not None else None

    for a in range(A):
        grid = M3[:, a, :]  # (S,F)
        grid_sd = SD3[:, a, :] if SD3 is not None else None

        for p in range(n_parts):
            j0 = p * max_cols
            j1 = min(F, (p + 1) * max_cols)

            cols_raw = feat_names[j0:j1]
            cols = [_wrap_header(c, wrap_header_at) for c in cols_raw]

            cell_text: List[List[str]] = []
            for s in range(S):
                row: List[str] = []
                for j in range(j0, j1):
                    row.append(_format_cell(grid[s, j], None if grid_sd is None else grid_sd[s, j], fmt=fmt))
                cell_text.append(row)

            fig_w = max(10.0, 0.75 * (j1 - j0))
            fig_h = max(2.8, 0.55 * S + 1.2)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            ax.axis("off")

            title = f"{title_prefix} (action a={a})"
            if n_parts > 1:
                title += f"  [features {j0}:{j1}]"
            ax.set_title(title)

            tbl = ax.table(
                cellText=cell_text,
                rowLabels=row_labels,
                colLabels=cols,
                loc="center",
                cellLoc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(cell_fontsize)
            tbl.scale(1.0, 1.35)

            for ci in range(j1 - j0):
                cell = tbl[(0, ci)]
                cell.get_text().set_rotation(int(header_rotation))
                cell.get_text().set_fontsize(header_fontsize)

            fig.tight_layout()
            figs[f"action_{a}_part_{p}"] = fig

    return figs
