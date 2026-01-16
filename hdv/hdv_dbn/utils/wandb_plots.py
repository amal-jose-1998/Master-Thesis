"""W&B plotting helpers for HDV DBN training."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import textwrap


def plot_pi_s0_bar(pi_s):
    """Bar plot of initial style distribution π_S."""
    pi_s = np.asarray(pi_s, dtype=np.float64).ravel()
    fig, ax = plt.subplots()
    ax.bar(np.arange(pi_s.size), pi_s)
    ax.set_title("Initial style distribution π_S")
    ax.set_xlabel("style s")
    ax.set_ylabel("probability")
    return fig

def plot_pi_a0_given_s0_heatmap(pi_a0_given_s0):
    """Heatmap of initial action distribution π(a0 | S0). Rows: style s, cols: action a."""
    P = np.asarray(pi_a0_given_s0, dtype=np.float64)
    fig, ax = plt.subplots()
    im = ax.imshow(P, aspect="auto")
    ax.set_title("Initial action distribution π(a0 | S0)")
    ax.set_xlabel("action a")
    ax.set_ylabel("style s")
    fig.colorbar(im, ax=ax)
    return fig

def plot_A_s_heatmap(A_s):
    """Heatmap of style transition A_S (rows: s_prev, cols: s_cur)."""
    A = np.asarray(A_s, dtype=np.float64)
    fig, ax = plt.subplots()
    im = ax.imshow(A, aspect="auto")
    ax.set_title("Style transition A_S: P(S_t=s | S_{t-1}=s_prev)")
    ax.set_xlabel("s_cur")
    ax.set_ylabel("s_prev")
    fig.colorbar(im, ax=ax)
    return fig


def plot_A_s_diag(A_s):
    """Line plot of stay probabilities diag(A_S)."""
    A = np.asarray(A_s, dtype=np.float64)
    d = np.diag(A)
    fig, ax = plt.subplots()
    ax.plot(np.arange(d.size), d, marker="o")
    ax.set_title("A_S diagonal: P(S_t=s | S_{t-1}=s)")
    ax.set_xlabel("style s")
    ax.set_ylabel("stay probability")
    return fig


def plot_A_a_heatmaps(A_a):
    """
    Heatmaps of action transition A_a for each style:
      A_a[s, a_prev, a_cur]
    Returns a dict: {"style_0": fig0, ...}
    """
    A = np.asarray(A_a, dtype=np.float64)
    S, A_prev, A_cur = A.shape
    figs = {}
    for s in range(S):
        fig, ax = plt.subplots()
        im = ax.imshow(A[s], aspect="auto")
        ax.set_title(f"Action transition A_a for style s={s}\nP(a_t | a_{{t-1}}, S_t=s)")
        ax.set_xlabel("a_cur")
        ax.set_ylabel("a_prev")
        fig.colorbar(im, ax=ax)
        figs[f"style_{s}"] = fig
    return figs


def plot_A_s_delta(A_prev, A_new):
    """Heatmap of ΔA_S = A_new - A_prev."""
    P = np.asarray(A_prev, dtype=np.float64)
    N = np.asarray(A_new, dtype=np.float64)
    D = N - P
    vmax = float(np.max(np.abs(D))) if np.any(D != 0.0) else 1.0
    fig, ax = plt.subplots()
    im = ax.imshow(D, aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_title("ΔA_S = A_S(new) - A_S(prev)")
    ax.set_xlabel("s_cur")
    ax.set_ylabel("s_prev")
    fig.colorbar(im, ax=ax)
    return fig


def plot_A_a_delta_per_style(Aa_prev, Aa_new):
    """ΔA_a heatmaps per style. Returns dict of figs keyed by style."""
    P = np.asarray(Aa_prev, dtype=np.float64)
    N = np.asarray(Aa_new, dtype=np.float64)
    D = N - P
    S = D.shape[0]
    figs = {}
    for s in range(S):
        vmax = float(np.max(np.abs(D[s]))) if np.any(D[s] != 0.0) else 1.0
        fig, ax = plt.subplots()
        im = ax.imshow(D[s], aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(f"ΔA_a for style s={s} = A_a(new) - A_a(prev)")
        ax.set_xlabel("a_cur")
        ax.set_ylabel("a_prev")
        fig.colorbar(im, ax=ax)
        figs[f"style_{s}"] = fig
    return figs


def plot_state_mass_bar(values, title, ylabel):
    """Generic bar plot over joint states (e.g., responsibility mass/fraction)."""
    fig, ax = plt.subplots()
    ax.bar(np.arange(values.size), values)
    ax.set_title(title)
    ax.set_xlabel("joint state z")
    ax.set_ylabel(ylabel)
    return fig


def plot_state_line(values, title, ylabel):
    """Generic line plot over joint states (e.g., mean norms, variance sums)."""
    fig, ax = plt.subplots()
    ax.plot(np.arange(values.size), values, marker="o")
    ax.set_title(title)
    ax.set_xlabel("joint state z")
    ax.set_ylabel(ylabel)
    return fig


def plot_lane_heatmap(lane_p):
    """Heatmap of p(lane_pos | z). Rows: z, cols: lane category."""
    fig, ax = plt.subplots()
    im = ax.imshow(lane_p, aspect="auto")
    ax.set_title("Lane categorical p(lane_pos | z)")
    ax.set_xlabel("lane_pos category")
    ax.set_ylabel("joint state z")
    fig.colorbar(im, ax=ax)
    return fig

def plot_bernoulli_means_per_state(values, num_states, title="Bernoulli features: mean p(x=1) per state", ylabel="mean p(x=1) across Bernoulli dims",):
    """Line plot of a single Bernoulli summary statistic per joint state."""
    fig, ax = plt.subplots()
    ax.plot(np.arange(num_states), values, marker="o")
    ax.set_title(title)
    ax.set_xlabel("joint state z")
    ax.set_ylabel(ylabel)
    return fig

def plot_bernoulli_feature_means_per_state(bern_p, bern_names=None, title: str = "Bernoulli features: p(x=1) per state"):
    """Plot per-feature Bernoulli probabilities per joint state.

    Parameters
    ----------
    bern_p : array-like, shape (K, B)
        Bernoulli probabilities p(x=1) for each joint state (rows) and binary feature (cols).
    bern_names : list[str] | None
        Optional names for the Bernoulli features.
    """
    P = np.asarray(bern_p, dtype=np.float64)
    if P.ndim != 2:
        raise ValueError(f"bern_p must be 2D (K,B), got shape {P.shape}")
    K, B = P.shape

    if bern_names is None or len(bern_names) != B:
        bern_names = [f"bern_{j}" for j in range(B)]

    fig, ax = plt.subplots()
    x = np.arange(K)
    for j in range(B):
        ax.plot(x, P[:, j], marker="o", label=str(bern_names[j]))

    ax.set_title(title)
    ax.set_xlabel("joint state z")
    ax.set_ylabel("p(x=1)")
    ax.legend(loc="best", fontsize=8)
    return fig


def plot_run_length_distribution(run_lengths, title="Run-length distribution", xlabel="segment length (timesteps)"):
    rl = np.asarray(run_lengths).ravel()
    rl = rl[np.isfinite(rl)]
    rl = rl[rl > 0]

    fig, ax = plt.subplots()
    if rl.size == 0:
        ax.text(0.5, 0.5, "No run lengths", ha="center", va="center")
        ax.set_axis_off()
        return fig

    # log-like bins without actually log-scaling axes (keeps it simple)
    max_rl = int(np.max(rl))
    bins = min(60, max(10, max_rl))
    ax.hist(rl, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("# segments")
    return fig

def plot_entropy_distribution(ent, title="Posterior entropy (normalized)", xlabel="H / log(K)"):
    e = np.asarray(ent, dtype=np.float64).ravel()
    e = e[np.isfinite(e)]
    fig, ax = plt.subplots()
    if e.size == 0:
        ax.text(0.5, 0.5, "No entropy values", ha="center", va="center")
        ax.set_axis_off()
        return fig
    ax.hist(e, bins=30)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("# timesteps")
    ax.set_xlim(0.0, 1.0)
    return fig


def plot_key_feature_per_feature(means, stds, feat_names, title_prefix="Posterior-weighted"):
    means = np.asarray(means, dtype=np.float64)
    stds  = np.asarray(stds, dtype=np.float64)
    K, F = means.shape

    figs = {}
    x = np.arange(K)

    for j in range(F):
        fname = str(feat_names[j])
        fig, ax = plt.subplots()
        ax.errorbar(x, means[:, j], yerr=stds[:, j], marker="o", linestyle="-", capsize=3)
        ax.set_title(f"{title_prefix}: {fname}")
        ax.set_xlabel("joint state z")
        ax.set_ylabel("value (mean ± std)")
        figs[fname] = fig

    return figs


def joint_grid_from_flat(values_flat, S, A):
    """
    Reshape joint-state flattened vector into a (S, A) grid using z = s*A + a.

    Parameters
    values_flat : array-like, shape (S*A,)
    S : int
    A : int

    Returns
    grid : np.ndarray, shape (S, A)
    """
    v = np.asarray(values_flat).ravel()  
    if v.size != S * A:
        raise ValueError(f"Expected values_flat of length {S*A}, got {v.size}")
    return v.reshape(S, A)


def plot_joint_grid_heatmap(values_flat, S, A, title, xlabel="action a", ylabel="style s"):
    """
    Heatmap over joint states laid out as a (style x action) grid.
    """
    grid = joint_grid_from_flat(values_flat, S, A)
    fig, ax = plt.subplots()
    im = ax.imshow(grid, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # tick labels
    ax.set_xticks(np.arange(A))
    ax.set_yticks(np.arange(S))

    fig.colorbar(im, ax=ax)
    return fig


def plot_joint_grid_annotated(values_flat, S, A, title, fmt="{:.2e}"):
    """
    Same as heatmap, but writes the numeric value in each cell.
    Useful for small S,A (like 3x3).
    """
    grid = joint_grid_from_flat(values_flat, S, A)
    fig, ax = plt.subplots()
    im = ax.imshow(grid, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("action a")
    ax.set_ylabel("style s")
    ax.set_xticks(np.arange(A))
    ax.set_yticks(np.arange(S))

    # annotate each cell
    for s in range(S):
        for a in range(A):
            ax.text(a, s, fmt.format(grid[s, a]), ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax)
    return fig


def plot_A_heatmap(A, title="Transition matrix"):
    """Generic heatmap for any transition matrix."""
    M = np.asarray(A, dtype=np.float64)
    fig, ax = plt.subplots()
    im = ax.imshow(M, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("state (current)")
    ax.set_ylabel("state (previous)")
    fig.colorbar(im, ax=ax)
    return fig


def plot_A_diag(A, title="Diagonal of transition matrix"):
    """Line plot of diag(A): stay probabilities per state."""
    M = np.asarray(A, dtype=np.float64)
    d = np.diag(M)
    fig, ax = plt.subplots()
    ax.plot(np.arange(d.size), d, marker="o")
    ax.set_title(title)
    ax.set_xlabel("state")
    ax.set_ylabel("stay probability")
    return fig


def plot_joint_sa_grid(values_flat, S, A, title="(Style × Action) grid", annotate=False):
    """
    Plot flattened joint-state values on a (S x A) grid using z = s*A + a.
    If annotate=True, prints values in each cell (good for small grids).
    """
    if annotate:
        return plot_joint_grid_annotated(values_flat, S, A, title=title, fmt="{:.3f}")
    return plot_joint_grid_heatmap(values_flat, S, A, title=title)


def plot_joint_index_grid(S, A, title="Joint index grid z = s*A + a"):
    """Annotated grid showing the joint-state index z in each (s,a) cell."""
    idx = np.arange(S * A, dtype=np.int64)
    return plot_joint_grid_annotated(idx, S, A, title=title, fmt="{:d}")


def plot_lc_heatmap(lc_p):
    """Heatmap of p(lc | z). Rows: z, cols: lc category (left, none, right)."""
    fig, ax = plt.subplots()
    im = ax.imshow(lc_p, aspect="auto")
    ax.set_title("Lane-change categorical p(lc | z)")
    ax.set_xlabel("lc category (left / none / right)")
    ax.set_ylabel("joint state z")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["left(-1)", "none(0)", "right(+1)"])
    fig.colorbar(im, ax=ax)
    return fig


#def plot_semantics_heatmap(means, feat_names, title="Semantics heatmap (means)"):
#    """
#    Heatmap of posterior-weighted semantic means.
#    Rows: joint state z
#    Cols: semantic feature
#    """
#    M = np.asarray(means, dtype=np.float64)
#    fig_w = max(8.0, 0.45 * M.shape[1])   # expand width with #features
#    fig_h = max(4.0, 0.35 * M.shape[0])   # expand height with #states
#    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
#
#    im = ax.imshow(M, aspect="auto")
#    ax.set_title(title)
#    ax.set_xlabel("semantic feature")
#    ax.set_ylabel("joint state z")
#
#    ax.set_xticks(np.arange(len(feat_names)))
#    ax.set_xticklabels([str(f) for f in feat_names], rotation=60, ha="right", fontsize=7)
#    ax.set_yticks(np.arange(M.shape[0]))
#    ax.tick_params(axis="y", labelsize=8)
#
#    fig.colorbar(im, ax=ax)
#    fig.tight_layout()
#    return fig


#def plot_semantics_by_style(means, feat_names, S, A, title_prefix="Semantics heatmap"):
#    """
#    Plot semantics heatmaps split by style.
#    For each style s, rows = action a, cols = semantic features.
#    Returns dict: {"style_0": fig0, ...}
#    """
#    M = np.asarray(means, dtype=np.float64)
#    K, F = M.shape
#    assert K == S * A, "Expected K = S * A"
#
#    figs = {}
#    for s in range(S):
#        rows = []
#        for a in range(A):
#            z = s * A + a
#            rows.append(M[z])
#        grid = np.vstack(rows)  # (A, F)
#
#        fig_w = max(8.0, 0.45 * F)
#        fig_h = max(3.0, 0.35 * A)
#        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
#
#        im = ax.imshow(grid, aspect="auto")
#        ax.set_title(f"{title_prefix} (style s={s})")
#        ax.set_xlabel("semantic feature")
#        ax.set_ylabel("action a")
#
#        ax.set_xticks(np.arange(F))
#        ax.set_xticklabels(feat_names, rotation=60, ha="right", fontsize=7)
#        ax.set_yticks(np.arange(A))
#
#        fig.colorbar(im, ax=ax)
#        fig.tight_layout()
#        figs[f"style_{s}"] = fig
#
#    return figs

def _format_cell(mean, std=None, fmt="{:.2f}", pm="±"):
    if std is None:
        try:
            return fmt.format(float(mean))
        except Exception:
            return str(mean)
    try:
        return f"{fmt.format(float(mean))}{pm}{fmt.format(float(std))}"
    except Exception:
        return str(mean)


def plot_semantics_table_by_style(
    means,
    feat_names,
    S,
    A,
    stds=None,
    title_prefix="Semantics (raw) table",
    max_cols=10,
    fmt="{:.2f}",
    wrap_header_at=14,          # wrap feature names at ~N chars per line
    header_rotation="auto",     # "auto" | 0 | 30 | 45 | 60 | 90
    header_fontsize=8,
    cell_fontsize=8,
    rowlabel_fontsize=8,
    base_col_width=0.75,        # inches per column baseline
    min_fig_w=10.0,
    max_fig_w=28.0,
    fig_h_per_row=0.55,
    extra_top_margin=0.18,      # increases room for wrapped headers
):
    """
    Render numeric tables (matplotlib table) for semantics.
    For each style s: rows=action a, cols=features.
    If many features, split into chunks of max_cols.

    Parameters
    ----------
    means : (K,F) array
    stds  : (K,F) array or None
    feat_names : list[str], length F
    S, A : ints, K == S*A
    max_cols : int, number of feature columns per figure
    fmt : format string for mean/std

    Returns
    -------
    figs : dict[str, matplotlib.figure.Figure]
    keys like: "style_0_part_0"
    """

    def _wrap(s: str, width: int) -> str:
        s = str(s)
        if width is None or width <= 0:
            return s
        # keep underscores; wrap on underscores/spaces reasonably
        # (textwrap will break long tokens; this is fine for feature names)
        return "\n".join(textwrap.wrap(s, width=width, break_long_words=True, break_on_hyphens=False))


    M = np.asarray(means, dtype=np.float64)
    K, F = M.shape
    assert K == S * A, f"Expected K=S*A={S*A}, got {K}"

    SD = None
    if stds is not None:
        SD = np.asarray(stds, dtype=np.float64)
        if SD.shape != M.shape:
            SD = None

    # Clean display names: keep feature name, drop condition suffix like " | right_rear_exist"
    # Example: "right_rear_dvx | right_rear_exist" -> "right_rear_dvx"
    feat_names = [str(x).split("|", 1)[0].strip() for x in feat_names]
    n_parts = int(np.ceil(F / max_cols))

    figs = {}
    row_labels = [f"a={a}" for a in range(A)]

    for s in range(S):
        grid = np.vstack([M[s * A + a] for a in range(A)])  # (A,F)
        grid_sd = None
        if SD is not None:
            grid_sd = np.vstack([SD[s * A + a] for a in range(A)])  # (A,F)

        for p in range(n_parts):
            j0 = p * max_cols
            j1 = min(F, (p + 1) * max_cols)

            cols_raw = feat_names[j0:j1]
            cols = [_wrap(c, wrap_header_at) for c in cols_raw]

            # build cell text
            cell_text = []
            for a in range(A):
                row = []
                for j in range(j0, j1):
                    if grid_sd is None:
                        row.append(_format_cell(grid[a, j], None, fmt=fmt))
                    else:
                        row.append(_format_cell(grid[a, j], grid_sd[a, j], fmt=fmt))
                cell_text.append(row)

            # ---------- figure sizing ----------
            ncol = (j1 - j0)

            # estimate needed width from header lengths
            raw_lens = [len(str(c)) for c in cols_raw]
            max_len = max(raw_lens) if raw_lens else 1

            # if headers are long, grow the per-col width a bit
            # (log-ish growth; avoids exploding width)
            len_factor = 1.0 + min(1.2, max(0.0, (max_len - 12) / 20.0))
            fig_w = max(min_fig_w, base_col_width * ncol * len_factor)
            fig_w = min(fig_w, max_fig_w)

            fig_h = max(2.8, fig_h_per_row * A + 1.2)

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

            # make rows a bit taller to accommodate wrapped headers
            # (header row index is 0 for col labels; data rows start at 1)
            tbl.scale(1.0, 1.35)

            # ---------- style header row ----------
            # Decide rotation if "auto"
            if header_rotation == "auto":
                # rotate if still too dense after wrapping or too many columns
                rot = 0
                if ncol >= 9:
                    rot = 45
                if max_len >= 22:
                    rot = 45
                if ncol >= 11:
                    rot = 60
            else:
                rot = int(header_rotation)

            # Set per-column widths (longer headers get slightly wider)
            # Table columns indices are 0..ncol-1 for header/data cols.
            # Row-label column has col index -1 internally (skip).
            total_len = float(sum(raw_lens)) if raw_lens else 1.0
            # base width in axis fraction for each column (normalized)
            # We distribute by header length but keep a floor.
            weights = []
            for L in raw_lens:
                weights.append(max(1.0, L / (total_len / max(1, ncol))))
            wsum = float(sum(weights)) if weights else 1.0
            col_widths = [w / wsum for w in weights]  # sum to ~1 in axis coords

            # Apply header formatting + width
            for ci in range(ncol):
                # header cell: row=0, col=ci
                cell = tbl[(0, ci)]
                cell.get_text().set_rotation(rot)
                cell.get_text().set_fontsize(header_fontsize)
                cell.get_text().set_fontweight("bold")
                cell.get_text().set_va("center")
                cell.get_text().set_ha("center")

                # widen/narrow the whole column (header + data cells)
                # cells exist for rows 0..A (0=header, 1..A=data)
                for ri in range(0, A + 1):
                    tbl[(ri, ci)].set_width(col_widths[ci])

            # Row label styling (row labels are at col=-1, rows 1..A)
            for ri in range(1, A + 1):
                try:
                    c = tbl[(ri, -1)]
                    c.get_text().set_fontsize(rowlabel_fontsize)
                    c.get_text().set_fontweight("bold")
                except Exception:
                    pass

            # extra top room for wrapped/rotated headers
            fig.subplots_adjust(top=1.0 - extra_top_margin)

            fig.tight_layout()
            figs[f"style_{s}_part_{p}"] = fig

    return figs