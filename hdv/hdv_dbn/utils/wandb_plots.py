"""W&B plotting helpers for HDV DBN training."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_pi_bar(pi_np, num_states):
    """Bar plot of initial distribution π_z over joint states."""
    fig, ax = plt.subplots()
    ax.bar(np.arange(num_states), pi_np)
    ax.set_title("π_z distribution")
    ax.set_xlabel("joint state z")
    ax.set_ylabel("probability")
    return fig


def plot_A_heatmap(A_np):
    """Heatmap of transition matrix A_zz (rows: current z, cols: next z')."""
    fig, ax = plt.subplots()
    im = ax.imshow(A_np, aspect="auto")
    ax.set_title("Transition matrix A_zz")
    ax.set_xlabel("z'")
    ax.set_ylabel("z")
    fig.colorbar(im, ax=ax)
    return fig


def plot_A_diag(A_np):
    """Line plot of diagonal entries of A_zz (stay probabilities)."""
    diag_A = np.diag(A_np)
    fig, ax = plt.subplots()
    ax.plot(np.arange(diag_A.size), diag_A, marker="o")
    ax.set_title("A_zz diagonal: P(Z_{t+1}=z | Z_t=z)")
    ax.set_xlabel("joint state z")
    ax.set_ylabel("stay probability")
    return fig


def plot_A_delta(A_prev, A_new):
    """
    Heatmap of ΔA = A_new - A_prev (centered at 0).
    Useful to see where transitions are still changing.
    """
    A_diff = A_new - A_prev
    vmax = float(np.max(np.abs(A_diff))) if np.any(A_diff != 0.0) else 1.0
    fig, ax = plt.subplots()
    im = ax.imshow(A_diff, aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_title("ΔA_zz = A_new - A_prev")
    ax.set_xlabel("z'")
    ax.set_ylabel("z")
    fig.colorbar(im, ax=ax)
    return fig


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

def plot_bernoulli_means_per_state(values, num_states):
    fig, ax = plt.subplots()
    ax.plot(np.arange(num_states), values, marker="o")
    ax.set_title("Bernoulli exists: mean p(exists) per state")
    ax.set_xlabel("joint state z")
    ax.set_ylabel("mean bern_p over exists dims")
    return fig

def plot_switch_rate_distribution(switch_rates, title="Switch rate per trajectory", xlabel="switches / (T-1)"):
    """Histogram of per-trajectory switch rates."""
    sr = np.asarray(switch_rates, dtype=np.float64).ravel()
    sr = sr[np.isfinite(sr)]
    fig, ax = plt.subplots()
    if sr.size == 0:
        ax.text(0.5, 0.5, "No valid trajectories", ha="center", va="center")
        ax.set_axis_off()
        return fig
    ax.hist(sr, bins=30)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("# trajectories")
    ax.set_xlim(0.0, 1.0)
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


def plot_key_feature_means(means, stds, feat_names, title="Posterior-weighted key feature means"):
    means = np.asarray(means, dtype=np.float64)
    stds  = np.asarray(stds, dtype=np.float64)
    K, F = means.shape

    fig, ax = plt.subplots()
    x = np.arange(F)

    for k in range(K):
        ax.errorbar(x, means[k], yerr=stds[k], marker="o", linestyle="-", capsize=3, label=f"state {k}")

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(feat_names, rotation=25, ha="right")
    ax.set_ylabel("value (mean ± std, posterior-weighted)")
    ax.legend()
    fig.tight_layout()
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
