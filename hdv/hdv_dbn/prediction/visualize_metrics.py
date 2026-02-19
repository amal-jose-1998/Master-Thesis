"""Visualization utilities for validation metrics (Hit@H and TTE)."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

def plot_confusion_matrix(cm, labels, title="Confusion Matrix", output_path=None, figsize=(12, 10)):
    """
    Plot a confusion matrix with semantic labels.

    Parameters
    cm : np.ndarray
        Confusion matrix of shape (N, N).
    labels : list of str
        Semantic labels for each class (length N).
    title : str
        Plot title.
    output_path : Path, optional
        If provided, saves figure to this path.
    figsize : tuple
        Figure size.
    Returns
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax, cbar=True, linewidths=0.5)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[visualize] Saved confusion matrix to {output_path}")
    return fig


def plot_time_to_hit_histogram(predictions, fps=25.0, stride_frames=10, bins=10, output_path=None, figsize=(10, 6)):
    """
    Histogram of time-to-hit (TTE) for predictions that hit.
    
    Parameters
    predictions : list of tuples
        Each tuple is (pred_z, true_z, hit_h, tte_steps) from ValidationStep.
    fps : float
        Frame rate (Hz) for converting steps to seconds.
    stride_frames : int
        Stride in frames per step.
    bins : int
        Number of histogram bins.
    output_path : Path, optional
        If provided, saves figure to this path.
    figsize : tuple
        Figure size.
    
    Returns
    fig : matplotlib.figure.Figure
    """
    
    delta_t = stride_frames / fps  # step duration in seconds
    
    # Extract TTE in seconds for hits only
    tte_seconds = [tte * delta_t for _, _, _, tte in predictions if tte is not None]
    
    if not tte_seconds:
        print("[visualize] No hits found, cannot plot TTE histogram")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    counts, edges, patches = ax.hist(tte_seconds, bins=bins, color='steelblue', 
                                      edgecolor='black', alpha=0.7)
    
    ax.set_xlabel('Time-to-Hit (seconds)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Time-to-Hit (Hits Only)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    stats_text = f'Mean: {np.mean(tte_seconds):.2f}s\nMedian: {np.median(tte_seconds):.2f}s\nN: {len(tte_seconds)}'
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[visualize] Saved TTE histogram to {output_path}")
    
    return fig


def plot_cumulative_hit_rate(predictions, fps=25.0, stride_frames=10, output_path=None, figsize=(10, 6)):
    """
    Cumulative hit rate over time.
    
    Shows the % of predictions that have hit by time T.
    
    Parameters
    predictions : list of tuples
        Each tuple is (pred_z, true_z, hit_h, tte_steps) from ValidationStep.
    fps : float
        Frame rate (Hz).
    stride_frames : int
        Stride in frames per step.
    output_path : Path, optional
        If provided, saves figure to this path.
    figsize : tuple
        Figure size.
    
    Returns
    fig : matplotlib.figure.Figure
    """
    
    delta_t = stride_frames / fps  # step duration in seconds
    
    # Extract TTE in seconds for all predictions
    tte_list = []
    for _, _, hit_h, tte in predictions:
        if tte is not None:
            tte_list.append(tte * delta_t)
        else:
            tte_list.append(np.inf)
    
    tte_list = np.array(tte_list)
    tte_finite = tte_list[np.isfinite(tte_list)]
    
    if len(tte_finite) == 0:
        print("[visualize] No hits found, cannot plot cumulative rate")
        return None
    
    # Sort and compute cumulative hit rate
    tte_sorted = np.sort(tte_finite)
    cumulative_hits = np.arange(1, len(tte_sorted) + 1)
    cumulative_rate = cumulative_hits / len(tte_list)
    
    # Time axis goes a bit beyond max tte
    time_axis = np.concatenate([[0], tte_sorted])
    rate_axis = np.concatenate([[0], cumulative_rate])
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time_axis, rate_axis, 'o-', linewidth=2, markersize=4, color='steelblue')
    ax.fill_between(time_axis, rate_axis, alpha=0.3, color='steelblue')
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Cumulative Hit Rate', fontsize=12)
    ax.set_title('Cumulative Prediction Hit Rate Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Add reference lines and stats
    hit_rate = len(tte_finite) / len(tte_list)
    ax.axhline(y=hit_rate, color='red', linestyle='--', alpha=0.5, label=f'Overall Hit Rate: {hit_rate:.1%}')
    ax.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[visualize] Saved cumulative hit rate to {output_path}")
    
    return fig


def plot_hit_count_summary(predictions, output_path=None, figsize=(8, 6)):
    """
    Simple bar chart showing hits vs misses.
    
    Parameters
    predictions : list of tuples
    output_path : Path, optional
    figsize : tuple
    
    Returns
    fig : matplotlib.figure.Figure
    """
    
    hits = sum(1 for _, _, hit_h, _ in predictions if hit_h)
    misses = len(predictions) - hits
    
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(['Hits', 'Misses'], [hits, misses], color=['steelblue', 'coral'], edgecolor='black', alpha=0.7)
    
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Overall Prediction Performance', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(hits, misses) * 1.15])
    
    # Add count and percentage labels
    for bar, count in zip(bars, [hits, misses]):
        height = bar.get_height()
        pct = count / len(predictions) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{int(count)}\n({pct:.1f}%)',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[visualize] Saved hit count summary to {output_path}")
    
    return fig


def visualize_all_metrics(*, predictions, output_dir, S, A, labels, fps=25.0, stride_frames=10):
    """
    Generate all visualization plots and save to output_dir.
    
    Parameters
    predictions : list of tuples
        Each tuple is (pred_z, true_z, hit_h, tte_steps) from ValidationStep.predict_one_trajectory.
    output_dir : Path
        Directory to save plots.
    S : int
        Number of styles.
    A : int
        Number of actions.
    labels : list of str
        Semantic labels for each (s, a) pair.
    fps : float
        Frame rate for TTE conversion.
    stride_frames : int
        Stride in frames per step.
    
    Returns
    dict
        Maps plot names to matplotlib figure objects.
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    figs = {}
    
    # Plot 1: Confusion matrix (computed from predictions)
    SA = S * A
    cm = np.zeros((SA, SA), dtype=int) # Empty confusion matrix. Rows are true labels, columns are predicted labels.
    for pred_z, true_z, _, _ in predictions:
        idx_pred = pred_z[0] * A + pred_z[1] # Turn predicted (s,a) into a single class index.
        idx_true = true_z[0] * A + true_z[1] # Turn true (s,a) into a single class index.
        cm[idx_true, idx_pred] += 1 # Increment the confusion matrix count for this true/pred pair.
    fig = plot_confusion_matrix(cm, labels, output_path=output_dir / "confusion_matrix.png")
    if fig:
        figs['confusion_matrix'] = fig
    
    # Plot 2: TTE histogram
    fig = plot_time_to_hit_histogram(predictions, fps=fps, stride_frames=stride_frames, output_path=output_dir / "tte_histogram.png")
    if fig:
        figs['tte_histogram'] = fig
    
    # Plot 3: Cumulative hit rate
    fig = plot_cumulative_hit_rate(predictions, fps=fps, stride_frames=stride_frames, output_path=output_dir / "cumulative_hit_rate.png")
    if fig:
        figs['cumulative_hit_rate'] = fig
    
    # Plot 4: Hit/miss summary
    fig = plot_hit_count_summary(predictions, output_path=output_dir / "hit_count_summary.png")
    if fig:
        figs['hit_count_summary'] = fig
    
    return figs
