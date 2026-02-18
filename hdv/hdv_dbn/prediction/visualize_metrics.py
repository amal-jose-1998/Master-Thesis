"""
Visualization utilities for validation metrics (Hit@H and TTE).
Replaces confusion matrix visualization with more informative graphs.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict


def plot_hit_rate_vs_horizon(
    predictions: List[Tuple],
    max_horizon: int = None,
    output_path: Path = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot hit rate as a function of horizon length.
    
    For each measured horizon H (1 to max_H), shows what % of predictions
    "hit" (appear at least once) within H steps.
    
    Parameters
    predictions : list of tuples
        Each tuple is (pred_z, true_z, hit_h, tte_steps) from ValidationStep.
    max_horizon : int, optional
        Maximum horizon to compute (default: longest tte_steps in data).
    output_path : Path, optional
        If provided, saves figure to this path.
    figsize : tuple
        Figure size (width, height) in inches.
    
    Returns
    fig : matplotlib.figure.Figure
    """
    
    # Extract TTE steps, excluding misses
    tte_list = [tte for _, _, _, tte in predictions if tte is not None]
    
    if not tte_list:
        print("[visualize] No hits found, cannot plot hit rate vs horizon")
        return None
    
    if max_horizon is None:
        max_horizon = max(tte_list)
    
    horizons = np.arange(1, max_horizon + 1)
    hit_rates = []
    hit_counts = []
    
    total_predictions = len(predictions)
    
    for h in horizons:
        # Count how many predictions hit within horizon h
        hits_at_h = sum(1 for _, _, _, tte in predictions if tte is not None and tte <= h)
        hit_rates.append(hits_at_h / total_predictions)
        hit_counts.append(hits_at_h)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(horizons, hit_rates, 'o-', linewidth=2, markersize=6, color='steelblue')
    ax.set_xlabel('Horizon (steps)', fontsize=12)
    ax.set_ylabel('Hit Rate', fontsize=12)
    ax.set_title('Prediction Hit Rate vs Horizon Length', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Add count annotations
    for h, rate, count in zip(horizons[::max(1, len(horizons)//5)], 
                               hit_rates[::max(1, len(horizons)//5)],
                               hit_counts[::max(1, len(horizons)//5)]):
        ax.annotate(f'{count}', xy=(h, rate), xytext=(0, 5), 
                   textcoords='offset points', ha='center', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[visualize] Saved hit rate vs horizon to {output_path}")
    
    return fig


def plot_time_to_hit_histogram(
    predictions: List[Tuple],
    fps: float = 25.0,
    stride_frames: int = 10,
    bins: int = 15,
    output_path: Path = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
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
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[visualize] Saved TTE histogram to {output_path}")
    
    return fig


def plot_cumulative_hit_rate(
    predictions: List[Tuple],
    fps: float = 25.0,
    stride_frames: int = 10,
    output_path: Path = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
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
    ax.axhline(y=hit_rate, color='red', linestyle='--', alpha=0.5, 
              label=f'Overall Hit Rate: {hit_rate:.1%}')
    ax.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[visualize] Saved cumulative hit rate to {output_path}")
    
    return fig


def plot_hit_count_summary(
    predictions: List[Tuple],
    output_path: Path = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
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
    bars = ax.bar(['Hits', 'Misses'], [hits, misses], color=['steelblue', 'coral'], 
                   edgecolor='black', alpha=0.7)
    
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Overall Prediction Performance', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(hits, misses) * 1.15])
    
    # Add count and percentage labels
    for bar, count in zip(bars, [hits, misses]):
        height = bar.get_height()
        pct = count / len(predictions) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, height,
               f'{int(count)}\n({pct:.1f}%)',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[visualize] Saved hit count summary to {output_path}")
    
    return fig


def visualize_all_metrics(
    predictions: List[Tuple],
    output_dir: Path,
    fps: float = 25.0,
    stride_frames: int = 10,
    max_horizon: int = None
) -> Dict[str, plt.Figure]:
    """
    Generate all visualization plots and save to output_dir.
    
    Parameters
    predictions : list of tuples
        Each tuple is (pred_z, true_z, hit_h, tte_steps) from ValidationStep.predict_one_trajectory.
    output_dir : Path
        Directory to save plots.
    fps : float
        Frame rate for TTE conversion.
    stride_frames : int
        Stride in frames per step.
    max_horizon : int, optional
        Maximum horizon for hit rate vs horizon plot.
    
    Returns
    dict
        Maps plot names to matplotlib figure objects.
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    figs = {}
    
    # Plot 1: Hit rate vs horizon
    fig = plot_hit_rate_vs_horizon(
        predictions,
        max_horizon=max_horizon,
        output_path=output_dir / "hit_rate_vs_horizon.png"
    )
    if fig:
        figs['hit_rate_vs_horizon'] = fig
    
    # Plot 2: TTE histogram
    fig = plot_time_to_hit_histogram(
        predictions,
        fps=fps,
        stride_frames=stride_frames,
        output_path=output_dir / "tte_histogram.png"
    )
    if fig:
        figs['tte_histogram'] = fig
    
    # Plot 3: Cumulative hit rate
    fig = plot_cumulative_hit_rate(
        predictions,
        fps=fps,
        stride_frames=stride_frames,
        output_path=output_dir / "cumulative_hit_rate.png"
    )
    if fig:
        figs['cumulative_hit_rate'] = fig
    
    # Plot 4: Hit/miss summary
    fig = plot_hit_count_summary(
        predictions,
        output_path=output_dir / "hit_count_summary.png"
    )
    if fig:
        figs['hit_count_summary'] = fig
    
    return figs
