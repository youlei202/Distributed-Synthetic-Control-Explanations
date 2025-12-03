"""Visualization utilities for DISCO explanations."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict
from disco.explain.te_curve import TEOutput


def plot_te_curve(
    te_output: TEOutput,
    title: str = "Treatment Effect Curve",
    figsize: tuple = (10, 6),
    show_auc: bool = True,
    show_flip: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot treatment effect curve with shaded AUC.

    Args:
        te_output: Treatment effect output
        title: Plot title
        figsize: Figure size
        show_auc: Whether to show AUC in legend
        show_flip: Whether to mark flip intensity
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])

    # Top plot: trajectories
    ax1.plot(te_output.theta, te_output.y_t, 'b-', label='Target', linewidth=2)
    ax1.plot(te_output.theta, te_output.y_syn, 'r--', label='Synthetic', linewidth=2)
    ax1.set_ylabel('Response g(f(x))')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title)

    # Bottom plot: treatment effect
    ax2.plot(te_output.theta, te_output.tau, 'g-', linewidth=2)
    ax2.fill_between(te_output.theta, 0, te_output.tau, alpha=0.3, color='green')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Intervention Intensity θ')
    ax2.set_ylabel('Treatment Effect τ')
    ax2.grid(True, alpha=0.3)

    if show_auc:
        ax2.text(0.02, 0.98, f'AUC-TE: {te_output.auc_abs:.3f}',
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    if show_flip and te_output.flip_theta is not None:
        ax1.axvline(x=te_output.flip_theta, color='purple', linestyle=':', alpha=0.7)
        ax2.axvline(x=te_output.flip_theta, color='purple', linestyle=':', alpha=0.7,
                   label=f'Flip at θ={te_output.flip_theta:.2f}')
        ax2.legend()

    # Mark peak
    if te_output.peak_theta is not None:
        ax2.plot(te_output.peak_theta, te_output.peak_tau, 'ro', markersize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_weight_distribution(
    weights: np.ndarray,
    peer_ids: Optional[List[str]] = None,
    threshold: float = 0.01,
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot distribution of synthetic control weights.

    Args:
        weights: Weight vector on simplex
        peer_ids: Optional peer device IDs
        threshold: Minimum weight to display
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Filter significant weights
    significant = weights > threshold
    w_sig = weights[significant]

    if peer_ids is not None:
        ids_sig = [peer_ids[i] for i in range(len(weights)) if significant[i]]
    else:
        ids_sig = [f'Peer {i}' for i in range(len(weights)) if significant[i]]

    # Sort by weight
    sorted_idx = np.argsort(w_sig)[::-1]
    w_sorted = w_sig[sorted_idx]
    ids_sorted = [ids_sig[i] for i in sorted_idx]

    # Bar plot
    x = np.arange(len(w_sorted))
    bars = ax.bar(x, w_sorted)

    # Color by magnitude
    colors = plt.cm.viridis(w_sorted / w_sorted.max())
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_xticks(x)
    ax.set_xticklabels(ids_sorted, rotation=45, ha='right')
    ax.set_ylabel('Weight')
    ax.set_title('Synthetic Control Weights')
    ax.grid(True, alpha=0.3, axis='y')

    # Add cumulative line
    ax2 = ax.twinx()
    cumsum = np.cumsum(w_sorted)
    ax2.plot(x, cumsum, 'r--', alpha=0.7, label='Cumulative')
    ax2.set_ylabel('Cumulative Weight', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_multiple_interventions(
    te_outputs: Dict[str, TEOutput],
    figsize: tuple = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot treatment effects for multiple interventions.

    Args:
        te_outputs: Dictionary mapping intervention names to TE outputs
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    n_interventions = len(te_outputs)
    fig, axes = plt.subplots(n_interventions, 2, figsize=figsize)

    if n_interventions == 1:
        axes = axes.reshape(1, -1)

    for i, (name, te_out) in enumerate(te_outputs.items()):
        # Left: trajectories
        ax_left = axes[i, 0]
        ax_left.plot(te_out.theta, te_out.y_t, 'b-', label='Target')
        ax_left.plot(te_out.theta, te_out.y_syn, 'r--', label='Synthetic')
        ax_left.set_ylabel('Response')
        ax_left.set_title(f'{name}: Trajectories')
        ax_left.legend()
        ax_left.grid(True, alpha=0.3)

        # Right: treatment effect
        ax_right = axes[i, 1]
        ax_right.plot(te_out.theta, te_out.tau, 'g-')
        ax_right.fill_between(te_out.theta, 0, te_out.tau, alpha=0.3, color='green')
        ax_right.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax_right.set_ylabel('τ')
        ax_right.set_title(f'AUC-TE: {te_out.auc_abs:.3f}')
        ax_right.grid(True, alpha=0.3)

        if i == n_interventions - 1:
            ax_left.set_xlabel('Intensity θ')
            ax_right.set_xlabel('Intensity θ')

    plt.suptitle('Treatment Effects Across Interventions', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_diagnostics(
    eps_pre: float,
    anchor_radius: float,
    weight_entropy: float,
    n_effective_peers: int,
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot diagnostic metrics as a dashboard.

    Args:
        eps_pre: Pre-window fit error
        anchor_radius: Weighted anchor radius
        weight_entropy: Weight distribution entropy
        n_effective_peers: Number of effective peers
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Metric bars
    metrics = {
        'Pre-window Fit': eps_pre,
        'Anchor Radius': anchor_radius,
        'Weight Entropy': weight_entropy,
        'Effective Peers': n_effective_peers / 20.0  # Normalize for display
    }

    for ax, (name, value) in zip(axes.flat, metrics.items()):
        # Single bar with color coding
        if name == 'Pre-window Fit':
            color = 'green' if value < 0.1 else 'orange' if value < 0.2 else 'red'
        elif name == 'Anchor Radius':
            color = 'green' if value < 1.0 else 'orange' if value < 2.0 else 'red'
        else:
            color = 'blue'

        ax.barh([0], [value], color=color, alpha=0.7)
        ax.set_xlim(0, max(1.0, value * 1.2))
        ax.set_title(name)
        ax.set_xlabel('Value')
        ax.set_yticks([])
        ax.text(value + 0.01, 0, f'{value:.3f}', va='center')

    plt.suptitle('DISCO Diagnostics', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig