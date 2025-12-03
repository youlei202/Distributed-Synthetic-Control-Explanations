"""Grid builders for pre-window and post-grid intensities."""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class Grids:
    """Container for pre-window and post-grid intensities."""

    theta0: np.ndarray  # Pre-window, e.g., [0, ε, 2ε]
    theta: np.ndarray  # Post-grid, e.g., [γ, ..., θ_max]

    def __post_init__(self):
        """Validate grid configuration."""
        # Ensure arrays
        self.theta0 = np.asarray(self.theta0)
        self.theta = np.asarray(self.theta)

        # Check pre-window starts at 0
        if len(self.theta0) > 0 and self.theta0[0] != 0:
            raise ValueError("Pre-window must start at 0")

        # Check post-grid starts after pre-window
        if len(self.theta0) > 0 and len(self.theta) > 0:
            if self.theta[0] <= self.theta0[-1]:
                raise ValueError(
                    f"Post-grid must start after pre-window: "
                    f"{self.theta[0]} <= {self.theta0[-1]}"
                )


def create_grids(
    theta_max: float,
    epsilon: float,
    n_pre: int = 3,
    n_post: int = 8,
    gap_factor: float = 1.5,
) -> Grids:
    """Create standard grids for interventions.

    Args:
        theta_max: Maximum intervention intensity
        epsilon: Pre-window step size (e.g., 5-10% of feature scale)
        n_pre: Number of pre-window points (including 0)
        n_post: Number of post-grid points
        gap_factor: Multiplier for gap between pre and post

    Returns:
        Grids object with theta0 and theta
    """
    # Pre-window: [0, ε, 2ε, ...]
    theta0 = np.linspace(0, (n_pre - 1) * epsilon, n_pre)

    # Post-grid starts after gap
    gamma = theta0[-1] * gap_factor
    theta = np.linspace(gamma, theta_max, n_post)

    return Grids(theta0=theta0, theta=theta)


def create_adaptive_grids(
    X: np.ndarray,
    feat_idx: int,
    theta_max: float,
    n_pre: int = 3,
    n_post: int = 8,
    percentile: float = 10.0,
) -> Grids:
    """Create adaptive grids based on feature statistics.

    Args:
        X: Data matrix [n_samples, n_features]
        feat_idx: Feature index for adaptation
        theta_max: Maximum intervention intensity
        n_pre: Number of pre-window points
        n_post: Number of post-grid points
        percentile: Percentile for robust scale estimation

    Returns:
        Grids object adapted to feature scale
    """
    # Estimate robust scale
    feature = X[:, feat_idx]
    iqr = np.percentile(feature, 75) - np.percentile(feature, 25)
    mad = np.median(np.abs(feature - np.median(feature)))

    # Use more robust scale
    scale = max(iqr, 1.48 * mad, 1e-3)  # Avoid zero scale

    # Set epsilon as percentage of scale
    epsilon = percentile / 100.0 * scale

    return create_grids(
        theta_max=theta_max, epsilon=epsilon, n_pre=n_pre, n_post=n_post
    )


def validate_grids(grids: Grids) -> bool:
    """Validate grid configuration.

    Args:
        grids: Grids to validate

    Returns:
        True if valid

    Raises:
        ValueError: If grids are invalid
    """
    # Check monotonicity
    if len(grids.theta0) > 1:
        if not np.all(np.diff(grids.theta0) > 0):
            raise ValueError("Pre-window must be strictly increasing")

    if len(grids.theta) > 1:
        if not np.all(np.diff(grids.theta) > 0):
            raise ValueError("Post-grid must be strictly increasing")

    # Check separation
    if len(grids.theta0) > 0 and len(grids.theta) > 0:
        if grids.theta[0] <= grids.theta0[-1]:
            raise ValueError("Post-grid must start after pre-window")

    return True
