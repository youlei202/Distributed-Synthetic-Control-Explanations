"""K-nearest neighbor anchor selection with kernel weights."""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional
from sklearn.neighbors import NearestNeighbors


@dataclass
class AnchorSelection:
    """Container for selected anchors and their weights."""
    indices: np.ndarray  # Shape (K,) - indices into probe set
    weights: np.ndarray  # Shape (K,) - nonnegative, sum to 1
    distances: np.ndarray  # Shape (K,) - distances to x_star


class AnchorSelector:
    """Selects and weights anchor points near a query."""

    def __init__(
        self,
        emb: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        K: int = 50,
        tau: float = 0.5,
        kernel: str = "gaussian"
    ):
        """Initialize anchor selector.

        Args:
            emb: Embedding function (default: identity)
            K: Number of nearest neighbors
            tau: Kernel bandwidth parameter
            kernel: Kernel type ("gaussian", "exponential", "uniform")
        """
        self.emb = emb or (lambda X: X)
        self.K = K
        self.tau = tau
        self.kernel = kernel

    def select(self, X_probe: np.ndarray, x_star: np.ndarray) -> AnchorSelection:
        """Select K nearest anchors to x_star and compute weights.

        Args:
            X_probe: Probe set [m, d]
            x_star: Query point [1, d] or [d,]

        Returns:
            AnchorSelection with indices, weights, and distances
        """
        # Ensure x_star is 2D
        if x_star.ndim == 1:
            x_star = x_star.reshape(1, -1)

        # Apply embedding
        X_emb = self.emb(X_probe)
        x_emb = self.emb(x_star)

        # Find K nearest neighbors
        nn = NearestNeighbors(n_neighbors=min(self.K, len(X_probe)))
        nn.fit(X_emb)
        distances, indices = nn.kneighbors(x_emb)

        # Flatten results (since x_star is single point)
        distances = distances[0]
        indices = indices[0]

        # Compute kernel weights
        weights = self._compute_weights(distances)

        return AnchorSelection(
            indices=indices,
            weights=weights,
            distances=distances
        )

    def _compute_weights(self, distances: np.ndarray) -> np.ndarray:
        """Compute kernel weights from distances.

        Args:
            distances: Distances to anchors

        Returns:
            Normalized weights summing to 1
        """
        if self.kernel == "gaussian":
            # Gaussian RBF kernel
            weights = np.exp(-distances**2 / (2 * self.tau**2))

        elif self.kernel == "exponential":
            # Exponential kernel
            weights = np.exp(-distances / self.tau)

        elif self.kernel == "uniform":
            # Uniform weights
            weights = np.ones_like(distances)

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

        # Normalize to sum to 1
        weights = weights / weights.sum()

        return weights


def compute_anchor_radius(anchor_sel: AnchorSelection, X_probe: np.ndarray, x_star: np.ndarray) -> float:
    """Compute weighted anchor radius.

    Args:
        anchor_sel: Selected anchors with weights
        X_probe: Probe set
        x_star: Query point

    Returns:
        Weighted average distance to anchors
    """
    if x_star.ndim == 1:
        x_star = x_star.reshape(1, -1)

    anchors = X_probe[anchor_sel.indices]
    distances = np.linalg.norm(anchors - x_star, axis=1)

    # Weighted average distance
    radius = np.sum(anchor_sel.weights * distances)

    return radius


def validate_anchors(anchor_sel: AnchorSelection, tol: float = 1e-6) -> bool:
    """Validate anchor selection.

    Args:
        anchor_sel: Anchor selection to validate
        tol: Tolerance for weight sum

    Returns:
        True if valid

    Raises:
        ValueError: If invalid
    """
    # Check weights are nonnegative
    if np.any(anchor_sel.weights < 0):
        raise ValueError("Anchor weights must be nonnegative")

    # Check weights sum to 1
    if abs(anchor_sel.weights.sum() - 1.0) > tol:
        raise ValueError(f"Anchor weights must sum to 1, got {anchor_sel.weights.sum()}")

    # Check no duplicate indices
    if len(np.unique(anchor_sel.indices)) != len(anchor_sel.indices):
        raise ValueError("Duplicate anchor indices found")

    return True