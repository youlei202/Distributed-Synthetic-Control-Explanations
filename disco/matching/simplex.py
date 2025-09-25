"""Simplex projection utilities."""

import numpy as np


def project(v: np.ndarray, z: float = 1.0) -> np.ndarray:
    """Project vector onto the probability simplex.

    Solves: argmin ||w - v||^2 s.t. w >= 0, sum(w) = z

    Uses the sorting-based O(n log n) algorithm from:
    "Efficient Projections onto the l1-Ball for Learning in High Dimensions"
    by Duchi et al. (2008)

    Args:
        v: Input vector to project
        z: Simplex sum constraint (default=1 for probability simplex)

    Returns:
        Projected vector on the simplex
    """
    n = len(v)
    if n == 0:
        return v

    # Sort in descending order
    u = np.sort(v)[::-1]

    # Find the threshold
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - z))[0]

    if len(rho) > 0:
        theta = (cssv[rho[-1]] - z) / (rho[-1] + 1)
    else:
        theta = (cssv[-1] - z) / n

    # Project
    w = np.maximum(v - theta, 0)

    return w


def project_l2_ball(v: np.ndarray, radius: float = 1.0) -> np.ndarray:
    """Project vector onto L2 ball.

    Args:
        v: Input vector
        radius: Ball radius

    Returns:
        Projected vector
    """
    norm = np.linalg.norm(v)
    if norm <= radius:
        return v
    else:
        return v * (radius / norm)


def check_simplex(w: np.ndarray, tol: float = 1e-6) -> bool:
    """Check if vector is on the probability simplex.

    Args:
        w: Vector to check
        tol: Tolerance for constraints

    Returns:
        True if on simplex
    """
    return np.all(w >= -tol) and abs(w.sum() - 1.0) < tol


def simplex_projection_test():
    """Test simplex projection correctness."""
    # Test cases
    test_vectors = [
        np.array([0.5, 0.3, 0.2]),  # Already on simplex
        np.array([1.0, 1.0, 1.0]),  # Needs projection
        np.array([-0.5, 0.5, 1.5]),  # Mixed signs
        np.array([0.0, 0.0, 0.0]),  # Zero vector
    ]

    for v in test_vectors:
        w = project(v)
        assert check_simplex(w), f"Projection failed for {v}"
        # Check idempotence
        w2 = project(w)
        np.testing.assert_allclose(w, w2, rtol=1e-10)