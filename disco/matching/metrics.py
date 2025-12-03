"""Metrics and diagnostics for matching quality."""

import numpy as np
from typing import List, Tuple
from disco.matching.weights import MatchResult, MatchInputs


def compute_weight_sparsity(w: np.ndarray, threshold: float = 0.01) -> float:
    """Compute sparsity of weight vector.

    Args:
        w: Weight vector on simplex
        threshold: Threshold below which weights are considered zero

    Returns:
        Fraction of weights above threshold
    """
    return np.mean(w > threshold)


def compute_effective_peers(w: np.ndarray, threshold: float = 0.01) -> int:
    """Count effective number of peers.

    Args:
        w: Weight vector
        threshold: Minimum weight to be considered active

    Returns:
        Number of peers with weight above threshold
    """
    return np.sum(w > threshold)


def compute_weight_entropy(w: np.ndarray) -> float:
    """Compute entropy of weight distribution.

    Args:
        w: Weight vector on simplex

    Returns:
        Entropy (higher means more uniform)
    """
    w_pos = w[w > 0]
    if len(w_pos) == 0:
        return 0.0
    return -np.sum(w_pos * np.log(w_pos))


def compute_max_weight(w: np.ndarray) -> float:
    """Get maximum weight (concentration measure).

    Args:
        w: Weight vector

    Returns:
        Maximum weight value
    """
    return np.max(w)


def bootstrap_stability(
    solver: "WeightSolver",
    inputs: "MatchInputs",
    n_bootstrap: int = 100,
    sample_frac: float = 0.8,
    seed: int = 42
) -> dict:
    """Assess weight stability via bootstrap.

    Args:
        solver: Weight solver
        inputs: Original matching inputs
        n_bootstrap: Number of bootstrap samples
        sample_frac: Fraction of data to sample
        seed: Random seed

    Returns:
        Dictionary with stability metrics
    """
    rng = np.random.RandomState(seed)
    n_total = len(inputs.y_t)
    n_sample = int(n_total * sample_frac)

    weights = []
    eps_pres = []

    for _ in range(n_bootstrap):
        # Sample indices
        idx = rng.choice(n_total, n_sample, replace=True)

        # Create bootstrap inputs
        boot_inputs = MatchInputs(
            y_t=inputs.y_t[idx],
            X_peers=inputs.X_peers[idx, :],
            lam=inputs.lam
        )

        # Solve
        result = solver.solve(boot_inputs)
        weights.append(result.w)
        eps_pres.append(result.eps_pre)

    weights = np.array(weights)  # [n_bootstrap, n_peers]

    # Compute stability metrics
    mean_w = weights.mean(axis=0)
    std_w = weights.std(axis=0)
    cv_w = std_w / (mean_w + 1e-10)  # Coefficient of variation

    # Pairwise L1 distances
    l1_dists = []
    for i in range(n_bootstrap):
        for j in range(i+1, n_bootstrap):
            l1_dists.append(np.linalg.norm(weights[i] - weights[j], 1))

    return {
        "mean_weights": mean_w,
        "std_weights": std_w,
        "cv_weights": cv_w,
        "mean_l1_distance": np.mean(l1_dists),
        "std_l1_distance": np.std(l1_dists),
        "eps_pre_mean": np.mean(eps_pres),
        "eps_pre_std": np.std(eps_pres)
    }


def identify_active_peers(
    w: np.ndarray,
    peer_ids: List[str],
    threshold: float = 0.01
) -> List[Tuple[str, float]]:
    """Identify active peers and their weights.

    Args:
        w: Weight vector
        peer_ids: List of peer device IDs
        threshold: Minimum weight threshold

    Returns:
        List of (peer_id, weight) tuples sorted by weight
    """
    active = [(peer_ids[i], w[i]) for i in range(len(w)) if w[i] > threshold]
    return sorted(active, key=lambda x: x[1], reverse=True)