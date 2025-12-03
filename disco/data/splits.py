"""Data splitting utilities for non-IID device distributions."""

import numpy as np
from typing import List, Tuple, Literal
from sklearn.model_selection import KFold


def split_across_devices(
    X: np.ndarray,
    y: np.ndarray,
    N: int,
    strategy: Literal["iid", "label_skew", "feature_shift", "covariate_shift"] = "iid",
    seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split data across N devices with specified non-IID strategy.

    Args:
        X: Features [n_samples, n_features]
        y: Labels [n_samples]
        N: Number of devices
        strategy: Splitting strategy
        seed: Random seed

    Returns:
        List of (X_i, y_i) tuples for each device
    """
    rng = np.random.RandomState(seed)
    n = len(X)

    if strategy == "iid":
        # Random uniform split
        indices = rng.permutation(n)
        splits = np.array_split(indices, N)

    elif strategy == "label_skew":
        # Sort by label and create skewed splits
        sorted_idx = np.argsort(y)

        # Create concentration parameter for Dirichlet
        alpha = np.ones(N) * 0.5  # Lower alpha = more skew
        proportions = rng.dirichlet(alpha)

        # Assign samples with label bias
        splits = []
        start = 0
        for i, prop in enumerate(proportions):
            size = int(prop * n)
            if i == N - 1:  # Last device gets remainder
                size = n - start

            # Mix sorted (biased) and random samples
            bias_ratio = 0.7  # 70% biased, 30% random
            n_biased = int(size * bias_ratio)
            n_random = size - n_biased

            biased_idx = sorted_idx[start:start+n_biased]
            random_idx = rng.choice(n, n_random, replace=False)

            device_idx = np.concatenate([biased_idx, random_idx])
            splits.append(device_idx)
            start += n_biased

    elif strategy == "feature_shift":
        # Sort by first principal component
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1, random_state=seed)
        pc1 = pca.fit_transform(X).flatten()
        sorted_idx = np.argsort(pc1)

        # Split along PC1
        splits = np.array_split(sorted_idx, N)

    elif strategy == "covariate_shift":
        # Cluster features and assign clusters to devices
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=N, random_state=seed)
        clusters = kmeans.fit_predict(X)

        splits = []
        for i in range(N):
            device_idx = np.where(clusters == i)[0]
            # Add some samples from other clusters for diversity
            other_idx = np.where(clusters != i)[0]
            n_other = max(1, len(device_idx) // 10)  # 10% from other clusters
            if len(other_idx) > 0:
                other_samples = rng.choice(other_idx, min(n_other, len(other_idx)), replace=False)
                device_idx = np.concatenate([device_idx, other_samples])
            splits.append(device_idx)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Convert indices to data splits
    device_data = []
    for idx in splits:
        if len(idx) > 0:
            device_data.append((X[idx], y[idx]))
        else:
            # Ensure no empty devices
            fallback_idx = rng.choice(n, max(1, n // N), replace=False)
            device_data.append((X[fallback_idx], y[fallback_idx]))

    return device_data