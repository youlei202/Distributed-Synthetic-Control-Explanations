"""Embedding functions for anchor selection."""

import numpy as np
from typing import Callable, Literal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def identity_embedding(X: np.ndarray) -> np.ndarray:
    """Identity embedding (no transformation).

    Args:
        X: Input features [n_samples, n_features]

    Returns:
        Same as input
    """
    return X


def pca_embedding(X: np.ndarray, n_components: int = 10, seed: int = 42) -> np.ndarray:
    """PCA embedding for dimensionality reduction.

    Args:
        X: Input features [n_samples, n_features]
        n_components: Number of PCA components
        seed: Random seed

    Returns:
        PCA-transformed features [n_samples, n_components]
    """
    pca = PCA(n_components=min(n_components, X.shape[1]), random_state=seed)
    return pca.fit_transform(X)


def scaled_embedding(X: np.ndarray) -> np.ndarray:
    """Standardized embedding.

    Args:
        X: Input features

    Returns:
        Standardized features
    """
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def get_embedding_function(
    method: Literal["identity", "pca", "scaled"] = "identity",
    **kwargs
) -> Callable[[np.ndarray], np.ndarray]:
    """Get embedding function by name.

    Args:
        method: Embedding method name
        **kwargs: Additional arguments for the embedding

    Returns:
        Embedding function
    """
    if method == "identity":
        return identity_embedding
    elif method == "pca":
        n_comp = kwargs.get("n_components", 10)
        seed = kwargs.get("seed", 42)
        return lambda X: pca_embedding(X, n_comp, seed)
    elif method == "scaled":
        return scaled_embedding
    else:
        raise ValueError(f"Unknown embedding method: {method}")