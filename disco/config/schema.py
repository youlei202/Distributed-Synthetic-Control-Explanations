"""Configuration schema and validation."""

from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any
import numpy as np


@dataclass
class GridConfig:
    """Grid configuration for pre-window and post-grid."""
    theta0: np.ndarray  # Pre-window, e.g., [0, ε, 2ε]
    theta: np.ndarray   # Post-grid, e.g., [γ, ..., θ_max]

    def __post_init__(self):
        """Validate that post-grid starts after pre-window."""
        if len(self.theta0) > 0 and len(self.theta) > 0:
            assert self.theta[0] > self.theta0[-1], \
                f"Post-grid must start after pre-window: {self.theta[0]} <= {self.theta0[-1]}"


@dataclass
class AnchorConfig:
    """Anchor selection configuration."""
    K: int = 50  # Number of nearest neighbors
    tau: float = 0.5  # Kernel bandwidth
    embedding: Literal["identity", "pca"] = "identity"


@dataclass
class MatchingConfig:
    """Matching configuration."""
    lam: float = 0.01  # Ridge regularization
    solver: Literal["projected_grad", "newton"] = "projected_grad"
    max_iter: int = 1000
    tol: float = 1e-6


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    dataset: str
    N_devices: int
    non_iid: Literal["iid", "label_skew", "feature_shift", "covariate_shift"] = "label_skew"
    probe_size: int = 500
    seed: int = 42

    anchor: AnchorConfig = field(default_factory=AnchorConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    mode: Literal["s", "p"] = "p"  # S-mode (secure) or P-mode (proxy)

    dp_noise_std: float = 0.0  # Optional DP noise