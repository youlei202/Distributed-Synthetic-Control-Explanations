"""Intervention interface and base classes."""

from typing import Protocol, Optional
import numpy as np


class Intervention(Protocol):
    """Protocol for interpretable interventions."""

    name: str
    feat_idx: Optional[int]  # Feature index for tabular interventions
    theta_max: float

    def apply(self, X: np.ndarray, theta: float) -> np.ndarray:
        """Return X' with intervention along dimension d at intensity theta.

        Args:
            X: Input data shape [B, D]
            theta: Intervention intensity in [0, theta_max]

        Returns:
            Modified input with same shape as X
        """
        ...