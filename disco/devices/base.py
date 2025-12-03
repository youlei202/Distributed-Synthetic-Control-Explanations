"""Device interface and base classes."""

from typing import Protocol, Literal, Any
import numpy as np

Scalarization = Literal["prob", "logit", "identity"]
TaskType = Literal["regression", "classification"]


class Device(Protocol):
    """Protocol for heterogeneous model devices."""

    id: str
    n_outputs: int  # C (number of outputs)
    task: TaskType
    scalarization: Scalarization

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Return raw model outputs, shape [B, C] or [B,] for regression."""
        ...

    def g(self, raw: np.ndarray, target_class: int = 1) -> np.ndarray:
        """Scalarize raw outputs to shape [B,].

        Args:
            raw: Raw model outputs
            target_class: For classification, which class to extract (default=1 for binary)

        Returns:
            Scalarized values shape [B,]
        """
        ...