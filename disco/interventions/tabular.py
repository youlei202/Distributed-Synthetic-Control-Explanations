"""Tabular intervention implementations."""

import numpy as np
from typing import Optional, Tuple
from disco.interventions.base import Intervention


class FeatureAdd:
    """Additive intervention on a single feature."""

    def __init__(
        self,
        feat_idx: int,
        delta_max: float,
        clamp: Optional[Tuple[float, float]] = None,
        name: Optional[str] = None
    ):
        """Initialize additive intervention.

        Args:
            feat_idx: Feature index to modify
            delta_max: Maximum additive change
            clamp: Optional (min, max) bounds for the feature
            name: Intervention name
        """
        self.feat_idx = feat_idx
        self.theta_max = delta_max
        self.clamp = clamp
        self.name = name or f"add_feat{feat_idx}"

    def apply(self, X: np.ndarray, theta: float) -> np.ndarray:
        """Apply additive intervention."""
        X_new = X.copy()
        X_new[:, self.feat_idx] += theta

        if self.clamp is not None:
            X_new[:, self.feat_idx] = np.clip(
                X_new[:, self.feat_idx],
                self.clamp[0],
                self.clamp[1]
            )

        return X_new


class FeatureScale:
    """Multiplicative intervention on a single feature."""

    def __init__(
        self,
        feat_idx: int,
        scale_max: float,
        clamp: Optional[Tuple[float, float]] = None,
        name: Optional[str] = None
    ):
        """Initialize multiplicative intervention.

        Args:
            feat_idx: Feature index to modify
            scale_max: Maximum scale factor (e.g., 0.5 means up to 1.5x)
            clamp: Optional (min, max) bounds for the feature
            name: Intervention name
        """
        self.feat_idx = feat_idx
        self.theta_max = scale_max
        self.clamp = clamp
        self.name = name or f"scale_feat{feat_idx}"

    def apply(self, X: np.ndarray, theta: float) -> np.ndarray:
        """Apply multiplicative intervention."""
        X_new = X.copy()
        scale_factor = 1.0 + theta
        X_new[:, self.feat_idx] *= scale_factor

        if self.clamp is not None:
            X_new[:, self.feat_idx] = np.clip(
                X_new[:, self.feat_idx],
                self.clamp[0],
                self.clamp[1]
            )

        return X_new


class ClampedShift:
    """Shift feature with explicit bounds."""

    def __init__(
        self,
        feat_idx: int,
        shift_max: float,
        lower: float,
        upper: float,
        name: Optional[str] = None
    ):
        """Initialize clamped shift intervention.

        Args:
            feat_idx: Feature index to modify
            shift_max: Maximum shift amount
            lower: Lower bound for the feature
            upper: Upper bound for the feature
            name: Intervention name
        """
        self.feat_idx = feat_idx
        self.theta_max = shift_max
        self.lower = lower
        self.upper = upper
        self.name = name or f"clamped_feat{feat_idx}"

    def apply(self, X: np.ndarray, theta: float) -> np.ndarray:
        """Apply clamped shift intervention."""
        X_new = X.copy()

        # Shift towards upper bound
        current = X_new[:, self.feat_idx]
        target = current + theta * (self.upper - current)
        X_new[:, self.feat_idx] = np.clip(target, self.lower, self.upper)

        return X_new


class MonotoneShift:
    """Monotone shift ensuring direction consistency."""

    def __init__(
        self,
        feat_idx: int,
        shift_max: float,
        direction: int = 1,  # 1 for increase, -1 for decrease
        name: Optional[str] = None
    ):
        """Initialize monotone shift.

        Args:
            feat_idx: Feature index to modify
            shift_max: Maximum shift magnitude
            direction: 1 for increase, -1 for decrease
            name: Intervention name
        """
        self.feat_idx = feat_idx
        self.theta_max = shift_max
        self.direction = direction
        self.name = name or f"monotone_feat{feat_idx}"

    def apply(self, X: np.ndarray, theta: float) -> np.ndarray:
        """Apply monotone shift."""
        X_new = X.copy()
        X_new[:, self.feat_idx] += self.direction * abs(theta)
        return X_new