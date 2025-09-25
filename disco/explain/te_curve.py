"""Treatment effect curve computation and metrics."""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal
from disco.devices.base import Device
from disco.interventions.base import Intervention
from disco.interventions.grids import Grids


@dataclass
class TEOutput:
    """Treatment effect output container."""
    theta: np.ndarray         # Post-grid intensities
    y_t: np.ndarray          # Target trajectory
    y_syn: np.ndarray        # Synthetic trajectory
    tau: np.ndarray          # Treatment effect τ = y_t - y_syn
    auc_abs: float           # AUC of |τ|
    auc_signed: float        # AUC of τ (signed)
    flip_theta: Optional[float]  # Flip intensity (classification)
    peak_theta: Optional[float]  # Peak |τ| intensity
    peak_tau: float          # Peak |τ| value


class Explainer:
    """Computes treatment effect explanations."""

    def __init__(self, compute_flip: bool = True):
        """Initialize explainer.

        Args:
            compute_flip: Whether to compute flip intensity for classification
        """
        self.compute_flip = compute_flip

    def run(
        self,
        device_t: Device,
        y_syn: np.ndarray,
        x_star: np.ndarray,
        grids: Grids,
        intervention: Intervention,
        task: Literal["classification", "regression"],
        target_class: int = 1
    ) -> TEOutput:
        """Compute treatment effect explanation.

        Args:
            device_t: Target device
            y_syn: Synthetic counterfactual trajectory
            x_star: Private query point
            grids: Pre-window and post-grid
            intervention: Applied intervention
            task: Task type
            target_class: Target class for scalarization

        Returns:
            TEOutput with treatment effects and metrics
        """
        if x_star.ndim == 1:
            x_star = x_star.reshape(1, -1)

        # Compute target trajectory on post-grid
        n_post = len(grids.theta)
        y_t = np.zeros(n_post)

        for i, theta in enumerate(grids.theta):
            x_intervened = intervention.apply(x_star, theta)
            raw = device_t.predict_raw(x_intervened)
            y_t[i] = device_t.g(raw, target_class)[0]

        # Treatment effect
        tau = y_t - y_syn

        # Compute AUC-TE (trapezoidal integration)
        auc_abs = self._compute_auc(grids.theta, np.abs(tau))
        auc_signed = self._compute_auc(grids.theta, tau)

        # Peak deviation
        peak_idx = np.argmax(np.abs(tau))
        peak_theta = grids.theta[peak_idx]
        peak_tau = tau[peak_idx]

        # Flip intensity (classification only)
        flip_theta = None
        if self.compute_flip and task == "classification":
            flip_theta = self._find_flip_intensity(
                device_t, x_star, intervention,
                grids.theta, target_class
            )

        return TEOutput(
            theta=grids.theta,
            y_t=y_t,
            y_syn=y_syn,
            tau=tau,
            auc_abs=auc_abs,
            auc_signed=auc_signed,
            flip_theta=flip_theta,
            peak_theta=peak_theta,
            peak_tau=peak_tau
        )

    def _compute_auc(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute area under curve via trapezoidal rule.

        Args:
            x: X values (intensities)
            y: Y values (treatment effects)

        Returns:
            Area under curve
        """
        return np.trapz(y, x)

    def _find_flip_intensity(
        self,
        device: Device,
        x_star: np.ndarray,
        intervention: Intervention,
        theta_grid: np.ndarray,
        target_class: int
    ) -> Optional[float]:
        """Find minimum intensity that flips the prediction.

        Args:
            device: Target device
            x_star: Query point
            intervention: Applied intervention
            theta_grid: Intensity grid to search
            target_class: Current target class

        Returns:
            Flip intensity or None if no flip
        """
        # Get baseline prediction
        raw_base = device.predict_raw(x_star)
        if raw_base.shape[1] <= 1:
            # Regression or single output
            return None

        pred_base = np.argmax(raw_base[0])

        # Search for flip
        for theta in theta_grid:
            x_intervened = intervention.apply(x_star, theta)
            raw = device.predict_raw(x_intervened)
            pred = np.argmax(raw[0])

            if pred != pred_base:
                return theta

        return None


def compute_elasticity(
    te_output: TEOutput,
    window_size: int = 3
) -> np.ndarray:
    """Compute local elasticity of treatment effect.

    Args:
        te_output: Treatment effect output
        window_size: Window for local derivative estimation

    Returns:
        Elasticity values at each grid point
    """
    tau = te_output.tau
    theta = te_output.theta

    n = len(theta)
    elasticity = np.zeros(n)

    for i in range(n):
        # Get local window
        start = max(0, i - window_size // 2)
        end = min(n, i + window_size // 2 + 1)

        if end - start < 2:
            continue

        # Estimate local derivative
        local_theta = theta[start:end]
        local_tau = tau[start:end]

        # Finite differences
        if len(local_theta) >= 2:
            d_tau = np.gradient(local_tau)
            d_theta = np.gradient(local_theta)
            if abs(local_tau[i - start]) > 1e-10 and abs(d_theta[i - start]) > 1e-10:
                elasticity[i] = (d_tau[i - start] / local_tau[i - start]) / \
                               (d_theta[i - start] / local_theta[i - start])

    return elasticity