"""Utilities for PIER ground truth and metrics."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from disco.synth.generator import SyntheticScenario


def compute_pier_truth(
    scenario: SyntheticScenario,
    weights: np.ndarray,
    x_star: np.ndarray,
    theta: np.ndarray,
    intervention,
) -> np.ndarray:
    """Wrapper that uses the scenario's analytic factorization to compute PIER.

    Args:
        scenario: Loaded synthetic scenario
        weights: Peer weights over N-1 peers
        x_star: Query point (1, p) or (p,)
        theta: Post-grid intensities
        intervention: Intervention used for evaluation

    Returns:
        pier_true vector aligned with theta
    """
    return scenario.compute_pier_truth(
        weights=weights, x=x_star, theta=theta, intervention=intervention
    )


def compute_pier_metrics(
    tau_hat: np.ndarray,
    pier_true: np.ndarray,
    theta: np.ndarray,
    theta_break: Optional[float] = None,
) -> Dict[str, float]:
    """Compute PIER-specific metrics comparing estimated τ to true PIER.

    Metrics:
      - pier_corr: correlation between τ̂ and PIER_true (NaN-safe)
      - pier_nrmse: ||τ̂ - PIER_true||2 / (||PIER_true||2 + eps)
      - pier_auc_abs_diff: ∫ |τ̂ - PIER_true| dθ
      - onset_abs_err: |θ_onset(τ̂) - θ_break| where θ_onset is first non-zero index
    """
    tau_hat = np.asarray(tau_hat, dtype=float).reshape(-1)
    pier_true = np.asarray(pier_true, dtype=float).reshape(-1)
    theta = np.asarray(theta, dtype=float).reshape(-1)

    eps = 1e-12
    # Correlation (handle degenerate cases)
    if np.allclose(tau_hat, tau_hat[0]) or np.allclose(pier_true, pier_true[0]):
        pier_corr = float("nan")
    else:
        c = np.corrcoef(tau_hat, pier_true)
        pier_corr = float(c[0, 1])

    diff = tau_hat - pier_true
    pier_nrmse = float(np.linalg.norm(diff) / (np.linalg.norm(pier_true) + eps))
    pier_auc_abs_diff = float(np.trapz(np.abs(diff), theta))

    onset_abs_err = float("nan")
    if theta_break is not None and theta.size > 0:
        # First theta where |τ̂| exceeds tiny threshold
        idxs = np.where(np.abs(tau_hat) > 1e-8)[0]
        if idxs.size > 0:
            onset_theta = float(theta[idxs[0]])
            onset_abs_err = abs(onset_theta - float(theta_break))

    return {
        "pier_corr": pier_corr,
        "pier_nrmse": pier_nrmse,
        "pier_auc_abs_diff": pier_auc_abs_diff,
        "onset_abs_err": onset_abs_err,
    }


__all__ = ["compute_pier_truth", "compute_pier_metrics"]
