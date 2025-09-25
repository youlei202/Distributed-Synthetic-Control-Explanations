"""S-mode (secure) counterfactual builder."""

import numpy as np
from typing import List
from disco.devices.base import Device
from disco.interventions.base import Intervention
from disco.interventions.grids import Grids


class SModeCounterfactual:
    """Secure mode counterfactual builder.

    In S-mode, peers securely evaluate f_j(T_d(θ, x*)) without
    x* leaving the target device. This implementation simulates
    the secure computation.
    """

    def __init__(self, secure_channel: bool = False):
        """Initialize S-mode builder.

        Args:
            secure_channel: Whether to simulate secure channel (placeholder)
        """
        self.secure_channel = secure_channel

    def build(
        self,
        w: np.ndarray,
        devices: List[Device],
        x_star: np.ndarray,
        grids: Grids,
        intervention: Intervention,
        target_device: Device,
        target_class: int = 1
    ) -> np.ndarray:
        """Build synthetic counterfactual via secure evaluation.

        Args:
            w: Peer weights [N-1] on simplex
            devices: All devices (including target)
            x_star: Private query point [1, d] or [d,]
            grids: Pre-window and post-grid
            intervention: Intervention to apply
            target_device: Target device (to exclude from peers)
            target_class: Target class for scalarization

        Returns:
            y_syn: Synthetic counterfactual on post-grid [r,]
        """
        if x_star.ndim == 1:
            x_star = x_star.reshape(1, -1)

        # Get peer devices (exclude target)
        peer_devices = [d for d in devices if d.id != target_device.id]
        assert len(peer_devices) == len(w), \
            f"Weight dimension {len(w)} doesn't match peers {len(peer_devices)}"

        # Build synthetic trajectory on post-grid
        n_post = len(grids.theta)
        y_syn = np.zeros(n_post)

        for i, theta in enumerate(grids.theta):
            # Apply intervention
            x_intervened = intervention.apply(x_star, theta)

            # Weighted sum of peer responses
            response = 0.0
            for j, device in enumerate(peer_devices):
                # Secure evaluation (simulated here)
                raw = device.predict_raw(x_intervened)
                scalar = device.g(raw, target_class)
                response += w[j] * scalar[0]  # scalar[0] since single point

            y_syn[i] = response

        return y_syn


class SimulatedSecureChannel:
    """Simulated secure channel for S-mode (placeholder for real SMPC/TEE)."""

    def __init__(self, noise_level: float = 0.0):
        """Initialize simulated secure channel.

        Args:
            noise_level: Optional noise for differential privacy
        """
        self.noise_level = noise_level

    def secure_evaluate(
        self,
        device: Device,
        x: np.ndarray,
        target_class: int = 1
    ) -> float:
        """Securely evaluate device on input.

        Args:
            device: Device to evaluate
            x: Input (already intervened)
            target_class: Target class for scalarization

        Returns:
            Scalarized response with optional DP noise
        """
        raw = device.predict_raw(x)
        scalar = device.g(raw, target_class)

        if self.noise_level > 0:
            # Add Gaussian noise for DP
            scalar = scalar + np.random.normal(0, self.noise_level, scalar.shape)

        return scalar[0] if scalar.ndim > 0 else scalar