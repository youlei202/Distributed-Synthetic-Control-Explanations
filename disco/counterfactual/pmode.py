"""P-mode (proxy/anchor-transfer) counterfactual builder."""

import numpy as np
from typing import List, Tuple, Optional
from disco.devices.base import Device
from disco.interventions.base import Intervention
from disco.interventions.grids import Grids
from disco.anchors.knn import AnchorSelection


class PModeCounterfactual:
    """Proxy mode counterfactual builder using anchor transfer."""

    def __init__(self, compute_bias: bool = True):
        """Initialize P-mode builder.

        Args:
            compute_bias: Whether to compute proxy bias diagnostic
        """
        self.compute_bias = compute_bias

    def build(
        self,
        w: np.ndarray,
        devices: List[Device],
        X_probe: np.ndarray,
        anchor_sel: AnchorSelection,
        x_star: np.ndarray,
        grids: Grids,
        intervention: Intervention,
        target_device: Device,
        target_class: int = 1,
        peer_baseline: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """Build synthetic counterfactual via anchor transfer.

        Args:
            w: Peer weights [N-1] on simplex
            devices: All devices
            X_probe: Probe set [m, d]
            anchor_sel: Selected anchors with weights
            x_star: Private query [1, d] or [d,]
            grids: Pre-window and post-grid
            intervention: Intervention to apply
            target_device: Target device to exclude
            target_class: Target class for scalarization
            peer_baseline: Optional pre-computed g(f_j(x*)) for each peer

        Returns:
            (y_syn, B_proxy): Synthetic counterfactual and proxy bias
        """
        if x_star.ndim == 1:
            x_star = x_star.reshape(1, -1)

        # Get peer devices
        peer_devices = [d for d in devices if d.id != target_device.id]
        assert len(peer_devices) == len(w)

        # Get anchors
        anchors = X_probe[anchor_sel.indices]  # [K, d]
        alpha = anchor_sel.weights  # [K,]

        # Compute peer baselines if not provided
        if peer_baseline is None:
            peer_baseline = np.zeros(len(peer_devices))
            for j, device in enumerate(peer_devices):
                raw = device.predict_raw(x_star)
                peer_baseline[j] = device.g(raw, target_class)[0]

        # Pre-compute anchor increments for all peers and post-grid points
        n_post = len(grids.theta)
        n_anchors = len(anchor_sel.indices)
        n_peers = len(peer_devices)

        # Delta[j, i, t] = F_j(anchor_i, theta_t) - F_j(anchor_i, 0)
        delta = np.zeros((n_peers, n_anchors, n_post))

        for j, device in enumerate(peer_devices):
            # Baseline at theta=0
            baseline_anchors = device.predict_raw(anchors)
            baseline_scalar = device.g(baseline_anchors, target_class)  # [K,]

            # Compute for each post-grid point
            for t, theta in enumerate(grids.theta):
                anchors_intervened = intervention.apply(anchors, theta)
                raw = device.predict_raw(anchors_intervened)
                scalar = device.g(raw, target_class)  # [K,]
                delta[j, :, t] = scalar - baseline_scalar

        # Build synthetic trajectory
        y_syn = np.zeros(n_post)

        for t in range(n_post):
            # For each peer, approximate response at x*
            peer_responses = np.zeros(n_peers)
            for j in range(n_peers):
                # Anchor transfer approximation
                increment = np.sum(alpha * delta[j, :, t])
                peer_responses[j] = peer_baseline[j] + increment

            # Weighted combination
            y_syn[t] = np.sum(w * peer_responses)

        # Compute proxy bias if requested
        B_proxy = 0.0
        if self.compute_bias:
            B_proxy = self._compute_proxy_bias(
                w, peer_devices, anchors, alpha, x_star,
                grids, intervention, target_class
            )

        return y_syn, B_proxy

    def _compute_proxy_bias(
        self,
        w: np.ndarray,
        peer_devices: List[Device],
        anchors: np.ndarray,
        alpha: np.ndarray,
        x_star: np.ndarray,
        grids: Grids,
        intervention: Intervention,
        target_class: int
    ) -> float:
        """Estimate proxy bias.

        The proxy bias measures the approximation error from using
        anchor transfer instead of direct evaluation at x*.

        Args:
            Various inputs from build()

        Returns:
            Estimated proxy bias (L2 norm)
        """
        # Sample a few test points to estimate bias
        test_thetas = grids.theta[:min(3, len(grids.theta))]
        biases = []

        for theta in test_thetas:
            # True responses at x* (if we could compute them)
            x_intervened = intervention.apply(x_star, theta)

            true_response = 0.0
            approx_response = 0.0

            for j, device in enumerate(peer_devices):
                # True response
                raw_true = device.predict_raw(x_intervened)
                true_val = device.g(raw_true, target_class)[0]
                true_response += w[j] * true_val

                # Approximated via anchors
                raw_base = device.predict_raw(x_star)
                base_val = device.g(raw_base, target_class)[0]

                # Anchor increments
                anchors_intervened = intervention.apply(anchors, theta)
                raw_anchors = device.predict_raw(anchors_intervened)
                anchor_vals = device.g(raw_anchors, target_class)

                raw_anchors_base = device.predict_raw(anchors)
                anchor_base_vals = device.g(raw_anchors_base, target_class)

                increment = np.sum(alpha * (anchor_vals - anchor_base_vals))
                approx_val = base_val + increment
                approx_response += w[j] * approx_val

            biases.append(abs(true_response - approx_response))

        return np.mean(biases) if biases else 0.0