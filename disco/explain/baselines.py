"""Baselines and ablations for DISCO explanations."""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from disco.devices.base import Device
from disco.interventions.base import Intervention
from disco.interventions.grids import Grids
from disco.anchors.knn import AnchorSelection
from disco.explain.te_curve import Explainer


@dataclass
class BaselineOutput:
    """Output container for baseline methods."""
    theta: np.ndarray
    y_t: np.ndarray
    y_syn: np.ndarray
    tau: np.ndarray
    auc_abs: float
    flip_theta: Optional[float]
    meta: Dict[str, Any]


class BaselineRunner:
    """Runs various baseline methods for comparison with DISCO."""

    def __init__(self, explainer: Explainer = None):
        """Initialize baseline runner."""
        self.explainer = explainer or Explainer(compute_flip=True)

    def baseline_avg(
        self,
        devices: List[Device],
        x_star: np.ndarray,
        grids: Grids,
        intervention: Intervention,
        target_idx: int,
        target_class: int = 1
    ) -> BaselineOutput:
        """Average baseline: equal weights on all peers.

        Args:
            devices: All devices
            x_star: Query point
            grids: Pre-window and post-grid
            intervention: Applied intervention
            target_idx: Index of target device
            target_class: Target class for scalarization

        Returns:
            BaselineOutput with equal-weight synthetic trajectory
        """
        if x_star.ndim == 1:
            x_star = x_star.reshape(1, -1)

        target_device = devices[target_idx]
        peer_devices = [d for i, d in enumerate(devices) if i != target_idx]
        n_peers = len(peer_devices)

        # Equal weights
        w = np.ones(n_peers) / n_peers

        # Build synthetic trajectory
        y_syn = np.zeros(len(grids.theta))
        for i, theta in enumerate(grids.theta):
            x_intervened = intervention.apply(x_star, theta)
            response = 0.0
            for j, device in enumerate(peer_devices):
                raw = device.predict_raw(x_intervened)
                scalar = device.g(raw, target_class)
                response += w[j] * scalar[0]
            y_syn[i] = response

        # Compute target trajectory
        y_t = np.zeros(len(grids.theta))
        for i, theta in enumerate(grids.theta):
            x_intervened = intervention.apply(x_star, theta)
            raw = target_device.predict_raw(x_intervened)
            y_t[i] = target_device.g(raw, target_class)[0]

        # Treatment effect
        tau = y_t - y_syn
        auc_abs = np.trapz(np.abs(tau), grids.theta)

        # Flip intensity
        flip_theta = self._find_flip_intensity(
            target_device, x_star, intervention, grids.theta, target_class
        )

        return BaselineOutput(
            theta=grids.theta,
            y_t=y_t,
            y_syn=y_syn,
            tau=tau,
            auc_abs=auc_abs,
            flip_theta=flip_theta,
            meta={"method": "avg", "n_peers": n_peers}
        )

    def baseline_topk_avg(
        self,
        devices: List[Device],
        trajectories: Dict,
        anchor_sel: AnchorSelection,
        theta0: np.ndarray,
        x_star: np.ndarray,
        grids: Grids,
        intervention: Intervention,
        target_idx: int,
        Kp: int = 3,
        target_class: int = 1
    ) -> BaselineOutput:
        """Top-K average baseline: select K nearest peers by pre-window distance.

        Args:
            devices: All devices
            trajectories: Pre-computed trajectories
            anchor_sel: Anchor selection
            theta0: Pre-window intensities
            x_star: Query point
            grids: Pre-window and post-grid
            intervention: Applied intervention
            target_idx: Target device index
            Kp: Number of top peers to select
            target_class: Target class

        Returns:
            BaselineOutput with top-K peer average
        """
        if x_star.ndim == 1:
            x_star = x_star.reshape(1, -1)

        target_device = devices[target_idx]
        peer_devices = [d for i, d in enumerate(devices) if i != target_idx]

        # Compute distances to peers over pre-window
        peer_distances = []
        K = len(anchor_sel.indices)
        q = len(theta0)

        # Get target responses on anchors
        target_key = (target_device.id, intervention.name)
        target_traj = trajectories[target_key]  # [m, q]
        target_responses = target_traj[anchor_sel.indices, :].flatten()  # [K*q]

        for peer_device in peer_devices:
            peer_key = (peer_device.id, intervention.name)
            if peer_key not in trajectories:
                continue

            peer_traj = trajectories[peer_key]
            peer_responses = peer_traj[anchor_sel.indices, :].flatten()  # [K*q]

            # L2 distance between trajectories
            distance = np.linalg.norm(target_responses - peer_responses)
            peer_distances.append((distance, peer_device))

        # Select top-K nearest peers
        peer_distances.sort(key=lambda x: x[0])
        top_peers = [peer for _, peer in peer_distances[:min(Kp, len(peer_distances))]]

        # Equal weights among top-K peers
        w = np.ones(len(top_peers)) / len(top_peers)

        # Build synthetic trajectory
        y_syn = np.zeros(len(grids.theta))
        for i, theta in enumerate(grids.theta):
            x_intervened = intervention.apply(x_star, theta)
            response = 0.0
            for j, device in enumerate(top_peers):
                raw = device.predict_raw(x_intervened)
                scalar = device.g(raw, target_class)
                response += w[j] * scalar[0]
            y_syn[i] = response

        # Compute target trajectory
        y_t = np.zeros(len(grids.theta))
        for i, theta in enumerate(grids.theta):
            x_intervened = intervention.apply(x_star, theta)
            raw = target_device.predict_raw(x_intervened)
            y_t[i] = target_device.g(raw, target_class)[0]

        # Treatment effect
        tau = y_t - y_syn
        auc_abs = np.trapz(np.abs(tau), grids.theta)

        flip_theta = self._find_flip_intensity(
            target_device, x_star, intervention, grids.theta, target_class
        )

        return BaselineOutput(
            theta=grids.theta,
            y_t=y_t,
            y_syn=y_syn,
            tau=tau,
            auc_abs=auc_abs,
            flip_theta=flip_theta,
            meta={"method": "topk", "K": len(top_peers), "selected_peers": [d.id for d in top_peers]}
        )

    def baseline_oracle_ground_truth(
        self,
        devices: List[Device],
        trajectories: Dict,
        theta0: np.ndarray,
        x_star: np.ndarray,
        grids: Grids,
        intervention: Intervention,
        target_idx: int,
        target_class: int = 1,
        lam: float = 0.01,
        weight_solver: Optional[Any] = None,
    ) -> BaselineOutput:
        """Option B: oracle-w ground truth baseline shared across methods."""
        from disco.data.loaders import stack_prewindow
        from disco.matching.weights import MatchInputs, WeightSolver
        from disco.counterfactual.smode import SModeCounterfactual

        if x_star.ndim == 1:
            x_star = x_star.reshape(1, -1)

        if not trajectories:
            raise ValueError("Trajectories are required to compute oracle weights")

        target_device = devices[target_idx]

        sample_key = next(iter(trajectories))
        n_probes = trajectories[sample_key].shape[0]
        full_anchor = AnchorSelection(
            indices=np.arange(n_probes),
            weights=np.ones(n_probes) / n_probes,
            distances=np.zeros(n_probes),
        )

        y_pre, X_pre = stack_prewindow(
            trajectories=trajectories,
            target=target_device,
            devices=devices,
            anchor_sel=full_anchor,
            intervention=intervention,
            theta0=theta0,
        )

        if X_pre.shape[1] == 0:
            raise ValueError("Oracle baseline requires at least one peer device")

        solver = weight_solver or WeightSolver(method="projected_grad", max_iter=2000)
        w_star = solver.solve(MatchInputs(y_t=y_pre, X_peers=X_pre, lam=lam)).w
        w_star = np.asarray(w_star, dtype=float).reshape(-1)

        smode_builder = SModeCounterfactual()
        y_syn = smode_builder.build(
            w_star,
            devices,
            x_star,
            grids,
            intervention,
            target_device,
            target_class,
        )

        y_t = np.zeros(len(grids.theta))
        for i, theta in enumerate(grids.theta):
            x_intervened = intervention.apply(x_star, theta)
            raw = target_device.predict_raw(x_intervened)
            y_t[i] = target_device.g(raw, target_class)[0]

        tau = y_t - y_syn
        auc_abs = np.trapz(np.abs(tau), grids.theta)
        flip_theta = self._find_flip_intensity(
            target_device, x_star, intervention, grids.theta, target_class
        )

        return BaselineOutput(
            theta=grids.theta,
            y_t=y_t,
            y_syn=y_syn,
            tau=tau,
            auc_abs=auc_abs,
            flip_theta=flip_theta,
            meta={
                "method": "oracle_w",
                "lam": lam,
                "weights": w_star.tolist(),
                "n_probes": int(n_probes),
                "peer_ids": [d.id for d in devices if d.id != target_device.id],
                "weights_l1": float(np.sum(w_star)),
            },
        )

    def baseline_oracle(
        self,
        devices: List[Device],
        trajectories: Dict,
        theta0: np.ndarray,
        x_star: np.ndarray,
        grids: Grids,
        intervention: Intervention,
        target_idx: int,
        target_class: int = 1,
        lam: float = 0.01,
        weight_solver: Optional[Any] = None,
    ) -> BaselineOutput:
        """Backward-compatible alias for oracle-w baseline."""

        return self.baseline_oracle_ground_truth(
            devices=devices,
            trajectories=trajectories,
            theta0=theta0,
            x_star=x_star,
            grids=grids,
            intervention=intervention,
            target_idx=target_idx,
            target_class=target_class,
            lam=lam,
            weight_solver=weight_solver,
        )


    def baseline_glob_matching(
        self,
        devices: List[Device],
        trajectories: Dict,
        theta0: np.ndarray,
        x_star: np.ndarray,
        grids: Grids,
        intervention: Intervention,
        target_idx: int,
        weight_solver,
        target_class: int = 1
    ) -> BaselineOutput:
        """Global matching baseline: use all probes instead of anchor selection.

        Args:
            devices: All devices
            trajectories: Pre-computed trajectories
            theta0: Pre-window intensities
            x_star: Query point
            grids: Pre-window and post-grid
            intervention: Applied intervention
            target_idx: Target device index
            weight_solver: Weight solver instance
            target_class: Target class

        Returns:
            BaselineOutput with global matching weights
        """
        from disco.data.loaders import stack_prewindow
        from disco.matching.weights import MatchInputs

        if x_star.ndim == 1:
            x_star = x_star.reshape(1, -1)

        target_device = devices[target_idx]

        # Use all probes instead of anchor selection
        target_key = (target_device.id, intervention.name)
        target_traj = trajectories[target_key]  # [m, q]
        m = target_traj.shape[0]

        # Create fake anchor selection covering all probes
        class GlobalAnchorSelection:
            def __init__(self, m):
                self.indices = np.arange(m)
                self.weights = np.ones(m) / m

        global_anchor_sel = GlobalAnchorSelection(m)

        # Stack pre-window data with all probes
        y_t, X_peers = stack_prewindow(
            trajectories, target_device, devices,
            global_anchor_sel, intervention, theta0
        )

        # Solve for weights
        match_inputs = MatchInputs(y_t=y_t, X_peers=X_peers, lam=0.01)
        match_result = weight_solver.solve(match_inputs)

        # Build synthetic trajectory using S-mode approach
        from disco.counterfactual.smode import SModeCounterfactual
        smode_builder = SModeCounterfactual()
        y_syn = smode_builder.build(
            match_result.w, devices, x_star, grids,
            intervention, target_device, target_class
        )

        # Compute target trajectory
        y_t = np.zeros(len(grids.theta))
        for i, theta in enumerate(grids.theta):
            x_intervened = intervention.apply(x_star, theta)
            raw = target_device.predict_raw(x_intervened)
            y_t[i] = target_device.g(raw, target_class)[0]

        tau = y_t - y_syn
        auc_abs = np.trapz(np.abs(tau), grids.theta)

        flip_theta = self._find_flip_intensity(
            target_device, x_star, intervention, grids.theta, target_class
        )

        return BaselineOutput(
            theta=grids.theta,
            y_t=y_t,
            y_syn=y_syn,
            tau=tau,
            auc_abs=auc_abs,
            flip_theta=flip_theta,
            meta={
                "method": "glob",
                "eps_pre": match_result.eps_pre,
                "n_probes": m,
                "weights_sparsity": np.mean(match_result.w > 0.01)
            }
        )

    def baseline_unconstrained_ridge(
        self,
        devices: List[Device],
        trajectories: Dict,
        anchor_sel: AnchorSelection,
        theta0: np.ndarray,
        x_star: np.ndarray,
        grids: Grids,
        intervention: Intervention,
        target_idx: int,
        lam: float = 0.01,
        target_class: int = 1
    ) -> BaselineOutput:
        """Unconstrained ridge baseline: ridge without simplex constraints.

        Args:
            devices: All devices
            trajectories: Pre-computed trajectories
            anchor_sel: Anchor selection
            theta0: Pre-window intensities
            x_star: Query point
            grids: Pre-window and post-grid
            intervention: Applied intervention
            target_idx: Target device index
            lam: Ridge regularization
            target_class: Target class

        Returns:
            BaselineOutput with unconstrained ridge weights
        """
        from disco.data.loaders import stack_prewindow

        if x_star.ndim == 1:
            x_star = x_star.reshape(1, -1)

        target_device = devices[target_idx]

        # Stack pre-window data
        y_t, X_peers = stack_prewindow(
            trajectories, target_device, devices,
            anchor_sel, intervention, theta0
        )

        # Solve unconstrained ridge regression
        XtX = X_peers.T @ X_peers + lam * np.eye(X_peers.shape[1])
        Xty = X_peers.T @ y_t
        w_unconstrained = np.linalg.solve(XtX, Xty)

        # For comparison, also compute clipped weights
        w_clipped = np.maximum(w_unconstrained, 0)
        if w_clipped.sum() > 0:
            w_clipped = w_clipped / w_clipped.sum()

        # Use clipped weights for synthetic trajectory
        from disco.counterfactual.smode import SModeCounterfactual
        smode_builder = SModeCounterfactual()
        y_syn = smode_builder.build(
            w_clipped, devices, x_star, grids,
            intervention, target_device, target_class
        )

        # Compute target trajectory
        y_t_post = np.zeros(len(grids.theta))
        for i, theta in enumerate(grids.theta):
            x_intervened = intervention.apply(x_star, theta)
            raw = target_device.predict_raw(x_intervened)
            y_t_post[i] = target_device.g(raw, target_class)[0]

        tau = y_t_post - y_syn
        auc_abs = np.trapz(np.abs(tau), grids.theta)

        flip_theta = self._find_flip_intensity(
            target_device, x_star, intervention, grids.theta, target_class
        )

        return BaselineOutput(
            theta=grids.theta,
            y_t=y_t_post,
            y_syn=y_syn,
            tau=tau,
            auc_abs=auc_abs,
            flip_theta=flip_theta,
            meta={
                "method": "unconstrained_ridge",
                "w_raw": w_unconstrained,
                "w_clipped": w_clipped,
                "n_negative": np.sum(w_unconstrained < 0),
                "residual": np.linalg.norm(X_peers @ w_clipped - y_t)
            }
        )

    def baseline_local_perturbation(
        self,
        device_t: Device,
        x_star: np.ndarray,
        grids: Grids,
        intervention: Intervention,
        target_class: int = 1
    ) -> BaselineOutput:
        """Local perturbation baseline: target-only forward perturbation.

        Args:
            device_t: Target device
            x_star: Query point
            grids: Pre-window and post-grid
            intervention: Applied intervention
            target_class: Target class

        Returns:
            BaselineOutput with zero synthetic trajectory (pure target effect)
        """
        if x_star.ndim == 1:
            x_star = x_star.reshape(1, -1)

        # Compute target trajectory
        y_t = np.zeros(len(grids.theta))
        for i, theta in enumerate(grids.theta):
            x_intervened = intervention.apply(x_star, theta)
            raw = device_t.predict_raw(x_intervened)
            y_t[i] = device_t.g(raw, target_class)[0]

        # Baseline: no counterfactual (y_syn = 0 or initial prediction)
        raw_initial = device_t.predict_raw(x_star)
        y_initial = device_t.g(raw_initial, target_class)[0]
        y_syn = np.full_like(grids.theta, y_initial)

        tau = y_t - y_syn
        auc_abs = np.trapz(np.abs(tau), grids.theta)

        flip_theta = self._find_flip_intensity(
            device_t, x_star, intervention, grids.theta, target_class
        )

        return BaselineOutput(
            theta=grids.theta,
            y_t=y_t,
            y_syn=y_syn,
            tau=tau,
            auc_abs=auc_abs,
            flip_theta=flip_theta,
            meta={"method": "local_perturbation", "y_initial": y_initial}
        )

    def baseline_kernelshap(
        self,
        device_t: Device,
        x_star: np.ndarray,
        intervention: Intervention,
        grids: Grids,
        background: np.ndarray,
        target_class: int = 1
    ) -> Optional[Dict[str, Any]]:
        """KernelSHAP baseline (optional, requires shap library).

        Args:
            device_t: Target device
            x_star: Query point
            intervention: Applied intervention
            grids: Pre-window and post-grid
            background: Background samples for SHAP
            target_class: Target class

        Returns:
            Dictionary with SHAP values or None if shap not available
        """
        try:
            import shap
        except ImportError:
            print("SHAP library not available, skipping KernelSHAP baseline")
            return None

        if x_star.ndim == 1:
            x_star = x_star.reshape(1, -1)

        # Define prediction function for intervention
        def predict_fn(X):
            theta = grids.theta[0]  # Use first post-grid point
            X_intervened = intervention.apply(X, theta)
            raw = device_t.predict_raw(X_intervened)
            return device_t.g(raw, target_class)

        # Compute SHAP values
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(x_star)

        return {
            "method": "kernelshap",
            "shap_values": shap_values,
            "expected_value": explainer.expected_value,
            "feature_importance": np.abs(shap_values).mean(axis=0)
        }

    def _find_flip_intensity(
        self,
        device: Device,
        x_star: np.ndarray,
        intervention: Intervention,
        theta_grid: np.ndarray,
        target_class: int
    ) -> Optional[float]:
        """Find minimum intensity that flips prediction (helper method)."""
        if x_star.ndim == 1:
            x_star = x_star.reshape(1, -1)

        # Get baseline prediction
        raw_base = device.predict_raw(x_star)
        if raw_base.shape[1] <= 1:
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
