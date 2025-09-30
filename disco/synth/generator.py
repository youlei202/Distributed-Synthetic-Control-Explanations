"""Synthetic data generator utilities for DISCO."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class SyntheticParameters:
    """Shared parameters for synthetic devices."""

    base_weights: np.ndarray
    effect_weights: np.ndarray
    intensity_break: float
    theta_span: float
    noise_std: float


@dataclass
class SyntheticDevice:
    """Analytic synthetic device with controllable treatment effect."""

    id: str
    alpha: float
    params: SyntheticParameters
    base_offset: float
    noise_weights: np.ndarray
    task: str = "regression"
    scalarization: str = "identity"
    n_outputs: int = 1

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Evaluate device on inputs with intensity stored in first feature."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        intensity = X[:, 0]
        z = X[:, 1:]

        base_input = z @ self.params.base_weights + self.base_offset
        base = np.tanh(base_input)

        effect_scale = np.clip(intensity - self.params.intensity_break, 0.0, None) / self.params.theta_span
        effect = self.alpha * effect_scale * (z @ self.params.effect_weights)

        if self.noise_weights.size:
            noise_driver = z @ self.noise_weights
        else:
            noise_driver = np.zeros(len(intensity))
        noise = self.params.noise_std * np.tanh(noise_driver + 0.1 * intensity)

        return (base + effect + noise).reshape(-1, 1)

    def g(self, raw: np.ndarray, target_class: int = 0) -> np.ndarray:  # type: ignore[override]
        return np.asarray(raw).flatten()


@dataclass
class SyntheticScenario:
    """Container for synthetic devices and latent parameters."""

    devices: List[SyntheticDevice]
    theta0: np.ndarray
    theta: np.ndarray
    X_probe: np.ndarray
    x_star: np.ndarray
    params: SyntheticParameters

    def evaluate_device(
        self,
        device_idx: int,
        x: np.ndarray,
        theta_grid: np.ndarray,
        intervention
    ) -> np.ndarray:
        """Evaluate device along theta grid using provided intervention."""
        device = self.devices[device_idx]
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        responses = []
        for theta in theta_grid:
            x_theta = intervention.apply(x, float(theta))
            raw = device.predict_raw(x_theta)
            responses.append(device.g(raw)[0])
        return np.asarray(responses)

    def compute_tau(
        self,
        weights: np.ndarray,
        intervention,
        x: Optional[np.ndarray] = None,
        theta: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute ground-truth target, synthetic, and treatment effect curves."""
        if x is None:
            x = self.x_star
        if theta is None:
            theta = self.theta

        weights = np.asarray(weights)
        if len(weights) != len(self.devices) - 1:
            raise ValueError("Weight vector length must match number of peer devices")

        y_target = self.evaluate_device(0, x, theta, intervention)
        peer_curves = [self.evaluate_device(i + 1, x, theta, intervention) for i in range(len(weights))]
        if not peer_curves:
            y_syn = np.zeros_like(y_target)
        else:
            peer_stack = np.stack(peer_curves, axis=0)
            y_syn = weights @ peer_stack

        tau = y_target - y_syn
        return y_target, y_syn, tau

    def compute_pier_truth(
        self,
        weights: np.ndarray,
        x: Optional[np.ndarray],
        theta: Optional[np.ndarray],
    ) -> np.ndarray:
        """Compute ground-truth PIER curve at x, theta for this scenario.

        PIER (peer in-expressible residual) is the portion of the target effect
        that cannot be spanned by the peer mixture along the post-grid. In this
        synthetic generator, each device's effect component factorizes as
        s(theta) * h(x) * alpha_device. With oracle pre-window weights w, the
        peer-expressible component uses w @ alpha_peers. Therefore the residual
        amplitude is (alpha_target - w @ alpha_peers).

        Args:
            weights: Oracle or estimated peer weights over peers (length N-1)
            x: Query point (1, p) or (p,)
            theta: Post-grid intensities (r,)

        Returns:
            pier_true: ndarray of shape (r,) giving the PIER curve.
        """
        if x is None:
            x = self.x_star
        if theta is None:
            theta = self.theta

        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        theta = np.asarray(theta, dtype=float)

        # Effect direction h(x)
        z = x[:, 1:]
        if z.size:
            h = float(z @ self.params.effect_weights)
        else:
            h = 0.0

        # Effect scale s(theta)
        s = np.clip(theta - float(self.params.intensity_break), 0.0, None)
        span = float(max(self.params.theta_span, 1e-12))
        s = s / span

        # Amplitude gap (target vs peer mixture)
        alpha_target = float(self.devices[0].alpha)
        alpha_peers = np.array([d.alpha for d in self.devices[1:]], dtype=float)
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if weights.size != alpha_peers.size:
            raise ValueError("weights length must equal number of peers (N-1)")
        amp_gap = alpha_target - float(np.dot(weights, alpha_peers))

        return s * h * amp_gap

    def sample_dataset(
        self,
        device_idx: int,
        n_samples: int,
        rng: np.random.RandomState,
        theta_range: Optional[tuple[float, float]] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample features and labels for a specific device."""
        device = self.devices[device_idx]
        feature_dim = device.params.base_weights.size
        p = feature_dim + 1

        if theta_range is None:
            theta_low = float(self.theta0[0]) if self.theta0.size else 0.0
            theta_high = float(self.theta[-1])
        else:
            theta_low, theta_high = theta_range

        local_shift = 0.1 * (device_idx - (len(self.devices) - 1) / 2.0)
        thetas = rng.uniform(theta_low, theta_high, size=n_samples) + local_shift
        thetas = np.clip(thetas, theta_low, theta_high)

        if feature_dim > 0:
            z = rng.normal(loc=device.base_offset, scale=1.0, size=(n_samples, feature_dim))
        else:
            z = np.zeros((n_samples, 0))

        X = np.concatenate([thetas.reshape(-1, 1), z], axis=1)
        y = device.predict_raw(X).reshape(-1)
        return X, y

    def save(self, path: Path) -> None:
        """Persist scenario parameters for later reuse."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        alphas = np.array([device.alpha for device in self.devices], dtype=np.float32)
        base_offsets = np.array([device.base_offset for device in self.devices], dtype=np.float32)
        noise_weights = np.stack([device.noise_weights for device in self.devices], axis=0)
        np.savez(
            path,
            base_weights=self.params.base_weights,
            effect_weights=self.params.effect_weights,
            intensity_break=self.params.intensity_break,
            theta_span=self.params.theta_span,
            noise_std=self.params.noise_std,
            theta0=self.theta0,
            theta=self.theta,
            X_probe=self.X_probe,
            x_star=self.x_star,
            alphas=alphas,
            base_offsets=base_offsets,
            noise_weights=noise_weights,
        )

    @classmethod
    def load(cls, path: Path) -> "SyntheticScenario":
        """Load previously saved scenario."""
        data = np.load(path, allow_pickle=True)
        params = SyntheticParameters(
            base_weights=data["base_weights"],
            effect_weights=data["effect_weights"],
            intensity_break=float(data["intensity_break"]),
            theta_span=float(data["theta_span"]),
            noise_std=float(data["noise_std"]),
        )
        alphas = data["alphas"].astype(float)
        base_offsets = data["base_offsets"].astype(float)
        noise_weights = data["noise_weights"].astype(float)
        devices = [
            SyntheticDevice(
                f"device_{i}",
                alpha=float(alphas[i]),
                params=params,
                base_offset=float(base_offsets[i]),
                noise_weights=noise_weights[i],
            )
            for i in range(len(alphas))
        ]
        return cls(
            devices=devices,
            theta0=data["theta0"],
            theta=data["theta"],
            X_probe=data["X_probe"],
            x_star=data["x_star"],
            params=params,
        )


def create_scenario(
    p: int,
    N: int,
    alpha_target: float,
    theta0: np.ndarray,
    theta_post: np.ndarray,
    m_probe: int,
    rng: np.random.RandomState,
    *,
    base_jitter: float = 0.3,
    peer_alpha_std: float = 0.1,
    noise_std: float = 0.05,
) -> SyntheticScenario:
    """Sample a synthetic scenario consistent with the CLI pipeline."""
    if p < 2:
        raise ValueError("p must be at least 2 to allocate an intensity dimension")

    theta0 = np.asarray(theta0, dtype=float)
    theta_post = np.asarray(theta_post, dtype=float)

    feature_dim = p - 1
    base_weights = rng.normal(size=feature_dim)
    effect_weights = rng.normal(size=feature_dim)

    intensity_break = float(theta0.max()) if theta0.size else 0.0
    theta_span = float(max(theta_post.max() - intensity_break, 1e-6))
    params = SyntheticParameters(
        base_weights=base_weights,
        effect_weights=effect_weights,
        intensity_break=intensity_break,
        theta_span=theta_span,
        noise_std=noise_std,
    )

    alphas = rng.normal(scale=peer_alpha_std, size=N)
    alphas[0] = alpha_target

    base_offsets = rng.normal(scale=base_jitter, size=N)
    if feature_dim > 0:
        noise_weights = rng.normal(scale=1.0, size=(N, feature_dim))
    else:
        noise_weights = np.zeros((N, 0))

    devices = [
        SyntheticDevice(
            id=f"device_{i}",
            alpha=float(alphas[i]),
            params=params,
            base_offset=float(base_offsets[i]),
            noise_weights=noise_weights[i],
        )
        for i in range(N)
    ]

    X_probe = np.zeros((m_probe, p))
    if feature_dim > 0:
        X_probe[:, 1:] = rng.normal(size=(m_probe, feature_dim))

    x_star = np.zeros((1, p))
    if feature_dim > 0:
        x_star[:, 1:] = rng.normal(size=(1, feature_dim))

    return SyntheticScenario(
        devices=devices,
        theta0=theta0,
        theta=theta_post,
        X_probe=X_probe,
        x_star=x_star,
        params=params,
    )
