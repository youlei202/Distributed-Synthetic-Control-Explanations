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
        """Evaluate device on feature-only inputs."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        features = X

        base_input = features @ self.params.base_weights + self.base_offset
        base = np.tanh(base_input)

        effect = self.alpha * (features @ self.params.effect_weights)

        if self.noise_weights.size:
            noise_driver = features @ self.noise_weights
        else:
            noise_driver = np.zeros(len(features), dtype=float)
        noise = self.params.noise_std * np.tanh(noise_driver)

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
        self, device_idx: int, x: np.ndarray, theta_grid: np.ndarray, intervention
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
        theta: Optional[np.ndarray] = None,
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
        peer_curves = [
            self.evaluate_device(i + 1, x, theta, intervention)
            for i in range(len(weights))
        ]
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
        intervention,
    ) -> np.ndarray:
        """Compute ground-truth PIER curve by evaluating the generator."""

        if x is None:
            x = self.x_star
        if theta is None:
            theta = self.theta

        _, _, tau = self.compute_tau(
            weights=weights,
            intervention=intervention,
            x=x,
            theta=theta,
        )
        return tau

    def sample_dataset(
        self,
        device_idx: int,
        n_samples: int,
        rng: np.random.RandomState,
        theta_range: Optional[tuple[float, float]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample features and labels for a specific device."""
        device = self.devices[device_idx]
        feature_dim = device.params.base_weights.size

        if theta_range is not None:
            # kept for API compatibility; ignored so we return baseline features
            pass

        features = rng.normal(
            loc=device.base_offset,
            scale=1.0,
            size=(n_samples, feature_dim),
        )

        y = device.predict_raw(features).reshape(-1)
        return features, y

    def save(self, path: Path) -> None:
        """Persist scenario parameters for later reuse."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        alphas = np.array([device.alpha for device in self.devices], dtype=np.float32)
        base_offsets = np.array(
            [device.base_offset for device in self.devices], dtype=np.float32
        )
        noise_weights = np.stack(
            [device.noise_weights for device in self.devices], axis=0
        )
        np.savez(
            path,
            base_weights=self.params.base_weights,
            effect_weights=self.params.effect_weights,
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
    if p < 1:
        raise ValueError("p must be at least 1 for feature-only generation")

    theta0 = np.asarray(theta0, dtype=float)
    theta_post = np.asarray(theta_post, dtype=float)

    feature_dim = p
    base_weights = rng.normal(size=feature_dim)
    effect_weights = rng.normal(size=feature_dim)
    params = SyntheticParameters(
        base_weights=base_weights,
        effect_weights=effect_weights,
        noise_std=noise_std,
    )

    alphas = rng.normal(scale=peer_alpha_std, size=N)
    alphas[0] = alpha_target

    base_offsets = rng.normal(scale=base_jitter, size=N)
    noise_weights = rng.normal(scale=1.0, size=(N, feature_dim))

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

    X_probe = rng.normal(size=(m_probe, feature_dim))
    x_star = rng.normal(size=(1, feature_dim))

    return SyntheticScenario(
        devices=devices,
        theta0=theta0,
        theta=theta_post,
        X_probe=X_probe,
        x_star=x_star,
        params=params,
    )
