"""Sklearn device adapters for tabular data."""

import numpy as np
from typing import Literal, Any
from disco.devices.base import Device, Scalarization, TaskType


class SklearnDevice:
    """Device adapter for sklearn models."""

    def __init__(
        self,
        id: str,
        model: Any,  # sklearn model
        task: TaskType,
        scalarization: Scalarization = "prob",
        temperature: float = 1.0
    ):
        """Initialize sklearn device.

        Args:
            id: Device identifier
            model: Trained sklearn model
            task: Task type (classification or regression)
            scalarization: How to scalarize outputs
            temperature: Temperature for probability calibration
        """
        self.id = id
        self.model = model
        self.task = task
        self.scalarization = scalarization
        self.temperature = temperature

        # Determine number of outputs
        if task == "regression":
            self.n_outputs = 1
        else:
            # Classification - get number of classes
            if hasattr(model, "classes_"):
                self.n_outputs = len(model.classes_)
            else:
                self.n_outputs = 2  # Default to binary

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Return raw model outputs."""
        if self.task == "regression":
            pred = self.model.predict(X)
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            return pred
        else:
            # Classification - return probabilities or decision values
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(X)
            elif hasattr(self.model, "decision_function"):
                dec = self.model.decision_function(X)
                if dec.ndim == 1:
                    # Binary classification
                    dec = np.column_stack([-dec, dec])
                return dec
            else:
                # Fallback to hard predictions
                pred = self.model.predict(X)
                n_classes = self.n_outputs
                probs = np.zeros((len(X), n_classes))
                for i, p in enumerate(pred):
                    probs[i, p] = 1.0
                return probs

    def _compute_probabilities(self, raw: np.ndarray) -> np.ndarray:
        """Convert raw classifier outputs to probabilities."""
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)

        # Apply temperature scaling if requested (pre-softmax)
        temperature = self.temperature if self.temperature > 0 else 1.0

        # If values already look like probabilities (non-negative and rows sum to ~1)
        if raw.min() >= 0 and raw.max() <= 1:
            row_sums = raw.sum(axis=1, keepdims=True)
            if np.allclose(row_sums, 1.0, atol=1e-6):
                return raw

        # Stable softmax
        shifted = raw / temperature
        shifted = shifted - shifted.max(axis=1, keepdims=True)
        exp_raw = np.exp(shifted)
        probs = exp_raw / exp_raw.sum(axis=1, keepdims=True)
        return probs

    def g(self, raw: np.ndarray, target_class: int = 1) -> np.ndarray:
        """Scalarize raw outputs."""
        if self.task == "regression":
            if self.scalarization == "identity":
                return raw.flatten()
            raise ValueError(f"Invalid scalarization {self.scalarization} for regression")

        # Classification scalarizations
        probs = self._compute_probabilities(raw)

        if self.scalarization == "prob":
            return probs[:, target_class]

        if self.scalarization == "logit":
            target_probs = np.clip(probs[:, target_class], 1e-10, 1 - 1e-10)
            return np.log(target_probs / (1 - target_probs))

        raise ValueError(f"Unknown scalarization: {self.scalarization}")


class XGBoostDevice(SklearnDevice):
    """Device adapter for XGBoost models."""

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Return raw XGBoost outputs."""
        import xgboost as xgb

        if self.task == "regression":
            pred = self.model.predict(X)
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            return pred
        else:
            # Classification - XGBoost specific
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(X)
            else:
                # Use raw predictions
                pred = self.model.predict(X, output_margin=True)
                if pred.ndim == 1:
                    # Binary classification
                    pred = np.column_stack([-pred, pred])
                return pred
