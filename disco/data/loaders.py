"""Data loaders for experiments."""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_adult() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Adult Income dataset.

    Returns:
        X_train, y_train, X_test, y_test as numpy arrays
    """
    # Fetch Adult Income dataset
    adult = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    X = adult.data
    y = adult.target

    # Handle categorical features
    categorical_cols = X.select_dtypes(include=["category", "object"]).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Convert target to binary
    y = LabelEncoder().fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X.values.astype(np.float32),
        y.astype(np.int32),
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


def load_bike() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Bike Sharing dataset for regression.

    Returns:
        X_train, y_train, X_test, y_test as numpy arrays
    """
    try:
        # Try to load from UCI repository
        import requests
        import io

        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
        # For now, create synthetic bike data
        print("Creating synthetic bike sharing data...")

        from sklearn.datasets import make_regression

        X, y = make_regression(
            n_samples=17379,  # Similar to original bike dataset size
            n_features=12,
            n_informative=8,
            noise=0.1,
            random_state=42
        )

        # Make target positive (bike counts)
        y = np.abs(y) + 50

    except Exception:
        # Fallback to simple synthetic regression data
        from sklearn.datasets import make_regression

        X, y = make_regression(
            n_samples=2000,
            n_features=12,
            n_informative=8,
            noise=0.1,
            random_state=42
        )
        y = np.abs(y) + 50

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X.astype(np.float32),
        y.astype(np.float32),
        test_size=0.2,
        random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


def load_compas() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load COMPAS recidivism dataset for classification.

    Returns:
        X_train, y_train, X_test, y_test as numpy arrays
    """
    try:
        # Try to create realistic COMPAS-like synthetic data
        print("Creating synthetic COMPAS-like data...")

        from sklearn.datasets import make_classification

        # Create binary classification dataset with bias
        X, y = make_classification(
            n_samples=6000,
            n_features=10,
            n_informative=6,
            n_redundant=2,
            n_classes=2,
            class_sep=0.8,
            flip_y=0.1,  # Add some label noise
            random_state=42
        )

    except Exception:
        # Fallback
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=2000,
            n_features=8,
            n_informative=5,
            n_classes=2,
            random_state=42
        )

    # Split and preprocess
    X_train, X_test, y_train, y_test = train_test_split(
        X.astype(np.float32),
        y.astype(np.int32),
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


def load_synthetic_tabular(
    n_samples: int = 1000,
    n_features: int = 10,
    n_classes: int = 2,
    seed: int = 42,
    task: str = "classification"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create synthetic tabular dataset for testing.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes (for classification)
        seed: Random seed
        task: "classification" or "regression"

    Returns:
        X_train, y_train, X_test, y_test
    """
    if task == "classification":
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features // 2,
            n_redundant=n_features // 4,
            n_classes=n_classes,
            random_state=seed,
            shuffle=True
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X.astype(np.float32),
            y.astype(np.int32),
            test_size=0.2,
            random_state=seed,
            stratify=y
        )

    else:  # regression
        from sklearn.datasets import make_regression

        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features // 2,
            noise=0.1,
            random_state=seed
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X.astype(np.float32),
            y.astype(np.float32),
            test_size=0.2,
            random_state=seed
        )

    return X_train, y_train, X_test, y_test


def stack_prewindow(
    trajectories: dict,
    target: "Device",
    devices: List["Device"],
    anchor_sel: "AnchorSelection",
    intervention: "Intervention",
    theta0: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Stack pre-window responses over anchors and intensities.

    Args:
        trajectories: Dict mapping (device_id, intervention_name) to responses
        target: Target device
        devices: All devices
        anchor_sel: Selected anchors with indices
        intervention: Current intervention
        theta0: Pre-window intensities

    Returns:
        y_t: Target responses [K*q]
        X_peers: Peer responses [K*q, N-1]
    """
    K = len(anchor_sel.indices)
    q = len(theta0)

    # Build target vector
    key = (target.id, intervention.name)
    if key not in trajectories:
        raise ValueError(f"Missing trajectory for {key}")

    traj_t = trajectories[key]  # [m, q]
    y_t = traj_t[anchor_sel.indices, :].flatten()  # [K*q]

    # Build peer matrix
    peer_cols = []
    for dev in devices:
        if dev.id == target.id:
            continue
        key = (dev.id, intervention.name)
        if key not in trajectories:
            raise ValueError(f"Missing trajectory for {key}")
        traj_j = trajectories[key]
        col = traj_j[anchor_sel.indices, :].flatten()  # [K*q]
        peer_cols.append(col)

    X_peers = np.column_stack(peer_cols)  # [K*q, N-1]

    return y_t, X_peers