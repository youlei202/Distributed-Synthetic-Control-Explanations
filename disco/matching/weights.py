"""Weight solver for synthetic control matching."""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal
from disco.matching.simplex import project


@dataclass
class MatchInputs:
    """Inputs for matching problem."""
    y_t: np.ndarray      # Target responses [K*q]
    X_peers: np.ndarray  # Peer responses [K*q, N-1]
    lam: float           # Ridge regularization


@dataclass
class MatchResult:
    """Results from matching."""
    w: np.ndarray        # Weights [N-1], on simplex
    eps_pre: float       # Average pre-window fit error
    residual: np.ndarray # Residual vector [K*q]
    n_iter: int          # Number of iterations (if iterative)


class WeightSolver:
    """Solves simplex-constrained ridge regression for synthetic control weights."""

    def __init__(
        self,
        method: Literal["projected_grad", "newton"] = "projected_grad",
        max_iter: int = 1000,
        tol: float = 1e-6,
        verbose: bool = False
    ):
        """Initialize weight solver.

        Args:
            method: Optimization method
            max_iter: Maximum iterations
            tol: Convergence tolerance
            verbose: Print convergence info
        """
        self.method = method
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def solve(self, inputs: MatchInputs) -> MatchResult:
        """Solve min ||X w - y||^2 + λ||w||^2 s.t. w≥0, 1^T w=1.

        Args:
            inputs: Matching inputs

        Returns:
            MatchResult with weights and diagnostics
        """
        if self.method == "projected_grad":
            return self._solve_projected_gradient(inputs)
        elif self.method == "newton":
            return self._solve_projected_newton(inputs)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _solve_projected_gradient(self, inputs: MatchInputs) -> MatchResult:
        """Projected gradient descent solver."""
        X = inputs.X_peers
        y = inputs.y_t
        lam = inputs.lam

        n_peers = X.shape[1]

        # Precompute for efficiency
        XtX = X.T @ X + lam * np.eye(n_peers)
        Xty = X.T @ y

        # Initialize weights uniformly
        w = np.ones(n_peers) / n_peers

        # Lipschitz constant for step size
        L = np.linalg.norm(XtX, 2)
        step_size = 1.0 / L

        for it in range(self.max_iter):
            w_old = w.copy()

            # Gradient
            grad = XtX @ w - Xty

            # Gradient step
            w_tmp = w - step_size * grad

            # Project onto simplex
            w = project(w_tmp, z=1.0)

            # Check convergence
            if np.linalg.norm(w - w_old) < self.tol:
                if self.verbose:
                    print(f"Converged in {it+1} iterations")
                break
        else:
            if self.verbose:
                print(f"Max iterations {self.max_iter} reached")

        # Compute diagnostics
        residual = X @ w - y
        eps_pre = np.linalg.norm(residual) / len(y)

        return MatchResult(
            w=w,
            eps_pre=eps_pre,
            residual=residual,
            n_iter=it + 1
        )

    def _solve_projected_newton(self, inputs: MatchInputs) -> MatchResult:
        """Projected Newton solver (for small N-1)."""
        X = inputs.X_peers
        y = inputs.y_t
        lam = inputs.lam

        n_peers = X.shape[1]

        if n_peers > 100:
            if self.verbose:
                print(f"Warning: Newton method with {n_peers} peers may be slow")

        # Initialize weights
        w = np.ones(n_peers) / n_peers

        # Precompute
        XtX = X.T @ X + lam * np.eye(n_peers)
        Xty = X.T @ y

        for it in range(self.max_iter):
            w_old = w.copy()

            # Gradient and Hessian
            grad = XtX @ w - Xty
            hess = XtX  # Constant in this case

            # Newton direction (solve Hd = -g)
            try:
                direction = -np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                # Fall back to gradient if singular
                direction = -grad

            # Line search with projection
            alpha = 1.0
            for _ in range(20):  # Backtracking line search
                w_new = project(w + alpha * direction, z=1.0)
                if self._objective(w_new, X, y, lam) < self._objective(w, X, y, lam):
                    w = w_new
                    break
                alpha *= 0.5
            else:
                # No improvement, use projected gradient step
                w = project(w - grad / np.linalg.norm(XtX, 2), z=1.0)

            # Check convergence
            if np.linalg.norm(w - w_old) < self.tol:
                if self.verbose:
                    print(f"Newton converged in {it+1} iterations")
                break

        # Compute diagnostics
        residual = X @ w - y
        eps_pre = np.linalg.norm(residual) / len(y)

        return MatchResult(
            w=w,
            eps_pre=eps_pre,
            residual=residual,
            n_iter=it + 1
        )

    def _objective(self, w: np.ndarray, X: np.ndarray, y: np.ndarray, lam: float) -> float:
        """Compute objective value."""
        residual = X @ w - y
        return 0.5 * np.linalg.norm(residual)**2 + 0.5 * lam * np.linalg.norm(w)**2