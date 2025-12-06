"""
Shared utilities for Gauss–Newton experiments on a damped oscillation model.

Implements the forward model, analytic Jacobian, Gauss–Newton, and a
Levenberg–Marquardt style damping variant. Designed for small experiments,
not as a production optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np


def damped_oscillation(t: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Damped cosine model: f(t; beta) = A * exp(-lambda * t) * cos(omega * t + phi) + c.

    Parameters
    ----------
    t : np.ndarray
        1D array of time points.
    beta : np.ndarray
        Parameter vector [A, lambda, omega, phi, c].
    """
    A, lam, omega, phi, c = beta
    exp_term = np.exp(-lam * t)
    return A * exp_term * np.cos(omega * t + phi) + c


def residuals(beta: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Residuals r = f(beta) - y for least squares."""
    return damped_oscillation(t, beta) - y


def jacobian(beta: np.ndarray, t: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    """
    Analytic Jacobian of residuals. Shape (m, 5).

    dr/dA = exp(-lam t) cos(...)
    dr/dlam = -A t exp(-lam t) cos(...)
    dr/domega = -A t exp(-lam t) sin(...)
    dr/dphi = -A exp(-lam t) sin(...)
    dr/dc = 1
    """
    A, lam, omega, phi, _ = beta
    exp_term = np.exp(-lam * t)
    cos_term = np.cos(omega * t + phi)
    sin_term = np.sin(omega * t + phi)

    dA = exp_term * cos_term
    dlam = -A * t * exp_term * cos_term
    domega = -A * t * exp_term * sin_term
    dphi = -A * exp_term * sin_term
    dc = np.ones_like(t)
    return np.column_stack((dA, dlam, domega, dphi, dc))


@dataclass
class GNConfig:
    max_iter: int = 80
    step_tol: float = 1e-8
    grad_tol: float = 1e-6
    damping: bool = False
    mu_init: float = 1e-3
    mu_factor: float = 10.0
    use_normal_eq: bool = False


def _qr_step(J: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Solve min ||J delta + r|| via QR without forming normal equations."""
    Q, R = np.linalg.qr(J, mode="reduced")
    try:
        return np.linalg.solve(R, -Q.T @ r)
    except np.linalg.LinAlgError:
        # Fall back to least-squares if R is singular.
        return np.linalg.lstsq(J, -r, rcond=None)[0]


def _normal_eq_step(J: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Solve normal equations; used to expose conditioning effects."""
    H = J.T @ J
    try:
        return np.linalg.solve(H, -J.T @ r)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(J, -r, rcond=None)[0]


def _sanitize_beta(beta: np.ndarray) -> np.ndarray:
    """Keep parameters in a numerically reasonable range to avoid overflow."""
    beta = beta.copy()
    beta[0] = float(np.clip(beta[0], -5.0, 5.0))  # amplitude can be signed
    beta[1] = float(np.clip(beta[1], 0.0, 2.0))  # damping should stay non-negative
    beta[2] = float(np.clip(beta[2], 0.05, 8.0))  # frequency
    beta[3] = float(np.mod(beta[3], 2 * np.pi))  # wrap phase
    beta[4] = float(np.clip(beta[4], -5.0, 5.0))  # offset
    return beta


def gauss_newton(
    beta0: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    config: GNConfig | None = None,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """
    Basic Gauss–Newton iteration with optional damping (LM flavor) and QR solve.

    Returns
    -------
    beta : np.ndarray
        Final parameter estimate.
    history : list of dict
        Iteration diagnostics with keys: iter, sse, grad_norm, cond_J, cond_JtJ, mu.
    """
    if config is None:
        config = GNConfig()
    beta = _sanitize_beta(np.array(beta0, dtype=float))
    history: List[Dict[str, float]] = []
    mu = config.mu_init

    for k in range(config.max_iter):
        r = residuals(beta, t, y)
        J = jacobian(beta, t, y)
        grad = J.T @ r
        sse = 0.5 * float(r.T @ r)
        try:
            cond_J = float(np.linalg.cond(J))
            cond_JtJ = float(np.linalg.cond(J.T @ J))
        except np.linalg.LinAlgError:
            cond_J = float("inf")
            cond_JtJ = float("inf")
        history.append(
            {
                "iter": k,
                "sse": sse,
                "grad_norm": float(np.linalg.norm(grad)),
                "cond_J": cond_J,
                "cond_JtJ": cond_JtJ,
                "mu": mu if config.damping else 0.0,
            }
        )
        if np.linalg.norm(grad) < config.grad_tol:
            break

        if config.damping:
            # Levenberg–Marquardt style damping with simple gain update.
            A = J.T @ J + mu * np.eye(J.shape[1])
            step = np.linalg.solve(A, -grad)
            trial_beta = _sanitize_beta(beta + step)
            trial_r = residuals(trial_beta, t, y)
            trial_sse = 0.5 * float(trial_r.T @ trial_r)
            if trial_sse < sse:
                beta = trial_beta
                mu = max(mu / config.mu_factor, 1e-12)
            else:
                mu = mu * config.mu_factor
        else:
            step = _normal_eq_step(J, r) if config.use_normal_eq else _qr_step(J, r)
            if np.linalg.norm(step) <= config.step_tol * (np.linalg.norm(beta) + config.step_tol):
                break
            beta = _sanitize_beta(beta + step)

    return beta, history


def relative_error(est: np.ndarray, truth: np.ndarray) -> float:
    """Compute ||est - truth|| / ||truth||."""
    return float(np.linalg.norm(est - truth) / (np.linalg.norm(truth) + 1e-12))
