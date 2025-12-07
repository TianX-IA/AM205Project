"""Shared utilities for Gauss–Newton experiments on damped oscillation model.

Implements the forward model, analytic Jacobian, Gauss–Newton, and
Levenberg–Marquardt style damping variant.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np


def damped_oscillation(t: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Damped cosine model: f(t; beta) = A * exp(-lambda * t) * cos(omega * t + phi) + c.
    
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
    """Residuals r = y - f(beta) for least squares."""
    return y - damped_oscillation(t, beta)


def jacobian(beta: np.ndarray, t: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    """Analytic Jacobian of residuals r = y - f(beta). Shape (m, 5).
    
    Since dr/dbeta = -df/dbeta, we return -df/dbeta.
    Columns: [dA, dlam, domega, dphi, dc]
    """
    A, lam, omega, phi, _ = beta
    exp_term = np.exp(-lam * t)
    cos_term = np.cos(omega * t + phi)
    sin_term = np.sin(omega * t + phi)
    return np.column_stack((
        -exp_term * cos_term,                    # dr/dA = -df/dA
        A * t * exp_term * cos_term,             # dr/dlam = -df/dlam
        A * t * exp_term * sin_term,             # dr/domega = -df/domega
        A * exp_term * sin_term,                 # dr/dphi = -df/dphi
        -np.ones_like(t)                         # dr/dc = -df/dc
    ))


@dataclass
class GNConfig:
    """Configuration for Gauss–Newton algorithm."""
    max_iter: int = 80
    step_tol: float = 1e-8
    grad_tol: float = 1e-6
    damping: bool = False          # Use Levenberg–Marquardt damping
    mu_init: float = 1e-3          # Initial damping parameter
    mu_factor: float = 10.0        # Factor for updating mu
    use_normal_eq: bool = False    # Use normal equations instead of QR


def _qr_step(J: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Solve min ||J delta + r|| via QR without forming normal equations."""
    Q, R = np.linalg.qr(J, mode="reduced")
    try:
        return np.linalg.solve(R, -Q.T @ r)
    except np.linalg.LinAlgError:
        # Fall back to least-squares if R is singular
        return np.linalg.lstsq(J, -r, rcond=None)[0]


def _normal_eq_step(J: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Solve normal equations (J^T J) delta = -J^T r."""
    try:
        return np.linalg.solve(J.T @ J, -J.T @ r)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(J, -r, rcond=None)[0]


def _sanitize_beta(beta: np.ndarray) -> np.ndarray:
    """Keep parameters in numerically reasonable range to avoid overflow."""
    beta = beta.copy()
    beta[0] = float(np.clip(beta[0], -5.0, 5.0))      # amplitude
    beta[1] = float(np.clip(beta[1], 0.0, 2.0))      # damping (non-negative)
    beta[2] = float(np.clip(beta[2], 0.05, 8.0))     # frequency
    beta[3] = float(np.mod(beta[3], 2 * np.pi))      # phase (wrap)
    beta[4] = float(np.clip(beta[4], -5.0, 5.0))     # offset
    return beta


def gauss_newton(
    beta0: np.ndarray, t: np.ndarray, y: np.ndarray, config: GNConfig | None = None
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """Basic Gauss–Newton iteration with optional damping (LM flavor) and QR solve.
    
    Returns
    -------
    beta : np.ndarray
        Final parameter estimate.
    history : list of dict
        Iteration diagnostics with keys: iter, sse, grad_norm, cond_J, cond_JtJ, cond_JtJ_mu, mu.
    """
    if config is None:
        config = GNConfig()
    beta = _sanitize_beta(np.array(beta0, dtype=float))
    history: List[Dict[str, float]] = []
    mu = config.mu_init

    for k in range(config.max_iter):
        # Compute residuals, Jacobian, and gradient
        r = residuals(beta, t, y)
        J = jacobian(beta, t, y)
        grad = J.T @ r
        sse = 0.5 * float(r.T @ r)
        
        # Compute condition numbers
        try:
            cond_J = float(np.linalg.cond(J))
            cond_JtJ = float(np.linalg.cond(J.T @ J))
            cond_JtJ_mu = float(np.linalg.cond(J.T @ J + mu * np.eye(J.shape[1]))) if config.damping else cond_JtJ
        except np.linalg.LinAlgError:
            cond_J = cond_JtJ = cond_JtJ_mu = float("inf")
        
        history.append({
            "iter": k, "sse": sse, "grad_norm": float(np.linalg.norm(grad)),
            "cond_J": cond_J, "cond_JtJ": cond_JtJ, "cond_JtJ_mu": cond_JtJ_mu,
            "mu": mu if config.damping else 0.0,
        })
        
        # Check convergence
        if np.linalg.norm(grad) < config.grad_tol:
            break

        if config.damping:
            # Levenberg–Marquardt: (J^T J + mu*I) delta = -J^T r
            A = J.T @ J + mu * np.eye(J.shape[1])
            step = np.linalg.solve(A, -grad)
            trial_beta = _sanitize_beta(beta + step)
            trial_sse = 0.5 * float(residuals(trial_beta, t, y).T @ residuals(trial_beta, t, y))
            
            # Update damping parameter based on step quality
            if trial_sse < sse:
                beta = trial_beta
                mu = max(mu / config.mu_factor, 1e-12)  # Decrease mu (approach GN)
            else:
                mu = mu * config.mu_factor  # Increase mu (approach gradient descent)
        else:
            # Standard Gauss–Newton step
            step = _normal_eq_step(J, r) if config.use_normal_eq else _qr_step(J, r)
            if np.linalg.norm(step) <= config.step_tol * (np.linalg.norm(beta) + config.step_tol):
                break
            beta = _sanitize_beta(beta + step)

    return beta, history


def relative_error(est: np.ndarray, truth: np.ndarray) -> float:
    """Compute ||est - truth|| / ||truth||."""
    return float(np.linalg.norm(est - truth) / (np.linalg.norm(truth) + 1e-12))
