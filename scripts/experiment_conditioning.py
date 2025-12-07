"""Experiment E2: Ill-Conditioning (Normal Equations vs QR vs LM).

Study numerical stability when J or J^T J becomes ill-conditioned.
Compare:
1. GN using normal equations
2. GN using QR factorization
3. LM using normal equations
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gn_utils import GNConfig, gauss_newton, relative_error, damped_oscillation


def main() -> None:
    # Generate synthetic data with long time horizon (t ∈ [0, 25]) to induce ill-conditioning
    rng = np.random.default_rng(123)
    beta_true = np.array([0.8, 0.08, 1.6, 0.9, 0.05])
    t = np.linspace(0.0, 25.0, 320)
    y = damped_oscillation(t, beta_true) + rng.normal(0.0, 0.08, size=t.shape)
    beta0 = np.array([1.2, 0.02, 1.0, 0.1, -0.2])

    # Three methods: GN(normal eq), GN(QR), LM(normal eq)
    config_gn_ne = GNConfig(max_iter=60, grad_tol=1e-6, use_normal_eq=True, damping=False)
    config_gn_qr = GNConfig(max_iter=60, grad_tol=1e-6, use_normal_eq=False, damping=False)
    config_lm = GNConfig(max_iter=60, grad_tol=1e-6, use_normal_eq=True, damping=True, mu_init=1e-3, mu_factor=10.0)

    beta_gn_ne, hist_gn_ne = gauss_newton(beta0, t, y, config_gn_ne)
    beta_gn_qr, hist_gn_qr = gauss_newton(beta0, t, y, config_gn_qr)
    beta_lm, hist_lm = gauss_newton(beta0, t, y, config_lm)

    # Save final metrics
    Path("data").mkdir(exist_ok=True)
    pd.DataFrame([
        {"method": "GN(NormalEq)", "sse": hist_gn_ne[-1]["sse"],
         "param_error": relative_error(beta_gn_ne, beta_true),
         "cond_J_last": hist_gn_ne[-1]["cond_J"], "cond_JtJ_last": hist_gn_ne[-1]["cond_JtJ"]},
        {"method": "GN(QR)", "sse": hist_gn_qr[-1]["sse"],
         "param_error": relative_error(beta_gn_qr, beta_true),
         "cond_J_last": hist_gn_qr[-1]["cond_J"], "cond_JtJ_last": hist_gn_qr[-1]["cond_JtJ"]},
        {"method": "LM(NormalEq)", "sse": hist_lm[-1]["sse"],
         "param_error": relative_error(beta_lm, beta_true),
         "cond_J_last": hist_lm[-1]["cond_J"], "cond_JtJ_last": hist_lm[-1]["cond_JtJ"],
         "cond_JtJ_mu_last": hist_lm[-1]["cond_JtJ_mu"]},
    ]).to_csv("data/conditioning_final_metrics.csv", index=False)

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # Plot 1: Conditioning over iterations
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    for hist, label, color in [(hist_gn_ne, "GN(NormalEq)", "#d95f02"),
                                (hist_gn_qr, "GN(QR)", "#7570b3"),
                                (hist_lm, "LM", "#1b9e77")]:
        plt.semilogy([h["iter"] for h in hist], [h["cond_J"] for h in hist], label=f"{label} cond(J)", color=color)
    plt.ylabel("cond(J)")
    plt.title("Conditioning of Jacobian and J^T J")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    for hist, label, color in [(hist_gn_ne, "GN(NormalEq)", "#d95f02"),
                                (hist_gn_qr, "GN(QR)", "#7570b3"),
                                (hist_lm, "LM", "#1b9e77")]:
        plt.semilogy([h["iter"] for h in hist], [h["cond_JtJ"] for h in hist],
                     "--", label=f"{label} cond(JᵀJ)", color=color)
    plt.semilogy([h["iter"] for h in hist_lm], [h["cond_JtJ_mu"] for h in hist_lm],
                 "-", label="LM cond(JᵀJ + μI)", color="#1b9e77", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Condition number")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "conditioning_numbers.png", dpi=200)
    plt.close()

    # Plot 2: SSE vs iteration comparison
    plt.figure(figsize=(7, 4.5))
    for hist, label, marker, color in [(hist_gn_ne, "GN(NormalEq)", "o", "#d95f02"),
                                        (hist_gn_qr, "GN(QR)", "s", "#7570b3"),
                                        (hist_lm, "LM(NormalEq)", "^", "#1b9e77")]:
        plt.semilogy([h["iter"] for h in hist], [h["sse"] for h in hist],
                     f"{marker}-", label=label, color=color)
    plt.xlabel("Iteration")
    plt.ylabel("SSE")
    plt.title("Objective decrease: GN vs LM")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "conditioning_objective.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
