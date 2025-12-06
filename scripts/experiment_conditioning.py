"""
Experiment E3: conditioning of J and comparison of QR vs normal equations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gn_utils import GNConfig, gauss_newton, relative_error, damped_oscillation


def main() -> None:
    rng = np.random.default_rng(123)
    beta_true = np.array([0.8, 0.08, 1.6, 0.9, 0.05])
    t = np.linspace(0.0, 25.0, 320)
    clean = damped_oscillation(t, beta_true)
    y = clean + rng.normal(0.0, 0.08, size=t.shape)

    beta0 = np.array([1.2, 0.02, 1.0, 0.1, -0.2])

    config_qr = GNConfig(max_iter=60, grad_tol=1e-6, use_normal_eq=False)
    config_ne = GNConfig(max_iter=60, grad_tol=1e-6, use_normal_eq=True)

    beta_qr, hist_qr = gauss_newton(beta0, t, y, config_qr)
    beta_ne, hist_ne = gauss_newton(beta0, t, y, config_ne)

    # Save final metrics for the report.
    final_records = pd.DataFrame(
        [
            {
                "method": "QR",
                "sse": hist_qr[-1]["sse"],
                "param_error": relative_error(beta_qr, beta_true),
                "cond_J_last": hist_qr[-1]["cond_J"],
            },
            {
                "method": "NormalEq",
                "sse": hist_ne[-1]["sse"],
                "param_error": relative_error(beta_ne, beta_true),
                "cond_J_last": hist_ne[-1]["cond_J"],
            },
        ]
    )
    final_records.to_csv("data/conditioning_final_metrics.csv", index=False)

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # Conditioning over iterations
    plt.figure(figsize=(7, 4.5))
    plt.semilogy(
        [h["iter"] for h in hist_qr], [h["cond_J"] for h in hist_qr], label="QR cond(J)"
    )
    plt.semilogy(
        [h["iter"] for h in hist_qr],
        [h["cond_JtJ"] for h in hist_qr],
        "--",
        label="QR cond(JᵀJ)",
    )
    plt.semilogy(
        [h["iter"] for h in hist_ne],
        [h["cond_J"] for h in hist_ne],
        label="Normal Eq cond(J)",
        color="#d95f02",
    )
    plt.semilogy(
        [h["iter"] for h in hist_ne],
        [h["cond_JtJ"] for h in hist_ne],
        "--",
        label="Normal Eq cond(JᵀJ)",
        color="#7570b3",
    )
    plt.xlabel("Iteration")
    plt.ylabel("Condition number")
    plt.title("Conditioning of Jacobian and JTJ")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "conditioning_numbers.png", dpi=200)
    plt.close()

    # Objective path comparison
    plt.figure(figsize=(7, 4.5))
    plt.semilogy(
        [h["iter"] for h in hist_qr],
        [h["sse"] for h in hist_qr],
        "o-",
        label="QR solve",
    )
    plt.semilogy(
        [h["iter"] for h in hist_ne],
        [h["sse"] for h in hist_ne],
        "s-",
        label="Normal equations",
    )
    plt.xlabel("Iteration")
    plt.ylabel("SSE")
    plt.title("Objective decrease vs solver")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "conditioning_objective.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
