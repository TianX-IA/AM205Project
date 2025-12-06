"""
Experiment E4: show stabilization from Levenberg–Marquardt damping.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gn_utils import GNConfig, gauss_newton, relative_error, damped_oscillation


def main() -> None:
    rng = np.random.default_rng(99)
    beta_true = np.array([1.1, 0.12, 3.2, 0.4, 0.05])
    t = np.linspace(0.0, 12.0, 250)
    clean = damped_oscillation(t, beta_true)
    y = clean + rng.normal(0.0, 0.07, size=t.shape)

    # Poor starting point; wrong sign on lambda and mismatched frequency.
    beta0 = np.array([2.5, -0.05, 6.0, -2.0, 0.5])

    config_gn = GNConfig(max_iter=45, grad_tol=1e-6, damping=False)
    config_lm = GNConfig(
        max_iter=60, grad_tol=1e-6, damping=True, mu_init=1e-2, mu_factor=5.0
    )

    beta_plain, hist_plain = gauss_newton(beta0, t, y, config_gn)
    beta_damped, hist_damped = gauss_newton(beta0, t, y, config_lm)

    results = pd.DataFrame(
        [
            {
                "method": "GN",
                "sse": hist_plain[-1]["sse"],
                "param_error": relative_error(beta_plain, beta_true),
            },
            {
                "method": "LM",
                "sse": hist_damped[-1]["sse"],
                "param_error": relative_error(beta_damped, beta_true),
            },
        ]
    )
    results.to_csv("data/damping_results.csv", index=False)

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(7, 4.5))
    plt.semilogy(
        [h["iter"] for h in hist_plain], [h["sse"] for h in hist_plain], "o-", label="GN"
    )
    plt.semilogy(
        [h["iter"] for h in hist_damped],
        [h["sse"] for h in hist_damped],
        "s-",
        label="LM (damped)",
    )
    plt.xlabel("Iteration")
    plt.ylabel("SSE")
    plt.title("Damping prevents divergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "damping_objective.png", dpi=200)
    plt.close()

    # Show fitted curves for the final parameters.
    plt.figure(figsize=(7, 4.5))
    plt.scatter(t, y, s=12, color="#555555", alpha=0.6, label="noisy data")
    plt.plot(t, clean, color="black", lw=1.5, label="clean signal")
    plt.plot(t, damped_oscillation(t, beta_plain), "--", color="#d95f02", label="GN fit")
    plt.plot(
        t, damped_oscillation(t, beta_damped), "-", color="#1b9e77", label="LM fit"
    )
    plt.xlabel("t")
    plt.ylabel("y")
    plt.title("Model fit: plain GN vs damped LM")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "damping_fits.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
