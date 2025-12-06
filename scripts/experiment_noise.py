"""
Experiment E2: sensitivity to measurement noise.
Generates fresh noisy draws for multiple sigma values and fits with Gauss–Newton.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gn_utils import GNConfig, gauss_newton, relative_error, damped_oscillation


def main() -> None:
    # Use the clean signal from the lowest-noise dataset as baseline.
    base_df = pd.read_csv("data/synthetic_sigma0p01.csv")
    t = base_df["t"].to_numpy()
    clean = base_df["y_clean"].to_numpy()

    beta_true = np.array([1.3, 0.15, 2.0, 0.5, 0.1])
    sigma_levels = [0.01, 0.05, 0.1, 0.2]
    trials = 20
    rng = np.random.default_rng(7)

    config = GNConfig(max_iter=60, grad_tol=1e-6)
    records = []

    # Slightly biased starting point to make the fit work a bit for GN.
    base_start = beta_true * np.array([0.9, 1.1, 0.8, 1.0, 1.0])
    base_start[3] += 0.2

    for sigma in sigma_levels:
        for trial in range(trials):
            noise = rng.normal(0.0, sigma, size=t.shape)
            y_noisy = clean + noise
            beta_hat, hist = gauss_newton(base_start, t, y_noisy, config)
            final = hist[-1]
            records.append(
                {
                    "sigma": sigma,
                    "trial": trial,
                    "sse": final["sse"],
                    "param_error": relative_error(beta_hat, beta_true),
                    "grad_norm": final["grad_norm"],
                }
            )

    out_df = pd.DataFrame(records)
    out_df.to_csv("data/noise_sensitivity_results.csv", index=False)

    summary = (
        out_df.groupby("sigma")
        .agg(
            param_error_median=("param_error", "median"),
            param_error_iqr=("param_error", lambda x: np.subtract(*np.percentile(x, [75, 25]))),
            sse_median=("sse", "median"),
            sse_iqr=("sse", lambda x: np.subtract(*np.percentile(x, [75, 25]))),
        )
        .reset_index()
    )

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.errorbar(
        summary["sigma"],
        summary["param_error_median"],
        yerr=0.5 * summary["param_error_iqr"],
        fmt="o-",
        capsize=4,
        color="#d95f02",
        ecolor="#1b9e77",
    )
    plt.xlabel("Noise level σ")
    plt.ylabel("Relative parameter error")
    plt.title("Noise sensitivity (Gauss–Newton)")
    plt.tight_layout()
    plt.savefig(plots_dir / "noise_param_error.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.errorbar(
        summary["sigma"],
        summary["sse_median"],
        yerr=0.5 * summary["sse_iqr"],
        fmt="o-",
        capsize=4,
        color="#4daf4a",
        ecolor="#377eb8",
    )
    plt.xlabel("Noise level σ")
    plt.ylabel("Final SSE")
    plt.title("Noise vs final objective")
    plt.tight_layout()
    plt.savefig(plots_dir / "noise_sse.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
