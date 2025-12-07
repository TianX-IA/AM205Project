"""Experiment E3: Sensitivity to Measurement Noise.

Quantify how noise affects parameter recovery.
- Fixed initialization
- Noise levels σ ∈ {0.01, 0.05, 0.10, 0.20}
- For each σ, run 20 trials and compute median parameter error and median SSE
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gn_utils import GNConfig, gauss_newton, relative_error, damped_oscillation


def main() -> None:
    # Generate clean signal as baseline
    beta_true = np.array([1.3, 0.15, 2.0, 0.5, 0.1])
    t = np.linspace(0.0, 10.0, 200)
    clean = damped_oscillation(t, beta_true)
    rng = np.random.default_rng(7)
    config = GNConfig(max_iter=60, grad_tol=1e-6)
    
    # Slightly biased starting point
    base_start = beta_true * np.array([0.9, 1.1, 0.8, 1.0, 1.0])
    base_start[3] += 0.2

    # Run trials for each noise level
    records = []
    for sigma in [0.01, 0.05, 0.1, 0.2]:
        for trial in range(20):
            y_noisy = clean + rng.normal(0.0, sigma, size=t.shape)
            beta_hat, hist = gauss_newton(base_start.copy(), t, y_noisy, config)
            final = hist[-1]
            records.append({
                "sigma": sigma, "trial": trial, "sse": final["sse"],
                "param_error": relative_error(beta_hat, beta_true),
                "grad_norm": final["grad_norm"],
            })

    # Save results
    out_df = pd.DataFrame(records)
    Path("data").mkdir(exist_ok=True)
    out_df.to_csv("data/noise_sensitivity_results.csv", index=False)

    # Compute summary statistics (median and IQR)
    summary = out_df.groupby("sigma").agg(
        param_error_median=("param_error", "median"),
        param_error_iqr=("param_error", lambda x: np.percentile(x, 75) - np.percentile(x, 25)),
        sse_median=("sse", "median"),
        sse_iqr=("sse", lambda x: np.percentile(x, 75) - np.percentile(x, 25)),
    ).reset_index()

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # Plot 1: Parameter error vs noise level
    plt.figure(figsize=(6, 4))
    plt.errorbar(summary["sigma"], summary["param_error_median"],
                 yerr=0.5 * summary["param_error_iqr"], fmt="o-", capsize=4,
                 color="#d95f02", ecolor="#1b9e77")
    plt.xlabel("Noise level σ")
    plt.ylabel("Relative parameter error")
    plt.title("Noise sensitivity (parameter error vs σ)")
    plt.tight_layout()
    plt.savefig(plots_dir / "noise_param_error.png", dpi=200)
    plt.close()

    # Plot 2: Final SSE vs noise level
    plt.figure(figsize=(6, 4))
    plt.errorbar(summary["sigma"], summary["sse_median"],
                 yerr=0.5 * summary["sse_iqr"], fmt="o-", capsize=4,
                 color="#4daf4a", ecolor="#377eb8")
    plt.xlabel("Noise level σ")
    plt.ylabel("Final SSE")
    plt.title("Noise sensitivity (final SSE vs σ)")
    plt.tight_layout()
    plt.savefig(plots_dir / "noise_sse.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
