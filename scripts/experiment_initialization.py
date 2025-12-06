"""
Experiment E1: sensitivity to initialization for Gauss–Newton on synthetic data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gn_utils import GNConfig, gauss_newton, relative_error
from gn_utils import damped_oscillation  # noqa: F401 (kept for reference)


def sample_initializations(
    beta_true: np.ndarray, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Draw random initial guesses around the truth with controlled perturbations.
    """
    scales = np.array([0.6, 0.8, 0.8, 0.7, 0.5])
    raw = beta_true * (1.0 + rng.normal(0.0, scales, size=(n_samples, beta_true.size)))
    raw[:, 0] = np.clip(raw[:, 0], 0.1, 3.0)  # amplitude
    raw[:, 1] = np.clip(raw[:, 1], 0.01, 0.8)  # lambda
    raw[:, 2] = np.clip(raw[:, 2], 0.2, 5.0)  # omega
    raw[:, 3] = np.mod(raw[:, 3], 2 * np.pi)  # phase
    raw[:, 4] = np.clip(raw[:, 4], -1.5, 1.5)  # offset
    return raw


def main() -> None:
    data_path = Path("data/synthetic_sigma0p05.csv")
    df = pd.read_csv(data_path)
    t = df["t"].to_numpy()
    y = df["y"].to_numpy()
    beta_true = np.array([1.3, 0.15, 2.0, 0.5, 0.1])

    rng = np.random.default_rng(42)
    n_samples = 80
    inits = sample_initializations(beta_true, n_samples, rng)
    results = []
    config = GNConfig(max_iter=80, grad_tol=1e-6)

    for beta0 in inits:
        beta_hat, hist = gauss_newton(beta0, t, y, config)
        final = hist[-1]
        results.append(
            {
                "beta0_A": beta0[0],
                "beta0_lambda": beta0[1],
                "beta0_omega": beta0[2],
                "beta0_phi": beta0[3],
                "beta0_c": beta0[4],
                "sse": final["sse"],
                "grad_norm": final["grad_norm"],
                "cond_J": final["cond_J"],
                "param_error": relative_error(beta_hat, beta_true),
                "converged": final["grad_norm"] < config.grad_tol * 5,
            }
        )

    res_df = pd.DataFrame(results)
    res_df.to_csv("data/init_sensitivity_results.csv", index=False)

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # Histogram of final SSE
    plt.figure(figsize=(6, 4))
    plt.hist(res_df["sse"], bins=20, color="#377eb8", edgecolor="black", alpha=0.8)
    plt.xlabel("Final SSE")
    plt.ylabel("Count")
    plt.title("Initialization sensitivity: final SSE distribution")
    plt.tight_layout()
    plt.savefig(plots_dir / "init_sse_hist.png", dpi=200)
    plt.close()

    # Scatter in (omega, lambda) space colored by final SSE
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(
        res_df["beta0_omega"],
        res_df["beta0_lambda"],
        c=res_df["sse"],
        cmap="viridis",
        s=50,
        alpha=0.9,
        edgecolor="k",
    )
    plt.colorbar(sc, label="Final SSE")
    plt.scatter(
        [beta_true[2]], [beta_true[1]], color="red", marker="x", s=80, label="truth"
    )
    plt.xlabel(r"$\omega_0$ (initial)")
    plt.ylabel(r"$\lambda_0$ (initial)")
    plt.title("Basins (color = final SSE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "init_basins.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
