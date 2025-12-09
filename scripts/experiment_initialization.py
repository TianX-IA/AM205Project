"""Experiment E1: Sensitivity to initialization (GN vs LM).

Demonstrates that GN is highly sensitive to initialization, while LM has a 
substantially larger basin of attraction.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gn_utils import GNConfig, gauss_newton, relative_error, damped_oscillation


def sample_initializations(beta_true: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Draw random initial guesses around the truth with controlled perturbations."""
    scales = np.array([0.6, 0.8, 0.8, 0.7, 0.5])
    raw = beta_true * (1.0 + rng.normal(0.0, scales, size=(n_samples, beta_true.size)))
    raw[:, 0] = np.clip(raw[:, 0], 0.1, 3.0)      # amplitude
    raw[:, 1] = np.clip(raw[:, 1], 0.01, 0.8)    # lambda (damping)
    raw[:, 2] = np.clip(raw[:, 2], 0.2, 5.0)     # omega (frequency)
    raw[:, 3] = np.mod(raw[:, 3], 2 * np.pi)     # phase
    raw[:, 4] = np.clip(raw[:, 4], -1.5, 1.5)    # offset
    return raw


def create_bad_initializations(beta_true: np.ndarray) -> list[np.ndarray]:
    """Create intentionally bad initializations to test robustness."""
    return [
        beta_true * np.array([-1, 1, 1, 1, 1]),  # wrong sign amplitude
        beta_true * np.array([1, 1, 2, 1, 1]),  # doubled omega
        beta_true * np.array([-1, 1, 2, 1, 1]),  # both wrong
        beta_true * np.array([2.5, 0.5, 3.0, 1.5, 2.0]),  # far from truth
        beta_true.copy() + np.array([0, -beta_true[1] - 0.1, 0, 0, 0]),  # negative damping
    ]


def main() -> None:
    # Generate synthetic data with moderate noise (σ = 0.05)
    beta_true = np.array([1.3, 0.15, 2.0, 0.5, 0.1], dtype=float)
    rng = np.random.default_rng(42)
    t = np.linspace(0.0, 10.0, 200)
    y = damped_oscillation(t, beta_true) + rng.normal(0.0, 0.05, size=t.shape)

    # Combine random and intentionally bad initializations
    # Ensure all initializations are np.ndarray for type consistency
    inits = [np.array(b, dtype=float) for b in sample_initializations(beta_true, 60, rng)]
    inits += create_bad_initializations(beta_true)
    config_gn = GNConfig(max_iter=80, grad_tol=1e-6, damping=False)
    config_lm = GNConfig(max_iter=80, grad_tol=1e-6, damping=True, mu_init=1e-3, mu_factor=10.0)

    # Run both methods for each initialization
    results = []
    for beta0 in inits:
        # Ensure beta0 is a float array for robustness
        beta0 = np.asarray(beta0, dtype=float)
        beta_gn, hist_gn = gauss_newton(beta0, t, y, config_gn)
        beta_lm, hist_lm = gauss_newton(beta0, t, y, config_lm)
        final_gn, final_lm = hist_gn[-1], hist_lm[-1]
        results.append({
            "beta0_A": beta0[0], "beta0_lambda": beta0[1], "beta0_omega": beta0[2],
            "beta0_phi": beta0[3], "beta0_c": beta0[4],
            "gn_sse": final_gn["sse"], "gn_iterations": len(hist_gn),
            "gn_converged": final_gn["grad_norm"] < config_gn.grad_tol,
            "gn_param_error": relative_error(beta_gn, beta_true), "gn_grad_norm": final_gn["grad_norm"],
            "lm_sse": final_lm["sse"], "lm_iterations": len(hist_lm),
            "lm_converged": final_lm["grad_norm"] < config_lm.grad_tol,
            "lm_param_error": relative_error(beta_lm, beta_true), "lm_grad_norm": final_lm["grad_norm"],
        })

    res_df = pd.DataFrame(results)
    Path("data").mkdir(exist_ok=True)
    res_df.to_csv("data/init_sensitivity_results.csv", index=False)

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # Plot 1: Convergence rate bar chart
    gn_conv, lm_conv = res_df["gn_converged"].sum(), res_df["lm_converged"].sum()
    total = len(res_df)
    plt.figure(figsize=(6, 4))
    x = np.arange(2)
    plt.bar(x - 0.175, [gn_conv, lm_conv], 0.35, label="Converged", color="#1b9e77")
    plt.bar(x + 0.175, [total - gn_conv, total - lm_conv], 0.35, label="Failed", color="#d95f02")
    plt.ylabel("Count")
    plt.title("Convergence rate: GN vs LM")
    plt.xticks(x, ["GN", "LM"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "init_convergence_rate.png", dpi=200)
    plt.close()

    # Plot 2: Basins of attraction comparison
    vmax = res_df[["gn_sse", "lm_sse"]].max().max()
    plt.figure(figsize=(12, 5))
    for idx, (method, col) in enumerate([("GN", "gn_sse"), ("LM", "lm_sse")], 1):
        plt.subplot(1, 2, idx)
        sc = plt.scatter(res_df["beta0_omega"], res_df["beta0_lambda"], c=res_df[col],
                        cmap="viridis", s=50, alpha=0.9, edgecolors="k", vmin=0, vmax=vmax)
        plt.colorbar(sc, label="Final SSE")
        plt.scatter([beta_true[2]], [beta_true[1]], color="red", marker="x", s=80, label="truth")
        plt.xlabel(r"$\omega_0$ (initial)")
        plt.ylabel(r"$\lambda_0$ (initial)")
        plt.title(f"{method}: Basins (color = final SSE)")
        plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "init_basins_comparison.png", dpi=200)
    plt.close()

    # Plot 3: Convergence trajectories (example where GN fails but LM succeeds)
    gn_failed_lm_succeeded = res_df[(~res_df["gn_converged"]) & (res_df["lm_converged"])]
    if len(gn_failed_lm_succeeded) > 0:
        # Select case with largest GN SSE to make the difference more visible
        gn_failed_lm_succeeded_sorted = gn_failed_lm_succeeded.sort_values("gn_sse", ascending=False)
        idx = gn_failed_lm_succeeded_sorted.index[0]
        beta0_sel = np.array([res_df.loc[idx, f"beta0_{k}"] for k in ["A", "lambda", "omega", "phi", "c"]], dtype=float)
        
        _, hist_gn_sel = gauss_newton(beta0_sel, t, y, config_gn)
        _, hist_lm_sel = gauss_newton(beta0_sel, t, y, config_lm)
        
        # Extract SSE values and filter out inf/nan
        gn_iters = [h["iter"] for h in hist_gn_sel]
        gn_sse = [h["sse"] for h in hist_gn_sel]
        lm_iters = [h["iter"] for h in hist_lm_sel]
        lm_sse = [h["sse"] for h in hist_lm_sel]
        
        gn_sse_clean = [s if np.isfinite(s) else 1e10 for s in gn_sse]
        lm_sse_clean = [s if np.isfinite(s) else 1e10 for s in lm_sse]
        
        all_sse = gn_sse_clean + lm_sse_clean
        y_min = max(1e-6, min(all_sse) * 0.1)
        y_max = min(1e10, max(all_sse) * 10) if max(all_sse) < 1e9 else 1e10
        
        plt.figure(figsize=(8, 6))
        plt.plot(gn_iters, gn_sse_clean, "o-", label="GN", color="#d95f02", 
                linewidth=3, markersize=10, alpha=1.0, zorder=3, 
                markerfacecolor="#d95f02", markeredgecolor="white", markeredgewidth=1.5)
        plt.plot(lm_iters, lm_sse_clean, "s-", label="LM", color="#1b9e77", 
                linewidth=3, markersize=10, alpha=1.0, zorder=2,
                markerfacecolor="#1b9e77", markeredgecolor="white", markeredgewidth=1.5)
        plt.yscale("log")
        plt.xlabel("Iteration", fontsize=13)
        plt.ylabel("SSE", fontsize=13)
        plt.title("Convergence trajectories: GN vs LM\n(Example: GN fails, LM succeeds)", 
                 fontsize=13, fontweight="bold")
        plt.ylim(y_min, y_max)
        plt.xlim(-0.5, max(max(gn_iters), max(lm_iters)) + 0.5)
        plt.legend(fontsize=12, loc="best", framealpha=0.9)
        plt.grid(True, alpha=0.4, which="both", linestyle="--")
        plt.tight_layout()
        plt.savefig(plots_dir / "init_trajectories.png", dpi=200, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()
