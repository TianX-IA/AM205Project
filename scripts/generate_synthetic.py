"""
Generate synthetic damped oscillation data with Gaussian noise and store CSVs in data/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from gn_utils import damped_oscillation


def generate_datasets(
    noise_levels: Iterable[float],
    beta_true: np.ndarray,
    n_points: int = 200,
    t_max: float = 10.0,
    seed: int = 0,
    out_dir: Path | str = "data",
) -> None:
    """Create one CSV per noise level with columns t, y, y_clean, noise."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, t_max, n_points)
    clean = damped_oscillation(t, beta_true)

    meta_rows = []
    for sigma in noise_levels:
        noise = rng.normal(0.0, sigma, size=t.shape)
        y_noisy = clean + noise
        df = pd.DataFrame(
            {"t": t, "y": y_noisy, "y_clean": clean, "noise": noise}
        )
        sigma_tag = f"{sigma:.2f}".replace(".", "p")
        fname = out_path / f"synthetic_sigma{sigma_tag}.csv"
        df.to_csv(fname, index=False)
        meta_rows.append({"sigma": sigma, "file": str(fname.name)})

    meta = pd.DataFrame(meta_rows)
    meta.to_csv(out_path / "synthetic_metadata.csv", index=False)


def main() -> None:
    beta_true = np.array([1.3, 0.15, 2.0, 0.5, 0.1])
    noise_levels = (0.01, 0.05, 0.10, 0.20)
    generate_datasets(noise_levels, beta_true)


if __name__ == "__main__":
    main()
