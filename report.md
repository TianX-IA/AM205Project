Gauss–Newton for Nonlinear Least Squares: Convergence, Damping, and Sensitivity to Initialization
================================================================================================

Model and objective
-------------------

- Damped oscillation: \(f(t;\beta)=A e^{-\lambda t}\cos(\omega t+\phi)+c\), \(\beta=(A,\lambda,\omega,\phi,c)\).
- Residuals and loss: \(r_i(\beta)=f(t_i,\beta)-y_i,\quad S(\beta)=\tfrac12\sum_i r_i(\beta)^2=\tfrac12\lVert r(\beta)\rVert_2^2\).
- Linearization \(r(\beta+\Delta)\approx r(\beta)+J(\beta)\Delta\) with Jacobian \(J_{ij}=\partial r_i/\partial \beta_j\).
- Gauss–Newton step solves \(\min_\Delta \lVert J\Delta+r\rVert_2\); implementation uses QR rather than explicitly forming \((J^\top J)^{-1}\).
- Damping (LM): \((J^\top J+\mu I)\Delta=-J^\top r\) with \(\mu\) shrunk on successful steps and enlarged otherwise.

Implementation notes
--------------------

- Code in `scripts/gn_utils.py` (analytic Jacobian, GN + LM, parameter clipping to avoid overflow).
- Synthetic data generator `scripts/generate_synthetic.py` stores CSVs in `data/` with multiple noise levels.
- Each experiment is a separate script in `scripts/` and writes plots to `plots/` and metrics to `data/`.

Experiment highlights
---------------------

**E1 – Initialization sensitivity (`scripts/experiment_initialization.py`)**
- 80 random starts around truth (\(\sigma=0.05\) noise, `data/synthetic_sigma0p05.csv`); only 51% met the gradient tolerance.
- Median final SSE \(0.26\); 95th percentile SSE \(80.1\); median relative parameter error 1.67 (large spread).
- Figures: `plots/init_sse_hist.png` (SSE histogram) and `plots/init_basins.png` ((\(\omega_0,\lambda_0\)) scatter colored by final SSE).

**E2 – Noise sensitivity (`scripts/experiment_noise.py`)**
- 20 trials per \(\sigma\in\{0.01,0.05,0.10,0.20\}\) from a mildly biased start.
- Median parameter error grows roughly linearly with noise: 0.0015 → 0.0286 as \(\sigma\) increases; median SSE rises from 0.01 → 3.77.
- Figures: `plots/noise_param_error.png` (error vs noise with IQR bars) and `plots/noise_sse.png` (objective vs noise).

**E3 – Conditioning and solver choice (`scripts/experiment_conditioning.py`)**
- Long horizon data (\(t\in[0,25]\)) yields \(\kappa(J)\approx 8.4\times 10^2\); \(\kappa(J^\top J)\) is correspondingly larger.
- Final metrics: QR solve SSE 12.77, relative parameter error 3.22; normal equations SSE 12.74, error 2.61 (slightly better here but higher sensitivity to conditioning early on).
- Figures: `plots/conditioning_numbers.png` (cond(J) and cond(\(J^\top J\)) per iteration) and `plots/conditioning_objective.png` (SSE paths).

**E4 – Damping vs plain GN (`scripts/experiment_damping.py`)**
- Bad start with wrong sign on \(\lambda\) and doubled frequency; plain GN diverged (final SSE 1590, error 2.05) while LM converged (SSE 0.576, error 0.0047).
- Figures: `plots/damping_objective.png` (objective trajectories) and `plots/damping_fits.png` (data + fits).

Data and reproducibility
------------------------

- Synthetic CSVs and experiment outputs live in `data/`; plots are in `plots/`.
- Run experiments with `conda run -n nlp_env python scripts/<experiment>.py` (set `MPLCONFIGDIR=.mplconfig` if matplotlib cache is unwritable).
