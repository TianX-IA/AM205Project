# Nonlinear Least Squares: Gauss–Newton and Levenberg–Marquardt  
### A Focused Computational Study Using the Damped Oscillation Model  
*(AM205 Final Project — Rooted in Heath Chapter 6: Optimization)*

This project presents a **clean, focused, and computationally substantial** exploration of two classical nonlinear least squares algorithms:

---

## Project Structure

```
AM205Project/
├── README.md
├── report.md
├── .gitignore
└── scripts/
    ├── gn_utils.py
    ├── experiment_initialization.py
    ├── experiment_conditioning.py
    └── experiment_noise.py
```

**Note:** The `data/` and `plots/` directories are automatically created when running experiments. They are excluded from version control (see `.gitignore`).

---

## Quick Start

### Requirements

```bash
pip install numpy pandas matplotlib
```

### Running Experiments

Each experiment is self-contained and generates its own synthetic data. Run them individually:

```bash
# E1: Sensitivity to Initialization (GN vs LM)
python scripts/experiment_initialization.py

# E2: Ill-Conditioning (Normal Equations vs QR vs LM)
python scripts/experiment_conditioning.py

# E3: Sensitivity to Measurement Noise
python scripts/experiment_noise.py
```

**Output:**
- Results are saved to `data/` directory (CSV files)
- Plots are saved to `plots/` directory (PNG files)

### Running All Experiments

```bash
# Run all three experiments sequentially
python scripts/experiment_initialization.py && \
python scripts/experiment_conditioning.py && \
python scripts/experiment_noise.py
```

---

This project presents a **clean, focused, and computationally substantial** exploration of two classical nonlinear least squares algorithms:

- **Gauss–Newton (GN)**  
- **Levenberg–Marquardt (LM)**  

The study centers on a single realistic model — **damped oscillation** — and investigates **three essential numerical phenomena**:

1. **Sensitivity to initialization**  
2. **Ill-conditioning of the Gauss–Newton linear subproblem**  
3. **Effect of measurement noise**

These experiments align directly with *Heath, Scientific Computing, Section 6.6* and emphasize numerical stability, conditioning, and robustness — all core themes of AM205 scientific computing.

---

# 0. Problem Setup

We study parameter estimation for the nonlinear model

$$
f(t;\beta) = A e^{-\lambda t} \cos(\omega t + \phi) + c, \quad
\beta = (A, \lambda, \omega, \phi, c).
$$

Given data $y_i$, residuals are

$$
r_i(\beta) = y_i - f(t_i;\beta),
$$

and the nonlinear least squares objective is

$$
S(\beta) = \tfrac12 \|r(\beta)\|^2.
$$

Let $J(\beta)$ denote the Jacobian of $r(\beta)$.

---

# 1. Algorithms Implemented

## Gauss–Newton (GN)
Approximates the Hessian using $J^T J$:

$$
(J^T J)s_k = -J^T r.
$$

A classical local method: fast when residuals are small, but sensitive to initialization and conditioning.

---

## Levenberg–Marquardt (LM)
Adds a damping/regularization term:

$$
(J^T J + \mu_k I)s_k = -J^T r.
$$

- If a step succeeds → decrease $\mu_k$ (approach GN)  
- If a step fails → increase $\mu_k$ (approach gradient descent)

LM stabilizes GN when the linear subproblem is **ill-conditioned** or when initialization is **poor**, as emphasized in *Heath 6.6.2*.

---

# 2. Experiments

We present **three core experiments**, each illuminating a key numerical property of GN and LM.  
All figures in the report correspond directly to these sections.

---

# **E1 — Sensitivity to Initialization (GN vs LM)**

### Goal  
Demonstrate that GN is highly sensitive to initialization, while LM has a substantially larger **basin of attraction**.

### Setup  
- Moderate noise ($\sigma = 0.05$)  
- 40–80 random initializations around the true parameters  
- Several intentionally bad initializations  
  (e.g., wrong sign of λ, doubled ω)

For each run record:

- Convergence or divergence  
- Iterations  
- Final SSE  
- Parameter error

### Expected Results  
- **GN**: many failures unless started close to the true solution  
- **LM**: consistently convergent; rescuing bad initializations  
- Matches Heath’s statement:  
  *“GN may fail unless the initial guess is sufficiently close.”*

### Suggested Plots  
- Convergence rate bar chart  
- Scatter: initialization → final SSE  
- LM vs GN convergence trajectories (for selected runs)

---

# **E2 — Ill-Conditioning of the Gauss–Newton Subproblem**  
### (Normal Equations vs QR vs LM)

### Goal  
Study numerical stability when $J$ or $J^T J$ becomes ill-conditioned — a central issue in NLS.

### Setup  
- Long time horizon (e.g. $t \in [0, 25]$)  
- Compute conditioning of:
  - $J$
  - $J^T J$
  - $J^T J + \mu I$ (LM)

Compare:

1. **GN using normal equations**  
2. **GN using QR factorization**  
3. **LM using normal equations**

### Expected Results  
- $J^T J$ becomes extremely ill-conditioned  
- GN(normal eq) exhibits instability or erratic steps  
- GN(QR) is more stable but still sensitive  
- **LM dramatically improves stability** by shifting the spectrum and regularizing the system  
- Confirms *Heath 6.6.2* on LM’s role in “ill-conditioned or rank-deficient least squares problems”

### Suggested Plots  
- cond(J), cond(J^T J), cond(J^T J+μI) vs iteration  
- SSE vs iteration  
- Comparison of step quality

---

# **E3 — Sensitivity to Measurement Noise**

### Goal  
Quantify how noise affects parameter recovery.

### Setup  
- Fixed initialization  
- Noise levels  
  $$
  \sigma \in \{0.01, 0.05, 0.10, 0.20\}
  $$
- For each $\sigma$, run 20 trials and compute:
  - median parameter error  
  - median SSE  

### Expected Results  
- Parameter error grows approximately linearly with $\sigma$  
- SSE increases significantly with $\sigma$  
- Noise affects both GN and LM similarly; this experiment provides completeness and insight into data sensitivity

### Suggested Plots  
- Parameter error vs $\sigma$  
- SSE vs $\sigma$  

---

# 3. Discussion & Takeaways

### Gauss–Newton  
- Efficient when residuals are small  
- Very sensitive to initialization  
- Suffers in ill-conditioning due to $J^T J$

### Levenberg–Marquardt  
- More stable across all experiments  
- Handles bad initializations  
- Regularizes ill-conditioned subproblems  
- Provides predictable convergence behavior

### Overall  
**LM offers greater robustness at modest cost**, while GN is fast only under ideal conditions.  
These findings align cleanly with the theoretical descriptions in *Heath, Ch. 6.6*.

# End of README.md
