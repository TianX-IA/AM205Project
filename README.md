# Nonlinear Least Squares: Newton, Gauss–Newton, and Levenberg–Marquardt  
### Experimental Study for Damped Oscillation Model

This project investigates the numerical behavior of three classical optimization methods for nonlinear least squares (NLS):

- **Newton’s Method (full Hessian)**
- **Gauss–Newton (GN)**
- **Levenberg–Marquardt (LM)**

All experiments follow the formulation in *Heath, Scientific Computing, Ch. 6.6*, and use a **damped oscillation model**

$$
f(t;\theta) = \lambda e^{-\alpha t}\sin(\omega t), \quad 
\theta = (\lambda, \alpha, \omega).
$$

The residuals are
$$
r_i(\theta) = y_i - f(t_i;\theta)
$$
and the objective is
$$
\phi(\theta) = \tfrac12 \|r(\theta)\|^2.
$$

We compare Newton, GN, and LM on several aspects of nonlinear least squares:  
**initialization sensitivity, noise robustness, quality of the GN Hessian approximation, conditioning of the linear LS subproblem, and overall method robustness.**

---

## 0. Algorithms Implemented

### Newton Method (full Hessian)
$$
\nabla\phi(\theta)=J^T r, \qquad
H_\phi(\theta)=J^TJ+\sum_{i=1}^m r_i(\theta) H_{r_i}(\theta)
$$
$$
H_\phi(\theta_k)s_k = -\nabla\phi(\theta_k)
$$

### Gauss–Newton (GN)
Drop the second-order term:
$$
(J^T J)s_k = -J^T r.
$$

### Levenberg–Marquardt (LM)
Add regularization to handle ill-conditioning:
$$
(J^T J + \mu_k I)s_k = -J^T r.
$$
$\mu_k$ decreases when the iteration is successful and increases otherwise.

All solvers use identical convergence criteria:  
gradient tolerance, small step size, and max iterations.

---

# **E1 – Sensitivity to Initialization**

### Goal
Compare how robust Newton, GN, and LM are to different starting points.

### Setup
- Generate data with moderate noise (e.g., σ = 0.05).  
- Run 80 random initializations around the true parameters and several **bad initializations**  
  (wrong sign of λ, doubled frequency ω, etc.).  
- For each method record:
  - convergence / divergence  
  - iterations  
  - final SSE  
  - parameter error  

### Outputs
- Histogram of final SSE for each method  
- Scatter plot of initial guesses colored by convergence quality  
- Table summarizing convergence rates

### Expected Behavior
- **Newton** and **GN** often diverge from bad initializations  
- **LM** has the largest basin of attraction and strongest robustness  
- Confirms textbook statement: *GN may fail unless started sufficiently close to the solution*

---

# **E2 – Sensitivity to Measurement Noise**

### Goal
Quantify how noise in the data affects parameter estimation.

### Setup
- Fix one initialization.  
- Noise levels:  
  $$
  \sigma \in \{0.01,0.05,0.10,0.20\}.
  $$
- For each σ run 20 trials, compute:
  - median parameter error  
  - median SSE  

### Outputs
- Parameter error vs noise (with IQR bars)  
- SSE vs noise  

### Expected Behavior
- Parameter error increases roughly linearly with noise  
- SSE grows significantly as σ increases  
- Noise affects **all methods** similarly; algorithmic differences are not dominant here

---

# **E3 – Small vs Large Residual: Validity of the GN Approximation**

### Goal
Test when the GN approximation  
$$
H_\phi \approx J^T J
$$  
is accurate, and when the omitted curvature term  
$$
\sum_i r_i H_{r_i}
$$  
becomes significant.

### Setup
Two datasets:

1. **Small residual**: model is correct + tiny noise (σ = 0.01)  
2. **Large residual**: model is slightly mis-specified (e.g., add a phase shift in data generation)

Run Newton, GN, LM from the same good initialization.

### Outputs
- Convergence curves φ(θ_k)  
- Bar plot comparing magnitudes of  
  - $ \|J^TJ\| $  
  - $ \left\|\sum r_i H_{r_i} \right\| $  
  at the solution  

### Expected Behavior
- Small residual: Newton ≈ GN (curvature term negligible)  
- Large residual: Newton performs differently; GN slows or fails  
- LM remains stable but may not match Newton accuracy  

---

# **E4 – Ill-Conditioned Linear Subproblems & Role of LM**

### Goal
Demonstrate why LM is crucial when  
$$
J \ \text{or} \ J^T J
$$
is ill-conditioned or nearly rank–deficient.

### Setup
- Long time horizon (e.g., t ∈ [0, 25]) → Jacobian becomes ill-conditioned  
- Compare:
  1. GN + Normal Equations (solve $J^T J s = -J^T r$)  
  2. GN + QR Factorization  
  3. LM + Normal Equations  

### Measurements
- cond(J) and cond(JᵀJ + μI) per iteration  
- SSE trajectories  
- Step quality and convergence rate  

### Expected Behavior
- cond(JᵀJ) extremely large  
- GN(normal eq) unstable  
- GN(QR) more stable but still sensitive  
- **LM provides the strongest numerical stability**  
  due to the shift λI improving conditioning  

This directly verifies textbook Section 6.6.2.

---

# **E5 – Overall Comparison and Practical Recommendations**

| Scenario | Newton | Gauss–Newton | LM |
|---------|--------|---------------|----|
| Good initialization, small residual | ✔ Fast | ✔ Fast | ✔ Slightly slower |
| Mildly bad initialization | ✖ Often diverges | △ Sometimes converges | ✔ Robust |
| Ill-conditioned J | ✖ Very unstable | △ Sensitive | ✔ Most stable |
| Large residual / model mismatch | ✔ Can help | ✖ GN inaccurate | △ Moderately robust |
| Implementation cost | ✖ High (Hessian) | ✔ Low | ✔ Medium |

### Final Takeaways
- **GN** is efficient and accurate when residuals are small and conditioning is reasonable.  
- **Newton** is powerful but unstable and expensive; best used only when very close to the optimum.  
- **LM** is the most robust overall and handles poor initialization and ill-conditioning gracefully.  

---

# Suggested Directory Structure

```
.
├── data/
├── scripts/
│   ├── experiment_init.py
│   ├── experiment_noise.py
│   ├── experiment_residuals.py
│   ├── experiment_conditioning.py
│   ├── experiment_newton_gn_lm.py
│   └── utils_solvers.py
└── plots/
```

# End of README.md
