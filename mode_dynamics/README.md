# Mode Dynamics

Compact experiment for studying whether gradient-based learning evolves in a small number of dominant update modes.

## Project story (end-to-end)

Train a small network on `y = sin(x)`, record parameter updates during optimization, and test whether those updates are low-rank in parameter space.

Pipeline:

$$
\theta_t \rightarrow v_t \rightarrow \{u_k\} \rightarrow a_{k,t} \rightarrow c_k(t) \rightarrow \phi_k(x,t)
$$

where:

- \(v_t = \theta_{t+1} - \theta_t\) is the update velocity
- \(u_k\) are parameter-space modes (right singular vectors of the velocity matrix)
- \(a_{k,t} = u_k^\top v_t\) are per-step mode amplitudes
- \(c_k(t) = \sum_{\tau < t} a_{k,\tau}\) are cumulative mode coordinates
- \(\phi_k(x,t) = \nabla_\theta f(x;\theta_t)^\top u_k\) are induced functional directions

## Core equations

Gradient descent:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t),
\quad
v_t = -\eta \nabla_\theta L(\theta_t)
$$

Small-step loss change:

$$
L_{t+1} - L_t \approx \nabla L(\theta_t)^\top v_t = -\frac{1}{\eta}\|v_t\|^2
$$

Velocity matrix and mode extraction:

$$
\mathbf{V} =
\begin{bmatrix}
v_0^T \\
v_1^T \\
\vdots \\
v_{T-1}^T
\end{bmatrix}
\in \mathbb{R}^{T \times P},
\qquad
\mathbf{V} = Q \Sigma W^T
$$

Mode decomposition:

$$
v_t = \sum_k a_{k,t}u_k,
\quad
a_{k,t}=u_k^\top v_t,
\quad
\rho_{k,t}=\frac{a_{k,t}^2}{\sum_j a_{j,t}^2}
$$

## What to look for

- **Low-dimensional learning:** a small number of PCs explain most trajectory variance.
- **Mode dominance:** \(\rho_{1,t}\) (or top-\(K\) sum) stays high for much of training.
- **Dissipation:** \(\|v_t\|\) decreases over time.
- **Behavioral meaning:** perturbing along \(u_k\) changes function outputs in interpretable ways.

## Files

- `mode_dynamics.py`: reusable module (dataset, model, tracking, PCA)
- `experiment.py`: script that runs the full analysis and produces plots
- `mode_dynamics_walkthrough.ipynb`: notebook walkthrough with visual diagnostics

## Run

From repository root:

```bash
uv run python mode_dynamics/experiment.py
```

Save PNG plots:

```bash
uv run python mode_dynamics/experiment.py --save-plots
```
