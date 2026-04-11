# Mode Dynamics

Goal: describe neural network learning as a structured superposition of collective modes whose amplitudes evolve over time and whose directions map to interpretable functions or behaviors. In favorable regimes, this decomposition may become effectively low-dimensional.

## End-to-End Story (First Principles)

We start from gradient descent and progressively build a mode-level description of learning:

$$
\theta_t \;\rightarrow\; v_t \;\rightarrow\; \{u_k\} \;\rightarrow\; a_{k,t} \;\rightarrow\; \phi_k(x) \;\rightarrow\; \Delta f_t(x)\ \text{reconstruction}
$$

## Step 1) Set up the simplest task

Use a single-pattern dataset:

$$
y = \sin(x), \quad x \in [0, 2\pi]
$$

Why: this is the cleanest case where one dominant learning mode is plausible.

## Step 2) Train a small, readable model

Use a tiny MLP (1 input, 1 hidden layer, small width), MSE loss, and plain SGD.

Why: simpler optimization dynamics are easier to interpret.

## Step 3) Save checkpoints over training

At regular intervals, store:
- parameters $\theta_t$
- train loss
- model predictions on a fixed $x$-grid

This gives both parameter-space and function-space trajectories.

## Step 4) Compute update velocities

From checkpoints:

$$
v_t = \theta_{t+1} - \theta_t = -\eta \nabla_\theta L(\theta_t)
$$

Stack them into:

$$
\mathbf{V} =
\begin{bmatrix}
v_0^T \\
v_1^T \\
\vdots \\
v_{T-1}^T
\end{bmatrix}
\in \mathbb{R}^{T \times P}
$$

$\mathbf{V}$ is the raw learning-dynamics object.

## Step 5) Extract learning modes

Run SVD/PCA on $\mathbf{V}$:

$$
\mathbf{V} = Q \Sigma W^T
$$

Take right singular vectors $u_k$ as collective parameter-space modes.

Primary question: does one mode dominate on single-sine?

## Step 6) Measure mode contribution over time

Project each update onto modes:

$$
a_{k,t} = u_k^T v_t
$$

Energy share per step:

$$
\rho_{k,t} = \frac{a_{k,t}^2}{\sum_j a_{j,t}^2}
$$

This quantifies which modes are doing the learning at each time.

## Step 7) Map modes to visible function patterns

For each important $u_k$, compute induced function direction:

$$
\phi_k(x) \approx \frac{f(x,\theta+\epsilon u_k)-f(x,\theta)}{\epsilon}
$$

This is the bridge from internal mode to observable behavior.

## Step 8) Reconstruct function change from modes

Use:

$$
\Delta f_t(x) \approx \sum_k a_{k,t}\phi_k(x)
$$

Then check whether top 1-2 modes explain most of the actual output change.

## Step 9) Plot only core diagnostics

- loss vs training step
- predictions at early/mid/late checkpoints
- explained variance of modes
- $\rho_{k,t}$ over time
- top induced function $\phi_1(x)$
- actual vs reconstructed output change

## Step 10) Success criteria (single-sine stage)

Success means:
- one mode explains most variance in $\mathbf{V}$
- one mode carries most learning energy ($\rho_{1,t}$ high)
- $\phi_1(x)$ matches the residual learning direction
- 1-mode reconstruction captures most $\Delta f_t(x)$

## Step 11) Scale gradually after baseline works

1. $y=\sin(x)$
2. $y=\sin(x)+0.5\sin(2x)$
3. richer frequency mixtures
4. noisy synthetic data
5. real data

## Project files

- `mode_dynamics.py`: dataset/model/training/PCA utilities
- `experiment.py`: script run of the baseline pipeline
- `mode_dynamics_walkthrough.ipynb`: exploratory walkthrough and plots

## Run

From repository root:

```bash
uv run python mode_dynamics/experiment.py
```

Save PNG plots:

```bash
uv run python mode_dynamics/experiment.py --save-plots
```
