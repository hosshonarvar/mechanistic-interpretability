# Mode Dynamics of Neural Network Learning

**Goal:** describe neural network learning as a structured superposition of collective modes whose amplitudes evolve over time, where each mode induces an interpretable function or behavior in the model’s output. In favorable regimes, this decomposition may become effectively low-dimensional.

---

## Key Hypothesis

Learning dynamics are structured and decomposable into collective modes, and these modes correspond to meaningful patterns in function space.

---

## End-to-End Story (First Principles)

We learn a shared mode basis $\{u_k\}$ from training dynamics and use it to describe how the model evolves in function space.

### 1) Local learning dynamics (core pipeline)

$$
\theta_t 
\;\rightarrow\; 
v_t = \theta_{t+1} - \theta_t 
\;\rightarrow\; 
\{u_k\} 
\;\rightarrow\; 
a_{k,t} = u_k^\top v_t 
\;\rightarrow\; 
\phi_k(x, \theta_t) 
\;\rightarrow\; 
\Delta f_t(x)
$$

This is the fundamental object: **how the model changes at each step**.

---

### 2) Local reconstruction (validated)

$$
\Delta f_t(x) 
\;\approx\; 
\sum_k a_{k,t}\,\phi_k(x, \theta_t)
$$

This describes learning as a superposition of mode-induced function updates.

---

### 3) Global function evolution (path-integrated)

$$
f_t(x) 
\;=\; 
f_0(x) 
\;+\; 
\sum_{\tau=0}^{t-1} \Delta f_\tau(x)
$$

Combining with mode decomposition:

$$
f_t(x) - f_0(x)
\;\approx\;
\sum_{\tau=0}^{t-1} \sum_k a_{k,\tau}\,\phi_k(x, \theta_\tau)
$$

---

### ⚠️ Important clarification

A single fixed linear expansion:

$$
f_t(x) \approx f_{\text{ref}}(x) + \sum_k \alpha_{k,t}\,\phi_k^{\text{ref}}(x)
$$

is **not generally valid**, because:

- $\nabla_\theta f(x, \theta)$ changes during training  
- neural networks are nonlinear in parameters  
- learning is path-dependent  

---

## Core Equations

### A) Update-space decomposition

Let:

$$
v_t = \theta_{t+1} - \theta_t
$$

Decompose:

$$
v_t = \sum_k a_{k,t} u_k, \qquad a_{k,t} = u_k^\top v_t
$$

Induced function:

$$
\phi_k(x, \theta_t) 
\;\approx\; 
\frac{f(x, \theta_t + \epsilon u_k) - f(x, \theta_t)}{\epsilon}
$$

Then:

$$
\Delta f_t(x)
\;=\;
f(x, \theta_{t+1}) - f(x, \theta_t)
\;\approx\;
\sum_k a_{k,t}\,\phi_k(x, \theta_t)
$$

Per-step energy:

$$
\rho_{k,t} = \frac{a_{k,t}^2}{\sum_j a_{j,t}^2}
$$

---

## Notation at a glance

- $u_k$: parameter-space mode (learned from dynamics)  
- $a_{k,t}$: coefficient of update $v_t$  
- $\phi_k(x, \theta_t)$: induced function at time $t$  
- $\rho_{k,t}$: mode energy contribution  

---

## Step 1) Set up the simplest task

$$
y = \sin(x), \quad x \in [0, 2\pi]
$$

Why: clean single-pattern system.

---

## Step 2) Train a small, readable model

- tiny MLP  
- 1 hidden layer  
- MSE loss  
- SGD  

---

## Step 3) Save checkpoints

Store:

- $\theta_t$  
- loss  
- predictions on a fixed grid  

---

## Step 4) Compute update velocities

$$
v_t = \theta_{t+1} - \theta_t
$$

Stack:

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

---

## Step 5) Extract learning modes

$$
\mathbf{V} = Q \Sigma W^T
$$

Take:

$$
u_k = \text{columns of } W
$$

These are collective parameter-space modes.

---

## Step 6) Measure mode contribution

$$
a_{k,t} = u_k^\top v_t
$$

$$
\rho_{k,t} = \frac{a_{k,t}^2}{\sum_j a_{j,t}^2}
$$

---

## Step 7) Map modes to function space

$$
\phi_k(x, \theta_t) 
\;\approx\; 
\frac{f(x, \theta_t + \epsilon u_k) - f(x, \theta_t)}{\epsilon}
$$

Interpretation:

$\phi_k(x)$ is the pattern added to the output when moving along $u_k$.

---

## Step 8) Reconstruct function updates

$$
\Delta f_t(x) 
\;\approx\; 
\sum_k a_{k,t}\,\phi_k(x, \theta_t)
$$

This is the primary validation step.

---

## Step 9) Diagnostics

Plot:

- loss vs epoch  
- predictions over time  
- explained variance of modes  
- $\rho_{k,t}$ over time  
- $\phi_k(x, t)$  
- $\Delta f_t$ reconstruction  

---

## Step 10) Success criteria

- modes explain most variance in $\mathbf{V}$  
- few modes dominate $\rho_{k,t}$  
- $\phi_k(x)$ are structured  
- $\Delta f_t(x)$ is well reconstructed with small $K$  

---

## Step 11) Scaling plan

1. $\sin(x)$  
2. $\sin(x) + 0.5\sin(2x)$  
3. richer mixtures  
4. noisy data  
5. real data  

---

## Key Insight

Learning is best described as:

> a superposition of mode-induced function updates over time

not:

> a static decomposition of the final function

---

## Project files

- `mode_dynamics.py`  
- `experiment.py`  
- `mode_dynamics_walkthrough.ipynb`  

---

## Run

From repository root:

```bash
uv run python mode_dynamics/experiment.py
```

Save PNG plots:

```bash
uv run python mode_dynamics/experiment.py --save-plots
```
