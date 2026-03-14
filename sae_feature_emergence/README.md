# SAE feature emergence

We study when and how SAE-discovered features emerge and stabilize during transformer training. We train a small transformer, train the same SAE at each checkpoint on one layer, and measure feature stability across time. Pipeline is built step by step; each step is runnable. A findings notebook loads saved results and presents the report.

**Layout**  
- Top level: `findings.ipynb`, `README.md` (notebook and docs).
- `scripts/`: all Python code (config, model, train, data, SAE, stability, ablation, max-activating, feature_dynamics, plots). Run from repo root via `make` (see main repo Makefile).
- Data and outputs stay here: `checkpoints/`, `activations/`, `sae_models/`, `results/`.

---

## Research question and hypotheses

**Feature emergence dynamics**  
Do SAE-discovered features emerge **gradually** during training or via **sudden phase transitions**? When do they become stable, and do they causally contribute to the model?

**Experimental setup**  
Train one SAE per checkpoint (same architecture, same layer) across training; match features across checkpoints (Hungarian on decoder directions); measure **drift** (1 − similarity) between consecutive pairs. Then test causality (ablation) and interpretability (max-activating examples).

**Hypotheses**
- **H1 — Gradual:** Feature identity stabilizes gradually over training.
- **H2 — Sharp:** Feature identity stabilizes sharply after a critical loss regime.
- **H3 — Reorganization:** Features form early but reorganize later.
 
**How we distinguish them**
- **Stability:** Drift = 1 − mean cosine similarity of matched feature directions (lower drift = more stable). Plots: drift vs step and vs loss, with regime-change shading.
- **Causal validation:** Ablate top-k SAE feature contributions; measure ΔCE. Random-k control confirms the effect is specific to top features.
- **Interpretability:** Max-activating examples show which token/context makes each feature fire (with synthetic data, mainly token detectors).

**Scope**  
One model size, one layer, one SAE config (fixed across checkpoints).

---

## Implementation plan

**Pipeline steps (1–12):** each step is runnable and produces something. The **findings notebook** loads those outputs and interprets them (drift, regime change, H1/H2/H3, ablation, max-activating examples). Run a step, then you can inspect its output in the notebook or in the terminal.

| Step | What we do | What we get | How we check it |
|------|------------|-------------|-----------------|
| 1 | Hypothesis: research question, H1/H2/H3 | Hypothesis doc | Written down; we can point to it |
| 2 | Config: model (layers, d_model, …), training (batch, lr, checkpoint steps), SAE (n_features, L1, which layer), paths | Config we can load | Load config; print e.g. checkpoint steps |
| 3 | Model and data: decoder-only transformer, residual at one layer, synthetic token batches | Model and data loader | One forward pass; print batch, logits, residual shapes |
| 4 | Training loop; save checkpoints and loss history at configured steps | `checkpoints/`, loss history | Run training; list checkpoint files, print final loss |
| 5 | Load checkpoint, run model, save residual activations at one layer per checkpoint | `activations/step_*.pt` | Run for one checkpoint; show activation shape and stats |
| 6 | Implement SAE; train on one checkpoint's activations | One trained SAE | Train SAE; print/plot reconstruction error; save SAE |
| 7 | Same SAE config; train one SAE per checkpoint | `sae_models/sae_step_*.pt` | Run for all steps; list saved SAE files |
| 8 | Load two SAEs, feature directions, Hungarian match; compute similarity and drift | Similarity and drift for one pair | Run for one pair; print similarity and drift |
| 9 | Loop consecutive checkpoint pairs; save metrics to JSON | `results/stability_results.json` | Produce file; print table |
| 10 | Read stability and loss; write drift plots (with regime-change shading) | `results/*.png` | Generate PNGs; confirm saved |
| 11 | Ablate top-k features, patch resid, measure ΔCE; random-k control; save | `results/ablation_results.json` | Print/save ablation result |
| 12 | Max-activating examples per feature (token + context) | `results/max_activating_results.json` | Run script; inspect examples in notebook |
| 13 | Feature dynamics: dominant token and consistency per (step, feature) | `results/feature_dynamics.json` | Run `make sae-feature-dynamics` (optional STEPS=1000,2000,3000,4000); table + consistency-vs-step plot in notebook |

**Findings notebook**  
Loads the outputs from the steps above (stability_results.json, plots, loss history, ablation, max_activating_results.json, feature_dynamics.json if present). Presents drift and regime change, interprets H1/H2/H3, summarizes ablation (causal + random-k), max-activating examples, and feature dynamics (dominant token and consistency over steps; phase transition in interpretability). Lists caveats and further experiments.

---

## How to run

From the **repo root** (parent of `sae_feature_emergence/`), with `uv`:

1. **`make sae-all`** — runs the full pipeline (train → activations → SAEs → stability → plots → ablation).
2. **`make help`** — lists all targets (for individual steps or optional runs like `sae-max-activating`, `sae-feature-dynamics`).
3. Open **`sae_feature_emergence/findings.ipynb`** to view the report.
