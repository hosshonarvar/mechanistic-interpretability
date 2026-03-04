# SAE feature emergence

We study when and how SAE-discovered features emerge and stabilize during transformer training. We train a small transformer, train the same SAE at each checkpoint on one layer, and measure feature stability across time. Pipeline is built step by step; each step is runnable. A findings notebook loads saved results and presents the report.

---

## Research question and hypotheses

**Question**  
How do interpretable (SAE-discovered) features emerge and stabilize during transformer training? Do they emerge gradually or abruptly? When do they become stable causal contributors?

**Hypotheses**
- **H1 — Gradual:** Feature identity stabilizes gradually over training.
- **H2 — Sharp:** Feature identity stabilizes sharply after a critical loss regime.
- **H3 — Reorganization:** Features form early but reorganize later.
 
**How we distinguish them**
- Metrics: cosine similarity, drift, sparsity, logit contribution
- Plots: similarity and drift vs step (and vs loss)
- Validation: zero or patch 2–3 features, measure logit impact

**Scope**  
One model size, one layer, one SAE config (fixed across checkpoints).

---

## Implementation plan

**Pipeline steps (1–12):** each step is runnable and produces something. The **findings notebook** loads those outputs and investigates them (interpretation, H1/H2/H3, ablation). Run a step, then you can inspect its output in the notebook or in the terminal.

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
| 10 | Read stability and loss; write plots | `results/*.png` | Generate PNGs; confirm saved |
| 11 | (Optional) Ablate top-k features, patch resid, measure cross-entropy change | Ablation result | Print/save ablation result |
| 12 | One script runs steps 4–10 (and optionally 11); README says how to run and where results go | Pipeline | Run script; artifacts in checkpoints/, activations/, sae_models/, results/ |

**Findings notebook**  
Loads the outputs from the steps above (e.g. stability_results.json, plots, loss history, ablation). Presents them and interprets (what the curves suggest re H1/H2/H3; what ablation shows). Run the pipeline (or individual steps) first; then open the notebook to investigate.
