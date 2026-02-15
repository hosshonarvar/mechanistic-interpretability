# Logit Lens & Activation Patching

A mechanistic interpretability project: **logit lens** (how predictions form across layers) and **activation patching** (how much patching the residual stream recovers performance) on a language model.

## What's in this project

- **Logit lens:** At each layer we project the residual stream through the unembedding matrix and read off the top predicted token. This shows how "ready" the model's prediction is at each layer.
- **Activation patching:** We compare a *clean* run (model gets the right context) vs a *corrupted* run (wrong context). Then we patch the residual stream from the clean run into the corrupted run at each layer and measure how much the output recovers (logit difference toward the correct answer).

## Setup

This project uses the repo's **uv** environment. From the repo root:

```bash
uv sync
```

Then run the notebook with the same Python/uv kernel (e.g. select the venv that `uv` uses).

## How to run

Open `logit_lens_and_patching.ipynb` and run all cells. The notebook uses **TransformerLens** to load a small model (e.g. GPT-2 Small), runs one clean/corrupted pair, then:

1. Computes logit lens at each layer (top-1 token from residual stream at last position).
2. Runs residual-stream patching at each layer and computes patching effect.

Results are plotted at the end (logit lens by layer, patching effect by layer).

## Results (example)

- **Logit lens:** By layer L, the residual stream often already predicts the final answer; the plot shows when the "correct" token first appears in the top-1.
- **Patching effect:** Patching at later layers usually recovers more performance; the curve shows which layers are most important for the task.

## References

- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/)
- [Interpretability in the Wild (logit lens)](https://www.lesswrong.com/posts/cgBQxf2KPHvjBz7g8/interpretability-in-the-wild)

## Author

Hoss Honarvar
