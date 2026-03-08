# Run from repo root. Uses uv for Python env.

.PHONY: help sae-config sae-model sae-train sae-activations sae-train-sae sae-train-sae-all sae-stability sae-stability-all sae-plots sae-ablation

help:
	@echo "SAE feature emergence (run from repo root):"
	@echo "  make sae-config        Load config and print checkpoint steps, paths, etc."
	@echo "  make sae-model        One forward pass; print batch, logits, residual shapes."
	@echo "  make sae-train        Train transformer; save checkpoints and loss history."
	@echo "  make sae-activations   Save residual activations per checkpoint (optional: STEP=100)."
	@echo "  make sae-train-sae     Train SAE on one checkpoint (optional: STEP=1000)."
	@echo "  make sae-train-sae-all Train one SAE per checkpoint."
	@echo "  make sae-stability     Similarity and drift for one SAE pair (STEP_A=100 STEP_B=400)."
	@echo "  make sae-stability-all Consecutive pairs -> stability_results.json."
	@echo "  make sae-plots        Stability and loss -> results/*.png."
	@echo "  make sae-ablation     Ablate top-k features at step; print/save ΔCE (optional: STEP=2000, --save)."

sae-config:
	uv run python sae_feature_emergence/config.py

sae-model:
	uv run python sae_feature_emergence/model.py

sae-train:
	uv run python sae_feature_emergence/train.py

sae-activations:
	uv run python sae_feature_emergence/collect_activations.py $(STEP)

sae-train-sae:
	uv run python sae_feature_emergence/train_sae.py $(STEP)

sae-train-sae-all:
	uv run python sae_feature_emergence/train_sae.py --all

STEP_A ?= 100
STEP_B ?= 400
sae-stability:
	uv run python sae_feature_emergence/stability.py $(STEP_A) $(STEP_B)

sae-stability-all:
	uv run python sae_feature_emergence/stability.py --all

sae-plots:
	uv run python sae_feature_emergence/plots.py

STEP ?=
sae-ablation:
	uv run python sae_feature_emergence/ablation.py $(STEP) --save
