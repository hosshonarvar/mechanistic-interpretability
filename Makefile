# Run from repo root. Uses uv for Python env.

.PHONY: help sae-config sae-model sae-train sae-activations sae-train-sae sae-train-sae-all sae-stability sae-stability-all sae-plots sae-ablation sae-max-activating sae-all

help:
	@echo "SAE feature emergence (run from repo root):"
	@echo "  make sae-config        Load config and print checkpoint steps, paths, etc."
	@echo "  make sae-model        One forward pass; print batch, logits, residual shapes."
	@echo "  make sae-train        Train transformer; save checkpoints and loss history."
	@echo "  make sae-activations   Save residual activations per checkpoint (optional: STEP=100)."
	@echo "  make sae-train-sae     Train SAE on one checkpoint (optional: STEP=1000)."
	@echo "  make sae-train-sae-all Train one SAE per checkpoint."
	@echo "  make sae-stability     One SAE pair: use STEP_A=N STEP_B=M; else use sae-stability-all."
	@echo "  make sae-stability-all Consecutive pairs -> stability_results.json."
	@echo "  make sae-plots        Stability and loss -> results/*.png."
	@echo "  make sae-ablation     Ablate top-k features at step; print/save ΔCE (optional: STEP=2000, --save)."
	@echo "  make sae-max-activating Max-activating examples per feature -> results/max_activating_results.json (optional: STEP=4000)."
	@echo "  make sae-feature-dynamics Dominant token & consistency per step -> results/feature_dynamics.json (optional: STEPS=1000,2000,3000,4000)."
	@echo "  make sae-all          Full pipeline: train -> activations -> train-sae-all -> stability-all -> plots -> ablation."

sae-config:
	uv run python sae_feature_emergence/scripts/config.py

sae-model:
	uv run python sae_feature_emergence/scripts/model.py

sae-train:
	uv run python sae_feature_emergence/scripts/train.py

sae-activations:
	uv run python sae_feature_emergence/scripts/collect_activations.py $(STEP)

sae-train-sae:
	uv run python sae_feature_emergence/scripts/train_sae.py $(STEP)

sae-train-sae-all:
	uv run python sae_feature_emergence/scripts/train_sae.py --all

sae-stability:
	uv run python sae_feature_emergence/scripts/stability.py $(STEP_A) $(STEP_B)

sae-stability-all:
	uv run python sae_feature_emergence/scripts/stability.py --all

sae-plots:
	uv run python sae_feature_emergence/scripts/plots.py

STEP ?=
sae-ablation:
	uv run python sae_feature_emergence/scripts/ablation.py $(STEP) --save

sae-max-activating:
	uv run python sae_feature_emergence/scripts/max_activating.py $(STEP) --save

STEPS ?= 1000,2000,3000,4000
sae-feature-dynamics:
	uv run python sae_feature_emergence/scripts/feature_dynamics.py --steps "$(STEPS)" --save

sae-all: sae-train sae-activations sae-train-sae-all sae-stability-all sae-plots sae-ablation
	@echo "Pipeline complete. See sae_feature_emergence/results/ and findings notebook."
