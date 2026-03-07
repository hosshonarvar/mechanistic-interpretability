# Run from repo root. Uses uv for Python env.

.PHONY: help sae-config sae-model sae-train sae-activations sae-train-sae sae-train-sae-all

help:
	@echo "SAE feature emergence (run from repo root):"
	@echo "  make sae-config        Load config and print checkpoint steps, paths, etc."
	@echo "  make sae-model         One forward pass; print batch, logits, residual shapes."
	@echo "  make sae-train         Train transformer; save checkpoints and loss history."
	@echo "  make sae-activations   Save residual activations per checkpoint (optional: STEP=100)."
	@echo "  make sae-train-sae     Train SAE on one checkpoint (optional: STEP=1000)."
	@echo "  make sae-train-sae-all Train one SAE per checkpoint."

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
