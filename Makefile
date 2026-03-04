# Run from repo root. Uses uv for Python env.

.PHONY: help sae-config sae-model sae-train

help:
	@echo "SAE feature emergence (run from repo root):"
	@echo "  make sae-config   Load config and print checkpoint steps, paths, etc."
	@echo "  make sae-model    One forward pass; print batch, logits, residual shapes."
	@echo "  make sae-train   Train transformer; save checkpoints and loss history."

sae-config:
	uv run python sae_feature_emergence/config.py

sae-model:
	uv run python sae_feature_emergence/model.py

sae-train:
	uv run python sae_feature_emergence/train.py
