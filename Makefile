# Run from repo root. Uses uv for Python env.

.PHONY: help sae-config sae-model

help:
	@echo "SAE feature emergence (run from repo root):"
	@echo "  make sae-config   Load config and print checkpoint steps, paths, etc."
	@echo "  make sae-model    One forward pass; print batch, logits, residual shapes."

sae-config:
	uv run python sae_feature_emergence/config.py

sae-model:
	uv run python sae_feature_emergence/model.py
