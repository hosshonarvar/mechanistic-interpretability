# Run from repo root. Uses uv for Python env.

.PHONY: help sae-config

help:
	@echo "SAE feature emergence (run from repo root):"
	@echo "  make sae-config   Load config and print checkpoint steps, paths, etc."

sae-config:
	uv run python sae_feature_emergence/config.py
