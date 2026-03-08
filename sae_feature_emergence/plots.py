"""Plotting functions for stability vs step and vs loss. Call from findings notebook or CLI (saves PNGs)."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from config import RESULTS_DIR


def plot_stability_vs_step(stability: list[dict], save_path: Path | None = None) -> None:
    """Plot similarity and drift vs step_b. If save_path set, save and close; else show()."""
    step_b = [r["step_b"] for r in stability]
    similarity = [r["similarity"] for r in stability]
    drift = [r["drift"] for r in stability]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    ax1.plot(step_b, similarity, "o-")
    ax1.set_ylabel("Similarity")
    ax1.set_title("Feature stability vs checkpoint step")
    ax2.plot(step_b, drift, "o-")
    ax2.set_xlabel("Step (end of pair)")
    ax2.set_ylabel("Drift")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_stability_vs_loss(
    stability: list[dict],
    loss_by_step: dict[int, float],
    save_path: Path | None = None,
) -> None:
    """Plot similarity and drift vs loss at step_b. If save_path set, save and close; else show()."""
    step_b = [r["step_b"] for r in stability]
    similarity = [r["similarity"] for r in stability]
    drift = [r["drift"] for r in stability]
    losses = [loss_by_step.get(s) for s in step_b]
    if any(l is None for l in losses):
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    ax1.plot(losses, similarity, "o-")
    ax1.set_ylabel("Similarity")
    ax1.set_title("Feature stability vs loss")
    ax2.plot(losses, drift, "o-")
    ax2.set_xlabel("Loss (at step_b)")
    ax2.set_ylabel("Drift")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def main() -> None:
    """Load data, call plot functions with save_path (for make sae-plots)."""
    results_dir = Path(RESULTS_DIR)
    stability_path = results_dir / "stability_results.json"
    loss_path = results_dir / "loss_history.json"
    if not stability_path.exists():
        raise SystemExit(f"Missing {stability_path}. Run make sae-stability-all.")
    with open(stability_path) as f:
        stability = json.load(f)
    loss_by_step = {}
    if loss_path.exists():
        with open(loss_path) as f:
            loss_list = json.load(f)
        loss_by_step = {x["step"]: x["loss"] for x in loss_list}
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_stability_vs_step(stability, save_path=results_dir / "stability_vs_step.png")
    print(f"Saved {results_dir / 'stability_vs_step.png'}")
    step_b = [r["step_b"] for r in stability]
    if loss_by_step and all(s in loss_by_step for s in step_b):
        plot_stability_vs_loss(
            stability, loss_by_step, save_path=results_dir / "stability_vs_loss.png"
        )
        print(f"Saved {results_dir / 'stability_vs_loss.png'}")


if __name__ == "__main__":
    main()
