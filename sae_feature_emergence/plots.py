"""Read stability_results.json and loss_history.json; write similarity/drift vs step and vs loss to results/*.png."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from config import RESULTS_DIR


def main() -> None:
    results_dir = Path(RESULTS_DIR)
    stability_path = results_dir / "stability_results.json"
    loss_path = results_dir / "loss_history.json"
    if not stability_path.exists():
        raise SystemExit(f"Missing {stability_path}. Run make sae-stability-all.")
    with open(stability_path) as f:
        stability = json.load(f)
    step_b = [r["step_b"] for r in stability]
    similarity = [r["similarity"] for r in stability]
    drift = [r["drift"] for r in stability]
    losses = None
    if loss_path.exists():
        with open(loss_path) as f:
            loss_list = json.load(f)
        loss_by_step = {x["step"]: x["loss"] for x in loss_list}
        losses = [loss_by_step.get(s) for s in step_b]
        if any(l is None for l in losses):
            losses = None

    results_dir.mkdir(parents=True, exist_ok=True)

    # Similarity and drift vs step
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    ax1.plot(step_b, similarity, "o-")
    ax1.set_ylabel("Similarity")
    ax1.set_title("Feature stability vs checkpoint step")
    ax2.plot(step_b, drift, "o-")
    ax2.set_xlabel("Step (end of pair)")
    ax2.set_ylabel("Drift")
    plt.tight_layout()
    out_step = results_dir / "stability_vs_step.png"
    plt.savefig(out_step)
    plt.close()
    print(f"Saved {out_step}")

    # Similarity and drift vs loss (if we have loss at step_b)
    if losses is not None:
        fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
        ax1.plot(losses, similarity, "o-")
        ax1.set_ylabel("Similarity")
        ax1.set_title("Feature stability vs loss")
        ax2.plot(losses, drift, "o-")
        ax2.set_xlabel("Loss (at step_b)")
        ax2.set_ylabel("Drift")
        plt.tight_layout()
        out_loss = results_dir / "stability_vs_loss.png"
        plt.savefig(out_loss)
        plt.close()
        print(f"Saved {out_loss}")


if __name__ == "__main__":
    main()
