"""Plotting functions for stability vs step and vs loss. Call from findings notebook or CLI (saves PNGs)."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from config import RESULTS_DIR

# Style
DRIFT_COLOR = "#1a5fb4"
TRANSITION_ALPHA = 0.2
TRANSITION_COLOR = "#1a5fb4"
GRID_ALPHA = 0.35


def _phase_transition_bounds(stability: list[dict]) -> tuple[int | None, int | None]:
    """Return (step_lo, step_hi) where drift drops the most (phase transition), or (None, None)."""
    if len(stability) < 2:
        return None, None
    step_b = [r["step_b"] for r in stability]
    drift = [r["drift"] for r in stability]
    drops = [drift[i] - drift[i + 1] for i in range(len(drift) - 1)]
    i_max = drops.index(max(drops))
    return step_b[i_max], step_b[i_max + 1]


def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=GRID_ALPHA, linestyle="--")
    ax.set_axisbelow(True)


def plot_stability_vs_step(stability: list[dict], save_path: Path | None = None) -> None:
    """Plot drift vs step_b with phase-transition shading. If save_path set, save and close; else show()."""
    step_b = [r["step_b"] for r in stability]
    drift = [r["drift"] for r in stability]
    fig, ax = plt.subplots(figsize=(7, 4))
    _style_axis(ax)
    step_lo, step_hi = _phase_transition_bounds(stability)
    if step_lo is not None and step_hi is not None:
        ax.axvspan(step_lo, step_hi, alpha=TRANSITION_ALPHA, color=TRANSITION_COLOR, zorder=0)
        ax.annotate(
            "regime change",
            xy=((step_lo + step_hi) / 2, 1.0),
            ha="center",
            va="top",
            fontsize=9,
            color=TRANSITION_COLOR,
            style="italic",
        )
    ax.plot(step_b, drift, "o-", color=DRIFT_COLOR, linewidth=2, markersize=7, zorder=2)
    ax.set_xlabel("Training step (end of interval; drift = previous checkpoint → this)", fontsize=10)
    ax.set_ylabel("Drift (1 − similarity)", fontsize=11)
    ax.set_title("Feature stability vs checkpoint step", fontsize=12, fontweight="medium")
    ax.set_ylim(0, 1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_stability_vs_loss(
    stability: list[dict],
    loss_by_step: dict[int, float],
    save_path: Path | None = None,
) -> None:
    """Plot drift vs loss at step_b with phase-transition shading. If save_path set, save and close; else show()."""
    step_b = [r["step_b"] for r in stability]
    drift = [r["drift"] for r in stability]
    losses = [loss_by_step.get(s) for s in step_b]
    if any(l is None for l in losses):
        return
    step_lo, step_hi = _phase_transition_bounds(stability)
    loss_lo = loss_by_step.get(step_lo) if step_lo is not None else None
    loss_hi = loss_by_step.get(step_hi) if step_hi is not None else None
    fig, ax = plt.subplots(figsize=(7, 4))
    _style_axis(ax)
    if loss_lo is not None and loss_hi is not None and loss_lo != loss_hi:
        lo, hi = min(loss_lo, loss_hi), max(loss_lo, loss_hi)
        ax.axvspan(lo, hi, alpha=TRANSITION_ALPHA, color=TRANSITION_COLOR, zorder=0)
        ax.annotate(
            "regime change",
            xy=((lo + hi) / 2, 1.0),
            ha="center",
            va="top",
            fontsize=9,
            color=TRANSITION_COLOR,
            style="italic",
        )
    ax.plot(losses, drift, "o-", color=DRIFT_COLOR, linewidth=2, markersize=7, zorder=2)
    ax.set_xlabel("Loss (at end of interval)", fontsize=11)
    ax.set_ylabel("Drift (1 − similarity)", fontsize=11)
    ax.set_title("Feature stability vs loss", fontsize=12, fontweight="medium")
    ax.set_ylim(0, 1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
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
