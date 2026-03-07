"""Load two SAEs, match features (Hungarian), compute similarity and drift for one pair or all consecutive pairs."""

import argparse
import json
import torch
from scipy.optimize import linear_sum_assignment

from config import SAE_DIR, RESULTS_DIR


def load_directions(step: int) -> torch.Tensor:
    """Load SAE for step, return L2-normalized feature directions (n_features, d_model)."""
    path = SAE_DIR / f"sae_step_{step}.pt"
    if not path.exists():
        raise FileNotFoundError(f"No SAE: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    W = ckpt["state_dict"]["W"]
    # Rows are decoder directions; normalize to unit length for cosine similarity
    norms = W.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return (W / norms).numpy()


def similarity_and_drift(step_a: int, step_b: int) -> tuple[float, float]:
    """
    Hungarian match between two SAEs' feature directions; return mean cosine similarity
    and mean drift (1 - similarity) over matched pairs.
    """
    D_a = load_directions(step_a)
    D_b = load_directions(step_b)
    # Cosine similarity matrix (both already unit norm)
    sim_matrix = D_a @ D_b.T
    # Hungarian: maximize total similarity = minimize -similarity
    cost = -sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost)
    matched_sims = sim_matrix[row_ind, col_ind]
    mean_similarity = float(matched_sims.mean())
    mean_drift = 1.0 - mean_similarity
    return mean_similarity, mean_drift


def run_all_pairs() -> None:
    """Consecutive SAE pairs; save to stability_results.json; print table."""
    steps = sorted(
        int(p.stem.split("_")[2])  # sae_step_100 -> 100
        for p in SAE_DIR.glob("sae_step_*.pt")
    )
    if len(steps) < 2:
        raise SystemExit("Need at least two SAEs. Run make sae-train-sae-all.")
    pairs = [(steps[i], steps[i + 1]) for i in range(len(steps) - 1)]
    results = []
    for step_a, step_b in pairs:
        mean_sim, mean_drift = similarity_and_drift(step_a, step_b)
        results.append({
            "step_a": step_a,
            "step_b": step_b,
            "similarity": round(mean_sim, 4),
            "drift": round(mean_drift, 4),
        })
        print(f"  {step_a} -> {step_b}: similarity = {mean_sim:.4f}, drift = {mean_drift:.4f}")
    out_path = RESULTS_DIR / "stability_results.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Similarity and drift for one SAE pair or all consecutive pairs.")
    parser.add_argument("step_a", type=int, nargs="?", default=None, help="First checkpoint step.")
    parser.add_argument("step_b", type=int, nargs="?", default=None, help="Second checkpoint step.")
    parser.add_argument("--all", action="store_true", help="Loop consecutive pairs; save to stability_results.json.")
    args = parser.parse_args()

    if args.all:
        run_all_pairs()
        return
    if args.step_a is None or args.step_b is None:
        parser.error("Give step_a and step_b, or use --all.")
    mean_sim, mean_drift = similarity_and_drift(args.step_a, args.step_b)
    print(f"Pair ({args.step_a}, {args.step_b}): mean similarity = {mean_sim:.4f}, mean drift = {mean_drift:.4f}")


if __name__ == "__main__":
    main()
