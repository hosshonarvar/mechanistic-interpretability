"""Load two SAEs, match features (Hungarian), compute similarity and drift for one pair."""

import argparse
import torch
from scipy.optimize import linear_sum_assignment

from config import SAE_DIR


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Similarity and drift for one SAE pair.")
    parser.add_argument("step_a", type=int, help="First checkpoint step.")
    parser.add_argument("step_b", type=int, help="Second checkpoint step.")
    args = parser.parse_args()

    mean_sim, mean_drift = similarity_and_drift(args.step_a, args.step_b)
    print(f"Pair ({args.step_a}, {args.step_b}): mean similarity = {mean_sim:.4f}, mean drift = {mean_drift:.4f}")


if __name__ == "__main__":
    main()
