"""
Track feature semantics over time: for each (step, feature_idx), compute dominant token
and consistency (how token-specific the feature is). Shows whether interpretability
emerges gradually or in a phase transition alongside drift.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from config import (
    ModelConfig,
    TrainConfig,
    SAEConfig,
    CHECKPOINTS_DIR,
    SAE_DIR,
    RESULTS_DIR,
    get_device,
)
from model import SmallTransformer
from data import get_batch
from sae import SAE

SEED = 7777
N_BATCHES_DEFAULT = 30
TOP_K_POSITIONS = 20  # use top-k max-activating positions to compute dominant token
FEATURE_INDICES_DEFAULT = [0, 2, 4, 5]  # include comma feature (2) and a few others


def load_model_at_step(step: int, device: torch.device) -> SmallTransformer:
    ckpt_path = CHECKPOINTS_DIR / f"step_{step}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint: {ckpt_path}")
    model = SmallTransformer(ModelConfig()).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


def load_sae_at_step(step: int, device: torch.device) -> SAE:
    path = SAE_DIR / f"sae_step_{step}.pt"
    if not path.exists():
        raise FileNotFoundError(f"No SAE: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    sae = SAE(cfg).to(device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    return sae


def dominant_token_and_consistency(
    F_cat: torch.Tensor,
    tokens_cat: torch.Tensor,
    feat_idx: int,
    B: int,
    T: int,
    top_k: int = TOP_K_POSITIONS,
) -> dict:
    """For one feature, get dominant token_id among top-k max-activating positions and consistency (fraction same token)."""
    vals = F_cat[:, feat_idx]
    k = min(top_k, vals.shape[0])
    _, top_flat_idx = vals.topk(k, dim=0)
    token_ids = []
    for flat_idx in top_flat_idx.tolist():
        flat_idx = int(flat_idx)
        batch_in_run = flat_idx // (B * T)
        rest = flat_idx % (B * T)
        bi = rest // T
        pos = rest % T
        row = batch_in_run * B + bi
        token_ids.append(int(tokens_cat[row, pos].item()))
    if not token_ids:
        return {"dominant_token_id": None, "dominant_char": ".", "consistency": 0.0, "mean_act": 0.0}
    cnt = Counter(token_ids)
    dominant_id, count = cnt.most_common(1)[0]
    consistency = count / len(token_ids)
    mean_act = float(vals[top_flat_idx].mean().item())
    char = chr(dominant_id) if 32 <= dominant_id < 127 else "."
    return {
        "dominant_token_id": dominant_id,
        "dominant_char": char,
        "consistency": round(consistency, 4),
        "mean_act": round(mean_act, 5),
    }


def run_feature_dynamics(
    steps: list[int],
    feature_indices: list[int],
    n_batches: int = N_BATCHES_DEFAULT,
) -> dict:
    """For each step and each feature index, compute dominant token and consistency; return JSON-serializable dict."""
    device = get_device()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    sae_cfg = SAEConfig()
    B, T = train_cfg.batch_size, train_cfg.seq_len

    out = {"steps": steps, "feature_indices": feature_indices, "n_batches": n_batches, "features": []}

    for feat_idx in feature_indices:
        by_step = {}
        for step in steps:
            model = load_model_at_step(step, device)
            sae = load_sae_at_step(step, device)
            all_f = []
            all_tokens = []
            with torch.no_grad():
                for b in range(n_batches):
                    batch = get_batch(B, T, model_cfg.vocab_size, device, seed=SEED + b)
                    resid = model.get_resid_at_layer(batch, sae_cfg.sae_layer)
                    flat = resid.flatten(0, 1)
                    f = sae.encode(flat)
                    all_f.append(f.cpu())
                    all_tokens.append(batch.cpu())
            F_cat = torch.cat(all_f, dim=0)
            tokens_cat = torch.cat(all_tokens, dim=0)
            if feat_idx >= F_cat.shape[1]:
                by_step[str(step)] = {"dominant_token_id": None, "dominant_char": ".", "consistency": 0.0, "mean_act": 0.0}
                continue
            rec = dominant_token_and_consistency(F_cat, tokens_cat, feat_idx, B, T)
            by_step[str(step)] = rec
        out["features"].append({"feature_idx": feat_idx, "by_step": by_step})

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track feature semantics (dominant token, consistency) across steps."
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="1000,2000,3000,4000",
        help="Comma-separated steps (default: 1000,2000,3000,4000).",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=",".join(map(str, FEATURE_INDICES_DEFAULT)),
        help=f"Comma-separated feature indices (default: {FEATURE_INDICES_DEFAULT}).",
    )
    parser.add_argument("--n-batches", type=int, default=N_BATCHES_DEFAULT)
    parser.add_argument("--save", action="store_true", help="Save to results/feature_dynamics.json")
    args = parser.parse_args()

    steps = [int(s) for s in args.steps.split(",")]
    feature_indices = [int(f) for f in args.features.split(",")]

    result = run_feature_dynamics(steps=steps, feature_indices=feature_indices, n_batches=args.n_batches)

    print("Feature dynamics (dominant token and consistency per step)")
    print("Consistency = fraction of top-20 max-activating positions with the same token_id.\n")
    for feat in result["features"]:
        print(f"  Feature {feat['feature_idx']}:")
        for step in result["steps"]:
            rec = feat["by_step"][str(step)]
            tid = rec["dominant_token_id"]
            char = rec["dominant_char"]
            cons = rec["consistency"]
            act = rec["mean_act"]
            print(f"    step {step}: token_id={tid} '{char}'  consistency={cons:.2f}  mean_act={act:.4f}")
        print()

    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / "feature_dynamics.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
