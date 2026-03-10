"""Collect max-activating examples per SAE feature: which token/context makes each feature fire most."""

import argparse
import json
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
N_BATCHES_DEFAULT = 50
N_FEATURES_DEFAULT = 8
N_EXAMPLES_PER_FEATURE_DEFAULT = 5
CONTEXT_LEN = 8


def token_ids_to_readable(ids: list[int]) -> str:
    """Turn token ids (0..255) into readable string; non-printable as '.'."""
    return "".join(chr(i) if 32 <= i < 127 else "." for i in ids)


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


def run_max_activating(
    step: int,
    n_features: int = N_FEATURES_DEFAULT,
    n_examples_per_feature: int = N_EXAMPLES_PER_FEATURE_DEFAULT,
    n_batches: int = N_BATCHES_DEFAULT,
) -> dict:
    """For each of the first n_features, find top n_examples_per_feature (batch, pos) and record token + context."""
    device = get_device()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    sae_cfg = SAEConfig()

    model = load_model_at_step(step, device)
    sae = load_sae_at_step(step, device)

    B, T = train_cfg.batch_size, train_cfg.seq_len
    all_f: list[torch.Tensor] = []
    all_tokens: list[torch.Tensor] = []

    with torch.no_grad():
        for b in range(n_batches):
            batch = get_batch(
                B,
                T,
                model_cfg.vocab_size,
                device,
                seed=SEED + b,
            )
            resid = model.get_resid_at_layer(batch, sae_cfg.sae_layer)
            _, _, d = resid.shape
            flat = resid.flatten(0, 1)
            f = sae.encode(flat)
            all_f.append(f.cpu())
            all_tokens.append(batch.cpu())

    F_cat = torch.cat(all_f, dim=0)
    tokens_cat = torch.cat(all_tokens, dim=0)

    # F_cat: (N, n_features), N = n_batches * B * T. tokens_cat: (n_batches * B, T).
    n_features = min(n_features, F_cat.shape[1])
    out: dict[str, list] = {"step": step, "n_batches": n_batches, "features": []}

    for feat_idx in range(n_features):
        vals = F_cat[:, feat_idx]
        top_vals, top_flat_idx = vals.topk(n_examples_per_feature, dim=0)
        examples = []
        for rank, (v, flat_idx) in enumerate(zip(top_vals.tolist(), top_flat_idx.tolist())):
            flat_idx = int(flat_idx)
            # flat_idx -> which (batch_in_run, bi, pos): flat_idx = batch_in_run*(B*T) + bi*T + pos
            batch_in_run = flat_idx // (B * T)
            rest = flat_idx % (B * T)
            bi = rest // T
            pos = rest % T
            row = batch_in_run * B + bi
            token_id = int(tokens_cat[row, pos].item())
            start = max(0, pos - CONTEXT_LEN)
            context_ids = tokens_cat[row, start:pos].tolist()
            context_str = token_ids_to_readable(context_ids)
            examples.append({
                "rank": rank + 1,
                "activation": round(v, 5),
                "token_id": token_id,
                "token_char": chr(token_id) if 32 <= token_id < 127 else ".",
                "context_ids": context_ids,
                "context": context_str,
            })
        out["features"].append({"feature_idx": feat_idx, "examples": examples})

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect max-activating examples per SAE feature (step, token, context)."
    )
    parser.add_argument(
        "step",
        nargs="?",
        type=int,
        default=None,
        help="Checkpoint/SAE step (default: last available).",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=N_FEATURES_DEFAULT,
        help=f"Number of features to show (default: {N_FEATURES_DEFAULT}).",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=N_EXAMPLES_PER_FEATURE_DEFAULT,
        help=f"Max-activating examples per feature (default: {N_EXAMPLES_PER_FEATURE_DEFAULT}).",
    )
    parser.add_argument(
        "--n-batches",
        type=int,
        default=N_BATCHES_DEFAULT,
        help=f"Batches to scan (default: {N_BATCHES_DEFAULT}).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save to results/max_activating_results.json",
    )
    args = parser.parse_args()

    step = args.step
    if step is None:
        steps = sorted(
            int(p.stem.split("_")[2])
            for p in SAE_DIR.glob("sae_step_*.pt")
        )
        if not steps:
            raise SystemExit("No SAEs found. Run make sae-train-sae-all.")
        step = steps[-1]
        print(f"Using step {step} (last available SAE).")

    result = run_max_activating(
        step=step,
        n_features=args.n_features,
        n_examples_per_feature=args.n_examples,
        n_batches=args.n_batches,
    )

    # Pretty-print
    print(f"Step {step}, n_features={args.n_features}, n_examples_per_feature={args.n_examples}")
    for feat in result["features"]:
        print(f"\n  Feature {feat['feature_idx']}:")
        for ex in feat["examples"]:
            print(f"    activation={ex['activation']:.4f}  token_id={ex['token_id']} '{ex['token_char']}'  context='{ex['context']}'")

    if args.save:
        # JSON-serializable (no tensor)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / "max_activating_results.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
