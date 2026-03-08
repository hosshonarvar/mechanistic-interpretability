"""Step 11: Ablate top-k SAE features at a chosen step; measure ΔCE (causal validation)."""

import argparse
import json
import torch
from torch.nn import functional as F

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

ABLATION_SEED = 9999
N_BATCHES_DEFAULT = 30
TOP_K_DEFAULT = 5


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


def ablate_topk_contribution(
    resid: torch.Tensor, sae: SAE, k: int, device: torch.device
) -> torch.Tensor:
    """
    For each position, zero out the top-k feature contributions from the residual.
    resid: (B, T, d_model). Returns resid_ablated: (B, T, d_model).
    """
    B, T, d = resid.shape
    flat = resid.flatten(0, 1)
    with torch.no_grad():
        f = sae.encode(flat)
        n_features = f.shape[1]
        k = min(k, n_features)
        topk_vals, topk_idx = f.abs().topk(k, dim=1)
        f_zero_topk = f.clone()
        f_zero_topk.scatter_(1, topk_idx, 0.0)
        contrib_topk = sae.decode(f) - sae.decode(f_zero_topk)
    resid_ablated = resid - contrib_topk.view(B, T, d)
    return resid_ablated


def ablate_randomk_contribution(
    resid: torch.Tensor, sae: SAE, k: int, device: torch.device, seed: int
) -> torch.Tensor:
    """Same as ablate_topk but zero random k features (control)."""
    B, T, d = resid.shape
    flat = resid.flatten(0, 1)
    n_features = sae.config.n_features
    k = min(k, n_features)
    with torch.no_grad():
        f = sae.encode(flat)
        g = torch.Generator(device=device).manual_seed(seed)
        random_idx = torch.randperm(n_features, device=device, generator=g)[:k]
        random_idx = random_idx.unsqueeze(0).expand(flat.shape[0], -1)
        f_zero_rand = f.clone()
        f_zero_rand.scatter_(1, random_idx, 0.0)
        contrib_rand = sae.decode(f) - sae.decode(f_zero_rand)
    resid_ablated = resid - contrib_rand.view(B, T, d)
    return resid_ablated


def run_ablation(
    step: int,
    top_k: int = TOP_K_DEFAULT,
    n_batches: int = N_BATCHES_DEFAULT,
    run_random_control: bool = True,
) -> dict:
    device = get_device()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    sae_cfg = SAEConfig()

    model = load_model_at_step(step, device)
    sae = load_sae_at_step(step, device)

    ce_orig_list: list[float] = []
    ce_ablated_list: list[float] = []
    ce_random_list: list[float] = []

    with torch.no_grad():
        for b in range(n_batches):
            batch = get_batch(
                train_cfg.batch_size,
                train_cfg.seq_len,
                model_cfg.vocab_size,
                device,
                seed=ABLATION_SEED + b,
            )
            resid = model.get_resid_at_layer(batch, sae_cfg.sae_layer)
            resid_ablated = ablate_topk_contribution(resid, sae, top_k, device)

            logits_orig = model(batch)
            logits_ablated = model.forward_with_patched_resid(
                batch, sae_cfg.sae_layer, resid_ablated
            )

            logits_flat = logits_orig[:, :-1].reshape(-1, model_cfg.vocab_size)
            targets = batch[:, 1:].reshape(-1)
            ce_orig = F.cross_entropy(logits_flat, targets).item()
            ce_orig_list.append(ce_orig)

            logits_abl_flat = logits_ablated[:, :-1].reshape(-1, model_cfg.vocab_size)
            ce_ablated = F.cross_entropy(logits_abl_flat, targets).item()
            ce_ablated_list.append(ce_ablated)

            if run_random_control:
                resid_random = ablate_randomk_contribution(
                    resid, sae, top_k, device, seed=ABLATION_SEED + b + 10000
                )
                logits_random = model.forward_with_patched_resid(
                    batch, sae_cfg.sae_layer, resid_random
                )
                ce_random = F.cross_entropy(
                    logits_random[:, :-1].reshape(-1, model_cfg.vocab_size), targets
                ).item()
                ce_random_list.append(ce_random)

    ce_orig_mean = sum(ce_orig_list) / len(ce_orig_list)
    ce_ablated_mean = sum(ce_ablated_list) / len(ce_ablated_list)
    delta_ce = ce_ablated_mean - ce_orig_mean

    result = {
        "step": step,
        "top_k": top_k,
        "n_batches": n_batches,
        "ce_original": round(ce_orig_mean, 6),
        "ce_ablated": round(ce_ablated_mean, 6),
        "delta_ce": round(delta_ce, 6),
    }
    if run_random_control and ce_random_list:
        ce_random_mean = sum(ce_random_list) / len(ce_random_list)
        result["ce_random_ablated"] = round(ce_random_mean, 6)
        result["delta_ce_random"] = round(ce_random_mean - ce_orig_mean, 6)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablate top-k SAE features at a step; measure ΔCE (step 11)."
    )
    parser.add_argument(
        "step",
        nargs="?",
        type=int,
        default=None,
        help="Checkpoint/SAE step (default: last in sae_checkpoint_steps).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K_DEFAULT,
        help=f"Number of top features to ablate (default: {TOP_K_DEFAULT}).",
    )
    parser.add_argument(
        "--n-batches",
        type=int,
        default=N_BATCHES_DEFAULT,
        help=f"Batches for averaging CE (default: {N_BATCHES_DEFAULT}).",
    )
    parser.add_argument(
        "--no-random",
        action="store_true",
        help="Skip random-k control.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save result to results/ablation_results.json.",
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

    result = run_ablation(
        step=step,
        top_k=args.top_k,
        n_batches=args.n_batches,
        run_random_control=not args.no_random,
    )

    print(f"Step {step}, top_k={result['top_k']}, n_batches={result['n_batches']}")
    print(f"  CE (original): {result['ce_original']}")
    print(f"  CE (ablated):  {result['ce_ablated']}")
    print(f"  ΔCE (top-k):   {result['delta_ce']}")
    if "delta_ce_random" in result:
        print(f"  ΔCE (random-k): {result['delta_ce_random']}")

    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / "ablation_results.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
