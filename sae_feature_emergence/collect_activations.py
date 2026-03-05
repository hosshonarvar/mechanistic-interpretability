"""Load checkpoint(s), run model, save residual activations at SAE layer to activations/step_*.pt."""

import argparse
import torch

from config import (
    ModelConfig,
    TrainConfig,
    SAEConfig,
    CHECKPOINTS_DIR,
    ACTIVATIONS_DIR,
)
from model import SmallTransformer
from data import get_batch

# Same data for every checkpoint (fixed seed)
ACTIVATION_SEED = 12345
N_BATCHES = 200


def collect_for_step(step: int, device: torch.device) -> torch.Tensor:
    """Load checkpoint at step, run N_BATCHES batches, return residual (N, d_model)."""
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    sae_cfg = SAEConfig()

    ckpt_path = CHECKPOINTS_DIR / f"step_{step}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint: {ckpt_path}")

    model = SmallTransformer(model_cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    activations_list: list[torch.Tensor] = []
    with torch.no_grad():
        for b in range(N_BATCHES):
            batch = get_batch(
                train_cfg.batch_size,
                train_cfg.seq_len,
                model_cfg.vocab_size,
                device,
                seed=ACTIVATION_SEED + b,
            )
            resid = model.get_resid_at_layer(batch, sae_cfg.sae_layer)
            activations_list.append(resid.flatten(0, 1))

    return torch.cat(activations_list, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Save residual activations per checkpoint.")
    parser.add_argument("step", nargs="?", type=int, default=None, help="Single step (default: all).")
    args = parser.parse_args()

    train_cfg = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)

    if args.step is not None:
        steps = [args.step]
    else:
        steps = sorted(
            int(p.stem.split("_")[1])
            for p in CHECKPOINTS_DIR.glob("step_*.pt")
        )

    for step in steps:
        activations = collect_for_step(step, device)
        out_path = ACTIVATIONS_DIR / f"step_{step}.pt"
        torch.save(activations, out_path)
        print(f"step {step}: shape {activations.shape}, mean {activations.mean().item():.4f}, std {activations.std().item():.4f} -> {out_path.name}")


if __name__ == "__main__":
    main()
