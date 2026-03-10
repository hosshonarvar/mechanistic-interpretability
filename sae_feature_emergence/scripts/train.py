"""Training loop: train transformer, save checkpoints and loss history at configured steps."""

import json
import torch
from torch.nn import functional as F

from config import ModelConfig, TrainConfig, CHECKPOINTS_DIR, RESULTS_DIR, get_device
from model import SmallTransformer
from data import get_batch


def train() -> None:
    train_cfg = TrainConfig()
    model_cfg = ModelConfig()
    device = get_device()
    print("Device:", device)
    torch.manual_seed(train_cfg.seed)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = SmallTransformer(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)

    checkpoint_steps = set(train_cfg.checkpoint_steps)
    loss_history: list[dict] = []

    for step in range(1, train_cfg.total_steps + 1):
        batch = get_batch(
            train_cfg.batch_size,
            train_cfg.seq_len,
            model_cfg.vocab_size,
            device,
            seed=train_cfg.seed + step,
        )
        logits = model(batch)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, model_cfg.vocab_size),
            batch[:, 1:].reshape(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append({"step": step, "loss": round(loss.item(), 6)})

        if step in checkpoint_steps:
            path = CHECKPOINTS_DIR / f"step_{step}.pt"
            torch.save({"step": step, "model_state_dict": model.state_dict()}, path)
            print(f"Checkpoint step {step} -> {path.name}")

    loss_path = RESULTS_DIR / "loss_history.json"
    with open(loss_path, "w") as f:
        json.dump(loss_history, f, indent=0)
    print(f"Loss history -> {loss_path}")
    print(f"Final loss: {loss_history[-1]['loss']}")


if __name__ == "__main__":
    train()
