"""Synthetic token batches for training and activation collection."""

import torch

from config import TrainConfig, ModelConfig


def get_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    seed: int | None = None,
) -> torch.Tensor:
    """Random token ids (B, T)."""
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randint(0, vocab_size, (batch_size, seq_len), device=device)


if __name__ == "__main__":
    from config import TrainConfig, ModelConfig, get_device
    train = TrainConfig()
    model_cfg = ModelConfig()
    device = get_device()
    batch = get_batch(train.batch_size, train.seq_len, model_cfg.vocab_size, device, seed=42)
    print("Batch shape:", batch.shape)
