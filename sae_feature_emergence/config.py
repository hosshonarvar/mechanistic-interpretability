"""Config: model, training, SAE, and paths."""

from dataclasses import dataclass
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
ACTIVATIONS_DIR = PROJECT_ROOT / "activations"
SAE_DIR = PROJECT_ROOT / "sae_models"
RESULTS_DIR = PROJECT_ROOT / "results"


def get_device() -> torch.device:
    """CUDA if available, else MPS (Apple Silicon), else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class ModelConfig:
    n_layers: int = 2
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    vocab_size: int = 256
    max_len: int = 64
    dropout: float = 0.1


@dataclass
class TrainConfig:
    batch_size: int = 32
    seq_len: int = 32
    lr: float = 3e-4
    total_steps: int = 4000
    checkpoint_steps: tuple = (100, 200, 400, 600, 800, 1000, 1300, 1600, 2000, 2500, 3000, 3500, 4000)
    seed: int = 42


@dataclass
class SAEConfig:
    d_model: int = 128
    n_features: int = 512
    l1_coeff: float = 1e-3
    lr: float = 1e-3
    batch_size: int = 512
    n_epochs: int = 1000
    sae_layer: int = 0
    sae_checkpoint_steps: tuple = (100, 200, 400, 600, 800, 1000, 1300, 1600, 2000, 2500, 3000, 3500, 4000)


if __name__ == "__main__":
    train = TrainConfig()
    print("Checkpoint steps:", train.checkpoint_steps)
    print("Total steps:", train.total_steps)
    model = ModelConfig()
    print("d_model:", model.d_model, "n_layers:", model.n_layers)
    sae = SAEConfig()
    print("SAE layer:", sae.sae_layer, "n_features:", sae.n_features)
    print("Paths:", CHECKPOINTS_DIR, ACTIVATIONS_DIR, SAE_DIR, RESULTS_DIR)
