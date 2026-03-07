"""Sparse autoencoder on residual stream: encoder (ReLU) + L1, decoder, tied weights."""

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import SAEConfig


class SAE(nn.Module):
    """Tied-weight SAE: x -> ReLU(x W^T) -> f, x_hat = f W. L1 on f."""

    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        # One matrix (n_features, d_model): encoder is x @ W.T, decoder is f @ W
        self.W = nn.Parameter(torch.empty(config.n_features, config.d_model))
        nn.init.xavier_uniform_(self.W)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, d_model) -> (B, n_features), ReLU."""
        return F.relu(F.linear(x, self.W))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """(B, n_features) -> (B, d_model)."""
        return F.linear(f, self.W.t())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (x_hat, f)."""
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f

    def loss(
        self, x: torch.Tensor, x_hat: torch.Tensor, f: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """MSE, L1, and total loss."""
        mse = F.mse_loss(x, x_hat)
        l1 = f.abs().mean()
        total = mse + self.config.l1_coeff * l1
        return total, mse, l1


if __name__ == "__main__":
    cfg = SAEConfig()
    sae = SAE(cfg)
    x = torch.randn(4, cfg.d_model)
    x_hat, f = sae(x)
    total, mse, l1 = sae.loss(x, x_hat, f)
    print("x", x.shape, "f", f.shape, "x_hat", x_hat.shape)
    print("mse", mse.item(), "l1", l1.item(), "total", total.item())
