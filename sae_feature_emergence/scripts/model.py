"""Small decoder-only transformer. Exposes residual at one layer for SAE."""

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import ModelConfig, SAEConfig, TrainConfig


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(
            config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class SmallTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_len, config.d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        x = self.embed(x) + self.pos_embed[:, :T, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    def get_resid_at_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Residual stream at layer_idx (after that block). Shape (B, T, d_model)."""
        B, T = x.shape
        x = self.embed(x) + self.pos_embed[:, :T, :]
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == layer_idx:
                return x
        return x

    def forward_with_patched_resid(
        self, x: torch.Tensor, layer_idx: int, patched_resid: torch.Tensor
    ) -> torch.Tensor:
        """Forward with residual at layer_idx replaced by patched_resid (B, T, d_model)."""
        B, T = x.shape
        x = self.embed(x) + self.pos_embed[:, :T, :]
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == layer_idx:
                x = patched_resid
                break
        for block in self.blocks[layer_idx + 1 :]:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


if __name__ == "__main__":
    from config import get_device
    from data import get_batch

    device = get_device()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    sae_cfg = SAEConfig()

    model = SmallTransformer(model_cfg).to(device)
    batch = get_batch(
        train_cfg.batch_size, train_cfg.seq_len, model_cfg.vocab_size, device, seed=42
    )

    logits = model(batch)
    resid = model.get_resid_at_layer(batch, sae_cfg.sae_layer)

    print("Batch shape:", batch.shape)
    print("Logits shape:", logits.shape)
    print("Residual shape (SAE layer):", resid.shape)
