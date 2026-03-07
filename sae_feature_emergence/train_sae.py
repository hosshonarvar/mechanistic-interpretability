"""Train SAE on one checkpoint's activations; print/plot reconstruction error; save SAE."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from config import SAEConfig, ACTIVATIONS_DIR, SAE_DIR, RESULTS_DIR
from sae import SAE 


def train_sae(step: int, device: torch.device) -> None:
    cfg = SAEConfig()
    act_path = ACTIVATIONS_DIR / f"step_{step}.pt"
    if not act_path.exists():
        raise FileNotFoundError(f"No activations: {act_path}")

    activations = torch.load(act_path, map_location=device, weights_only=True)
    dataset = TensorDataset(activations)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )

    sae = SAE(cfg).to(device)
    optimizer = torch.optim.AdamW(sae.parameters(), lr=cfg.lr)

    history: list[dict] = []
    for epoch in range(cfg.n_epochs):
        sae.train()
        epoch_mse, epoch_l1, n_batches = 0.0, 0.0, 0
        for (x,) in loader:
            x = x.to(device)
            x_hat, f = sae(x)
            total, mse, l1 = sae.loss(x, x_hat, f)
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            epoch_mse += mse.item()
            epoch_l1 += l1.item()
            n_batches += 1
        epoch_mse /= n_batches
        epoch_l1 /= n_batches
        history.append({"epoch": epoch + 1, "mse": epoch_mse, "l1": epoch_l1})
        if (epoch + 1) % 200 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}: mse={epoch_mse:.6f}, l1={epoch_l1:.6f}")

    # Plot reconstruction error
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
        ax1.plot([h["epoch"] for h in history], [h["mse"] for h in history])
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("MSE")
        ax1.set_title("Reconstruction error")
        ax2.plot([h["epoch"] for h in history], [h["l1"] for h in history])
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("L1 (mean |f|)")
        ax2.set_title("Sparsity")
        plt.tight_layout()
        plot_path = Path(RESULTS_DIR) / f"sae_step_{step}_train.png"
        Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot -> {plot_path}")
    except Exception as e:
        print(f"Plot skip: {e}")

    SAE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAE_DIR / f"sae_step_{step}.pt"
    torch.save(
        {"step": step, "config": cfg, "state_dict": sae.state_dict()},
        out_path,
    )
    print(f"SAE -> {out_path}")
    print(f"Final mse={history[-1]['mse']:.6f}, l1={history[-1]['l1']:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAE on one or all checkpoints' activations.")
    parser.add_argument("step", type=int, nargs="?", default=None, help="Checkpoint step (default 1000).")
    parser.add_argument("--all", action="store_true", help="Train one SAE per checkpoint.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.all:
        steps = sorted(
            int(p.stem.split("_")[1])
            for p in ACTIVATIONS_DIR.glob("step_*.pt")
        )
        if not steps:
            raise SystemExit("No activations found. Run make sae-activations first.")
        for step in steps:
            train_sae(step, device)
        print("Saved SAEs:", [f"sae_step_{s}.pt" for s in steps])
    else:
        train_sae(args.step if args.step is not None else 1000, device)
