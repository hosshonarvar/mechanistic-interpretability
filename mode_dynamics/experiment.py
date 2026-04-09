import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from mode_dynamics import (
    SineMLP,
    build_sine_dataset,
    pca_from_weight_trajectory,
    train_with_tracking,
)


def main(save_plots: bool = False):
    np.random.seed(0)
    torch.manual_seed(0)

    n_samples = 5000
    x_train, y_train = build_sine_dataset(n_samples=n_samples, seed=0)
    hidden_dim = 64
    model = SineMLP(hidden_dim=hidden_dim)
    n_epochs = 80000

    snapshot_epochs = [1000, 5000, 10000, 20000, 40000, 80000]
    tracked = train_with_tracking(
        model=model,
        x_train=x_train,
        y_train=y_train,
        n_epochs=n_epochs,
        lr=0.01,
        checkpoint_every=10,
        snapshot_epochs=snapshot_epochs,
    )

    train_losses = tracked["train_losses"]
    checkpoint_epochs = tracked["checkpoint_epochs"]
    snapshots = tracked["snapshots"]
    weight_matrix = tracked["checkpoint_weights"]
    _, explained_var_ratio, cumulative_explained = pca_from_weight_trajectory(weight_matrix)

    epochs_axis = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs_axis, train_losses, label="Train loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Training loss on y = sin(x)")
    plt.legend()
    plt.tight_layout()
    if save_plots:
        plt.savefig("loss_vs_epochs.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    for i in range(weight_matrix.shape[1]):
        plt.plot(checkpoint_epochs, weight_matrix[:, i], color="tab:blue", alpha=0.12, linewidth=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Parameter value")
    plt.title("Weights vs epoch (all flattened parameters)")
    plt.tight_layout()
    if save_plots:
        plt.savefig("weights_vs_epochs.png", dpi=150)
    plt.close()

    x_plot = torch.linspace(0.0, 2.0 * np.pi, 400).view(-1, 1)
    y_true = torch.sin(x_plot).numpy()
    plt.figure(figsize=(7, 4))
    plt.plot(x_plot.numpy(), y_true, label="True sin(x)", linewidth=2)
    for ep in snapshot_epochs:
        temp_model = SineMLP(hidden_dim=hidden_dim)
        temp_model.load_state_dict(snapshots[ep])
        temp_model.eval()
        with torch.no_grad():
            y_pred = temp_model(x_plot).numpy()
        plt.plot(x_plot.numpy(), y_pred, label=f"Epoch {ep}", alpha=0.85)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Function fit over training")
    plt.legend()
    plt.tight_layout()
    if save_plots:
        plt.savefig("function_predictions_over_time.png", dpi=150)
    plt.close()

    components = np.arange(1, len(cumulative_explained) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(components, cumulative_explained, marker="o", markersize=3)
    plt.axhline(0.9, color="gray", linestyle="--", linewidth=1, label="90% variance")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA of weight trajectory")
    plt.ylim(0.0, 1.01)
    plt.legend()
    plt.tight_layout()
    if save_plots:
        plt.savefig("pca_cumulative_explained_variance.png", dpi=150)
    plt.close()

    print(f"First PC explained variance ratio: {float(explained_var_ratio[0]):.4f}")
    if save_plots:
        print("Plots were saved in the current folder.")
    else:
        print("No PNGs saved (pass --save-plots to save files).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-plots", action="store_true", help="Save PNG plot files.")
    args = parser.parse_args()
    main(save_plots=args.save_plots)
