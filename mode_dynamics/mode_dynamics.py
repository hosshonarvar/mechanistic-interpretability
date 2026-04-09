import copy

import numpy as np
import torch
import torch.nn as nn


def build_sine_dataset(n_samples: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 2.0 * np.pi, size=(n_samples, 1)).astype(np.float32)
    y = np.sin(x).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)


class SineMLP(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


def flatten_parameters(model: nn.Module) -> np.ndarray:
    with torch.no_grad():
        return torch.cat([p.detach().view(-1) for p in model.parameters()]).cpu().numpy()


def train_with_tracking(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    n_epochs: int = 80000,
    lr: float = 0.01,
    checkpoint_every: int = 10,
    snapshot_epochs: list[int] | None = None,
):
    if snapshot_epochs is None:
        snapshot_epochs = [1000, 5000, 10000, 20000, 40000, 80000]

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    checkpoint_weights = [flatten_parameters(model)]
    snapshots = {0: copy.deepcopy(model.state_dict())}
    train_losses = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(x_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()

        train_losses.append(float(loss.item()))
        if epoch % checkpoint_every == 0:
            checkpoint_weights.append(flatten_parameters(model))

        if epoch in snapshot_epochs:
            snapshots[epoch] = copy.deepcopy(model.state_dict())

    checkpoint_epochs = np.arange(0, n_epochs + 1, checkpoint_every)
    return {
        "train_losses": np.array(train_losses),
        "checkpoint_epochs": checkpoint_epochs,
        "checkpoint_weights": np.stack(checkpoint_weights, axis=0),
        "snapshots": snapshots,
    }


def pca_from_weight_trajectory(weight_matrix: np.ndarray):
    delta_w = weight_matrix - weight_matrix[0:1]
    _, singular_vals, _ = np.linalg.svd(delta_w, full_matrices=False)
    explained_var_ratio = (singular_vals**2) / np.sum(singular_vals**2)
    cumulative_explained = np.cumsum(explained_var_ratio)
    return delta_w, explained_var_ratio, cumulative_explained


# Backward-compatible aliases for older notebook/script names.
make_dataset = build_sine_dataset
SmallSineNet = SineMLP
flatten_params = flatten_parameters
train_and_track = train_with_tracking
