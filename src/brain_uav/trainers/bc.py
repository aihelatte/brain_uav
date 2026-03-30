from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def train_behavior_cloning(
    actor: nn.Module,
    dataset_path: str | Path,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str = "cpu",
) -> list[float]:
    payload = np.load(dataset_path)
    obs = torch.tensor(payload["observations"], dtype=torch.float32)
    actions = torch.tensor(payload["actions"], dtype=torch.float32)
    loader = DataLoader(TensorDataset(obs, actions), batch_size=batch_size, shuffle=True)
    actor.to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history: list[float] = []
    for _ in range(epochs):
        running = 0.0
        count = 0
        for batch_obs, batch_actions in loader:
            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)
            pred = actor(batch_obs)
            loss = criterion(pred, batch_actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * len(batch_obs)
            count += len(batch_obs)
        history.append(running / max(count, 1))
    actor.to("cpu")
    return history

