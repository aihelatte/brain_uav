"""Behavior cloning training loop.

这个模块用于把基线轨迹变成监督学习训练，先让策略学会一个“像样的起步”。
"""

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
    verbose: bool = True,
) -> list[float]:
    """Train actor by supervised learning on (state, action) pairs."""

    payload = np.load(dataset_path)
    obs = torch.tensor(payload["observations"], dtype=torch.float32)
    actions = torch.tensor(payload["actions"], dtype=torch.float32)
    loader = DataLoader(TensorDataset(obs, actions), batch_size=batch_size, shuffle=True)
    actor.to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history: list[float] = []
    if verbose:
        print(f"[BC] dataset={dataset_path} samples={len(obs)} batch_size={batch_size} epochs={epochs}")
    for epoch_idx in range(epochs):
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
        epoch_loss = running / max(count, 1)
        history.append(epoch_loss)
        if verbose:
            print(f"[BC] epoch {epoch_idx + 1}/{epochs} loss={epoch_loss:.6f}")
    actor.to("cpu")
    return history
