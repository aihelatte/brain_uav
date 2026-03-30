from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..config import ExperimentConfig
from ..scripts.common import make_actor
from ..trainers import train_behavior_cloning
from ..utils.io import save_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Train behavior cloning initialization.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", choices=["snn", "ann"], default="snn")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg = ExperimentConfig()
    data = np.load(args.dataset)
    actor = make_actor(cfg, args.model, data["observations"].shape[1], data["actions"].shape[1])
    history = train_behavior_cloning(
        actor,
        args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=cfg.training.actor_lr,
    )
    output = args.output or Path(f"outputs/bc_{args.model}.pt")
    save_checkpoint(
        output,
        {
            "model_type": args.model,
            "state_dict": actor.state_dict(),
            "loss_history": history,
        },
    )
    print(f"Saved BC checkpoint to {output}")
    print(f"Final BC loss: {history[-1]:.6f}")


if __name__ == "__main__":
    main()

