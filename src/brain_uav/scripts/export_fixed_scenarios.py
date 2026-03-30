from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import ExperimentConfig
from ..envs import StaticNoFlyTrajectoryEnv
from ..utils.io import ensure_parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Export fixed evaluation scenarios.")
    parser.add_argument("--output", type=Path, default=Path("data/fixed_scenarios.json"))
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()

    cfg = ExperimentConfig()
    env = StaticNoFlyTrajectoryEnv(cfg.scenario, cfg.rewards, seed=args.seed)
    scenarios = []
    for idx in range(args.count):
        env.reset(seed=args.seed + idx)
        scenario = env.export_scenario()
        scenarios.append(
            {
                "state": scenario["state"].tolist(),
                "goal": scenario["goal"].tolist(),
                "zones": [
                    {"center_xy": zone["center_xy"].tolist(), "radius": zone["radius"]}
                    for zone in scenario["zones"]
                ],
            }
        )
    target = ensure_parent(args.output)
    target.write_text(json.dumps(scenarios, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(scenarios)} fixed scenarios to {target}")


if __name__ == "__main__":
    main()

