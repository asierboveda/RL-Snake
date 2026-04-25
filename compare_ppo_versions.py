from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_ppo import evaluate_against_bots


def parse_args():
    parser = argparse.ArgumentParser(description="Compare PPO v3 and tactical PPO v4.")
    parser.add_argument("--v3-model", default="models/ppo_headless/ppo_snake.zip")
    parser.add_argument("--v4-model", default="models/ppo_headless_v4/best_model/best_model.zip")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--turn-limit", type=int, default=900)
    parser.add_argument("--out", default="logs/ppo_headless_v4/v3_vs_v4_comparison.json")
    return parser.parse_args()


def main():
    args = parse_args()
    v3_path = Path(args.v3_model)
    v4_path = Path(args.v4_model)
    if not v3_path.exists():
        raise SystemExit(f"PPO v3 model not found: {v3_path}")
    if not v4_path.exists():
        raise SystemExit(
            f"PPO v4 model not found: {v4_path}. "
            "Train v4 first with train_ppo.py before comparing."
        )

    report = {
        "v3_model": str(v3_path),
        "v4_model": str(v4_path),
        "episodes": args.episodes,
        "seed": args.seed,
        "metrics": {
            "reward": "mean_reward",
            "score": "mean_score",
            "fruit": "mean_fruit_score",
            "survival": "mean_survival_turns",
            "early_death_rate": "early_death_rate",
            "invalid_action_rate": "invalid_action_rate",
            "win_rate": "win_rate",
            "kills": "mean_kills",
        },
        "v3": evaluate_against_bots(str(v3_path), args.episodes, args.seed, args.turn_limit),
        "v4": evaluate_against_bots(str(v4_path), args.episodes, args.seed, args.turn_limit),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
