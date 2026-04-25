from __future__ import annotations

import argparse
import json
from pathlib import Path

from ppo_env import PPOHeadlessSnakeEnv, run_policy_episode


def parse_args():
    parser = argparse.ArgumentParser(description="Run a reproducible PPO demo/replay v4.")
    parser.add_argument("--model-path", default="models/ppo_headless_v4/best_model/best_model.zip")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bot-kind", choices=("random", "greedy", "survival"), default="random")
    parser.add_argument("--out", default="logs/ppo_headless_v4/demo_replay.json")
    parser.add_argument("--turn-limit", type=int, default=900)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit(
            "Stable-Baselines3/Gymnasium no estan instalados. "
            "Instala dependencias con: pip install stable-baselines3 gymnasium"
        ) from exc

    env = PPOHeadlessSnakeEnv(bot_kind=args.bot_kind, turn_limit=args.turn_limit)
    model = PPO.load(args.model_path, env=env)
    metrics = run_policy_episode(env, model, deterministic=True, seed=args.seed)
    replay = env.replay_dict()
    replay["demo_metrics"] = metrics
    replay["model_path"] = args.model_path
    replay["bot_kind"] = args.bot_kind

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(replay, indent=2), encoding="utf-8")
    print(json.dumps({"replay": str(out), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
