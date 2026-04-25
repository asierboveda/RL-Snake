from __future__ import annotations

import argparse
import json
from pathlib import Path

from ppo_env import PPOHeadlessSnakeEnv, RandomPolicy, run_policy_episode, summarize_episode_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO Snake policy.")
    parser.add_argument("--model-path", default="models/ppo_headless/ppo_snake.zip")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--bot-kind", choices=("random", "greedy", "survival"), default="random")
    parser.add_argument("--out", default="logs/ppo_headless/evaluation.json")
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
    random_policy = RandomPolicy(env.action_space)

    ppo_episodes = [
        run_policy_episode(env, model, deterministic=True, seed=args.seed + i)
        for i in range(args.episodes)
    ]
    random_episodes = [
        run_policy_episode(env, random_policy, deterministic=False, seed=args.seed + 10_000 + i)
        for i in range(args.episodes)
    ]
    report = {
        "model_path": args.model_path,
        "bot_kind": args.bot_kind,
        "observation": "rl_observation_v1_features",
        "reward": "rl_reward.RewardConfig default + invalid action penalty",
        "ppo": summarize_episode_metrics(ppo_episodes),
        "random_policy_baseline": summarize_episode_metrics(random_episodes),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
