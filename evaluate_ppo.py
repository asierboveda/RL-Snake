from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ppo_env import PPOHeadlessSnakeEnv, RandomPolicy, run_policy_episode, summarize_episode_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO Snake policy v4.")
    parser.add_argument("--model-path", default="models/ppo_headless_v4/best_model/best_model.zip")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--bot-kind", choices=("random", "greedy", "survival"), default="random")
    parser.add_argument("--out", default="logs/ppo_headless_v4/evaluation.json")
    parser.add_argument("--csv-out", default="logs/ppo_headless_v4/evaluation.csv")
    parser.add_argument("--turn-limit", type=int, default=900)
    return parser.parse_args()


def evaluate_against_bots(model_path: str, episodes: int, seed: int, turn_limit: int):
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit(
            "Stable-Baselines3/Gymnasium no estan instalados. "
            "Instala dependencias con: pip install stable-baselines3 gymnasium"
        ) from exc

    results = {}
    for bot_kind in ("random", "greedy", "survival"):
        env = PPOHeadlessSnakeEnv(bot_kind=bot_kind, turn_limit=turn_limit)
        model = PPO.load(model_path, env=env)
        random_policy = RandomPolicy(env.action_space)

        ppo_eps = [
            run_policy_episode(env, model, deterministic=True, seed=seed + i)
            for i in range(episodes)
        ]
        rand_eps = [
            run_policy_episode(env, random_policy, deterministic=False, seed=seed + 10_000 + i)
            for i in range(episodes)
        ]
        results[bot_kind] = {
            "ppo": summarize_episode_metrics(ppo_eps),
            "random_baseline": summarize_episode_metrics(rand_eps),
        }
    return results


def main():
    args = parse_args()
    report = evaluate_against_bots(args.model_path, args.episodes, args.seed, args.turn_limit)
    report["model_path"] = args.model_path
    report["observation"] = "rl_observation_v1_features_v4"
    report["reward"] = "rl_reward.RewardConfig v4 (relative actions + tactical + fruit shaping)"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    csv_out = Path(args.csv_out)
    primary = report.get(args.bot_kind, {})
    rows = []
    for policy in ("ppo", "random_baseline"):
        row = {"policy": policy, **primary.get(policy, {})}
        rows.append(row)
    if rows:
        with csv_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
