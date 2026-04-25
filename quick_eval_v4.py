from __future__ import annotations

import json
from pathlib import Path

from stable_baselines3 import PPO
from ppo_env import PPOHeadlessSnakeEnv, RandomPolicy, run_policy_episode, summarize_episode_metrics


def quick_eval(model_path: str, episodes_per_bot: int = 5, seed: int = 5000):
    results = {}
    for bot_kind in ("random", "greedy", "survival"):
        env = PPOHeadlessSnakeEnv(bot_kind=bot_kind, turn_limit=900)
        model = PPO.load(model_path, env=env)
        random_policy = RandomPolicy(env.action_space)

        ppo_eps = [
            run_policy_episode(env, model, deterministic=True, seed=seed + i)
            for i in range(episodes_per_bot)
        ]
        rand_eps = [
            run_policy_episode(env, random_policy, deterministic=False, seed=seed + 10_000 + i)
            for i in range(episodes_per_bot)
        ]
        results[bot_kind] = {
            "ppo": summarize_episode_metrics(ppo_eps),
            "random_baseline": summarize_episode_metrics(rand_eps),
        }
    return results


if __name__ == "__main__":
    model_path = "models/ppo_headless_v4/best_model/best_model.zip"
    report = quick_eval(model_path, episodes_per_bot=5, seed=5000)
    report["model_path"] = model_path
    out = Path("logs/ppo_headless_v4/evaluation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
