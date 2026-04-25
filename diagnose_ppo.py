from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from ppo_env import PPOHeadlessSnakeEnv, RandomPolicy, run_policy_episode
from rl_reward import RewardConfig, compute_reward


def diagnose(model_path: str, episodes: int = 100, seed: int = 1000, bot_kind: str = "random"):
    env = PPOHeadlessSnakeEnv(bot_kind=bot_kind, turn_limit=900)
    model = PPO.load(model_path, env=env)
    random_policy = RandomPolicy(env.action_space)

    all_metrics = []
    for policy_name, policy in [("ppo", model), ("random", random_policy)]:
        stats = {
            "episodes": [],
            "actions": [],
            "invalid_actions": 0,
            "total_steps": 0,
            "silly_deaths": 0,
            "combat_deaths": 0,
            "early_deaths": 0,  # before turn 50
            "rewards": [],
            "scores": [],
            "fruit_scores": [],
            "survival_turns": [],
            "wins": 0,
            "kills": [],
            "reward_components": {"survival": 0.0, "fruit": 0.0, "kill": 0.0, "death": 0.0, "win": 0.0, "invalid": 0.0},
        }
        for i in range(episodes):
            obs, info = env.reset(seed=seed + i + (10000 if policy_name == "random" else 0))
            terminated = False
            truncated = False
            ep_reward = 0.0
            ep_actions = []
            while not (terminated or truncated):
                action, _ = policy.predict(obs, deterministic=(policy_name == "ppo"))
                action = int(action)
                # Reconstruct what ppo_env.step does to compute reward components
                prev_state = env.previous_state
                controlled_action = env.action_labels[action] if 0 <= action < len(env.action_labels) else env._safe_fallback_action(prev_state)
                is_invalid = controlled_action not in env.base_env.legal_actions(env.controlled_player)

                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                ep_actions.append(controlled_action)
                stats["total_steps"] += 1
                if is_invalid:
                    stats["invalid_actions"] += 1
                    stats["reward_components"]["invalid"] += -2.0

                # Decompose reward manually for PPO env
                # Note: the actual reward already includes invalid penalty; we want component breakdown
                # We can recompute base reward without invalid penalty
                base_reward = compute_reward(
                    prev_state,
                    controlled_action,
                    env.previous_state,
                    env.controlled_player,
                    {},
                    env.reward_config,
                )
                # classify components roughly
                if base_reward == env.reward_config.survival_reward:
                    stats["reward_components"]["survival"] += base_reward
                if base_reward > 0 and base_reward != env.reward_config.survival_reward and base_reward != env.reward_config.win_reward:
                    # could be fruit or kill
                    if info.get("fruit_score", 0) > 0:
                        stats["reward_components"]["fruit"] += base_reward
                    else:
                        stats["reward_components"]["kill"] += base_reward
                if base_reward < 0 and base_reward != -2.0:
                    stats["reward_components"]["death"] += base_reward
                if base_reward == env.reward_config.win_reward:
                    stats["reward_components"]["win"] += base_reward

            stats["episodes"].append(ep_reward)
            stats["scores"].append(info["score"])
            stats["fruit_scores"].append(info["fruit_score"])
            stats["survival_turns"].append(info["turn"])
            stats["kills"].append(info["kills"])
            if info.get("winner_id") == env.controlled_player:
                stats["wins"] += 1
            if not info["alive"]:
                if info["turn"] < 50:
                    stats["early_deaths"] += 1
                # infer death cause from last action and state
                # We don't have direct cause, but we can use the fact that if _is_silly_death returns True for the action vs prev_state before death...
                # For simplicity, just classify by action vs board bounds/body
                # We'll skip detailed cause for now and rely on silly/combat approximation later
            stats["actions"].extend(ep_actions)

        action_dist = dict(Counter(stats["actions"]))
        total = stats["total_steps"] or 1
        summary = {
            "policy": policy_name,
            "episodes": episodes,
            "mean_reward": float(np.mean(stats["episodes"])),
            "mean_score": float(np.mean(stats["scores"])),
            "mean_fruit_score": float(np.mean(stats["fruit_scores"])),
            "mean_survival_turns": float(np.mean(stats["survival_turns"])),
            "mean_kills": float(np.mean(stats["kills"])),
            "win_rate": stats["wins"] / episodes,
            "invalid_action_rate": stats["invalid_actions"] / total,
            "early_death_rate": stats["early_deaths"] / episodes,
            "action_distribution": action_dist,
            "reward_components": {k: v / episodes for k, v in stats["reward_components"].items()},
        }
        all_metrics.append(summary)
        print(json.dumps(summary, indent=2))

    out = Path("logs/ppo_headless/diagnosis.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    diagnose("models/ppo_headless/ppo_snake.zip", episodes=100, seed=2000)
