from __future__ import annotations

import argparse
import json
from pathlib import Path

from ppo_env import PPOHeadlessSnakeEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO headless for Snake battle royale.")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bot-kind", choices=("random", "greedy", "survival"), default="random")
    parser.add_argument("--model-path", default="models/ppo_headless/ppo_snake")
    parser.add_argument("--log-dir", default="logs/ppo_headless")
    parser.add_argument("--turn-limit", type=int, default=900)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise SystemExit(
            "Stable-Baselines3/Gymnasium no estan instalados. "
            "Instala dependencias con: pip install stable-baselines3 gymnasium"
        ) from exc

    log_dir = Path(args.log_dir)
    model_path = Path(args.model_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    env = Monitor(
        PPOHeadlessSnakeEnv(seed=args.seed, bot_kind=args.bot_kind, turn_limit=args.turn_limit),
        filename=str(log_dir / "monitor.csv"),
    )
    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        learning_rate=3e-4,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    model.save(str(model_path))

    metadata = {
        "algorithm": "PPO",
        "timesteps": args.timesteps,
        "seed": args.seed,
        "bot_kind": args.bot_kind,
        "observation": "rl_observation_v1_features",
        "reward": "rl_reward.RewardConfig default + invalid action penalty",
        "model_path": str(model_path.with_suffix(".zip")),
    }
    (model_path.parent / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
