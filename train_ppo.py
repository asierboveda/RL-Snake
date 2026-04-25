from __future__ import annotations

import argparse
import json
from pathlib import Path

from ppo_env import PPOHeadlessSnakeEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO headless for Snake battle royale v4.")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bot-kind", choices=("random", "greedy", "survival"), default="random")
    parser.add_argument("--model-path", default="models/ppo_headless_v4/ppo_snake")
    parser.add_argument("--log-dir", default="logs/ppo_headless_v4")
    parser.add_argument("--turn-limit", type=int, default=900)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback
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

    eval_env = Monitor(
        PPOHeadlessSnakeEnv(seed=args.seed + 1, bot_kind=args.bot_kind, turn_limit=args.turn_limit),
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_path.parent / "best_model"),
        log_path=str(log_dir / "eval"),
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=args.eval_episodes,
    )

    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        learning_rate=3e-4,
        device="cpu",
    )
    model.learn(total_timesteps=args.timesteps, callback=eval_callback, progress_bar=False)
    model.save(str(model_path))

    metadata = {
        "algorithm": "PPO",
        "timesteps": args.timesteps,
        "seed": args.seed,
        "bot_kind": args.bot_kind,
        "observation": "rl_observation_v1_features_v4",
        "reward": "rl_reward.RewardConfig v4 (relative actions + tactical + fruit shaping)",
        "model_path": str(model_path.with_suffix(".zip")),
    }
    (model_path.parent / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
