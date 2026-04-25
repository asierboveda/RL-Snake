from __future__ import annotations

import random
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from baseline_bots import GreedyPlayer, RandomPlayer, SurvivalPlayer
from board_state import BoardState
from rl_observation import FEATURE_SET_V1, build_observation
from rl_reward import RewardConfig, compute_reward, _is_silly_death
from snake_env import SnakeEnv

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised only without optional deps
    class _FallbackEnv:
        pass

    class _Discrete:
        def __init__(self, n: int):
            self.n = n

        def sample(self) -> int:
            return random.randrange(self.n)

    class _Box:
        def __init__(self, low, high, shape, dtype):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class _Spaces:
        Box = _Box
        Discrete = _Discrete

    class _Gym:
        Env = _FallbackEnv

    gym = _Gym()
    spaces = _Spaces()


RELATIVE_ACTIONS: Tuple[str, ...] = ("FORWARD", "LEFT", "RIGHT")
OBSERVATION_NAME = "rl_observation_v1_features"
INVALID_ACTION_PENALTY = -2.0
BotFactory = Callable[[int, str, object], object]

_ABSOLUTE_FROM_RELATIVE = {
    "N": {"FORWARD": "N", "LEFT": "W", "RIGHT": "E", "BACK": "S"},
    "S": {"FORWARD": "S", "LEFT": "E", "RIGHT": "W", "BACK": "N"},
    "E": {"FORWARD": "E", "LEFT": "N", "RIGHT": "S", "BACK": "W"},
    "W": {"FORWARD": "W", "LEFT": "S", "RIGHT": "N", "BACK": "E"},
}


def _relative_to_absolute(direction: str, relative: str) -> str:
    return _ABSOLUTE_FROM_RELATIVE.get(direction, _ABSOLUTE_FROM_RELATIVE["N"])[relative]


def _legal_relative_actions(state: BoardState, player_id: int, base_env: SnakeEnv) -> Tuple[str, ...]:
    """Return relative actions that map to legal absolute actions."""
    direction = state.snakes[player_id].head[2]
    legal_abs = set(base_env.legal_actions(player_id))
    legal_rel = []
    for rel in RELATIVE_ACTIONS:
        abs_action = _relative_to_absolute(direction, rel)
        if abs_action in legal_abs:
            legal_rel.append(rel)
    return tuple(legal_rel)


def make_bot_factories(kind: str) -> Tuple[BotFactory, BotFactory, BotFactory]:
    bot_cls = {
        "random": RandomPlayer,
        "greedy": GreedyPlayer,
        "survival": SurvivalPlayer,
    }.get(kind.lower())
    if bot_cls is None:
        raise ValueError("bot kind must be one of: random, greedy, survival")
    return (bot_cls, bot_cls, bot_cls)


class PPOHeadlessSnakeEnv(gym.Env):
    """Single-agent Gymnasium wrapper: PPO controls snake 0, bots control 1..3."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        seed: Optional[int] = None,
        bot_kind: str = "random",
        controlled_player: int = 0,
        initial_fruits: int = 5,
        turn_limit: int = 900,
        reward_config: RewardConfig = RewardConfig(),
    ):
        self.controlled_player = controlled_player
        self.initial_fruits = initial_fruits
        self.turn_limit = turn_limit
        self.reward_config = reward_config
        self.action_labels = RELATIVE_ACTIONS
        self.bot_factories = make_bot_factories(bot_kind)
        self.bot_kind = bot_kind
        self.base_env = SnakeEnv(seed=seed, initial_fruits=initial_fruits, turn_limit=turn_limit)
        self.bots: List[object] = []
        self.previous_state: Optional[BoardState] = None
        self.last_invalid_action = False

        obs_dim = len(FEATURE_SET_V1.feature_names)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(RELATIVE_ACTIONS))

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, object]] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        state = self.base_env.reset(seed=seed)
        game = self.base_env.game
        colors = ("G", "B", "R", "Y")
        self.bots = [
            factory(player_id, colors[player_id], game)
            for player_id, factory in zip((1, 2, 3), self.bot_factories)
        ]
        self.previous_state = state
        self.last_invalid_action = False
        return self._observation(state), self._info(state, reward=0.0)

    def step(self, action: int):
        if self.previous_state is None:
            raise RuntimeError("call reset() before step()")

        action_int = int(action)
        direction = self.previous_state.snakes[self.controlled_player].head[2]

        if action_int < 0 or action_int >= len(RELATIVE_ACTIONS):
            intended_relative = "FORWARD"
            self.last_invalid_action = True
        else:
            intended_relative = RELATIVE_ACTIONS[action_int]
            intended_absolute = _relative_to_absolute(direction, intended_relative)
            self.last_invalid_action = intended_absolute not in self.base_env.legal_actions(self.controlled_player)

        if self.last_invalid_action:
            executed_relative = self._safe_fallback_relative_action(self.previous_state)
        else:
            executed_relative = intended_relative

        executed_absolute = _relative_to_absolute(direction, executed_relative)

        actions = {self.controlled_player: executed_absolute}
        for offset, bot in enumerate(self.bots, start=1):
            actions[offset] = bot.play_board_state(self.previous_state)

        transition = self.base_env.step(actions)
        reward = compute_reward(
            self.previous_state,
            executed_absolute,
            transition.state,
            self.controlled_player,
            transition.info,
            self.reward_config,
        )
        if self.last_invalid_action:
            reward += INVALID_ACTION_PENALTY

        controlled = transition.state.snakes[self.controlled_player]
        terminated = transition.done or not controlled.alive
        truncated = bool(self.base_env.game and self.base_env.game.turn >= self.turn_limit and not terminated)
        self.previous_state = transition.state
        return self._observation(transition.state), float(reward), bool(terminated), truncated, self._info(
            transition.state,
            reward=reward,
            invalid_action=self.last_invalid_action,
            relative_action=executed_relative,
            absolute_action=executed_absolute,
            intended_relative=intended_relative,
        )

    def replay_dict(self) -> Dict[str, object]:
        return self.base_env.to_replay_dict()

    def _observation(self, state: BoardState) -> np.ndarray:
        snake = state.snakes[self.controlled_player]
        if not snake.alive:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        obs = build_observation(state, self.controlled_player)["features"].astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)

    def _info(
        self,
        state: BoardState,
        reward: float,
        invalid_action: bool = False,
        relative_action: Optional[str] = None,
        absolute_action: Optional[str] = None,
        intended_relative: Optional[str] = None,
    ) -> Dict[str, object]:
        snake = state.snakes[self.controlled_player]
        kill_score = max(0, snake.score - snake.fruit_score)
        info = {
            "observation": OBSERVATION_NAME,
            "bot_kind": self.bot_kind,
            "turn": state.turn,
            "alive": snake.alive,
            "score": snake.score,
            "fruit_score": snake.fruit_score,
            "kills": kill_score // 30,
            "winner_id": state.winner_id,
            "terminal_reason": state.terminal_reason,
            "reward": float(reward),
            "invalid_action": invalid_action,
            "relative_action": relative_action,
            "absolute_action": absolute_action,
            "intended_relative": intended_relative,
        }
        if not snake.alive and self.previous_state is not None:
            prev_snake = self.previous_state.snakes[self.controlled_player]
            if prev_snake.alive:
                info["death_cause"] = "silly" if _is_silly_death(self.previous_state, prev_snake, absolute_action or prev_snake.head[2]) else "combat"
        return info

    def _safe_fallback_relative_action(self, state: BoardState) -> str:
        legal_rel = _legal_relative_actions(state, self.controlled_player, self.base_env)
        if legal_rel:
            return legal_rel[0]
        return "FORWARD"


def run_policy_episode(
    env: PPOHeadlessSnakeEnv,
    policy,
    deterministic: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    terminated = False
    truncated = False
    invalid_actions = 0
    steps = 0
    relative_actions_taken = []
    absolute_actions_taken = []
    death_cause = None
    while not (terminated or truncated):
        action, _ = policy.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        steps += 1
        if info.get("invalid_action"):
            invalid_actions += 1
        relative_actions_taken.append(info.get("relative_action"))
        absolute_actions_taken.append(info.get("absolute_action"))
        if not info.get("alive") and death_cause is None:
            death_cause = info.get("death_cause", "unknown")
    info = dict(info)
    info["episode_reward"] = total_reward
    info["survival_turns"] = info["turn"]
    info["win"] = info.get("winner_id") == env.controlled_player
    info["invalid_actions"] = invalid_actions
    info["steps"] = steps
    info["relative_action_distribution"] = {a: relative_actions_taken.count(a) for a in set(relative_actions_taken)}
    info["absolute_action_distribution"] = {a: absolute_actions_taken.count(a) for a in set(absolute_actions_taken)}
    info["death_cause"] = death_cause
    return info


class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, obs, deterministic: bool = True):
        return self.action_space.sample(), None


def summarize_episode_metrics(episodes: Iterable[Dict[str, object]]) -> Dict[str, float]:
    data = list(episodes)
    if not data:
        return {
            "episodes": 0.0,
            "mean_reward": 0.0,
            "mean_score": 0.0,
            "mean_fruit_score": 0.0,
            "mean_survival_turns": 0.0,
            "mean_kills": 0.0,
            "win_rate": 0.0,
            "invalid_action_rate": 0.0,
            "early_death_rate": 0.0,
        }
    total_steps = sum(row.get("steps", 0) for row in data) or 1
    return {
        "episodes": float(len(data)),
        "mean_reward": float(np.mean([row["episode_reward"] for row in data])),
        "mean_score": float(np.mean([row["score"] for row in data])),
        "mean_fruit_score": float(np.mean([row["fruit_score"] for row in data])),
        "mean_survival_turns": float(np.mean([row["survival_turns"] for row in data])),
        "mean_kills": float(np.mean([row["kills"] for row in data])),
        "win_rate": float(np.mean([1.0 if row["win"] else 0.0 for row in data])),
        "invalid_action_rate": float(sum(row.get("invalid_actions", 0) for row in data)) / total_steps,
        "early_death_rate": float(sum(1.0 for row in data if row.get("survival_turns", 0) < 50)) / len(data),
    }
