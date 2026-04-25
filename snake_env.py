import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from board_state import BOARD_COLS, BOARD_ROWS, BoardState, FruitState, SnakeState, determine_winner
from SnakeGame import SnakeGame


ActionInput = Union[Mapping[int, str], Sequence[str]]
VALID_ACTIONS = ("N", "S", "E", "W")


@dataclass(frozen=True)
class EnvTransition:
    state: BoardState
    rewards: Dict[int, float]
    done: bool
    info: Dict[str, object]


@dataclass(frozen=True)
class ReplayStep:
    turn: int
    actions: Dict[int, str]
    state: BoardState
    rewards: Dict[int, float]
    done: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "turn": self.turn,
            "actions": {str(player_id): action for player_id, action in self.actions.items()},
            "state": self.state.to_dict(),
            "rewards": {str(player_id): reward for player_id, reward in self.rewards.items()},
            "done": self.done,
        }


class SnakeEnv:
    """Headless Gym-like adapter around the existing SnakeGame engine."""

    def __init__(self, seed: Optional[int] = None, initial_fruits: int = 5, turn_limit: int = 900):
        self.seed = seed
        self.initial_fruits = initial_fruits
        self.turn_limit = turn_limit
        self.game: Optional[SnakeGame] = None
        self.replay: List[ReplayStep] = []
        self._last_scores: Dict[int, int] = {}
        self._last_alive: Dict[int, bool] = {}

    def reset(self, seed: Optional[int] = None) -> BoardState:
        if seed is not None:
            self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        self.game = SnakeGame()
        self.game.setNoise(0.0)
        for _ in range(self.initial_fruits):
            self.game.addRandomFruit()

        self.replay = []
        self._last_scores = {idx: snake.getScore() for idx, snake in enumerate(self.game.snakes)}
        self._last_alive = {idx: snake.isAlive for idx, snake in enumerate(self.game.snakes)}
        return self.board_state()

    def step(self, actions: ActionInput) -> EnvTransition:
        self._require_game()
        normalized_actions = self._normalize_actions(actions)
        previous_scores = {idx: snake.getScore() for idx, snake in enumerate(self.game.snakes)}
        previous_alive = {idx: snake.isAlive for idx, snake in enumerate(self.game.snakes)}

        for player_id, action in normalized_actions.items():
            self.game.movePlayer(player_id, action)

        self.game.checkMovements()
        self.game.update()
        self.game.turn += 1

        state = self.board_state()
        rewards = self._compute_rewards(previous_scores, previous_alive)
        done = self.game.turn >= self.turn_limit or not self.game.gameIsAlive()
        transition = EnvTransition(
            state=state,
            rewards=rewards,
            done=done,
            info={
                "turn": self.game.turn,
                "scores": self.game.getScores(),
                "legal_actions": {
                    player_id: self.legal_actions(player_id)
                    for player_id in range(len(self.game.snakes))
                },
            },
        )
        self.replay.append(
            ReplayStep(
                turn=self.game.turn,
                actions=normalized_actions,
                state=state,
                rewards=rewards,
                done=done,
            )
        )
        self._last_scores = {idx: snake.getScore() for idx, snake in enumerate(self.game.snakes)}
        self._last_alive = {idx: snake.isAlive for idx, snake in enumerate(self.game.snakes)}
        return transition

    def legal_actions(self, player_id: int) -> Tuple[str, ...]:
        self._require_game()
        snake = self.game.snakes[player_id]
        if not snake.isAlive:
            return ()
        head = snake.body[0]
        candidates = {
            "N": (head[0] - 1, head[1]),
            "S": (head[0] + 1, head[1]),
            "E": (head[0], head[1] + 1),
            "W": (head[0], head[1] - 1),
        }
        occupied = {
            (piece[0], piece[1])
            for other in self.game.snakes
            if other.isAlive
            for piece in other.body
        }
        return tuple(
            action
            for action in VALID_ACTIONS
            if self._is_inside(candidates[action]) and candidates[action] not in occupied
        )

    def board_state(self) -> BoardState:
        self._require_game()
        snakes = tuple(
            SnakeState(
                player_id=idx,
                label=("A", "B", "C", "D")[idx],
                color=snake.color,
                alive=snake.isAlive,
                body=self._body_for_contract(snake.body),
                score=snake.getScore(),
                fruit_score=snake.getFruitScore(),
            )
            for idx, snake in enumerate(self.game.snakes)
        )
        winner_id, terminal_reason = determine_winner(snakes)
        game_alive = self.game.gameIsAlive()
        return BoardState(
            turn=self.game.turn,
            rows=BOARD_ROWS,
            cols=BOARD_COLS,
            snakes=snakes,
            fruits=tuple(
                FruitState(row=fruit.pos[0], col=fruit.pos[1], value=fruit.value, time_left=fruit.timeLeft)
                for fruit in self.game.fruits
                if self._is_inside((fruit.pos[0], fruit.pos[1]))
            ),
            game_alive=game_alive,
            winner_id=None if game_alive else winner_id,
            terminal_reason=None if game_alive else terminal_reason,
        )

    def to_replay_dict(self) -> Dict[str, object]:
        return {
            "seed": self.seed,
            "initial_fruits": self.initial_fruits,
            "turn_limit": self.turn_limit,
            "steps": [step.to_dict() for step in self.replay],
        }

    def _normalize_actions(self, actions: ActionInput) -> Dict[int, str]:
        if isinstance(actions, Mapping):
            normalized = {int(player_id): action for player_id, action in actions.items()}
        else:
            normalized = {player_id: action for player_id, action in enumerate(actions)}

        for player_id in range(len(self.game.snakes)):
            normalized.setdefault(player_id, self.game.snakes[player_id].body[0][2])
        invalid = {
            player_id: action
            for player_id, action in normalized.items()
            if action not in VALID_ACTIONS
        }
        if invalid:
            raise ValueError(f"invalid actions: {invalid}")
        return normalized

    def _compute_rewards(self, previous_scores: Dict[int, int], previous_alive: Dict[int, bool]) -> Dict[int, float]:
        rewards = {}
        for idx, snake in enumerate(self.game.snakes):
            reward = float(snake.getScore() - previous_scores[idx])
            if previous_alive[idx] and not snake.isAlive:
                reward -= 100.0
            rewards[idx] = reward
        return rewards

    def _body_for_contract(self, body: Iterable[Sequence[Union[int, str]]]) -> Tuple[Tuple[int, int, str], ...]:
        in_bounds = tuple(
            (int(piece[0]), int(piece[1]), str(piece[2]))
            for piece in body
            if self._is_inside((int(piece[0]), int(piece[1])))
        )
        if in_bounds:
            return in_bounds
        return ((0, 0, "N"),)

    def _is_inside(self, pos: Tuple[int, int]) -> bool:
        return 0 <= pos[0] < BOARD_ROWS and 0 <= pos[1] < BOARD_COLS

    def _require_game(self) -> None:
        if self.game is None:
            raise RuntimeError("call reset() before using the environment")
