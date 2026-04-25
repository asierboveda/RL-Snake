from __future__ import annotations

import random
from collections import deque
from typing import Iterable, List, Optional, Sequence, Tuple

from board_state import BoardState


VALID_ACTIONS = ("N", "S", "E", "W")
ACTION_DELTAS = {
    "N": (-1, 0),
    "S": (1, 0),
    "E": (0, 1),
    "W": (0, -1),
}


def _rows(source) -> int:
    return source.rows if hasattr(source, "rows") else source.rSize


def _cols(source) -> int:
    return source.cols if hasattr(source, "cols") else source.cSize


def _snakes(source):
    return source.snakes


def _fruits(source):
    return source.fruits


def _snake_alive(snake) -> bool:
    return bool(snake.alive if hasattr(snake, "alive") else snake.isAlive)


def _snake_head(snake) -> Tuple[int, int, str]:
    return tuple(snake.head if hasattr(snake, "head") else snake.body[0])  # type: ignore[arg-type]


def _snake_body(snake) -> Sequence[Sequence[object]]:
    return snake.body


def _snake_fruit_score(snake) -> int:
    return int(snake.fruit_score if hasattr(snake, "fruit_score") else snake.getFruitScore())


def _snake_score(snake) -> int:
    return int(snake.score if hasattr(snake, "score") else snake.getScore())


def _player_id(snake) -> int:
    return int(snake.player_id if hasattr(snake, "player_id") else snake.playerNumber)


def _is_hunter(snake) -> bool:
    return bool(snake.is_hunter if hasattr(snake, "is_hunter") else _snake_fruit_score(snake) >= 120)


def _head_pos(snake) -> Tuple[int, int]:
    head = _snake_head(snake)
    return int(head[0]), int(head[1])


def _current_direction(snake) -> str:
    head = _snake_head(snake)
    if len(head) >= 3 and str(head[2]) in VALID_ACTIONS:
        return str(head[2])
    return "N"


def _occupied_positions(source, alive_only: bool = True) -> set[Tuple[int, int]]:
    occupied: set[Tuple[int, int]] = set()
    for snake in _snakes(source):
        if alive_only and not _snake_alive(snake):
            continue
        for cell in _snake_body(snake):
            occupied.add((int(cell[0]), int(cell[1])))
    return occupied


def _is_inside(source, pos: Tuple[int, int]) -> bool:
    return 0 <= pos[0] < _rows(source) and 0 <= pos[1] < _cols(source)


def _step(pos: Tuple[int, int], action: str) -> Tuple[int, int]:
    dr, dc = ACTION_DELTAS[action]
    return pos[0] + dr, pos[1] + dc


def _legal_actions(source, player_id: int) -> Tuple[str, ...]:
    snake = _snakes(source)[player_id]
    if not _snake_alive(snake):
        return ()

    head = _head_pos(snake)
    occupied = _occupied_positions(source, alive_only=True)
    actions: List[str] = []
    for action in VALID_ACTIONS:
        next_pos = _step(head, action)
        if _is_inside(source, next_pos) and next_pos not in occupied:
            actions.append(action)
    return tuple(actions)


def _fruit_positions(source) -> List[Tuple[int, int]]:
    positions: List[Tuple[int, int]] = []
    for fruit in _fruits(source):
        if hasattr(fruit, "row") and hasattr(fruit, "col"):
            positions.append((int(fruit.row), int(fruit.col)))
        else:
            positions.append((int(fruit.pos[0]), int(fruit.pos[1])))
    return positions


def _enemy_targets(source, player_id: int, weaker_only: bool = False) -> List[Tuple[int, int]]:
    my_snake = _snakes(source)[player_id]
    my_score = _snake_fruit_score(my_snake)
    targets: List[Tuple[int, int]] = []
    for idx, snake in enumerate(_snakes(source)):
        if idx == player_id or not _snake_alive(snake):
            continue
        if weaker_only and _snake_fruit_score(snake) >= my_score:
            continue
        targets.extend((_head_pos(snake) if i == 0 else (int(cell[0]), int(cell[1])))
                       for i, cell in enumerate(_snake_body(snake)))
    return targets


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _reachable_space(source, player_id: int, action: str) -> int:
    snake = _snakes(source)[player_id]
    if not _snake_alive(snake):
        return 0

    start = _step(_head_pos(snake), action)
    if not _is_inside(source, start):
        return 0

    occupied = _occupied_positions(source, alive_only=True)
    occupied.discard(start)
    queue = deque([start])
    visited = {start}

    while queue:
        row, col = queue.popleft()
        for delta in ACTION_DELTAS.values():
            nxt = (row + delta[0], col + delta[1])
            if nxt in visited or not _is_inside(source, nxt) or nxt in occupied:
                continue
            visited.add(nxt)
            queue.append(nxt)

    return len(visited)


def _wall_distance(source, pos: Tuple[int, int]) -> int:
    return min(pos[0], _rows(source) - 1 - pos[0], pos[1], _cols(source) - 1 - pos[1])


class BaselinePlayer:
    def __init__(self, playerID, color, game=None):
        self.playerID = playerID
        self.color = color
        self.game = game

    def _source(self, state):
        if state is not None and (hasattr(state, "snakes") or isinstance(state, BoardState)):
            return state
        return self.game if self.game is not None else state

    def _fallback_action(self, source) -> str:
        if source is None:
            return "N"
        snake = _snakes(source)[self.playerID]
        return _current_direction(snake)

    def _play(self, source) -> str:
        if source is None:
            return "N"
        snake = _snakes(source)[self.playerID]
        if not _snake_alive(snake):
            return self._fallback_action(source)

        legal = _legal_actions(source, self.playerID)
        if not legal:
            return self._fallback_action(source)
        return self.select_action(source, snake, legal)

    def play(self, state):
        return self._play(self._source(state))

    def play_board_state(self, board_state: BoardState):
        return self._play(board_state)

    def legal_actions(self, source) -> Tuple[str, ...]:
        return _legal_actions(source, self.playerID)

    def select_action(self, source, snake, legal: Tuple[str, ...]) -> str:  # pragma: no cover - abstract hook
        raise NotImplementedError


class RandomPlayer(BaselinePlayer):
    def select_action(self, source, snake, legal: Tuple[str, ...]) -> str:
        return random.choice(list(legal))


class GreedyPlayer(BaselinePlayer):
    def _target_positions(self, source) -> List[Tuple[int, int]]:
        snake = _snakes(source)[self.playerID]
        if _is_hunter(snake):
            rival_targets = _enemy_targets(source, self.playerID, weaker_only=True)
            if rival_targets:
                return rival_targets
        return _fruit_positions(source)

    def select_action(self, source, snake, legal: Tuple[str, ...]) -> str:
        head = _head_pos(snake)
        targets = self._target_positions(source)
        if not targets:
            return legal[0]

        def ranking(action: str):
            next_pos = _step(head, action)
            distance = min(_manhattan(next_pos, target) for target in targets)
            space = _reachable_space(source, self.playerID, action)
            return (distance, -space, _wall_distance(source, next_pos), action)

        return min(legal, key=ranking)

    def findCloserFruit(self, my_snake):
        source = self.game if self.game is not None else getattr(my_snake, "game", None)
        fruits = _fruit_positions(source) if source is not None else []
        head = _head_pos(my_snake)
        if not fruits:
            if source is None:
                return (0, 0)
            return (_rows(source) // 2, _cols(source) // 2)
        return min(fruits, key=lambda pos: _manhattan(head, pos))

    def findCloserRival(self, my_snake, weaker_rivals):
        targets = []
        for rival in weaker_rivals:
            targets.extend((_head_pos(rival) if idx == 0 else (int(cell[0]), int(cell[1])))
                           for idx, cell in enumerate(_snake_body(rival)))
        source = self.game if self.game is not None else getattr(my_snake, "game", None)
        head = _head_pos(my_snake)
        if not targets:
            if source is None:
                return (0, 0)
            return (_rows(source) // 2, _cols(source) // 2)
        return min(targets, key=lambda pos: _manhattan(head, pos))

    def setDirection(self, headPos, goal):
        if goal[0] > headPos[0]:
            return "S"
        if goal[0] < headPos[0]:
            return "N"
        if goal[1] > headPos[1]:
            return "E"
        return "W"


class SurvivalPlayer(BaselinePlayer):
    def select_action(self, source, snake, legal: Tuple[str, ...]) -> str:
        head = _head_pos(snake)

        def ranking(action: str):
            next_pos = _step(head, action)
            space = _reachable_space(source, self.playerID, action)
            wall = _wall_distance(source, next_pos)
            return (space, wall, -_manhattan(next_pos, head), action)

        return max(legal, key=ranking)


class AggressivePlayer(BaselinePlayer):
    def _targets(self, source) -> List[Tuple[int, int]]:
        enemy_targets = _enemy_targets(source, self.playerID, weaker_only=False)
        if enemy_targets:
            return enemy_targets
        return _fruit_positions(source)

    def select_action(self, source, snake, legal: Tuple[str, ...]) -> str:
        head = _head_pos(snake)
        targets = self._targets(source)
        if not targets:
            return legal[0]

        def ranking(action: str):
            next_pos = _step(head, action)
            distance = min(_manhattan(next_pos, target) for target in targets)
            space = _reachable_space(source, self.playerID, action)
            return (distance, -space, _wall_distance(source, next_pos), action)

        return min(legal, key=ranking)


class HybridPlayer(BaselinePlayer):
    def select_action(self, source, snake, legal: Tuple[str, ...]) -> str:
        head = _head_pos(snake)
        hunter = _is_hunter(snake)
        weak_rivals = _enemy_targets(source, self.playerID, weaker_only=True)
        fruits = _fruit_positions(source)

        if hunter and weak_rivals:
            targets = weak_rivals
        elif len(legal) <= 2 or max(_reachable_space(source, self.playerID, action) for action in legal) < 30:
            targets = []
        else:
            targets = fruits

        def ranking(action: str):
            next_pos = _step(head, action)
            space = _reachable_space(source, self.playerID, action)
            wall = _wall_distance(source, next_pos)
            if targets:
                distance = min(_manhattan(next_pos, target) for target in targets)
                target_bias = -distance * 2
            else:
                target_bias = 0
            return (space * 3 + wall + target_bias, -distance if targets else 0, action)

        return max(legal, key=ranking)


__all__ = [
    "AggressivePlayer",
    "BaselinePlayer",
    "GreedyPlayer",
    "HybridPlayer",
    "RandomPlayer",
    "SurvivalPlayer",
]
