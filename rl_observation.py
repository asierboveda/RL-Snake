from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from board_state import FRUIT_KILL_THRESHOLD, BoardState, SnakeState
from tactical_planner import compute_tactical_features


OWN_BODY_CHANNEL = 0
OWN_HEAD_CHANNEL = 1
ENEMY_BODY_CHANNEL = 2
ENEMY_HEAD_CHANNEL = 3
FRUIT_CHANNEL = 4
WALL_CHANNEL = 5
DANGEROUS_ENEMY_CHANNEL = 6
ATTACKABLE_ENEMY_CHANNEL = 7
IMMEDIATE_DANGER_CHANNEL = 8
FREE_CHANNEL = 9


@dataclass(frozen=True)
class FeatureSet:
    version: str
    spatial_channel_names: Tuple[str, ...]
    feature_names: Tuple[str, ...]
    score_scale: float = float(FRUIT_KILL_THRESHOLD)
    turn_limit: int = 900


FEATURE_SET_V1 = FeatureSet(
    version="rl_observation_v1",
    spatial_channel_names=(
        "own_body",
        "own_head",
        "enemy_bodies",
        "enemy_heads",
        "fruits",
        "walls_or_bounds",
        "dangerous_enemies",
        "attackable_enemies",
        "immediate_danger",
        "free_navigable",
    ),
    feature_names=(
        "own_fruit_score_norm",
        "rival_0_fruit_score_norm",
        "rival_1_fruit_score_norm",
        "rival_2_fruit_score_norm",
        "rival_0_score_delta_norm",
        "rival_1_score_delta_norm",
        "rival_2_score_delta_norm",
        "own_can_kill",
        "rival_0_can_kill",
        "rival_1_can_kill",
        "rival_2_can_kill",
        "own_length_norm",
        "rival_0_length_norm",
        "rival_1_length_norm",
        "rival_2_length_norm",
        "rival_0_alive",
        "rival_1_alive",
        "rival_2_alive",
        "nearest_fruit_distance_norm",
        "nearest_dangerous_enemy_distance_norm",
        "nearest_attackable_enemy_distance_norm",
        "alive_snake_count_norm",
        "turn_norm",
        "direction_N",
        "direction_S",
        "direction_E",
        "direction_W",
        "danger_forward",
        "danger_left",
        "danger_right",
        "forward_safe",
        "left_safe",
        "right_safe",
        "best_fruit_distance",
        "best_fruit_action_forward",
        "best_fruit_action_left",
        "best_fruit_action_right",
        "forward_free_space",
        "left_free_space",
        "right_free_space",
        "attack_available",
        "best_attack_distance",
        "best_attack_action_forward",
        "best_attack_action_left",
        "best_attack_action_right",
        "strong_enemy_risk_forward",
        "strong_enemy_risk_left",
        "strong_enemy_risk_right",
        "head_to_head_risk_forward",
        "head_to_head_risk_left",
        "head_to_head_risk_right",
    ),
)


def build_observation(
    board: BoardState,
    player_id: int,
    feature_set: FeatureSet = FEATURE_SET_V1,
) -> Dict[str, np.ndarray]:
    own = _find_snake(board.snakes, player_id)
    if own is None:
        raise ValueError(f"player_id {player_id} is not present in BoardState")
    if not own.alive:
        raise ValueError("cannot build an RL observation for a dead player")

    spatial = _build_spatial(board, own, feature_set)
    features = _build_features(board, own, feature_set)
    return {"spatial": spatial, "features": features}


def _build_spatial(board: BoardState, own: SnakeState, feature_set: FeatureSet) -> np.ndarray:
    spatial = np.zeros((len(feature_set.spatial_channel_names), board.rows, board.cols), dtype=np.float32)

    _mark_perimeter(spatial[WALL_CHANNEL], board.rows, board.cols)

    own_head = _head_pos(own)
    _mark_cell(spatial[OWN_HEAD_CHANNEL], own_head)
    for cell in _body_cells_without_head(own):
        _mark_cell(spatial[OWN_BODY_CHANNEL], cell)

    occupied = set(own.occupied_cells())
    dangerous_enemy_cells = set()
    attackable_enemy_cells = set()

    for enemy in _rivals(board.snakes, own.player_id):
        if not enemy.alive:
            continue
        enemy_head = _head_pos(enemy)
        _mark_cell(spatial[ENEMY_HEAD_CHANNEL], enemy_head)
        occupied.add(enemy_head)
        for cell in _body_cells_without_head(enemy):
            _mark_cell(spatial[ENEMY_BODY_CHANNEL], cell)
            occupied.add(cell)

        threat_cells = set(enemy.occupied_cells())
        if _is_attackable(own, enemy):
            attackable_enemy_cells.update(threat_cells)
        else:
            dangerous_enemy_cells.update(threat_cells)

    for fruit in board.fruits:
        _mark_cell(spatial[FRUIT_CHANNEL], (fruit.row, fruit.col))

    for cell in dangerous_enemy_cells:
        _mark_cell(spatial[DANGEROUS_ENEMY_CHANNEL], cell)
    for cell in attackable_enemy_cells:
        _mark_cell(spatial[ATTACKABLE_ENEMY_CHANNEL], cell)

    immediate_danger = _immediate_danger_cells(board, own, occupied, dangerous_enemy_cells)
    for cell in immediate_danger:
        _mark_cell(spatial[IMMEDIATE_DANGER_CHANNEL], cell)

    for row in range(board.rows):
        for col in range(board.cols):
            if (row, col) not in occupied and (row, col) not in immediate_danger:
                spatial[FREE_CHANNEL, row, col] = 1.0

    return spatial


def _build_features(board: BoardState, own: SnakeState, feature_set: FeatureSet) -> np.ndarray:
    rivals = _rivals(board.snakes, own.player_id)
    rival_slots = list(rivals[:3])
    while len(rival_slots) < 3:
        rival_slots.append(None)

    dangerous = [enemy for enemy in rivals if enemy.alive and not _is_attackable(own, enemy)]
    attackable = [enemy for enemy in rivals if enemy.alive and _is_attackable(own, enemy)]

    tf = compute_tactical_features(board, player_id=own.player_id)

    values = [
        _score_norm(own.fruit_score, feature_set),
        *[_score_norm(enemy.fruit_score, feature_set) if enemy is not None else 0.0 for enemy in rival_slots],
        *[
            _score_norm(own.fruit_score - enemy.fruit_score, feature_set) if enemy is not None else 0.0
            for enemy in rival_slots
        ],
        float(own.is_hunter),
        *[float(enemy.is_hunter) if enemy is not None else 0.0 for enemy in rival_slots],
        _length_norm(own, board),
        *[_length_norm(enemy, board) if enemy is not None else 0.0 for enemy in rival_slots],
        *[float(enemy.alive) if enemy is not None else 0.0 for enemy in rival_slots],
        _nearest_distance_norm(_head_pos(own), [(fruit.row, fruit.col) for fruit in board.fruits], board),
        _nearest_distance_norm(_head_pos(own), [_head_pos(enemy) for enemy in dangerous], board),
        _nearest_distance_norm(_head_pos(own), [_head_pos(enemy) for enemy in attackable], board),
        _alive_count_norm(board.snakes),
        min(float(board.turn) / float(feature_set.turn_limit), 1.0),
        *_direction_one_hot(own),
        *_relative_danger(board, own),
        *tf.to_array(),
    ]
    return np.asarray(values, dtype=np.float32)


def _find_snake(snakes: Sequence[SnakeState], player_id: int) -> Optional[SnakeState]:
    for snake in snakes:
        if snake.player_id == player_id:
            return snake
    return None


def _rivals(snakes: Sequence[SnakeState], player_id: int) -> Tuple[SnakeState, ...]:
    return tuple(sorted((snake for snake in snakes if snake.player_id != player_id), key=lambda snake: snake.player_id))


def _is_attackable(own: SnakeState, enemy: SnakeState) -> bool:
    return own.is_hunter and own.fruit_score > enemy.fruit_score


def _head_pos(snake: SnakeState) -> Tuple[int, int]:
    row, col, _ = snake.head
    return row, col


def _body_cells_without_head(snake: SnakeState) -> Tuple[Tuple[int, int], ...]:
    return tuple((row, col) for row, col, _ in snake.body[1:])


def _mark_cell(channel: np.ndarray, cell: Tuple[int, int]) -> None:
    row, col = cell
    channel[row, col] = 1.0


def _mark_perimeter(channel: np.ndarray, rows: int, cols: int) -> None:
    channel[0, :] = 1.0
    channel[rows - 1, :] = 1.0
    channel[:, 0] = 1.0
    channel[:, cols - 1] = 1.0


def _immediate_danger_cells(
    board: BoardState,
    own: SnakeState,
    occupied: Iterable[Tuple[int, int]],
    dangerous_enemy_cells: Iterable[Tuple[int, int]],
) -> Tuple[Tuple[int, int], ...]:
    own_head = _head_pos(own)
    occupied_set = set(occupied)
    dangerous_set = set(dangerous_enemy_cells)
    danger = set(occupied_set)
    for row, col in dangerous_set:
        for candidate in _neighbors(row, col):
            if _inside(candidate, board):
                danger.add(candidate)
    for candidate in _neighbors(*own_head):
        if not _inside(candidate, board):
            continue
        if candidate in occupied_set or candidate in dangerous_set:
            danger.add(candidate)
    return tuple(sorted(danger))


def _neighbors(row: int, col: int) -> Tuple[Tuple[int, int], ...]:
    return ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))


def _inside(cell: Tuple[int, int], board: BoardState) -> bool:
    row, col = cell
    return 0 <= row < board.rows and 0 <= col < board.cols


def _score_norm(score: int, feature_set: FeatureSet) -> float:
    return float(score) / feature_set.score_scale


def _length_norm(snake: SnakeState, board: BoardState) -> float:
    return float(len(snake.body)) / float(max(board.rows, board.cols))


def _nearest_distance_norm(
    origin: Tuple[int, int],
    targets: Sequence[Tuple[int, int]],
    board: BoardState,
) -> float:
    if not targets:
        return 1.0
    distance = min(_manhattan(origin, target) for target in targets)
    return float(distance) / float(board.rows + board.cols)


def _manhattan(first: Tuple[int, int], second: Tuple[int, int]) -> int:
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def _alive_count_norm(snakes: Sequence[SnakeState]) -> float:
    return float(sum(1 for snake in snakes if snake.alive)) / float(len(snakes) or 1)


_DIRECTION_ORDER = ("N", "S", "E", "W")

_RELATIVE_ACTIONS = {
    "N": {"forward": "N", "left": "W", "right": "E", "back": "S"},
    "S": {"forward": "S", "left": "E", "right": "W", "back": "N"},
    "E": {"forward": "E", "left": "N", "right": "S", "back": "W"},
    "W": {"forward": "W", "left": "S", "right": "N", "back": "E"},
}


def _direction_one_hot(snake: SnakeState) -> Tuple[float, float, float, float]:
    direction = snake.head[2]
    return tuple(1.0 if d == direction else 0.0 for d in _DIRECTION_ORDER)


def _relative_danger(board: BoardState, own: SnakeState) -> Tuple[float, float, float]:
    direction = own.head[2]
    mapping = _RELATIVE_ACTIONS.get(direction, _RELATIVE_ACTIONS["N"])
    occupied = set()
    for snake in board.snakes:
        if not snake.alive:
            continue
        for cell in snake.occupied_cells():
            occupied.add(cell)
    dangers = []
    for rel in ("forward", "left", "right"):
        action = mapping[rel]
        dr, dc = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}[action]
        nr, nc = own.head[0] + dr, own.head[1] + dc
        if not (0 <= nr < board.rows and 0 <= nc < board.cols):
            dangers.append(1.0)
        elif (nr, nc) in occupied:
            dangers.append(1.0)
        else:
            dangers.append(0.0)
    return tuple(dangers)


__all__ = [
    "ATTACKABLE_ENEMY_CHANNEL",
    "DANGEROUS_ENEMY_CHANNEL",
    "ENEMY_BODY_CHANNEL",
    "ENEMY_HEAD_CHANNEL",
    "FEATURE_SET_V1",
    "FREE_CHANNEL",
    "FRUIT_CHANNEL",
    "FeatureSet",
    "IMMEDIATE_DANGER_CHANNEL",
    "OWN_BODY_CHANNEL",
    "OWN_HEAD_CHANNEL",
    "WALL_CHANNEL",
    "build_observation",
]
