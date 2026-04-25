"""Tactical planner for Snake RL using BFS over BoardState.

Provides pathfinding features to help PPO make safer, goal-directed decisions
without replacing the learning algorithm.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from board_state import BoardState, SnakeState


# Relative action mapping based on current absolute direction
_RELATIVE_MAP = {
    "N": {"forward": "N", "left": "W", "right": "E", "back": "S"},
    "S": {"forward": "S", "left": "E", "right": "W", "back": "N"},
    "E": {"forward": "E", "left": "N", "right": "S", "back": "W"},
    "W": {"forward": "W", "left": "S", "right": "N", "back": "E"},
}

_ACTION_DELTAS = {
    "N": (-1, 0),
    "S": (1, 0),
    "E": (0, 1),
    "W": (0, -1),
}

_RELATIVE_ORDER = ("forward", "left", "right")


def _head_pos(snake: SnakeState) -> Tuple[int, int]:
    row, col, _ = snake.head
    return row, col


def _occupied_grid(board: BoardState, exclude_tail: bool = True) -> np.ndarray:
    """Return a boolean grid marking occupied cells for all alive snakes.
    If exclude_tail, remove the last segment of every snake."""
    grid = np.zeros((board.rows, board.cols), dtype=bool)
    for snake in board.snakes:
        if not snake.alive:
            continue
        cells = list(snake.occupied_cells())
        if exclude_tail and len(cells) > 1:
            cells = cells[:-1]
        for r, c in cells:
            if 0 <= r < board.rows and 0 <= c < board.cols:
                grid[r, c] = True
    return grid


def _bfs_distances(grid: np.ndarray, start: Tuple[int, int]) -> np.ndarray:
    """BFS returning distance grid from start. Unreachable cells are -1."""
    rows, cols = grid.shape
    distances = np.full((rows, cols), -1, dtype=np.int32)
    if grid[start]:
        return distances
    queue = deque([start])
    distances[start] = 0
    while queue:
        r, c = queue.popleft()
        d = distances[r, c] + 1
        nr, nc = r - 1, c
        if nr >= 0 and not grid[nr, nc] and distances[nr, nc] == -1:
            distances[nr, nc] = d
            queue.append((nr, nc))
        nr, nc = r + 1, c
        if nr < rows and not grid[nr, nc] and distances[nr, nc] == -1:
            distances[nr, nc] = d
            queue.append((nr, nc))
        nr, nc = r, c - 1
        if nc >= 0 and not grid[nr, nc] and distances[nr, nc] == -1:
            distances[nr, nc] = d
            queue.append((nr, nc))
        nr, nc = r, c + 1
        if nc < cols and not grid[nr, nc] and distances[nr, nc] == -1:
            distances[nr, nc] = d
            queue.append((nr, nc))
    return distances


def _bfs_reachable_count(grid: np.ndarray, start: Tuple[int, int]) -> int:
    """Count reachable cells from start using BFS."""
    dists = _bfs_distances(grid, start)
    return int(np.count_nonzero(dists >= 0))


class TacticalFeatures:
    """Container for tactical features computed by the planner."""

    def __init__(
        self,
        forward_safe: bool,
        left_safe: bool,
        right_safe: bool,
        best_fruit_distance: float,
        best_fruit_action_forward: bool,
        best_fruit_action_left: bool,
        best_fruit_action_right: bool,
        forward_free_space: float,
        left_free_space: float,
        right_free_space: float,
        attack_available: bool,
        best_attack_distance: float,
        best_attack_action_forward: bool,
        best_attack_action_left: bool,
        best_attack_action_right: bool,
        strong_enemy_risk_forward: bool,
        strong_enemy_risk_left: bool,
        strong_enemy_risk_right: bool,
        head_to_head_risk_forward: bool,
        head_to_head_risk_left: bool,
        head_to_head_risk_right: bool,
    ):
        self.forward_safe = forward_safe
        self.left_safe = left_safe
        self.right_safe = right_safe
        self.best_fruit_distance = best_fruit_distance
        self.best_fruit_action_forward = best_fruit_action_forward
        self.best_fruit_action_left = best_fruit_action_left
        self.best_fruit_action_right = best_fruit_action_right
        self.forward_free_space = forward_free_space
        self.left_free_space = left_free_space
        self.right_free_space = right_free_space
        self.attack_available = attack_available
        self.best_attack_distance = best_attack_distance
        self.best_attack_action_forward = best_attack_action_forward
        self.best_attack_action_left = best_attack_action_left
        self.best_attack_action_right = best_attack_action_right
        self.strong_enemy_risk_forward = strong_enemy_risk_forward
        self.strong_enemy_risk_left = strong_enemy_risk_left
        self.strong_enemy_risk_right = strong_enemy_risk_right
        self.head_to_head_risk_forward = head_to_head_risk_forward
        self.head_to_head_risk_left = head_to_head_risk_left
        self.head_to_head_risk_right = head_to_head_risk_right

    def to_array(self) -> Tuple[float, ...]:
        return (
            float(self.forward_safe),
            float(self.left_safe),
            float(self.right_safe),
            self.best_fruit_distance,
            float(self.best_fruit_action_forward),
            float(self.best_fruit_action_left),
            float(self.best_fruit_action_right),
            self.forward_free_space,
            self.left_free_space,
            self.right_free_space,
            float(self.attack_available),
            self.best_attack_distance,
            float(self.best_attack_action_forward),
            float(self.best_attack_action_left),
            float(self.best_attack_action_right),
            float(self.strong_enemy_risk_forward),
            float(self.strong_enemy_risk_left),
            float(self.strong_enemy_risk_right),
            float(self.head_to_head_risk_forward),
            float(self.head_to_head_risk_left),
            float(self.head_to_head_risk_right),
        )


def compute_tactical_features(board: BoardState, player_id: int) -> TacticalFeatures:
    """Compute tactical features for the given player using BFS."""
    own = None
    for snake in board.snakes:
        if snake.player_id == player_id:
            own = snake
            break
    if own is None or not own.alive:
        return TacticalFeatures(
            forward_safe=False, left_safe=False, right_safe=False,
            best_fruit_distance=1.0,
            best_fruit_action_forward=False, best_fruit_action_left=False, best_fruit_action_right=False,
            forward_free_space=0.0, left_free_space=0.0, right_free_space=0.0,
            attack_available=False, best_attack_distance=1.0,
            best_attack_action_forward=False, best_attack_action_left=False, best_attack_action_right=False,
            strong_enemy_risk_forward=False, strong_enemy_risk_left=False, strong_enemy_risk_right=False,
            head_to_head_risk_forward=False, head_to_head_risk_left=False, head_to_head_risk_right=False,
        )

    direction = own.head[2]
    rel_map = _RELATIVE_MAP.get(direction, _RELATIVE_MAP["N"])
    head = _head_pos(own)

    # Optimistic grid (tails excluded) for BFS
    grid_opt = _occupied_grid(board, exclude_tail=True)
    grid_opt[head] = False  # allow BFS to start from head

    # Pessimistic grid (tails included) for immediate safety
    grid_pes = _occupied_grid(board, exclude_tail=False)

    # Evaluate each relative action
    rel_positions: Dict[str, Tuple[int, int]] = {}
    rel_safe: Dict[str, bool] = {}
    rel_free_space: Dict[str, float] = {}
    for rel in _RELATIVE_ORDER:
        action = rel_map[rel]
        dr, dc = _ACTION_DELTAS[action]
        pos = (head[0] + dr, head[1] + dc)
        rel_positions[rel] = pos
        safe = (
            0 <= pos[0] < board.rows
            and 0 <= pos[1] < board.cols
            and not grid_pes[pos]
        )
        rel_safe[rel] = safe
        if safe:
            space = _bfs_reachable_count(grid_opt, pos)
            rel_free_space[rel] = min(space / (board.rows * board.cols), 1.0)
        else:
            rel_free_space[rel] = 0.0

    # Fruit distances via BFS from head
    fruit_positions = [(f.row, f.col) for f in board.fruits]
    best_fruit_dist = 1.0
    best_fruit_actions: Dict[str, bool] = {rel: False for rel in _RELATIVE_ORDER}

    if fruit_positions:
        distances = _bfs_distances(grid_opt, head)
        min_dist = None
        best_fruit = None
        for fp in fruit_positions:
            d = distances[fp]
            if d >= 0:
                if min_dist is None or d < min_dist:
                    min_dist = d
                    best_fruit = fp
        if min_dist is not None:
            best_fruit_dist = min(min_dist / (board.rows + board.cols), 1.0)
            for rel in _RELATIVE_ORDER:
                pos = rel_positions[rel]
                if rel_safe[rel]:
                    dist_from_here = abs(head[0] - best_fruit[0]) + abs(head[1] - best_fruit[1])
                    dist_from_there = abs(pos[0] - best_fruit[0]) + abs(pos[1] - best_fruit[1])
                    best_fruit_actions[rel] = dist_from_there < dist_from_here

    # Attack features (only if hunter)
    attackable_heads: List[Tuple[int, int]] = []
    for snake in board.snakes:
        if snake.alive and snake.player_id != player_id:
            if own.is_hunter and own.fruit_score > snake.fruit_score:
                attackable_heads.append(_head_pos(snake))

    attack_available = len(attackable_heads) > 0
    best_attack_dist = 1.0
    best_attack_actions: Dict[str, bool] = {rel: False for rel in _RELATIVE_ORDER}

    if attack_available:
        grid_attack = grid_opt.copy()
        for ep in attackable_heads:
            grid_attack[ep] = False
        distances = _bfs_distances(grid_attack, head)
        min_dist = None
        best_target = None
        for ep in attackable_heads:
            d = distances[ep]
            if d >= 0:
                if min_dist is None or d < min_dist:
                    min_dist = d
                    best_target = ep
        if min_dist is not None:
            best_attack_dist = min(min_dist / (board.rows + board.cols), 1.0)
            for rel in _RELATIVE_ORDER:
                pos = rel_positions[rel]
                if rel_safe[rel] and best_target is not None:
                    dist_from_here = abs(head[0] - best_target[0]) + abs(head[1] - best_target[1])
                    dist_from_there = abs(pos[0] - best_target[0]) + abs(pos[1] - best_target[1])
                    best_attack_actions[rel] = dist_from_there < dist_from_here

    # Strong enemy risk
    strong_cells = set()
    for snake in board.snakes:
        if not snake.alive or snake.player_id == player_id:
            continue
        if own.is_hunter and own.fruit_score > snake.fruit_score:
            continue
        for r, c in snake.occupied_cells():
            strong_cells.add((r, c))
    strong_risk = {rel: rel_positions[rel] in strong_cells for rel in _RELATIVE_ORDER}

    # Head-to-head risk
    enemy_heads = []
    for snake in board.snakes:
        if snake.alive and snake.player_id != player_id:
            enemy_heads.append((_head_pos(snake), snake))
    h2h_risk = {rel: False for rel in _RELATIVE_ORDER}
    for rel in _RELATIVE_ORDER:
        pos = rel_positions[rel]
        for eh, snake in enemy_heads:
            if abs(pos[0] - eh[0]) + abs(pos[1] - eh[1]) <= 1:
                if not (own.is_hunter and own.fruit_score > snake.fruit_score):
                    h2h_risk[rel] = True
                    break

    return TacticalFeatures(
        forward_safe=rel_safe["forward"],
        left_safe=rel_safe["left"],
        right_safe=rel_safe["right"],
        best_fruit_distance=best_fruit_dist,
        best_fruit_action_forward=best_fruit_actions["forward"],
        best_fruit_action_left=best_fruit_actions["left"],
        best_fruit_action_right=best_fruit_actions["right"],
        forward_free_space=rel_free_space["forward"],
        left_free_space=rel_free_space["left"],
        right_free_space=rel_free_space["right"],
        attack_available=attack_available,
        best_attack_distance=best_attack_dist,
        best_attack_action_forward=best_attack_actions["forward"],
        best_attack_action_left=best_attack_actions["left"],
        best_attack_action_right=best_attack_actions["right"],
        strong_enemy_risk_forward=strong_risk["forward"],
        strong_enemy_risk_left=strong_risk["left"],
        strong_enemy_risk_right=strong_risk["right"],
        head_to_head_risk_forward=h2h_risk["forward"],
        head_to_head_risk_left=h2h_risk["left"],
        head_to_head_risk_right=h2h_risk["right"],
    )


def recommended_fruit_action(board: BoardState, player_id: int) -> Optional[str]:
    """Return the relative action (forward/left/right) that best approaches the nearest safe fruit, or None."""
    tf = compute_tactical_features(board, player_id)
    if tf.best_fruit_action_forward and tf.forward_safe:
        return "forward"
    if tf.best_fruit_action_left and tf.left_safe:
        return "left"
    if tf.best_fruit_action_right and tf.right_safe:
        return "right"
    return None


def recommended_attack_action(board: BoardState, player_id: int) -> Optional[str]:
    """Return the relative action that best approaches an attackable enemy, or None."""
    tf = compute_tactical_features(board, player_id)
    if not tf.attack_available:
        return None
    if tf.best_attack_action_forward and tf.forward_safe:
        return "forward"
    if tf.best_attack_action_left and tf.left_safe:
        return "left"
    if tf.best_attack_action_right and tf.right_safe:
        return "right"
    return None
