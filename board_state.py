from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple


BOARD_ROWS = 44
BOARD_COLS = 44
PLAYER_LABELS = ("A", "B", "C", "D")
PLAYER_COLORS = ("G", "B", "R", "Y")
VALID_DIRECTIONS = ("N", "S", "E", "W")
FRUIT_KILL_THRESHOLD = 120
KILL_SCORE = 30
TERMINAL_REASONS = (
    "single_alive",
    "all_dead",
    "too_few_points",
    "score_threshold",
    "draw",
)

BodyCell = Tuple[int, int, str]


@dataclass(frozen=True)
class FruitState:
    row: int
    col: int
    value: int
    time_left: int

    def __post_init__(self):
        if self.value not in (10, 15, 20):
            raise ValueError("fruit value must be one of 10, 15, 20")
        if self.time_left < 0:
            raise ValueError("fruit time_left must be non-negative")

    def to_dict(self) -> Dict[str, int]:
        return {
            "row": self.row,
            "col": self.col,
            "value": self.value,
            "time_left": self.time_left,
        }


@dataclass(frozen=True)
class SnakeState:
    player_id: int
    label: str
    color: str
    alive: bool
    body: Tuple[BodyCell, ...]
    score: int
    fruit_score: int

    def __post_init__(self):
        if self.player_id < 0:
            raise ValueError("player_id must be non-negative")
        if self.label not in PLAYER_LABELS:
            raise ValueError(f"label must be one of {PLAYER_LABELS}")
        if self.color not in PLAYER_COLORS:
            raise ValueError(f"color must be one of {PLAYER_COLORS}")
        if not self.body:
            raise ValueError("body must contain at least the head")
        if self.score < 0 or self.fruit_score < 0:
            raise ValueError("scores must be non-negative")
        if self.fruit_score > self.score:
            raise ValueError("fruit_score cannot exceed total score")
        for row, col, direction in self.body:
            if direction not in VALID_DIRECTIONS:
                raise ValueError(f"direction must be one of {VALID_DIRECTIONS}")

    @property
    def head(self) -> BodyCell:
        return self.body[0]

    @property
    def is_hunter(self) -> bool:
        return self.fruit_score >= FRUIT_KILL_THRESHOLD

    def occupied_cells(self) -> Tuple[Tuple[int, int], ...]:
        return tuple((row, col) for row, col, _ in self.body)

    def overlaps(self, other: "SnakeState") -> bool:
        return bool(set(self.occupied_cells()) & set(other.occupied_cells()))

    def to_dict(self) -> Dict[str, object]:
        return {
            "player_id": self.player_id,
            "label": self.label,
            "color": self.color,
            "alive": self.alive,
            "head": self.head,
            "body": list(self.body),
            "score": self.score,
            "fruit_score": self.fruit_score,
            "is_hunter": self.is_hunter,
        }


@dataclass(frozen=True)
class CollisionOutcome:
    dead_ids: Tuple[int, ...]
    killer_id: Optional[int]
    points_awarded: int


@dataclass(frozen=True)
class BoardState:
    turn: int
    rows: int
    cols: int
    snakes: Sequence[SnakeState]
    fruits: Sequence[FruitState]
    game_alive: bool
    winner_id: Optional[int]
    terminal_reason: Optional[str]

    def __post_init__(self):
        if self.turn < 0:
            raise ValueError("turn must be non-negative")
        if self.rows != BOARD_ROWS or self.cols != BOARD_COLS:
            raise ValueError(f"board dimensions must be {BOARD_ROWS}x{BOARD_COLS}")
        ids = [snake.player_id for snake in self.snakes]
        if len(ids) != len(set(ids)):
            raise ValueError("snake player_id values must be unique")
        for snake in self.snakes:
            for row, col in snake.occupied_cells():
                if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
                    raise ValueError("snake body cells must be inside the board")
        for fruit in self.fruits:
            if fruit.row < 0 or fruit.row >= self.rows or fruit.col < 0 or fruit.col >= self.cols:
                raise ValueError("fruit cells must be inside the board")
        if self.game_alive and self.terminal_reason is not None:
            raise ValueError("terminal_reason must be None while game_alive is True")
        if not self.game_alive and self.terminal_reason not in TERMINAL_REASONS:
            raise ValueError(f"terminal_reason must be one of {TERMINAL_REASONS} when the game is over")
        if self.winner_id is not None and self.winner_id not in ids:
            raise ValueError("winner_id must reference an existing snake")

    def to_dict(self) -> Dict[str, object]:
        return {
            "turn": self.turn,
            "rows": self.rows,
            "cols": self.cols,
            "snakes": [snake.to_dict() for snake in self.snakes],
            "fruits": [fruit.to_dict() for fruit in self.fruits],
            "game_alive": self.game_alive,
            "winner_id": self.winner_id,
            "terminal_reason": self.terminal_reason,
        }


def resolve_collision(first: SnakeState, second: SnakeState) -> CollisionOutcome:
    if not first.overlaps(second):
        return CollisionOutcome(dead_ids=(), killer_id=None, points_awarded=0)

    if first.fruit_score == second.fruit_score:
        return CollisionOutcome(
            dead_ids=tuple(sorted((first.player_id, second.player_id))),
            killer_id=None,
            points_awarded=0,
        )

    if not first.is_hunter and not second.is_hunter:
        return CollisionOutcome(
            dead_ids=tuple(sorted((first.player_id, second.player_id))),
            killer_id=None,
            points_awarded=0,
        )

    killer, victim = (first, second) if first.fruit_score > second.fruit_score else (second, first)
    return CollisionOutcome(dead_ids=(victim.player_id,), killer_id=killer.player_id, points_awarded=KILL_SCORE)


def determine_winner(snakes: Sequence[SnakeState]) -> Tuple[Optional[int], str]:
    alive = [snake for snake in snakes if snake.alive]
    if len(alive) == 1:
        return alive[0].player_id, "single_alive"
    if len(alive) == 0:
        return None, "all_dead"

    max_score = max(snake.score for snake in alive)
    if max_score < FRUIT_KILL_THRESHOLD:
        return None, "too_few_points"

    winners = [snake for snake in alive if snake.score == max_score]
    if len(winners) == 1:
        return winners[0].player_id, "score_threshold"
    return None, "draw"
