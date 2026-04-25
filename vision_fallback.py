from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from board_state import BOARD_COLS, BOARD_ROWS, PLAYER_COLORS, VALID_DIRECTIONS, BoardState
from vision_parser import VisionParseResult


ACTION_ORDER = ("N", "E", "S", "W")
ACTION_DELTAS = {
    "N": (-1, 0),
    "S": (1, 0),
    "E": (0, 1),
    "W": (0, -1),
}

MODE_TRUST = "trust"
MODE_CONSERVATIVE = "conservative"
MODE_REUSE_LAST_RELIABLE = "reuse_last_reliable"
MODE_SAFE_ACTION = "safe_action"
MODE_DROP_FRAME = "drop_frame"


@dataclass(frozen=True)
class VisionFallbackThresholds:
    trust_confidence_min: float = 0.998
    reject_confidence_max: float = 0.995
    trust_snakes_confidence_min: float = 0.998
    trust_fruits_confidence_min: float = 0.998
    consistency_confidence_min: float = 0.95
    max_warnings_for_trust: int = 0
    max_warnings_for_conservative: int = 2
    max_turn_gap_for_conservative: int = 3
    max_score_jump_per_turn: int = 60


@dataclass(frozen=True)
class VisionFallbackDecision:
    mode: str
    accepted: bool
    board_state: Optional[BoardState]
    conservative_mode: bool
    request_next_frame: bool
    force_safe_action: bool
    safe_action: Optional[str]
    reasons: Tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "accepted": self.accepted,
            "conservative_mode": self.conservative_mode,
            "request_next_frame": self.request_next_frame,
            "force_safe_action": self.force_safe_action,
            "safe_action": self.safe_action,
            "reasons": list(self.reasons),
            "board_state": None if self.board_state is None else self.board_state.to_dict(),
        }


class VisionFallbackPolicy:
    def __init__(self, thresholds: VisionFallbackThresholds | None = None):
        self.thresholds = thresholds or VisionFallbackThresholds()
        self._last_reliable_state: Optional[BoardState] = None

    @property
    def last_reliable_state(self) -> Optional[BoardState]:
        return self._last_reliable_state

    def evaluate(
        self,
        parse_result: VisionParseResult,
        *,
        snake_id: int,
        last_action: str = "N",
    ) -> VisionFallbackDecision:
        critical_reasons = self._critical_reasons(parse_result, snake_id=snake_id)
        if critical_reasons:
            return self._critical_decision(
                parse_result=parse_result,
                snake_id=snake_id,
                last_action=last_action,
                reasons=critical_reasons,
            )

        conservative_reasons = self._conservative_reasons(parse_result)
        if conservative_reasons:
            return VisionFallbackDecision(
                mode=MODE_CONSERVATIVE,
                accepted=True,
                board_state=parse_result.board_state,
                conservative_mode=True,
                request_next_frame=False,
                force_safe_action=False,
                safe_action=None,
                reasons=tuple(conservative_reasons),
            )

        self._last_reliable_state = parse_result.board_state
        return VisionFallbackDecision(
            mode=MODE_TRUST,
            accepted=True,
            board_state=parse_result.board_state,
            conservative_mode=False,
            request_next_frame=False,
            force_safe_action=False,
            safe_action=None,
            reasons=(),
        )

    def _critical_reasons(self, parse_result: VisionParseResult, *, snake_id: int) -> list[str]:
        reasons: list[str] = []
        board = parse_result.board_state
        thresholds = self.thresholds
        component_confidence = parse_result.component_confidence
        my_color = _snake_color(snake_id)
        my_snake = _find_snake(board, snake_id)

        if parse_result.errors:
            reasons.append("parser-errors")
        if parse_result.confidence < thresholds.reject_confidence_max:
            reasons.append("confidence-below-reject")
        if component_confidence.get("consistency", 1.0) < thresholds.consistency_confidence_min:
            reasons.append("consistency-confidence-low")
        if len(parse_result.warnings) > thresholds.max_warnings_for_conservative:
            reasons.append("too-many-warnings")
        if any(
            warning.startswith(f"player {my_color} has score") and "no snake body detected" in warning
            for warning in parse_result.warnings
        ):
            reasons.append("critical-warning-own-body-missing")
        if my_snake is None:
            reasons.append("own-snake-missing")
        elif my_snake.alive and my_snake.head[2] not in VALID_DIRECTIONS:
            reasons.append("own-head-invalid")

        if self._last_reliable_state is not None:
            if board.turn < self._last_reliable_state.turn:
                reasons.append("turn-regression")
            if my_snake is not None:
                last_my_snake = _find_snake(self._last_reliable_state, snake_id)
                if last_my_snake is not None:
                    turn_delta = max(1, board.turn - self._last_reliable_state.turn)
                    score_jump = my_snake.score - last_my_snake.score
                    if score_jump > thresholds.max_score_jump_per_turn * turn_delta:
                        reasons.append("score-jump-too-large")

        return reasons

    def _conservative_reasons(self, parse_result: VisionParseResult) -> list[str]:
        reasons: list[str] = []
        thresholds = self.thresholds
        component_confidence = parse_result.component_confidence

        if parse_result.confidence < thresholds.trust_confidence_min:
            reasons.append("confidence-below-trust")
        if component_confidence.get("snakes", 1.0) < thresholds.trust_snakes_confidence_min:
            reasons.append("snakes-confidence-low")
        if component_confidence.get("fruits", 1.0) < thresholds.trust_fruits_confidence_min:
            reasons.append("fruits-confidence-low")
        if len(parse_result.warnings) > thresholds.max_warnings_for_trust:
            reasons.append("warnings-present")
        if any("fruit overlaps with snake" in warning for warning in parse_result.warnings):
            reasons.append("fruit-overlap-warning")

        if self._last_reliable_state is not None:
            turn_gap = parse_result.board_state.turn - self._last_reliable_state.turn
            if turn_gap > thresholds.max_turn_gap_for_conservative:
                reasons.append("turn-gap-large")

        return reasons

    def _critical_decision(
        self,
        *,
        parse_result: VisionParseResult,
        snake_id: int,
        last_action: str,
        reasons: Sequence[str],
    ) -> VisionFallbackDecision:
        if self._last_reliable_state is not None:
            return VisionFallbackDecision(
                mode=MODE_REUSE_LAST_RELIABLE,
                accepted=True,
                board_state=self._last_reliable_state,
                conservative_mode=True,
                request_next_frame=True,
                force_safe_action=False,
                safe_action=None,
                reasons=tuple(reasons),
            )

        safe_action = choose_safe_action(
            parse_result.board_state,
            snake_id=snake_id,
            preferred_action=last_action,
        )
        if safe_action is None:
            return VisionFallbackDecision(
                mode=MODE_DROP_FRAME,
                accepted=False,
                board_state=None,
                conservative_mode=True,
                request_next_frame=True,
                force_safe_action=False,
                safe_action=None,
                reasons=tuple(reasons) + ("no-safe-action-available",),
            )

        return VisionFallbackDecision(
            mode=MODE_SAFE_ACTION,
            accepted=False,
            board_state=None,
            conservative_mode=True,
            request_next_frame=True,
            force_safe_action=True,
            safe_action=safe_action,
            reasons=tuple(reasons),
        )


def choose_safe_action(
    board_state: BoardState,
    *,
    snake_id: int,
    preferred_action: str = "N",
) -> Optional[str]:
    snake = _find_snake(board_state, snake_id)
    if snake is None or not snake.alive:
        return preferred_action if preferred_action in ACTION_DELTAS else "N"

    head_row, head_col, _ = snake.head
    occupied = {
        (row, col)
        for other in board_state.snakes
        if other.alive
        for row, col in other.occupied_cells()
    }

    safe_actions = []
    for action in ACTION_ORDER:
        delta_row, delta_col = ACTION_DELTAS[action]
        next_row = head_row + delta_row
        next_col = head_col + delta_col
        if next_row < 0 or next_row >= BOARD_ROWS or next_col < 0 or next_col >= BOARD_COLS:
            continue
        if (next_row, next_col) in occupied:
            continue
        safe_actions.append(action)

    if not safe_actions:
        return preferred_action if preferred_action in ACTION_DELTAS else "N"
    if preferred_action in safe_actions:
        return preferred_action
    return max(
        safe_actions,
        key=lambda action: (
            _enemy_distance_score(board_state, snake_id, head_row, head_col, action),
            -ACTION_ORDER.index(action),
        ),
    )


def _enemy_distance_score(
    board_state: BoardState,
    snake_id: int,
    head_row: int,
    head_col: int,
    action: str,
) -> int:
    delta_row, delta_col = ACTION_DELTAS[action]
    next_row = head_row + delta_row
    next_col = head_col + delta_col

    enemy_heads = [
        (snake.head[0], snake.head[1])
        for snake in board_state.snakes
        if snake.player_id != snake_id and snake.alive
    ]
    if not enemy_heads:
        return BOARD_ROWS + BOARD_COLS
    return min(abs(next_row - row) + abs(next_col - col) for row, col in enemy_heads)


def _find_snake(board_state: BoardState, snake_id: int):
    for snake in board_state.snakes:
        if snake.player_id == snake_id:
            return snake
    return None


def _snake_color(snake_id: int) -> str:
    if snake_id < 0 or snake_id >= len(PLAYER_COLORS):
        raise ValueError(f"snake_id must be in 0..{len(PLAYER_COLORS)-1}")
    return PLAYER_COLORS[snake_id]
