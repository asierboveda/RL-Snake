from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

from board_state import (
    BOARD_COLS,
    BOARD_ROWS,
    PLAYER_COLORS,
    PLAYER_LABELS,
    VALID_DIRECTIONS,
    BoardState,
    FruitState,
    SnakeState,
    determine_winner,
)
from vision_fruits import FruitsState, detect_fruits
from vision_grid import detect_grid_geometry
from vision_hud import HUDState, detect_hud
from vision_snakes import SnakesState, detect_snakes


DEFAULT_DEAD_BODY = ((0, 0, "N"),)


@dataclass(frozen=True)
class VisionParseResult:
    board_state: BoardState
    confidence: float
    component_confidence: Dict[str, float]
    errors: Tuple[str, ...]
    warnings: Tuple[str, ...]
    metadata: Dict[str, object]
    components: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "board_state": self.board_state.to_dict(),
            "confidence": self.confidence,
            "component_confidence": self.component_confidence,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "metadata": self.metadata,
            "components": self.components,
        }


class VisionParser:
    def parse(self, image) -> VisionParseResult:
        geometry = detect_grid_geometry(image)
        hud_state = detect_hud(image)
        snakes_state = detect_snakes(image)
        fruits_state = detect_fruits(image)

        errors: list[str] = []
        warnings: list[str] = []

        snakes, fruit_score_bounds = _build_snakes(hud_state, snakes_state, warnings)
        fruits = _build_fruits(fruits_state)
        _validate_consistency(snakes, fruits, errors, warnings)

        board_state = _build_board_state(hud_state.turn, snakes, fruits)
        component_confidence = _component_confidence(snakes_state, fruits_state, errors, warnings)
        confidence = round(
            0.25 * component_confidence["hud"]
            + 0.45 * component_confidence["snakes"]
            + 0.20 * component_confidence["fruits"]
            + 0.10 * component_confidence["consistency"],
            6,
        )

        metadata = {
            "geometry": geometry.to_dict(),
            "board_bbox": list(geometry.board_bbox),
            "hud_bbox": list(geometry.hud_bbox),
            "fruit_score_policy": "lower_bound_from_length_and_total_score",
            "fruit_score_bounds": fruit_score_bounds,
            "detector_thresholds": {
                "snakes": {"match_threshold": snakes_state.match_threshold},
                "fruits": {
                    "match_threshold": fruits_state.match_threshold,
                    "margin_ratio": fruits_state.margin_ratio,
                    "margin_delta": fruits_state.margin_delta,
                },
            },
        }
        components = {
            "hud": hud_state.to_dict(),
            "snakes": snakes_state.to_dict(),
            "fruits": fruits_state.to_dict(),
        }

        return VisionParseResult(
            board_state=board_state,
            confidence=confidence,
            component_confidence=component_confidence,
            errors=tuple(errors),
            warnings=tuple(warnings),
            metadata=metadata,
            components=components,
        )


def _build_snakes(
    hud_state: HUDState,
    snakes_state: SnakesState,
    warnings: list[str],
) -> Tuple[Tuple[SnakeState, ...], Dict[str, Dict[str, int]]]:
    snakes = []
    fruit_score_bounds: Dict[str, Dict[str, int]] = {}

    for player_id, color in enumerate(PLAYER_COLORS):
        label = PLAYER_LABELS[player_id]
        player = snakes_state.players.get(color)
        score = int(hud_state.scores.get(color, 0))

        if player is None or not player.board_body:
            alive = False
            body = DEFAULT_DEAD_BODY
            if score > 0:
                warnings.append(f"player {color} has score {score} but no snake body detected")
            lower_bound = 0
        else:
            alive = True
            body = tuple(
                (row, col, _canonical_direction(direction))
                for row, col, direction in player.board_body
            )
            lower_bound = min(score, max(0, (len(body) - 1) * 10))
            if player.direction is None:
                warnings.append(f"player {color} detected without head direction")

        fruit_score_bounds[color] = {"lower": lower_bound, "upper": score}
        snakes.append(
            SnakeState(
                player_id=player_id,
                label=label,
                color=color,
                alive=alive,
                body=body,
                score=score,
                fruit_score=lower_bound,
            )
        )

    return tuple(snakes), fruit_score_bounds


def _build_fruits(fruits_state: FruitsState) -> Tuple[FruitState, ...]:
    return tuple(
        FruitState(
            row=fruit.row,
            col=fruit.col,
            value=fruit.value,
            time_left=0,
        )
        for fruit in fruits_state.fruits
    )


def _build_board_state(
    turn: int,
    snakes: Sequence[SnakeState],
    fruits: Sequence[FruitState],
) -> BoardState:
    game_alive = sum(1 for snake in snakes if snake.alive) > 1
    if game_alive:
        winner_id = None
        terminal_reason = None
    else:
        winner_id, terminal_reason = determine_winner(snakes)
    return BoardState(
        turn=turn,
        rows=BOARD_ROWS,
        cols=BOARD_COLS,
        snakes=tuple(snakes),
        fruits=tuple(fruits),
        game_alive=game_alive,
        winner_id=winner_id,
        terminal_reason=terminal_reason,
    )


def _validate_consistency(
    snakes: Sequence[SnakeState],
    fruits: Sequence[FruitState],
    errors: list[str],
    warnings: list[str],
) -> None:
    occupied: Dict[Tuple[int, int], str] = {}
    for snake in snakes:
        if not snake.alive:
            continue
        for row, col in snake.occupied_cells():
            owner = occupied.get((row, col))
            if owner is not None and owner != snake.color:
                errors.append(
                    f"snake overlap detected at cell ({row}, {col}) between {owner} and {snake.color}"
                )
            else:
                occupied[(row, col)] = snake.color

    fruit_cells = set()
    for fruit in fruits:
        cell = (fruit.row, fruit.col)
        if cell in fruit_cells:
            errors.append(f"duplicate fruit detected at cell ({fruit.row}, {fruit.col})")
            continue
        fruit_cells.add(cell)
        if cell in occupied:
            warnings.append(
                f"fruit overlaps with snake {occupied[cell]} at cell ({fruit.row}, {fruit.col})"
            )


def _component_confidence(
    snakes_state: SnakesState,
    fruits_state: FruitsState,
    errors: Sequence[str],
    warnings: Sequence[str],
) -> Dict[str, float]:
    snakes_confidence = (
        sum(segment.confidence for segment in snakes_state.segments) / len(snakes_state.segments)
        if snakes_state.segments
        else 0.0
    )
    fruits_confidence = (
        sum(fruit.confidence for fruit in fruits_state.fruits) / len(fruits_state.fruits)
        if fruits_state.fruits
        else 1.0
    )
    consistency = max(0.0, 1.0 - 0.25 * len(errors) - 0.05 * len(warnings))

    return {
        "hud": 1.0,
        "snakes": round(snakes_confidence, 6),
        "fruits": round(fruits_confidence, 6),
        "consistency": round(consistency, 6),
    }


def _canonical_direction(direction: str) -> str:
    if direction in VALID_DIRECTIONS:
        return direction
    for char in direction:
        if char in VALID_DIRECTIONS:
            return char
    return "N"
