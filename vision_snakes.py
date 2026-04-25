from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from vision_match import best_template, confidence, image_pixels, load_template
from vision_grid import CELL_SIZE, GridGeometry, detect_grid_geometry


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "input" / str(CELL_SIZE)

PLAYER_ORDER = ("G", "B", "R", "Y")
CLASS_ORDER = ("snake_head", "snake_body", "snake_tail")
CHAIN_CLASS_PRIORITY = {"snake_head": 0, "snake_body": 1, "snake_tail": 2}
SNAKE_MATCH_MAX_ERROR = 0.02
SNAKE_MATCH_MARGIN_RATIO = 0.4
NON_SNAKE_TEMPLATES = ("bomb", "fruit10", "fruit15", "fruit20")


@dataclass(frozen=True)
class SnakeSegment:
    segment_class: str
    player: str
    row: int
    col: int
    direction: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    error: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "class": self.segment_class,
            "player": self.player,
            "cell": {"row": self.row, "col": self.col},
            "bbox": list(self.bbox),
            "direction": self.direction,
            "confidence": round(self.confidence, 6),
        }


@dataclass(frozen=True)
class PlayerSnake:
    player: str
    segments: Tuple[SnakeSegment, ...]
    ordered_segments: Tuple[SnakeSegment, ...]

    @property
    def head(self) -> Optional[SnakeSegment]:
        return next((segment for segment in self.segments if segment.segment_class == "snake_head"), None)

    @property
    def body(self) -> Tuple[SnakeSegment, ...]:
        return tuple(segment for segment in self.segments if segment.segment_class == "snake_body")

    @property
    def tail(self) -> Optional[SnakeSegment]:
        return next((segment for segment in self.segments if segment.segment_class == "snake_tail"), None)

    @property
    def direction(self) -> Optional[str]:
        if self.head is not None:
            return self.head.direction
        if self.ordered_segments:
            return self.ordered_segments[0].direction
        return None

    @property
    def confidence(self) -> float:
        if not self.segments:
            return 0.0
        return float(np.mean([segment.confidence for segment in self.segments]))

    @property
    def board_body(self) -> Tuple[Tuple[int, int, str], ...]:
        return tuple((segment.row, segment.col, segment.direction) for segment in self.ordered_segments)

    def to_dict(self) -> Dict[str, object]:
        return {
            "player": self.player,
            "direction": self.direction,
            "length": len(self.segments),
            "confidence": round(self.confidence, 6),
            "head": self.head.to_dict() if self.head else None,
            "body": [segment.to_dict() for segment in self.body],
            "tail": self.tail.to_dict() if self.tail else None,
            "board_body": [list(cell) for cell in self.board_body],
        }


@dataclass(frozen=True)
class SnakesState:
    geometry: GridGeometry
    segments: Tuple[SnakeSegment, ...]
    players: Dict[str, PlayerSnake]
    match_threshold: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "snakes": [segment.to_dict() for segment in self.segments],
            "players": [self.players[player].to_dict() for player in PLAYER_ORDER if player in self.players],
            "match_threshold": self.match_threshold,
        }


def detect_snakes(
    image,
    *,
    match_threshold: float = SNAKE_MATCH_MAX_ERROR,
    margin_ratio: float = SNAKE_MATCH_MARGIN_RATIO,
) -> SnakesState:
    geometry = detect_grid_geometry(image)
    pixels = image_pixels(image)
    snake_templates = _snake_templates()
    other_templates = _other_templates()

    segments = []
    for row in range(geometry.rows):
        for col in range(geometry.cols):
            x, y, _, _ = geometry.cell_bbox(row, col)
            crop = pixels[y:y + CELL_SIZE, x:x + CELL_SIZE, :]

            key, snake_error = best_template(crop, snake_templates)
            _, other_error = best_template(crop, other_templates)

            if snake_error > match_threshold:
                continue
            if snake_error >= other_error * margin_ratio:
                continue

            segment_class, player, direction = key
            segment_confidence = confidence(snake_error, other_error)
            segments.append(
                SnakeSegment(
                    segment_class=segment_class,
                    player=player,
                    row=row,
                    col=col,
                    direction=direction,
                    bbox=geometry.cell_bbox(row, col),
                    confidence=segment_confidence,
                    error=snake_error,
                )
            )

    sorted_segments = tuple(sorted(segments, key=_segment_sort_key))
    players = _group_by_player(sorted_segments)
    return SnakesState(
        geometry=geometry,
        segments=sorted_segments,
        players=players,
        match_threshold=match_threshold,
    )


def _group_by_player(segments: Iterable[SnakeSegment]) -> Dict[str, PlayerSnake]:
    grouped: Dict[str, list[SnakeSegment]] = {}
    for segment in segments:
        grouped.setdefault(segment.player, []).append(segment)

    players = {}
    for player in PLAYER_ORDER:
        player_segments = grouped.get(player)
        if not player_segments:
            continue
        ordered_segments = tuple(_reconstruct_snake_chain(player_segments))
        players[player] = PlayerSnake(
            player=player,
            segments=tuple(sorted(player_segments, key=_player_segment_sort_key)),
            ordered_segments=ordered_segments,
        )
    return players


def _reconstruct_snake_chain(segments: Iterable[SnakeSegment]) -> list[SnakeSegment]:
    pending = list(segments)
    head = next((segment for segment in pending if segment.segment_class == "snake_head"), None)
    if head is None:
        head = min(pending, key=lambda segment: (segment.row, segment.col))

    chain = [head]
    used = {(head.row, head.col)}

    while len(chain) < len(pending):
        current = chain[-1]
        candidates = [segment for segment in pending if (segment.row, segment.col) not in used]
        adjacent = [segment for segment in candidates if _manhattan(current, segment) == 1]

        pick_pool = adjacent if adjacent else candidates
        next_segment = min(
            pick_pool,
            key=lambda segment: (
                _manhattan(current, segment),
                CHAIN_CLASS_PRIORITY.get(segment.segment_class, 99),
                segment.row,
                segment.col,
            ),
        )
        chain.append(next_segment)
        used.add((next_segment.row, next_segment.col))

    return chain


def _manhattan(first: SnakeSegment, second: SnakeSegment) -> int:
    return abs(first.row - second.row) + abs(first.col - second.col)


def _segment_sort_key(segment: SnakeSegment) -> Tuple[int, int, int, int]:
    return (
        PLAYER_ORDER.index(segment.player),
        segment.row,
        segment.col,
        CLASS_ORDER.index(segment.segment_class),
    )


def _player_segment_sort_key(segment: SnakeSegment) -> Tuple[int, int, int]:
    return (
        CLASS_ORDER.index(segment.segment_class),
        segment.row,
        segment.col,
    )


@lru_cache(maxsize=1)
def _snake_templates() -> Dict[Tuple[str, str, str], np.ndarray]:
    templates = {}
    for path in ASSET_DIR.glob("snake*.png"):
        name = path.stem
        if name.startswith("snakehead_"):
            segment_class = "snake_head"
            rest = name[len("snakehead_"):]
        elif name.startswith("snaketail_"):
            segment_class = "snake_tail"
            rest = name[len("snaketail_"):]
        elif name.startswith("snake_"):
            segment_class = "snake_body"
            rest = name[len("snake_"):]
        else:
            continue
        player, direction = rest.split("_", 1)
        templates[(segment_class, player, direction)] = load_template(path)
    return templates


@lru_cache(maxsize=1)
def _other_templates() -> Dict[str, np.ndarray]:
    return {
        template_name: load_template(ASSET_DIR / f"{template_name}.png")
        for template_name in NON_SNAKE_TEMPLATES
    }
