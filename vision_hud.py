from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image

from vision_grid import CELL_SIZE, CELL_STRIDE, detect_grid_geometry


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "input" / str(CELL_SIZE)

HUD_SCORE_PLAYERS = ("G", "B", "R", "Y")
HUD_SCORE_COLORS = {"G": "g", "B": "b", "R": "r", "Y": "y"}
HUD_SCORE_POSITIONS = {
    "G": (2, 2),
    "B": (2, 10),
    "R": (2, 28),
    "Y": (2, 36),
}
HUD_TURN_POSITION = (1, 19)
HUD_TURN_COLOR = "g"
HUD_VALUE_WIDTH = 6 * CELL_STRIDE - 1
HUD_VALUE_HEIGHT = 3 * CELL_STRIDE - 1


@dataclass(frozen=True)
class HUDState:
    turn: int
    scores: Dict[str, int]
    turn_bbox: Tuple[int, int, int, int]
    score_bboxes: Dict[str, Tuple[int, int, int, int]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "turn_counter": {
                "class": "turn_counter",
                "bbox": list(self.turn_bbox),
                "value": self.turn,
            },
            "scores": [
                {
                    "class": f"score_{player}",
                    "player": player,
                    "bbox": list(self.score_bboxes[player]),
                    "value": self.scores[player],
                }
                for player in HUD_SCORE_PLAYERS
            ],
        }


def detect_hud(image) -> HUDState:
    detect_grid_geometry(image)
    pixels = _image_pixels(image)
    turn = _read_score_value(pixels, *HUD_TURN_POSITION, HUD_TURN_COLOR)
    scores = {
        player: _read_score_value(pixels, row, col, HUD_SCORE_COLORS[player])
        for player, (row, col) in HUD_SCORE_POSITIONS.items()
    }
    return HUDState(
        turn=turn,
        scores=scores,
        turn_bbox=_value_bbox(*HUD_TURN_POSITION),
        score_bboxes={player: _value_bbox(*HUD_SCORE_POSITIONS[player]) for player in HUD_SCORE_PLAYERS},
    )


def _read_score_value(pixels: np.ndarray, pos_row: int, pos_col: int, color: str) -> int:
    digits = [
        _read_digit(pixels, pos_row, pos_col + digit_index * 2, color)
        for digit_index in range(3)
    ]
    return digits[0] * 100 + digits[1] * 10 + digits[2]


def _read_digit(pixels: np.ndarray, pos_row: int, pos_col: int, color: str) -> int:
    errors = {
        digit: _digit_error(pixels, pos_row, pos_col, digit, color)
        for digit in range(10)
    }
    return min(errors, key=errors.get)


def _digit_error(pixels: np.ndarray, pos_row: int, pos_col: int, digit: int, color: str) -> float:
    error = 0.0
    for offset_row in range(3):
        for offset_col in range(2):
            y = (pos_row + offset_row) * CELL_STRIDE + 1
            x = (pos_col + offset_col) * CELL_STRIDE + 1
            crop = pixels[y:y + CELL_SIZE, x:x + CELL_SIZE, :]
            template = _template(digit, color, offset_row, offset_col)
            error += float(np.mean((crop - template) ** 2))
    return error


def _value_bbox(pos_row: int, pos_col: int) -> Tuple[int, int, int, int]:
    return (
        pos_col * CELL_STRIDE + 1,
        pos_row * CELL_STRIDE + 1,
        HUD_VALUE_WIDTH,
        HUD_VALUE_HEIGHT,
    )


def _image_pixels(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        pil_image = Image.fromarray(np.asarray(image)).convert("RGB")
    return np.asarray(pil_image, dtype=np.float32) / 255.0


@lru_cache(maxsize=None)
def _template(digit: int, color: str, offset_row: int, offset_col: int) -> np.ndarray:
    path = ASSET_DIR / f"{digit}{color}-{offset_row}-{offset_col}.png"
    if not path.exists():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
