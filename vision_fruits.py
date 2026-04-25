from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from vision_grid import CELL_SIZE, GridGeometry, detect_grid_geometry
from vision_match import best_template, confidence, image_pixels, load_template


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "input" / str(CELL_SIZE)

FRUIT_VALUES = (10, 15, 20)
FRUIT_MATCH_MAX_ERROR = 0.02
FRUIT_MATCH_MARGIN_RATIO = 0.4
FRUIT_MATCH_MARGIN_DELTA = 0.01


@dataclass(frozen=True)
class FruitDetection:
    value: int
    row: int
    col: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    error: float

    @property
    def fruit_class(self) -> str:
        return f"fruit_{self.value}"

    def to_dict(self) -> Dict[str, object]:
        return {
            "class": self.fruit_class,
            "value": self.value,
            "cell": {"row": self.row, "col": self.col},
            "bbox": list(self.bbox),
            "confidence": round(self.confidence, 6),
        }


@dataclass(frozen=True)
class FruitsState:
    geometry: GridGeometry
    fruits: Tuple[FruitDetection, ...]
    match_threshold: float
    margin_ratio: float
    margin_delta: float

    @property
    def by_value(self) -> Dict[int, Tuple[FruitDetection, ...]]:
        return {
            value: tuple(fruit for fruit in self.fruits if fruit.value == value)
            for value in FRUIT_VALUES
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "fruits": [fruit.to_dict() for fruit in self.fruits],
            "match_threshold": self.match_threshold,
            "margin_ratio": self.margin_ratio,
            "margin_delta": self.margin_delta,
        }


def detect_fruits(
    image,
    *,
    match_threshold: float = FRUIT_MATCH_MAX_ERROR,
    margin_ratio: float = FRUIT_MATCH_MARGIN_RATIO,
    margin_delta: float = FRUIT_MATCH_MARGIN_DELTA,
) -> FruitsState:
    geometry = detect_grid_geometry(image)
    pixels = image_pixels(image)
    fruit_templates = _fruit_templates()
    other_templates = _other_templates()

    fruits = []
    for row in range(geometry.rows):
        for col in range(geometry.cols):
            x, y, _, _ = geometry.cell_bbox(row, col)
            crop = pixels[y:y + CELL_SIZE, x:x + CELL_SIZE, :]

            value, fruit_error = best_template(crop, fruit_templates)
            _, other_error = best_template(crop, other_templates)
            if fruit_error > match_threshold:
                continue
            if fruit_error >= other_error * margin_ratio:
                continue
            if (other_error - fruit_error) < margin_delta:
                continue

            fruits.append(
                FruitDetection(
                    value=value,
                    row=row,
                    col=col,
                    bbox=geometry.cell_bbox(row, col),
                    confidence=confidence(fruit_error, other_error),
                    error=fruit_error,
                )
            )

    sorted_fruits = tuple(sorted(fruits, key=lambda fruit: (fruit.row, fruit.col, fruit.value)))
    return FruitsState(
        geometry=geometry,
        fruits=sorted_fruits,
        match_threshold=match_threshold,
        margin_ratio=margin_ratio,
        margin_delta=margin_delta,
    )


@lru_cache(maxsize=1)
def _fruit_templates() -> Dict[int, np.ndarray]:
    return {
        value: load_template(ASSET_DIR / f"fruit{value}.png")
        for value in FRUIT_VALUES
    }


@lru_cache(maxsize=1)
def _other_templates() -> Dict[str, np.ndarray]:
    templates = {"bomb": load_template(ASSET_DIR / "bomb.png")}
    for path in ASSET_DIR.glob("snake*.png"):
        templates[path.stem] = load_template(path)
    return templates
