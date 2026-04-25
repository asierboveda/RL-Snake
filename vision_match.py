from pathlib import Path
from typing import Mapping, Tuple, TypeVar

import numpy as np
from PIL import Image


K = TypeVar("K")


def image_pixels(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        pil_image = Image.fromarray(np.asarray(image)).convert("RGB")
    return np.asarray(pil_image, dtype=np.float32) / 255.0


def load_template(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def best_template(crop: np.ndarray, templates: Mapping[K, np.ndarray]) -> Tuple[K, float]:
    best_key = None
    best_error = float("inf")
    for key, template in templates.items():
        error = float(np.mean((crop - template) ** 2))
        if error < best_error:
            best_key = key
            best_error = error
    return best_key, best_error


def confidence(primary_error: float, secondary_error: float) -> float:
    if secondary_error <= 0.0:
        return 0.0
    margin = max(secondary_error - primary_error, 0.0)
    return min(1.0, margin / secondary_error)
