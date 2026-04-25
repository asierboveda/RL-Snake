from dataclasses import dataclass
from typing import Optional, Tuple


IMAGE_WIDTH = 793
IMAGE_HEIGHT = 901
BOARD_ROWS = 44
BOARD_COLS = 44
UPPER_PANEL_ROWS = 6
CELL_SIZE = 17
CELL_STRIDE = 18
BOARD_ORIGIN_X = 1
BOARD_ORIGIN_Y = 1 + UPPER_PANEL_ROWS * CELL_STRIDE
HUD_HEIGHT = BOARD_ORIGIN_Y
BOARD_PIXEL_HEIGHT = BOARD_ROWS * CELL_STRIDE


@dataclass(frozen=True)
class GridGeometry:
    image_width: int
    image_height: int
    rows: int
    cols: int
    upper_panel_rows: int
    cell_size: int
    cell_stride: int
    board_origin: Tuple[int, int]

    @classmethod
    def standard(cls) -> "GridGeometry":
        return cls(
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            rows=BOARD_ROWS,
            cols=BOARD_COLS,
            upper_panel_rows=UPPER_PANEL_ROWS,
            cell_size=CELL_SIZE,
            cell_stride=CELL_STRIDE,
            board_origin=(BOARD_ORIGIN_X, BOARD_ORIGIN_Y),
        )

    @property
    def hud_bbox(self) -> Tuple[int, int, int, int]:
        return (0, 0, self.image_width, self.board_origin[1])

    @property
    def board_bbox(self) -> Tuple[int, int, int, int]:
        return (0, self.board_origin[1], self.image_width, self.rows * self.cell_stride)

    def cell_bbox(self, row: int, col: int) -> Tuple[int, int, int, int]:
        self._validate_cell(row, col)
        x = self.board_origin[0] + col * self.cell_stride
        y = self.board_origin[1] + row * self.cell_stride
        return (x, y, self.cell_size, self.cell_size)

    def cell_center(self, row: int, col: int) -> Tuple[int, int]:
        x, y, width, height = self.cell_bbox(row, col)
        return (x + width // 2, y + height // 2)

    def pixel_to_cell(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        origin_x, origin_y = self.board_origin
        rel_x = x - origin_x
        rel_y = y - origin_y
        if rel_x < 0 or rel_y < 0:
            return None

        col = rel_x // self.cell_stride
        row = rel_y // self.cell_stride
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return None

        cell_x = rel_x % self.cell_stride
        cell_y = rel_y % self.cell_stride
        if cell_x >= self.cell_size or cell_y >= self.cell_size:
            return None
        return (row, col)

    def to_dict(self) -> dict:
        return {
            "image_size": {"width": self.image_width, "height": self.image_height},
            "grid_size": {"rows": self.rows, "cols": self.cols},
            "upper_panel_rows": self.upper_panel_rows,
            "cell_size": self.cell_size,
            "cell_stride": self.cell_stride,
            "board_origin": {"x": self.board_origin[0], "y": self.board_origin[1]},
        }

    def _validate_cell(self, row: int, col: int) -> None:
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            raise ValueError(f"cell ({row}, {col}) is outside {self.rows}x{self.cols}")


def detect_grid_geometry(image) -> GridGeometry:
    width, height = _image_size(image)
    if (width, height) != (IMAGE_WIDTH, IMAGE_HEIGHT):
        raise ValueError(f"expected image size {IMAGE_WIDTH}x{IMAGE_HEIGHT}, got {width}x{height}")
    return GridGeometry.standard()


def _image_size(image) -> Tuple[int, int]:
    if hasattr(image, "size"):
        return tuple(image.size)
    if hasattr(image, "shape") and len(image.shape) >= 2:
        return (int(image.shape[1]), int(image.shape[0]))
    raise TypeError("image must expose PIL .size or numpy-like .shape")
