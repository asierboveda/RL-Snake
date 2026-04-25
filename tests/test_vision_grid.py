import json
import unittest
from pathlib import Path

from PIL import Image

from vision_grid import (
    BOARD_COLS,
    BOARD_ROWS,
    CELL_SIZE,
    CELL_STRIDE,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    UPPER_PANEL_ROWS,
    GridGeometry,
    detect_grid_geometry,
)


ROOT = Path(__file__).resolve().parent.parent


class VisionGridTests(unittest.TestCase):
    def test_detect_grid_geometry_from_capture_size(self):
        with Image.open(ROOT / "output" / "snake000.png") as image:
            geometry = detect_grid_geometry(image)

        self.assertEqual(IMAGE_WIDTH, geometry.image_width)
        self.assertEqual(IMAGE_HEIGHT, geometry.image_height)
        self.assertEqual((0, 0, 793, 109), geometry.hud_bbox)
        self.assertEqual((0, 109, 793, 792), geometry.board_bbox)
        self.assertEqual((1, 109), geometry.board_origin)
        self.assertEqual(BOARD_ROWS, geometry.rows)
        self.assertEqual(BOARD_COLS, geometry.cols)
        self.assertEqual(UPPER_PANEL_ROWS, geometry.upper_panel_rows)
        self.assertEqual(CELL_SIZE, geometry.cell_size)
        self.assertEqual(CELL_STRIDE, geometry.cell_stride)

    def test_cell_bbox_matches_dataset_annotations(self):
        geometry = GridGeometry.standard()
        annotation = json.loads(
            (ROOT / "datasets" / "vision_captures" / "annotations" / "snake025.json").read_text(
                encoding="utf-8"
            )
        )

        for obj in annotation["objects"]["fruits"] + annotation["objects"]["snakes"]:
            row = obj["cell"]["row"]
            col = obj["cell"]["col"]
            self.assertEqual(tuple(obj["bbox"]), geometry.cell_bbox(row, col))

    def test_pixel_to_cell_uses_playable_cell_area_only(self):
        geometry = GridGeometry.standard()

        self.assertEqual((0, 0), geometry.pixel_to_cell(1, 109))
        self.assertEqual((0, 0), geometry.pixel_to_cell(17, 125))
        self.assertEqual((43, 43), geometry.pixel_to_cell(775, 883))
        self.assertIsNone(geometry.pixel_to_cell(0, 109))
        self.assertIsNone(geometry.pixel_to_cell(18, 109))
        self.assertIsNone(geometry.pixel_to_cell(1, 108))
        self.assertIsNone(geometry.pixel_to_cell(792, 900))

    def test_cell_center_round_trips_to_cell(self):
        geometry = GridGeometry.standard()

        for row, col in ((0, 0), (10, 10), (24, 13), (43, 43)):
            x, y = geometry.cell_center(row, col)
            self.assertEqual((row, col), geometry.pixel_to_cell(x, y))


if __name__ == "__main__":
    unittest.main()
