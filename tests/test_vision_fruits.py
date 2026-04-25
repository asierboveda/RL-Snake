import json
import unittest
from pathlib import Path

from PIL import Image

from vision_fruits import detect_fruits


ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS = ROOT / "datasets" / "vision_captures" / "annotations"


class VisionFruitTests(unittest.TestCase):
    def test_detect_fruits_matches_labeled_captures_with_perfect_precision(self):
        for annotation_path in sorted(ANNOTATIONS.glob("snake*.json")):
            with self.subTest(annotation=annotation_path.name):
                annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
                with Image.open(ROOT / annotation["image_path"]) as image:
                    fruits = detect_fruits(image)

                expected_by_cell_value = {
                    (fruit["cell"]["row"], fruit["cell"]["col"], fruit["value"])
                    for fruit in annotation["objects"]["fruits"]
                }
                expected_by_cell = {
                    (fruit["cell"]["row"], fruit["cell"]["col"])
                    for fruit in annotation["objects"]["fruits"]
                }
                detected_by_cell_value = {
                    (fruit.row, fruit.col, fruit.value)
                    for fruit in fruits.fruits
                }
                detected_by_cell = {
                    (fruit.row, fruit.col)
                    for fruit in fruits.fruits
                }

                self.assertEqual(expected_by_cell_value, detected_by_cell_value)
                self.assertEqual(expected_by_cell, detected_by_cell)
                self.assertTrue(all(fruit.confidence > 0.95 for fruit in fruits.fruits))

    def test_detect_fruits_keeps_border_and_busy_frame_cells(self):
        with Image.open(ROOT / "output" / "snake012.png") as image:
            fruits = detect_fruits(image)
        border_cells = {(fruit.row, fruit.col, fruit.value) for fruit in fruits.fruits}
        self.assertIn((0, 28, 15), border_cells)
        self.assertIn((1, 41, 20), border_cells)

        with Image.open(ROOT / "output" / "snake025.png") as image:
            fruits = detect_fruits(image)
        busy_cells = {(fruit.row, fruit.col, fruit.value) for fruit in fruits.fruits}
        self.assertEqual({(18, 1, 15), (18, 7, 20)}, busy_cells)

    def test_detect_fruits_serializes_contract_shape(self):
        with Image.open(ROOT / "output" / "snake000.png") as image:
            data = detect_fruits(image).to_dict()

        self.assertIn("fruits", data)
        self.assertTrue(data["fruits"])
        sample = data["fruits"][0]
        self.assertEqual(
            {"class", "value", "cell", "bbox", "confidence"},
            set(sample.keys()),
        )
        self.assertIn(sample["class"], {"fruit_10", "fruit_15", "fruit_20"})
        self.assertIn(sample["value"], {10, 15, 20})
        self.assertGreaterEqual(sample["confidence"], 0.0)
        self.assertLessEqual(sample["confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
