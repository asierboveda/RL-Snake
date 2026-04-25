import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets" / "vision_captures"
MANIFEST = DATASET_DIR / "manifest.json"


class VisionDatasetContractTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_manifest_selects_required_representative_phases(self):
        captures = self.manifest["captures"]
        phases = {capture["phase"] for capture in captures}
        splits = {capture["split"] for capture in captures}

        self.assertEqual({"inicio", "media", "caos"}, phases)
        self.assertEqual({"train", "val", "test"}, splits)
        self.assertEqual(3, len(captures))

    def test_manifest_paths_exist(self):
        self.assertTrue((ROOT / self.manifest["label_schema"]).is_file())
        for capture in self.manifest["captures"]:
            self.assertTrue((ROOT / capture["image_path"]).is_file(), capture["image_path"])
            self.assertTrue((ROOT / capture["annotation_path"]).is_file(), capture["annotation_path"])

    def test_annotations_match_geometry_and_manifest(self):
        expected_geometry = {
            "image_size": {"width": 793, "height": 901},
            "grid_size": {"rows": 44, "cols": 44},
            "upper_panel_rows": 6,
            "cell_size": 17,
            "cell_stride": 18,
            "board_origin": {"x": 1, "y": 109},
        }
        expected_board_bbox = [0, 109, 793, 792]

        for capture in self.manifest["captures"]:
            annotation = json.loads((ROOT / capture["annotation_path"]).read_text(encoding="utf-8"))
            self.assertEqual(capture["id"], annotation["capture_id"])
            self.assertEqual(capture["image_path"], annotation["image_path"])
            self.assertEqual(expected_geometry, annotation["geometry"])
            self.assertEqual(expected_board_bbox, annotation["board"]["bbox"])
            self.assertEqual(44, annotation["board"]["rows"])
            self.assertEqual(44, annotation["board"]["cols"])

    def test_annotations_use_declared_label_sets(self):
        label_sets = self.manifest["label_sets"]
        fruit_classes = set(label_sets["fruit_classes"])
        snake_classes = set(label_sets["snake_part_classes"])
        colors = set(label_sets["player_colors"])
        directions = set(label_sets["directions"])

        for capture in self.manifest["captures"]:
            annotation = json.loads((ROOT / capture["annotation_path"]).read_text(encoding="utf-8"))
            for fruit in annotation["objects"]["fruits"]:
                self.assertIn(fruit["class"], fruit_classes)
                self.assertEqual(f"fruit_{fruit['value']}", fruit["class"])
                self.assert_cell_and_bbox(fruit)

            for segment in annotation["objects"]["snakes"]:
                self.assertIn(segment["class"], snake_classes)
                self.assertIn(segment["player"], colors)
                self.assertIn(segment["direction"], directions)
                self.assert_cell_and_bbox(segment)

    def assert_cell_and_bbox(self, obj):
        row = obj["cell"]["row"]
        col = obj["cell"]["col"]
        self.assertGreaterEqual(row, 0)
        self.assertLess(row, 44)
        self.assertGreaterEqual(col, 0)
        self.assertLess(col, 44)
        self.assertEqual([1 + col * 18, 109 + row * 18, 17, 17], obj["bbox"])


if __name__ == "__main__":
    unittest.main()
