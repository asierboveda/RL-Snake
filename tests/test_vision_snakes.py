import json
import unittest
from pathlib import Path

from PIL import Image

from vision_snakes import detect_snakes


ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS = ROOT / "datasets" / "vision_captures" / "annotations"


class VisionSnakeTests(unittest.TestCase):
    def test_detect_snakes_matches_labeled_captures(self):
        for annotation_path in sorted(ANNOTATIONS.glob("snake*.json")):
            with self.subTest(annotation=annotation_path.name):
                annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
                with Image.open(ROOT / annotation["image_path"]) as image:
                    snakes = detect_snakes(image)

                expected = {
                    (
                        segment["class"],
                        segment["player"],
                        segment["cell"]["row"],
                        segment["cell"]["col"],
                        segment["direction"],
                        tuple(segment["bbox"]),
                    )
                    for segment in annotation["objects"]["snakes"]
                }
                detected = {
                    (
                        segment.segment_class,
                        segment.player,
                        segment.row,
                        segment.col,
                        segment.direction,
                        segment.bbox,
                    )
                    for segment in snakes.segments
                }
                self.assertEqual(expected, detected)
                self.assertTrue(all(segment.confidence > 0.95 for segment in snakes.segments))

    def test_detect_snakes_groups_players_and_chain(self):
        with Image.open(ROOT / "output" / "snake025.png") as image:
            snakes = detect_snakes(image)

        self.assertEqual({"B", "G", "R", "Y"}, set(snakes.players.keys()))
        self.assertEqual(
            [(0, 33, "W"), (0, 34, "W"), (0, 35, "W")],
            list(snakes.players["B"].board_body),
        )
        self.assertEqual(
            [(31, 14, "N"), (32, 14, "N"), (33, 14, "N")],
            list(snakes.players["R"].board_body),
        )

    def test_detect_snakes_serializes_contract_shape(self):
        with Image.open(ROOT / "output" / "snake012.png") as image:
            data = detect_snakes(image).to_dict()

        self.assertIn("snakes", data)
        self.assertIn("players", data)
        self.assertTrue(data["snakes"])

        first_segment = data["snakes"][0]
        self.assertEqual(
            {"class", "player", "cell", "bbox", "direction", "confidence"},
            set(first_segment.keys()),
        )
        self.assertGreaterEqual(first_segment["confidence"], 0.0)
        self.assertLessEqual(first_segment["confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
