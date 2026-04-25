import json
import unittest
from pathlib import Path

from PIL import Image

from vision_hud import HUD_SCORE_PLAYERS, detect_hud


ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS = ROOT / "datasets" / "vision_captures" / "annotations"


class VisionHUDTests(unittest.TestCase):
    def test_detect_hud_matches_labeled_captures(self):
        for annotation_path in sorted(ANNOTATIONS.glob("snake*.json")):
            with self.subTest(annotation=annotation_path.name):
                annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
                with Image.open(ROOT / annotation["image_path"]) as image:
                    hud = detect_hud(image)

                expected_scores = {
                    score["player"]: score["value"]
                    for score in annotation["hud"]["scores"]
                }
                self.assertEqual(annotation["hud"]["turn_counter"]["value"], hud.turn)
                self.assertEqual(expected_scores, hud.scores)

    def test_detect_hud_exposes_stable_score_order_and_bboxes(self):
        with Image.open(ROOT / "output" / "snake025.png") as image:
            hud = detect_hud(image)

        self.assertEqual(("G", "B", "R", "Y"), HUD_SCORE_PLAYERS)
        self.assertEqual((343, 19, 107, 53), hud.turn_bbox)
        self.assertEqual(
            {
                "G": (37, 37, 107, 53),
                "B": (181, 37, 107, 53),
                "R": (505, 37, 107, 53),
                "Y": (649, 37, 107, 53),
            },
            hud.score_bboxes,
        )

    def test_detect_hud_serializes_to_contract_shape(self):
        with Image.open(ROOT / "output" / "snake012.png") as image:
            data = detect_hud(image).to_dict()

        self.assertEqual("turn_counter", data["turn_counter"]["class"])
        self.assertEqual(12, data["turn_counter"]["value"])
        self.assertEqual(["score_G", "score_B", "score_R", "score_Y"], [s["class"] for s in data["scores"]])
        self.assertEqual([0, 10, 0, 20], [s["value"] for s in data["scores"]])


if __name__ == "__main__":
    unittest.main()
