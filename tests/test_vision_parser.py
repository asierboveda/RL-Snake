import json
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from RLPlayer import RLPlayer
from vision_fruits import FruitsState
from vision_grid import GridGeometry
from vision_hud import HUDState
from vision_parser import VisionParser
from vision_snakes import PlayerSnake, SnakeSegment, SnakesState


ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS = ROOT / "datasets" / "vision_captures" / "annotations"
PARSER = VisionParser()


class VisionParserTests(unittest.TestCase):
    def test_parse_rebuilds_board_state_from_labeled_captures(self):
        for annotation_path in sorted(ANNOTATIONS.glob("snake*.json")):
            with self.subTest(annotation=annotation_path.name):
                annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
                with Image.open(ROOT / annotation["image_path"]) as image:
                    result = PARSER.parse(image)

                board = result.board_state
                expected_scores = {
                    score["player"]: score["value"]
                    for score in annotation["hud"]["scores"]
                }
                expected_snakes = {}
                expected_heads = {}
                for segment in annotation["objects"]["snakes"]:
                    color = segment["player"]
                    expected_snakes.setdefault(color, set()).add(
                        (segment["cell"]["row"], segment["cell"]["col"], segment["direction"])
                    )
                    if segment["class"] == "snake_head":
                        expected_heads[color] = (
                            segment["cell"]["row"],
                            segment["cell"]["col"],
                            segment["direction"],
                        )

                self.assertEqual(annotation["hud"]["turn_counter"]["value"], board.turn)
                self.assertEqual(44, board.rows)
                self.assertEqual(44, board.cols)
                self.assertEqual(tuple(), result.errors)
                self.assertGreater(result.confidence, 0.95)

                snake_by_color = {snake.color: snake for snake in board.snakes}
                self.assertEqual({"G", "B", "R", "Y"}, set(snake_by_color.keys()))
                for color, expected_score in expected_scores.items():
                    snake = snake_by_color[color]
                    self.assertEqual(expected_score, snake.score)
                    self.assertEqual(expected_snakes[color], set(snake.body))
                    self.assertEqual(expected_heads[color], snake.head)

                expected_fruits = {
                    (fruit["cell"]["row"], fruit["cell"]["col"], fruit["value"])
                    for fruit in annotation["objects"]["fruits"]
                }
                parsed_fruits = {
                    (fruit.row, fruit.col, fruit.value)
                    for fruit in board.fruits
                }
                self.assertEqual(expected_fruits, parsed_fruits)

    def test_parse_output_is_consumable_by_rl_player(self):
        with Image.open(ROOT / "output" / "snake012.png") as image:
            result = PARSER.parse(image)

        action = RLPlayer(0, "G", game=None, epsilon=0.0, training_enabled=False).play_board_state(
            result.board_state
        )
        self.assertIn(action, {"N", "S", "E", "W"})

    def test_parse_keeps_hunter_threshold_conservative_without_fruit_history(self):
        geometry = GridGeometry.standard()
        segment = SnakeSegment(
            segment_class="snake_head",
            player="G",
            row=10,
            col=10,
            direction="N",
            bbox=geometry.cell_bbox(10, 10),
            confidence=0.999,
            error=0.0001,
        )
        snakes_state = SnakesState(
            geometry=geometry,
            segments=(segment,),
            players={"G": PlayerSnake(player="G", segments=(segment,), ordered_segments=(segment,))},
            match_threshold=0.02,
        )
        fruits_state = FruitsState(
            geometry=geometry,
            fruits=(),
            match_threshold=0.02,
            margin_ratio=0.4,
            margin_delta=0.01,
        )
        hud_state = HUDState(
            turn=50,
            scores={"G": 130, "B": 0, "R": 0, "Y": 0},
            turn_bbox=(343, 19, 107, 53),
            score_bboxes={
                "G": (37, 37, 107, 53),
                "B": (181, 37, 107, 53),
                "R": (505, 37, 107, 53),
                "Y": (649, 37, 107, 53),
            },
        )

        with patch("vision_parser.detect_grid_geometry", return_value=geometry), patch(
            "vision_parser.detect_hud", return_value=hud_state
        ), patch("vision_parser.detect_snakes", return_value=snakes_state), patch(
            "vision_parser.detect_fruits", return_value=fruits_state
        ):
            result = PARSER.parse(image=object())

        snake_g = next(snake for snake in result.board_state.snakes if snake.color == "G")
        self.assertEqual(130, snake_g.score)
        self.assertEqual(0, snake_g.fruit_score)
        self.assertFalse(snake_g.is_hunter)


if __name__ == "__main__":
    unittest.main()
