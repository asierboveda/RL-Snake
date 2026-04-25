import unittest

import numpy as np

from board_state import BOARD_COLS, BOARD_ROWS, BoardState, FruitState, SnakeState
from rl_observation import (
    FEATURE_SET_V1,
    ATTACKABLE_ENEMY_CHANNEL,
    DANGEROUS_ENEMY_CHANNEL,
    ENEMY_BODY_CHANNEL,
    ENEMY_HEAD_CHANNEL,
    FREE_CHANNEL,
    FRUIT_CHANNEL,
    IMMEDIATE_DANGER_CHANNEL,
    OWN_BODY_CHANNEL,
    OWN_HEAD_CHANNEL,
    WALL_CHANNEL,
    build_observation,
)


def make_board(snakes, fruits=(), turn=45):
    return BoardState(
        turn=turn,
        rows=BOARD_ROWS,
        cols=BOARD_COLS,
        snakes=snakes,
        fruits=fruits,
        game_alive=True,
        winner_id=None,
        terminal_reason=None,
    )


class RLObservationTests(unittest.TestCase):
    def test_builds_spatial_tensor_with_documented_channels(self):
        board = make_board(
            snakes=(
                SnakeState(
                    0,
                    "A",
                    "G",
                    True,
                    ((10, 10, "E"), (10, 9, "E")),
                    score=150,
                    fruit_score=130,
                ),
                SnakeState(
                    1,
                    "B",
                    "B",
                    True,
                    ((10, 12, "W"), (10, 13, "W")),
                    score=40,
                    fruit_score=40,
                ),
                SnakeState(2, "C", "R", False, ((20, 20, "N"),), score=0, fruit_score=0),
            ),
            fruits=(FruitState(row=8, col=10, value=20, time_left=10),),
        )

        observation = build_observation(board, player_id=0)
        spatial = observation["spatial"]

        self.assertEqual((len(FEATURE_SET_V1.spatial_channel_names), BOARD_ROWS, BOARD_COLS), spatial.shape)
        self.assertEqual(np.float32, spatial.dtype)
        self.assertEqual(1.0, spatial[OWN_HEAD_CHANNEL, 10, 10])
        self.assertEqual(1.0, spatial[OWN_BODY_CHANNEL, 10, 9])
        self.assertEqual(1.0, spatial[ENEMY_HEAD_CHANNEL, 10, 12])
        self.assertEqual(1.0, spatial[ENEMY_BODY_CHANNEL, 10, 13])
        self.assertEqual(1.0, spatial[FRUIT_CHANNEL, 8, 10])
        self.assertEqual(1.0, spatial[WALL_CHANNEL, 0, 10])
        self.assertEqual(1.0, spatial[WALL_CHANNEL, BOARD_ROWS - 1, 10])
        self.assertEqual(1.0, spatial[WALL_CHANNEL, 10, 0])
        self.assertEqual(1.0, spatial[WALL_CHANNEL, 10, BOARD_COLS - 1])
        self.assertEqual(1.0, spatial[ATTACKABLE_ENEMY_CHANNEL, 10, 12])
        self.assertEqual(0.0, spatial[DANGEROUS_ENEMY_CHANNEL, 10, 12])
        self.assertEqual(1.0, spatial[IMMEDIATE_DANGER_CHANNEL, 10, 12])
        self.assertEqual(0.0, spatial[FREE_CHANNEL, 10, 12])
        self.assertEqual(1.0, spatial[FREE_CHANNEL, 9, 10])

    def test_classifies_rivals_as_dangerous_until_own_score_can_kill(self):
        board = make_board(
            snakes=(
                SnakeState(0, "A", "G", True, ((10, 10, "E"),), score=110, fruit_score=110),
                SnakeState(1, "B", "B", True, ((10, 11, "W"),), score=10, fruit_score=10),
            )
        )

        spatial = build_observation(board, player_id=0)["spatial"]

        self.assertEqual(1.0, spatial[DANGEROUS_ENEMY_CHANNEL, 10, 11])
        self.assertEqual(0.0, spatial[ATTACKABLE_ENEMY_CHANNEL, 10, 11])

    def test_builds_stable_scalar_features_for_three_rival_slots(self):
        board = make_board(
            snakes=(
                SnakeState(0, "A", "G", True, ((10, 10, "E"), (10, 9, "E")), score=150, fruit_score=130),
                SnakeState(1, "B", "B", True, ((10, 12, "W"),), score=40, fruit_score=40),
                SnakeState(2, "C", "R", True, ((20, 10, "N"), (21, 10, "N")), score=200, fruit_score=150),
                SnakeState(3, "D", "Y", False, ((30, 30, "S"),), score=0, fruit_score=0),
            ),
            fruits=(FruitState(row=8, col=10, value=10, time_left=7),),
            turn=90,
        )

        features = build_observation(board, player_id=0)["features"]
        feature_by_name = dict(zip(FEATURE_SET_V1.feature_names, features))

        self.assertEqual((len(FEATURE_SET_V1.feature_names),), features.shape)
        self.assertEqual(np.float32, features.dtype)
        self.assertAlmostEqual(130 / 120, feature_by_name["own_fruit_score_norm"])
        self.assertAlmostEqual(40 / 120, feature_by_name["rival_0_fruit_score_norm"])
        self.assertAlmostEqual(150 / 120, feature_by_name["rival_1_fruit_score_norm"])
        self.assertEqual(0.0, feature_by_name["rival_2_alive"])
        self.assertAlmostEqual((130 - 40) / 120, feature_by_name["rival_0_score_delta_norm"])
        self.assertAlmostEqual((130 - 150) / 120, feature_by_name["rival_1_score_delta_norm"])
        self.assertEqual(1.0, feature_by_name["own_can_kill"])
        self.assertEqual(1.0, feature_by_name["rival_1_can_kill"])
        self.assertAlmostEqual(2 / max(BOARD_ROWS, BOARD_COLS), feature_by_name["own_length_norm"])
        self.assertAlmostEqual(2 / (BOARD_ROWS + BOARD_COLS), feature_by_name["nearest_fruit_distance_norm"])
        self.assertAlmostEqual(10 / (BOARD_ROWS + BOARD_COLS), feature_by_name["nearest_dangerous_enemy_distance_norm"])
        self.assertAlmostEqual(2 / (BOARD_ROWS + BOARD_COLS), feature_by_name["nearest_attackable_enemy_distance_norm"])
        self.assertAlmostEqual(3 / 4, feature_by_name["alive_snake_count_norm"])
        self.assertAlmostEqual(90 / FEATURE_SET_V1.turn_limit, feature_by_name["turn_norm"])
        # Tactical features should be present and finite
        self.assertIn("forward_safe", feature_by_name)
        self.assertIn("best_fruit_distance", feature_by_name)
        self.assertIn("attack_available", feature_by_name)
        self.assertTrue(np.isfinite(features).all())

    def test_rejects_unknown_or_dead_own_player(self):
        dead_board = make_board(
            snakes=(SnakeState(0, "A", "G", False, ((10, 10, "N"),), score=0, fruit_score=0),)
        )

        with self.assertRaises(ValueError):
            build_observation(dead_board, player_id=99)
        with self.assertRaises(ValueError):
            build_observation(dead_board, player_id=0)


if __name__ == "__main__":
    unittest.main()
