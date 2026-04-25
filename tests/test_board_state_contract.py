import unittest

from board_state import (
    BOARD_COLS,
    BOARD_ROWS,
    FRUIT_KILL_THRESHOLD,
    KILL_SCORE,
    BoardState,
    CollisionOutcome,
    FruitState,
    SnakeState,
    determine_winner,
    resolve_collision,
)


class BoardStateContractTests(unittest.TestCase):
    def test_board_state_serializes_required_turn_fields(self):
        state = BoardState(
            turn=7,
            rows=BOARD_ROWS,
            cols=BOARD_COLS,
            snakes=[
                SnakeState(
                    player_id=0,
                    label="A",
                    color="G",
                    alive=True,
                    body=((10, 10, "N"), (11, 10, "N")),
                    score=20,
                    fruit_score=20,
                )
            ],
            fruits=[FruitState(row=4, col=5, value=15, time_left=12)],
            game_alive=True,
            winner_id=None,
            terminal_reason=None,
        )

        data = state.to_dict()

        self.assertEqual(
            {
                "turn",
                "rows",
                "cols",
                "snakes",
                "fruits",
                "game_alive",
                "winner_id",
                "terminal_reason",
            },
            set(data),
        )
        self.assertEqual(7, data["turn"])
        self.assertEqual((10, 10, "N"), data["snakes"][0]["head"])
        self.assertEqual([(10, 10, "N"), (11, 10, "N")], data["snakes"][0]["body"])
        self.assertEqual({"row": 4, "col": 5, "value": 15, "time_left": 12}, data["fruits"][0])

    def test_collision_without_hunter_kills_both(self):
        outcome = resolve_collision(
            SnakeState(0, "A", "G", True, ((1, 1, "N"),), score=119, fruit_score=119),
            SnakeState(1, "B", "B", True, ((1, 1, "S"),), score=0, fruit_score=0),
        )

        self.assertEqual(CollisionOutcome(dead_ids=(0, 1), killer_id=None, points_awarded=0), outcome)

    def test_collision_hunter_kills_weaker_rival_and_scores(self):
        outcome = resolve_collision(
            SnakeState(0, "A", "G", True, ((1, 1, "N"),), score=130, fruit_score=120),
            SnakeState(1, "B", "B", True, ((1, 1, "S"),), score=80, fruit_score=80),
        )

        self.assertEqual(CollisionOutcome(dead_ids=(1,), killer_id=0, points_awarded=KILL_SCORE), outcome)

    def test_collision_between_equal_fruit_scores_kills_both_even_at_threshold(self):
        outcome = resolve_collision(
            SnakeState(0, "A", "G", True, ((1, 1, "N"),), score=130, fruit_score=120),
            SnakeState(1, "B", "B", True, ((1, 1, "S"),), score=130, fruit_score=120),
        )

        self.assertEqual(CollisionOutcome(dead_ids=(0, 1), killer_id=None, points_awarded=0), outcome)

    def test_winner_requires_single_alive_or_unique_score_at_threshold(self):
        snakes = (
            SnakeState(0, "A", "G", False, ((1, 1, "N"),), score=110, fruit_score=110),
            SnakeState(1, "B", "B", True, ((2, 2, "S"),), score=10, fruit_score=10),
        )
        self.assertEqual((1, "single_alive"), determine_winner(snakes))

        snakes = (
            SnakeState(0, "A", "G", True, ((1, 1, "N"),), score=FRUIT_KILL_THRESHOLD, fruit_score=120),
            SnakeState(1, "B", "B", True, ((2, 2, "S"),), score=110, fruit_score=110),
        )
        self.assertEqual((0, "score_threshold"), determine_winner(snakes))

        snakes = (
            SnakeState(0, "A", "G", True, ((1, 1, "N"),), score=110, fruit_score=110),
            SnakeState(1, "B", "B", True, ((2, 2, "S"),), score=100, fruit_score=100),
        )
        self.assertEqual((None, "too_few_points"), determine_winner(snakes))

    def test_board_state_validates_terminal_fields(self):
        snake_a = SnakeState(0, "A", "G", True, ((1, 1, "N"),), score=0, fruit_score=0)
        snake_b = SnakeState(1, "B", "B", True, ((2, 2, "S"),), score=0, fruit_score=0)

        with self.assertRaises(ValueError):
            BoardState(
                turn=0,
                rows=BOARD_ROWS,
                cols=BOARD_COLS,
                snakes=(snake_a, snake_b),
                fruits=(),
                game_alive=True,
                winner_id=None,
                terminal_reason="all_dead",
            )

        with self.assertRaises(ValueError):
            BoardState(
                turn=0,
                rows=BOARD_ROWS,
                cols=BOARD_COLS,
                snakes=(snake_a,),
                fruits=(),
                game_alive=False,
                winner_id=99,
                terminal_reason="single_alive",
            )


if __name__ == "__main__":
    unittest.main()
