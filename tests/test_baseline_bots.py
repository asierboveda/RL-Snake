import unittest

from board_state import BOARD_COLS, BOARD_ROWS, BoardState, FruitState, SnakeState
from AggressivePlayer import AggressivePlayer
from GreedyPlayer import GreedyPlayer
from HybridPlayer import HybridPlayer
from RandomPlayer import RandomPlayer
from SurvivalPlayer import SurvivalPlayer


def make_board(snakes, fruits=(), turn=1, game_alive=True, winner_id=None, terminal_reason=None):
    return BoardState(
        turn=turn,
        rows=BOARD_ROWS,
        cols=BOARD_COLS,
        snakes=snakes,
        fruits=fruits,
        game_alive=game_alive,
        winner_id=winner_id,
        terminal_reason=terminal_reason,
    )


class BaselineBotTests(unittest.TestCase):
    def test_random_player_returns_legal_action(self):
        board = make_board(
            snakes=(
                SnakeState(0, "A", "G", True, ((10, 10, "N"),), score=0, fruit_score=0),
                SnakeState(1, "B", "B", True, ((10, 12, "W"),), score=0, fruit_score=0),
            )
        )
        player = RandomPlayer(0, "G")

        self.assertIn(player.play_board_state(board), {"N", "S", "E", "W"})

    def test_greedy_player_chases_nearest_fruit(self):
        board = make_board(
            snakes=(SnakeState(0, "A", "G", True, ((10, 10, "N"),), score=0, fruit_score=0),),
            fruits=(FruitState(row=10, col=13, value=10, time_left=20),),
        )
        player = GreedyPlayer(0, "G", game=None)

        self.assertEqual("E", player.play_board_state(board))

    def test_survival_player_prefers_more_space(self):
        board = make_board(
            snakes=(
                SnakeState(0, "A", "G", True, ((10, 10, "N"),), score=0, fruit_score=0),
                SnakeState(1, "B", "B", True, ((10, 12, "W"),), score=0, fruit_score=0),
                SnakeState(2, "C", "R", True, ((11, 10, "N"),), score=0, fruit_score=0),
                SnakeState(3, "D", "Y", True, ((9, 10, "S"),), score=0, fruit_score=0),
            )
        )
        player = SurvivalPlayer(0, "G")

        self.assertEqual("E", player.play_board_state(board))

    def test_aggressive_player_targets_weaker_rival_when_hunter(self):
        board = make_board(
            snakes=(
                SnakeState(0, "A", "G", True, ((10, 10, "N"),), score=150, fruit_score=120),
                SnakeState(1, "B", "B", True, ((10, 13, "W"),), score=10, fruit_score=10),
            )
        )
        player = AggressivePlayer(0, "G")

        self.assertEqual("E", player.play_board_state(board))

    def test_hybrid_player_switches_to_survival_when_space_is_tight(self):
        board = make_board(
            snakes=(
                SnakeState(0, "A", "G", True, ((10, 10, "N"),), score=0, fruit_score=0),
                SnakeState(1, "B", "B", True, ((10, 11, "W"),), score=0, fruit_score=0),
                SnakeState(2, "C", "R", True, ((9, 10, "S"),), score=0, fruit_score=0),
            ),
            fruits=(FruitState(row=10, col=13, value=10, time_left=20),),
        )
        player = HybridPlayer(0, "G")

        self.assertEqual("S", player.play_board_state(board))


if __name__ == "__main__":
    unittest.main()
