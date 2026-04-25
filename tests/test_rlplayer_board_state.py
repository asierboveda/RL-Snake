import unittest

from board_state import BOARD_COLS, BOARD_ROWS, BoardState, FruitState, SnakeState
from RLPlayer import RLPlayer


def make_board(snakes, fruits=()):
    return BoardState(
        turn=1,
        rows=BOARD_ROWS,
        cols=BOARD_COLS,
        snakes=snakes,
        fruits=fruits,
        game_alive=True,
        winner_id=None,
        terminal_reason=None,
    )


class RLPlayerBoardStateTests(unittest.TestCase):
    def test_policy_state_uses_board_state_without_game(self):
        board = make_board(
            snakes=(
                SnakeState(0, "A", "G", True, ((0, 0, "N"),), score=120, fruit_score=120),
                SnakeState(1, "B", "B", True, ((0, 1, "W"),), score=20, fruit_score=20),
            )
        )
        player = RLPlayer(0, "G", game=None, epsilon=0.0, training_enabled=False)

        state = player.get_state_from_board(board)

        self.assertEqual(
            (
                True,   # north wall
                False,  # south is free
                True,   # east occupied by rival
                True,   # west wall
                False,  # goal is not north
                False,  # goal is not south
                True,   # hunter target is east
                False,  # goal is not west
                True,   # fruit_score >= 120
            ),
            state,
        )

    def test_safe_actions_use_board_occupancy(self):
        board = make_board(
            snakes=(
                SnakeState(0, "A", "G", True, ((10, 10, "N"),), score=0, fruit_score=0),
                SnakeState(1, "B", "B", True, ((10, 11, "W"),), score=0, fruit_score=0),
                SnakeState(2, "C", "R", True, ((9, 10, "S"),), score=0, fruit_score=0),
            )
        )
        player = RLPlayer(0, "G", game=None, epsilon=0.0, training_enabled=False)

        self.assertEqual(["S", "W"], player.get_safe_actions_from_board(board))

    def test_play_board_state_selects_best_safe_q_action(self):
        board = make_board(
            snakes=(SnakeState(0, "A", "G", True, ((10, 10, "N"),), score=0, fruit_score=0),),
            fruits=(FruitState(row=10, col=12, value=10, time_left=20),),
        )
        player = RLPlayer(0, "G", game=None, epsilon=0.0, training_enabled=False)
        state = player.get_state_from_board(board)
        player.q_table[state] = {"N": 0.0, "S": 0.0, "E": 5.0, "W": 1.0}

        self.assertEqual("E", player.play_board_state(board))

    def test_dead_player_returns_default_action_without_game(self):
        board = make_board(
            snakes=(SnakeState(0, "A", "G", False, ((10, 10, "N"),), score=0, fruit_score=0),),
            fruits=(),
        )
        player = RLPlayer(0, "G", game=None, epsilon=0.0, training_enabled=False)

        self.assertEqual("N", player.play_board_state(board))


if __name__ == "__main__":
    unittest.main()
