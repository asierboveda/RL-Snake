import unittest

from board_state import BoardState
from snake_env import SnakeEnv


class SnakeEnvTests(unittest.TestCase):
    def test_reset_is_reproducible_by_seed(self):
        first = SnakeEnv(seed=7, initial_fruits=3).reset()
        second = SnakeEnv(seed=7, initial_fruits=3).reset()

        self.assertIsInstance(first, BoardState)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(0, first.turn)
        self.assertEqual(3, len(first.fruits))

    def test_legal_actions_exclude_walls_and_occupied_cells(self):
        env = SnakeEnv(seed=1, initial_fruits=0)
        state = env.reset()
        snake = state.snakes[0]

        env.game.snakes[0].body = [[0, 0, "N"]]
        env.game.snakes[1].body = [[0, 1, "W"]]

        self.assertEqual(("S",), env.legal_actions(0))
        self.assertEqual(("S",), env.legal_actions(snake.player_id))

    def test_step_returns_transition_and_records_replay(self):
        env = SnakeEnv(seed=3, initial_fruits=0, turn_limit=5)
        initial_state = env.reset()

        transition = env.step({0: "S", 1: "S", 2: "N", 3: "N"})

        self.assertIsInstance(transition.state, BoardState)
        self.assertEqual(initial_state.turn + 1, transition.state.turn)
        self.assertEqual({0, 1, 2, 3}, set(transition.rewards))
        self.assertFalse(transition.done)
        self.assertEqual(1, len(env.replay))
        self.assertEqual({0: "S", 1: "S", 2: "N", 3: "N"}, env.replay[0].actions)

    def test_step_serializes_replay_logs_with_terminal_flag(self):
        env = SnakeEnv(seed=5, initial_fruits=0, turn_limit=1)
        env.reset()

        transition = env.step(["S", "S", "N", "N"])
        data = env.to_replay_dict()

        self.assertTrue(transition.done)
        self.assertEqual(5, data["seed"])
        self.assertEqual(1, data["turn_limit"])
        self.assertEqual(1, len(data["steps"]))
        self.assertTrue(data["steps"][0]["done"])
        self.assertIn("state", data["steps"][0])
        self.assertIn("rewards", data["steps"][0])


if __name__ == "__main__":
    unittest.main()
