import unittest

import numpy as np

from board_state import BOARD_COLS, BOARD_ROWS, BoardState, FruitState, SnakeState
from ppo_env import PPOHeadlessSnakeEnv, RandomPolicy, make_bot_factories, run_policy_episode, summarize_episode_metrics
from tactical_planner import compute_tactical_features, recommended_fruit_action, recommended_attack_action


class TacticalPlannerTests(unittest.TestCase):
    def _make_board(self, snakes, fruits, turn=0):
        return BoardState(
            turn=turn, rows=BOARD_ROWS, cols=BOARD_COLS, snakes=snakes, fruits=fruits,
            game_alive=True, winner_id=None, terminal_reason=None,
        )

    def test_bfs_finds_reachable_fruit(self):
        own = SnakeState(
            player_id=0, label="A", color="G", alive=True,
            body=((1, 1, "E"), (1, 0, "E")), score=0, fruit_score=0,
        )
        fruit = FruitState(row=1, col=5, value=10, time_left=50)
        board = self._make_board([own], [fruit])
        tf = compute_tactical_features(board, 0)
        self.assertTrue(tf.forward_safe)
        self.assertLess(tf.best_fruit_distance, 1.0)

    def test_bfs_avoids_walls_and_bodies(self):
        own = SnakeState(
            player_id=0, label="A", color="G", alive=True,
            body=((1, 1, "E"), (1, 0, "E")), score=0, fruit_score=0,
        )
        obstacle = SnakeState(
            player_id=1, label="B", color="B", alive=True,
            body=((1, 2, "E"), (1, 3, "E")), score=0, fruit_score=0,
        )
        board = self._make_board([own, obstacle], [])
        tf = compute_tactical_features(board, 0)
        # Forward (E) should be unsafe because obstacle occupies (1,2)
        self.assertFalse(tf.forward_safe)

    def test_attack_only_when_hunter(self):
        own_weak = SnakeState(
            player_id=0, label="A", color="G", alive=True,
            body=((5, 5, "N"), (5, 6, "N")), score=50, fruit_score=50,
        )
        enemy = SnakeState(
            player_id=1, label="B", color="B", alive=True,
            body=((3, 3, "N"), (3, 4, "N")), score=30, fruit_score=30,
        )
        board = self._make_board([own_weak, enemy], [])
        tf = compute_tactical_features(board, 0)
        self.assertFalse(tf.attack_available)
        self.assertEqual(tf.best_attack_distance, 1.0)

    def test_attack_available_when_hunter_and_stronger(self):
        own_hunter = SnakeState(
            player_id=0, label="A", color="G", alive=True,
            body=((5, 5, "N"), (5, 6, "N")), score=130, fruit_score=130,
        )
        enemy = SnakeState(
            player_id=1, label="B", color="B", alive=True,
            body=((3, 3, "N"), (3, 4, "N")), score=50, fruit_score=50,
        )
        board = self._make_board([own_hunter, enemy], [])
        tf = compute_tactical_features(board, 0)
        self.assertTrue(tf.attack_available)
        self.assertLess(tf.best_attack_distance, 1.0)

    def test_recommended_fruit_action_points_toward_fruit(self):
        own = SnakeState(
            player_id=0, label="A", color="G", alive=True,
            body=((5, 5, "N"), (5, 6, "N")), score=0, fruit_score=0,
        )
        fruit = FruitState(row=3, col=5, value=10, time_left=50)
        board = self._make_board([own], [fruit])
        rec = recommended_fruit_action(board, 0)
        # Fruit is north, snake faces north -> forward is north
        self.assertEqual(rec, "forward")

    def test_stronger_enemies_marked_as_risk(self):
        own = SnakeState(
            player_id=0, label="A", color="G", alive=True,
            body=((5, 5, "N"), (5, 6, "N")), score=50, fruit_score=50,
        )
        strong = SnakeState(
            player_id=1, label="B", color="B", alive=True,
            body=((4, 5, "N"), (4, 6, "N")), score=150, fruit_score=150,
        )
        board = self._make_board([own, strong], [])
        tf = compute_tactical_features(board, 0)
        # Forward from (5,5) when facing N goes to (4,5), occupied by strong enemy -> risk
        self.assertTrue(tf.strong_enemy_risk_forward)


class PPOHeadlessSnakeEnvTests(unittest.TestCase):
    def test_reset_returns_fixed_finite_vector_observation(self):
        env = PPOHeadlessSnakeEnv(seed=11, turn_limit=20)

        obs, info = env.reset(seed=11)

        self.assertEqual((51,), obs.shape)
        self.assertEqual(np.float32, obs.dtype)
        self.assertTrue(np.isfinite(obs).all())
        self.assertEqual("rl_observation_v1_features", info["observation"])
        self.assertEqual(env.action_labels, ("FORWARD", "LEFT", "RIGHT"))

    def test_step_runs_controlled_snake_against_random_bots(self):
        env = PPOHeadlessSnakeEnv(seed=12, turn_limit=20)
        obs, _ = env.reset(seed=12)

        next_obs, reward, terminated, truncated, info = env.step(1)

        self.assertEqual(obs.shape, next_obs.shape)
        self.assertTrue(np.isfinite(next_obs).all())
        self.assertIsInstance(float(reward), float)
        self.assertIn("score", info)
        self.assertIn("fruit_score", info)
        self.assertIn("kills", info)
        self.assertFalse(terminated and truncated)

    def test_make_bot_factories_supports_random_greedy_and_survival(self):
        for name in ("random", "greedy", "survival"):
            factories = make_bot_factories(name)
            self.assertEqual(3, len(factories))

    def test_invalid_action_is_replaced_and_penalized(self):
        env = PPOHeadlessSnakeEnv(seed=13, turn_limit=20)
        env.reset(seed=13)
        for _ in range(5):
            _, reward, terminated, truncated, info = env.step(0)
            self.assertIn("invalid_action", info)
            self.assertIn("relative_action", info)
            self.assertIn("absolute_action", info)
            self.assertIn("intended_relative", info)
            if terminated or truncated:
                break

    def test_run_policy_episode_returns_metrics(self):
        env = PPOHeadlessSnakeEnv(seed=14, turn_limit=20)
        policy = RandomPolicy(env.action_space)
        metrics = run_policy_episode(env, policy, seed=14)
        self.assertIn("episode_reward", metrics)
        self.assertIn("survival_turns", metrics)
        self.assertIn("invalid_actions", metrics)
        self.assertIn("relative_action_distribution", metrics)
        self.assertIn("absolute_action_distribution", metrics)
        self.assertIn("win", metrics)

    def test_summarize_episode_metrics(self):
        episodes = [
            {"episode_reward": -10, "score": 5, "fruit_score": 5, "survival_turns": 10, "kills": 0, "win": False, "invalid_actions": 1, "steps": 10},
            {"episode_reward": -20, "score": 3, "fruit_score": 3, "survival_turns": 5, "kills": 0, "win": False, "invalid_actions": 0, "steps": 5},
        ]
        summary = summarize_episode_metrics(episodes)
        self.assertEqual(summary["episodes"], 2.0)
        self.assertAlmostEqual(summary["mean_reward"], -15.0)
        self.assertAlmostEqual(summary["invalid_action_rate"], 1.0 / 15.0)
        self.assertAlmostEqual(summary["early_death_rate"], 1.0)

    def test_observation_has_no_nan(self):
        env = PPOHeadlessSnakeEnv(seed=15, turn_limit=30)
        obs, _ = env.reset(seed=15)
        self.assertFalse(np.isnan(obs).any())
        for _ in range(10):
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            self.assertFalse(np.isnan(obs).any())
            if terminated or truncated:
                break


if __name__ == "__main__":
    unittest.main()
