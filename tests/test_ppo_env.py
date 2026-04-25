import unittest

import numpy as np

from ppo_env import ACTIONS, PPOHeadlessSnakeEnv, make_bot_factories


class PPOHeadlessSnakeEnvTests(unittest.TestCase):
    def test_reset_returns_fixed_finite_vector_observation(self):
        env = PPOHeadlessSnakeEnv(seed=11, turn_limit=20)

        obs, info = env.reset(seed=11)

        self.assertEqual((23,), obs.shape)
        self.assertEqual(np.float32, obs.dtype)
        self.assertTrue(np.isfinite(obs).all())
        self.assertEqual("rl_observation_v1_features", info["observation"])
        self.assertEqual(ACTIONS, env.action_labels)

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


if __name__ == "__main__":
    unittest.main()
