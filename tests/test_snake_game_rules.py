import unittest

from SnakeGame import SnakeGame


class SnakeGameRulesTests(unittest.TestCase):
    def test_kill_award_does_not_increase_fruit_score(self):
        game = SnakeGame()
        hunter = game.snakes[0]
        victim = game.snakes[1]

        hunter.body = [[5, 5, "N"]]
        hunter.points = 120
        hunter.fruit_points = 120
        victim.body = [[5, 5, "S"]]
        victim.points = 80
        victim.fruit_points = 80

        for snake in game.snakes[2:]:
            snake.isAlive = False

        game.checkMovements()

        self.assertTrue(hunter.isAlive)
        self.assertFalse(victim.isAlive)
        self.assertEqual(150, hunter.getScore())
        self.assertEqual(120, hunter.getFruitScore())


if __name__ == "__main__":
    unittest.main()
