import unittest

from board_state import BOARD_COLS, BOARD_ROWS, BoardState, FruitState, SnakeState
from vision_fallback import (
    MODE_CONSERVATIVE,
    MODE_REUSE_LAST_RELIABLE,
    MODE_SAFE_ACTION,
    MODE_TRUST,
    VisionFallbackPolicy,
    VisionFallbackThresholds,
)
from vision_parser import VisionParseResult


def make_board(*, turn: int, snakes, fruits=()):
    return BoardState(
        turn=turn,
        rows=BOARD_ROWS,
        cols=BOARD_COLS,
        snakes=snakes,
        fruits=fruits,
        game_alive=sum(1 for snake in snakes if snake.alive) > 1,
        winner_id=None,
        terminal_reason=None,
    )


def make_parse_result(
    board: BoardState,
    *,
    confidence: float = 0.999,
    snakes_confidence: float = 0.999,
    fruits_confidence: float = 0.999,
    consistency_confidence: float = 1.0,
    errors=(),
    warnings=(),
):
    return VisionParseResult(
        board_state=board,
        confidence=confidence,
        component_confidence={
            "hud": 1.0,
            "snakes": snakes_confidence,
            "fruits": fruits_confidence,
            "consistency": consistency_confidence,
        },
        errors=tuple(errors),
        warnings=tuple(warnings),
        metadata={},
        components={},
    )


class VisionFallbackPolicyTests(unittest.TestCase):
    def setUp(self):
        self.thresholds = VisionFallbackThresholds(
            trust_confidence_min=0.998,
            reject_confidence_max=0.995,
            trust_snakes_confidence_min=0.998,
            trust_fruits_confidence_min=0.998,
            consistency_confidence_min=0.95,
            max_warnings_for_trust=0,
            max_warnings_for_conservative=2,
            max_turn_gap_for_conservative=3,
            max_score_jump_per_turn=60,
        )
        self.policy = VisionFallbackPolicy(self.thresholds)

    def _base_snakes(self, my_score=0):
        return (
            SnakeState(0, "A", "G", True, ((10, 10, "N"), (11, 10, "N")), score=my_score, fruit_score=0),
            SnakeState(1, "B", "B", True, ((10, 14, "W"),), score=0, fruit_score=0),
            SnakeState(2, "C", "R", True, ((20, 20, "N"),), score=0, fruit_score=0),
            SnakeState(3, "D", "Y", True, ((30, 30, "N"),), score=0, fruit_score=0),
        )

    def test_trust_mode_accepts_and_updates_last_reliable(self):
        board = make_board(turn=10, snakes=self._base_snakes())
        decision = self.policy.evaluate(make_parse_result(board), snake_id=0, last_action="N")

        self.assertEqual(MODE_TRUST, decision.mode)
        self.assertTrue(decision.accepted)
        self.assertIs(decision.board_state, board)
        self.assertIs(self.policy.last_reliable_state, board)

    def test_conservative_mode_for_marginal_confidence(self):
        board = make_board(turn=11, snakes=self._base_snakes())
        decision = self.policy.evaluate(
            make_parse_result(board, confidence=0.9975),
            snake_id=0,
            last_action="N",
        )

        self.assertEqual(MODE_CONSERVATIVE, decision.mode)
        self.assertTrue(decision.accepted)
        self.assertTrue(decision.conservative_mode)
        self.assertIn("confidence-below-trust", decision.reasons)

    def test_critical_errors_reuse_last_reliable_state(self):
        trusted_board = make_board(turn=12, snakes=self._base_snakes())
        self.policy.evaluate(make_parse_result(trusted_board), snake_id=0, last_action="N")

        corrupted_board = make_board(turn=13, snakes=self._base_snakes())
        decision = self.policy.evaluate(
            make_parse_result(corrupted_board, errors=("snake overlap detected at cell (10,10)",)),
            snake_id=0,
            last_action="E",
        )

        self.assertEqual(MODE_REUSE_LAST_RELIABLE, decision.mode)
        self.assertTrue(decision.accepted)
        self.assertIs(decision.board_state, trusted_board)
        self.assertIn("parser-errors", decision.reasons)

    def test_critical_without_history_forces_safe_action(self):
        risky_board = make_board(
            turn=3,
            snakes=(
                SnakeState(0, "A", "G", True, ((0, 0, "E"),), score=0, fruit_score=0),
                SnakeState(1, "B", "B", True, ((0, 1, "W"),), score=0, fruit_score=0),
                SnakeState(2, "C", "R", True, ((20, 20, "N"),), score=0, fruit_score=0),
                SnakeState(3, "D", "Y", True, ((30, 30, "N"),), score=0, fruit_score=0),
            ),
            fruits=(FruitState(1, 0, 10, 0),),
        )
        decision = self.policy.evaluate(
            make_parse_result(risky_board, errors=("duplicate fruit detected at cell (1,0)",)),
            snake_id=0,
            last_action="E",
        )

        self.assertEqual(MODE_SAFE_ACTION, decision.mode)
        self.assertFalse(decision.accepted)
        self.assertTrue(decision.force_safe_action)
        self.assertEqual("S", decision.safe_action)

    def test_score_jump_too_large_triggers_reuse_last_reliable(self):
        trusted_board = make_board(turn=50, snakes=self._base_snakes(my_score=10))
        self.policy.evaluate(make_parse_result(trusted_board), snake_id=0, last_action="N")

        implausible_board = make_board(turn=51, snakes=self._base_snakes(my_score=200))
        decision = self.policy.evaluate(
            make_parse_result(implausible_board, confidence=0.9999),
            snake_id=0,
            last_action="N",
        )

        self.assertEqual(MODE_REUSE_LAST_RELIABLE, decision.mode)
        self.assertIn("score-jump-too-large", decision.reasons)

    def test_keep_preferred_action_if_it_is_safe(self):
        board = make_board(turn=1, snakes=self._base_snakes())
        decision = self.policy.evaluate(
            make_parse_result(board, errors=("synthetic parser error",)),
            snake_id=0,
            last_action="E",
        )
        self.assertEqual(MODE_SAFE_ACTION, decision.mode)
        self.assertEqual("E", decision.safe_action)


if __name__ == "__main__":
    unittest.main()
