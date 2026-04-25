import unittest

from vision_validation import evaluate_vision_parser, validation_report_markdown


class VisionValidationTests(unittest.TestCase):
    def test_validation_metrics_match_current_labeled_dataset(self):
        report = evaluate_vision_parser(min_hard_cases=20)
        metrics = report.metrics

        self.assertEqual(3, metrics.captures)
        self.assertGreaterEqual(metrics.turn_accuracy, 0.99)
        self.assertGreaterEqual(metrics.score_exact_rate, 0.99)
        self.assertGreaterEqual(metrics.snake_cell_precision, 0.99)
        self.assertGreaterEqual(metrics.snake_cell_recall, 0.99)
        self.assertGreaterEqual(metrics.snake_head_accuracy, 0.99)
        self.assertGreaterEqual(metrics.fruit_precision, 0.99)
        self.assertGreaterEqual(metrics.fruit_recall, 0.99)
        self.assertGreaterEqual(metrics.mean_parser_confidence, 0.95)
        self.assertEqual(0, metrics.total_errors)

    def test_validation_report_contains_at_least_twenty_hard_cases_with_comments(self):
        report = evaluate_vision_parser(min_hard_cases=20)

        self.assertGreaterEqual(len(report.hard_cases), 20)
        for case in report.hard_cases[:20]:
            self.assertTrue(case.comment.strip())
            self.assertGreaterEqual(case.confidence, 0.0)
            self.assertLessEqual(case.confidence, 1.0)

    def test_markdown_report_includes_hard_case_section(self):
        report = evaluate_vision_parser(min_hard_cases=20)
        markdown = validation_report_markdown(report)

        self.assertIn("# VI-06 Validacion del VisionParser contra dataset", markdown)
        self.assertIn("## Casos dificiles comentados (top 20)", markdown)
        self.assertIn("| # | captura | componente | entidad | celda | confianza | match | comentario |", markdown)


if __name__ == "__main__":
    unittest.main()
