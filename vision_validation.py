import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image

from board_state import PLAYER_COLORS
from vision_parser import VisionParser


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "datasets" / "vision_captures" / "manifest.json"


@dataclass(frozen=True)
class ValidationMetrics:
    captures: int
    turn_accuracy: float
    score_exact_rate: float
    snake_cell_precision: float
    snake_cell_recall: float
    snake_cell_f1: float
    snake_head_accuracy: float
    fruit_precision: float
    fruit_recall: float
    fruit_f1: float
    mean_parser_confidence: float
    total_errors: int
    total_warnings: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "captures": self.captures,
            "turn_accuracy": self.turn_accuracy,
            "score_exact_rate": self.score_exact_rate,
            "snake_cell_precision": self.snake_cell_precision,
            "snake_cell_recall": self.snake_cell_recall,
            "snake_cell_f1": self.snake_cell_f1,
            "snake_head_accuracy": self.snake_head_accuracy,
            "fruit_precision": self.fruit_precision,
            "fruit_recall": self.fruit_recall,
            "fruit_f1": self.fruit_f1,
            "mean_parser_confidence": self.mean_parser_confidence,
            "total_errors": self.total_errors,
            "total_warnings": self.total_warnings,
        }


@dataclass(frozen=True)
class CaptureValidation:
    capture_id: str
    turn_ok: bool
    score_exact: float
    snake_cell_precision: float
    snake_cell_recall: float
    snake_head_accuracy: float
    fruit_precision: float
    fruit_recall: float
    parser_confidence: float
    errors: Tuple[str, ...]
    warnings: Tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "capture_id": self.capture_id,
            "turn_ok": self.turn_ok,
            "score_exact": self.score_exact,
            "snake_cell_precision": self.snake_cell_precision,
            "snake_cell_recall": self.snake_cell_recall,
            "snake_head_accuracy": self.snake_head_accuracy,
            "fruit_precision": self.fruit_precision,
            "fruit_recall": self.fruit_recall,
            "parser_confidence": self.parser_confidence,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class HardCase:
    rank: int
    capture_id: str
    component: str
    entity: str
    cell: Tuple[int, int]
    confidence: float
    matched: bool
    comment: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "rank": self.rank,
            "capture_id": self.capture_id,
            "component": self.component,
            "entity": self.entity,
            "cell": list(self.cell),
            "confidence": self.confidence,
            "matched": self.matched,
            "comment": self.comment,
        }


@dataclass(frozen=True)
class VisionValidationReport:
    metrics: ValidationMetrics
    captures: Tuple[CaptureValidation, ...]
    hard_cases: Tuple[HardCase, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "metrics": self.metrics.to_dict(),
            "captures": [capture.to_dict() for capture in self.captures],
            "hard_cases": [case.to_dict() for case in self.hard_cases],
        }


def evaluate_vision_parser(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    parser: VisionParser | None = None,
    min_hard_cases: int = 20,
) -> VisionValidationReport:
    parser = parser or VisionParser()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

    capture_reports: list[CaptureValidation] = []
    hard_case_candidates: list[Tuple[float, str, str, str, Tuple[int, int], float, bool, str]] = []

    aggregate = {
        "turn_ok": 0,
        "score_exact_hits": 0,
        "score_total": 0,
        "snake_tp": 0,
        "snake_fp": 0,
        "snake_fn": 0,
        "head_hits": 0,
        "head_total": 0,
        "fruit_tp": 0,
        "fruit_fp": 0,
        "fruit_fn": 0,
        "parser_confidence_sum": 0.0,
        "errors": 0,
        "warnings": 0,
    }

    for capture in manifest["captures"]:
        capture_id = capture["id"]
        annotation = json.loads((ROOT / capture["annotation_path"]).read_text(encoding="utf-8"))

        with Image.open(ROOT / capture["image_path"]) as image:
            parsed = parser.parse(image)

        gt_scores = {entry["player"]: entry["value"] for entry in annotation["hud"]["scores"]}
        gt_turn = int(annotation["hud"]["turn_counter"]["value"])
        gt_snake_cells_by_player, gt_heads_by_player, gt_snake_segments = _ground_truth_snakes(annotation)
        gt_fruits = {
            (fruit["cell"]["row"], fruit["cell"]["col"], int(fruit["value"]))
            for fruit in annotation["objects"]["fruits"]
        }

        board = parsed.board_state
        predicted_scores = {snake.color: snake.score for snake in board.snakes}
        predicted_snake_cells_by_player = {
            snake.color: {(row, col) for row, col, _ in snake.body}
            for snake in board.snakes
            if snake.alive
        }
        predicted_heads = {
            snake.color: _normalize_direction_cell(snake.head)
            for snake in board.snakes
            if snake.alive
        }
        predicted_fruits = {(fruit.row, fruit.col, fruit.value) for fruit in board.fruits}

        turn_ok = board.turn == gt_turn
        if turn_ok:
            aggregate["turn_ok"] += 1

        score_hits = sum(1 for color in PLAYER_COLORS if predicted_scores.get(color) == gt_scores.get(color))
        aggregate["score_exact_hits"] += score_hits
        aggregate["score_total"] += len(PLAYER_COLORS)

        snake_tp, snake_fp, snake_fn = _set_confusion(
            _flatten_player_cells(predicted_snake_cells_by_player),
            _flatten_player_cells(gt_snake_cells_by_player),
        )
        aggregate["snake_tp"] += snake_tp
        aggregate["snake_fp"] += snake_fp
        aggregate["snake_fn"] += snake_fn

        head_total = len(gt_heads_by_player)
        head_hits = sum(
            1
            for color, gt_head in gt_heads_by_player.items()
            if predicted_heads.get(color) == gt_head
        )
        aggregate["head_total"] += head_total
        aggregate["head_hits"] += head_hits

        fruit_tp, fruit_fp, fruit_fn = _set_confusion(predicted_fruits, gt_fruits)
        aggregate["fruit_tp"] += fruit_tp
        aggregate["fruit_fp"] += fruit_fp
        aggregate["fruit_fn"] += fruit_fn

        aggregate["parser_confidence_sum"] += parsed.confidence
        aggregate["errors"] += len(parsed.errors)
        aggregate["warnings"] += len(parsed.warnings)

        capture_reports.append(
            CaptureValidation(
                capture_id=capture_id,
                turn_ok=turn_ok,
                score_exact=score_hits / len(PLAYER_COLORS),
                snake_cell_precision=_precision(snake_tp, snake_fp),
                snake_cell_recall=_recall(snake_tp, snake_fn),
                snake_head_accuracy=(head_hits / head_total) if head_total else 1.0,
                fruit_precision=_precision(fruit_tp, fruit_fp),
                fruit_recall=_recall(fruit_tp, fruit_fn),
                parser_confidence=parsed.confidence,
                errors=parsed.errors,
                warnings=parsed.warnings,
            )
        )

        hard_case_candidates.extend(
            _extract_hard_cases(
                capture_id=capture_id,
                parsed=parsed.to_dict(),
                gt_snake_segments=gt_snake_segments,
                gt_fruits=gt_fruits,
                gt_occupied_cells={segment[2:4] for segment in gt_snake_segments},
            )
        )

    captures_count = len(capture_reports)
    snake_precision = _precision(aggregate["snake_tp"], aggregate["snake_fp"])
    snake_recall = _recall(aggregate["snake_tp"], aggregate["snake_fn"])
    fruit_precision = _precision(aggregate["fruit_tp"], aggregate["fruit_fp"])
    fruit_recall = _recall(aggregate["fruit_tp"], aggregate["fruit_fn"])
    metrics = ValidationMetrics(
        captures=captures_count,
        turn_accuracy=aggregate["turn_ok"] / captures_count if captures_count else 0.0,
        score_exact_rate=aggregate["score_exact_hits"] / aggregate["score_total"] if aggregate["score_total"] else 0.0,
        snake_cell_precision=snake_precision,
        snake_cell_recall=snake_recall,
        snake_cell_f1=_f1(snake_precision, snake_recall),
        snake_head_accuracy=aggregate["head_hits"] / aggregate["head_total"] if aggregate["head_total"] else 0.0,
        fruit_precision=fruit_precision,
        fruit_recall=fruit_recall,
        fruit_f1=_f1(fruit_precision, fruit_recall),
        mean_parser_confidence=aggregate["parser_confidence_sum"] / captures_count if captures_count else 0.0,
        total_errors=aggregate["errors"],
        total_warnings=aggregate["warnings"],
    )

    hard_cases = _rank_hard_cases(hard_case_candidates, min_hard_cases=min_hard_cases)
    return VisionValidationReport(
        metrics=metrics,
        captures=tuple(capture_reports),
        hard_cases=tuple(hard_cases),
    )


def validation_report_markdown(report: VisionValidationReport) -> str:
    lines = [
        "# VI-06 Validacion del VisionParser contra dataset",
        "",
        "## Metricas agregadas",
        "",
        "| metrica | valor |",
        "| --- | --- |",
    ]
    metrics = report.metrics.to_dict()
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"| `{key}` | `{value:.6f}` |")
        else:
            lines.append(f"| `{key}` | `{value}` |")

    lines.extend(
        [
            "",
            "## Metricas por captura",
            "",
            "| captura | turn_ok | score_exact | snake_precision | snake_recall | head_accuracy | fruit_precision | fruit_recall | parser_confidence |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for capture in report.captures:
        lines.append(
            f"| `{capture.capture_id}` | `{capture.turn_ok}` | `{capture.score_exact:.3f}` | "
            f"`{capture.snake_cell_precision:.3f}` | `{capture.snake_cell_recall:.3f}` | "
            f"`{capture.snake_head_accuracy:.3f}` | `{capture.fruit_precision:.3f}` | "
            f"`{capture.fruit_recall:.3f}` | `{capture.parser_confidence:.6f}` |"
        )

    lines.extend(
        [
            "",
            "## Casos dificiles comentados (top 20)",
            "",
            "| # | captura | componente | entidad | celda | confianza | match | comentario |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for case in report.hard_cases[:20]:
        lines.append(
            f"| {case.rank} | `{case.capture_id}` | `{case.component}` | `{case.entity}` | "
            f"`({case.cell[0]}, {case.cell[1]})` | `{case.confidence:.6f}` | `{case.matched}` | {case.comment} |"
        )

    return "\n".join(lines) + "\n"


def _ground_truth_snakes(annotation: Dict[str, object]) -> Tuple[Dict[str, set], Dict[str, Tuple[int, int, str]], set]:
    cells_by_player: Dict[str, set] = {}
    heads_by_player: Dict[str, Tuple[int, int, str]] = {}
    segments = set()
    for segment in annotation["objects"]["snakes"]:
        player = segment["player"]
        row = int(segment["cell"]["row"])
        col = int(segment["cell"]["col"])
        direction = _normalize_direction(segment["direction"])
        segment_class = segment["class"]
        cells_by_player.setdefault(player, set()).add((row, col))
        segments.add((player, segment_class, row, col, direction))
        if segment_class == "snake_head":
            heads_by_player[player] = (row, col, direction)
    return cells_by_player, heads_by_player, segments


def _extract_hard_cases(
    *,
    capture_id: str,
    parsed: Dict[str, object],
    gt_snake_segments: set,
    gt_fruits: set,
    gt_occupied_cells: set,
) -> List[Tuple[float, str, str, str, Tuple[int, int], float, bool, str]]:
    candidates = []
    snake_segments = parsed["components"]["snakes"]["snakes"]
    for entry in snake_segments:
        row = int(entry["cell"]["row"])
        col = int(entry["cell"]["col"])
        direction = _normalize_direction(entry["direction"])
        key = (entry["player"], entry["class"], row, col, direction)
        matched = key in gt_snake_segments
        conf = float(entry.get("confidence", 0.0))
        difficulty = (1.0 - conf) + (0.5 if not matched else 0.0) + _cell_difficulty_boost(row, col, gt_occupied_cells)
        comment = _hard_case_comment(
            component="snake",
            matched=matched,
            row=row,
            col=col,
            confidence=conf,
            crowded=_has_adjacent_occupied(row, col, gt_occupied_cells),
        )
        candidates.append(
            (difficulty, capture_id, "snake", f"{entry['player']}:{entry['class']}", (row, col), conf, matched, comment)
        )

    fruits = parsed["components"]["fruits"]["fruits"]
    for entry in fruits:
        row = int(entry["cell"]["row"])
        col = int(entry["cell"]["col"])
        value = int(entry["value"])
        key = (row, col, value)
        matched = key in gt_fruits
        conf = float(entry.get("confidence", 0.0))
        difficulty = (1.0 - conf) + (0.5 if not matched else 0.0) + _cell_difficulty_boost(row, col, gt_occupied_cells)
        comment = _hard_case_comment(
            component="fruit",
            matched=matched,
            row=row,
            col=col,
            confidence=conf,
            crowded=_has_adjacent_occupied(row, col, gt_occupied_cells),
        )
        candidates.append(
            (difficulty, capture_id, "fruit", f"fruit_{value}", (row, col), conf, matched, comment)
        )
    return candidates


def _rank_hard_cases(
    candidates: Iterable[Tuple[float, str, str, str, Tuple[int, int], float, bool, str]],
    *,
    min_hard_cases: int,
) -> List[HardCase]:
    sorted_candidates = sorted(
        list(candidates),
        key=lambda item: (-item[0], item[1], item[2], item[4][0], item[4][1], item[3]),
    )
    if len(sorted_candidates) < min_hard_cases:
        raise ValueError(
            f"not enough hard-case candidates ({len(sorted_candidates)}) to satisfy minimum {min_hard_cases}"
        )
    selected = sorted_candidates[:min_hard_cases]
    return [
        HardCase(
            rank=index + 1,
            capture_id=capture_id,
            component=component,
            entity=entity,
            cell=cell,
            confidence=confidence,
            matched=matched,
            comment=comment,
        )
        for index, (_, capture_id, component, entity, cell, confidence, matched, comment) in enumerate(selected)
    ]


def _flatten_player_cells(player_cells: Dict[str, set]) -> set:
    return {(player, row, col) for player, cells in player_cells.items() for row, col in cells}


def _normalize_direction_cell(cell: Tuple[int, int, str]) -> Tuple[int, int, str]:
    return (int(cell[0]), int(cell[1]), _normalize_direction(cell[2]))


def _normalize_direction(direction: str) -> str:
    if direction in {"N", "S", "E", "W"}:
        return direction
    for char in direction:
        if char in {"N", "S", "E", "W"}:
            return char
    return "N"


def _set_confusion(predicted: set, expected: set) -> Tuple[int, int, int]:
    tp = len(predicted & expected)
    fp = len(predicted - expected)
    fn = len(expected - predicted)
    return tp, fp, fn


def _precision(tp: int, fp: int) -> float:
    return tp / (tp + fp) if (tp + fp) else 1.0


def _recall(tp: int, fn: int) -> float:
    return tp / (tp + fn) if (tp + fn) else 1.0


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _cell_difficulty_boost(row: int, col: int, occupied_cells: set) -> float:
    boost = 0.0
    if row in (0, 43) or col in (0, 43):
        boost += 0.2
    if _has_adjacent_occupied(row, col, occupied_cells):
        boost += 0.2
    return boost


def _has_adjacent_occupied(row: int, col: int, occupied_cells: set) -> bool:
    neighbors = ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
    return any(neighbor in occupied_cells for neighbor in neighbors)


def _hard_case_comment(
    *,
    component: str,
    matched: bool,
    row: int,
    col: int,
    confidence: float,
    crowded: bool,
) -> str:
    notes = []
    if row in (0, 43) or col in (0, 43):
        notes.append("zona de borde")
    if crowded:
        notes.append("alta densidad local")
    if confidence < 0.995:
        notes.append("confianza relativa baja")
    if not notes:
        notes.append("caso de referencia estable")
    status = "coincide con ground truth" if matched else "desviacion respecto a ground truth"
    component_text = "segmento serpiente" if component == "snake" else "fruta"
    return f"{component_text}: {status}; " + ", ".join(notes)
