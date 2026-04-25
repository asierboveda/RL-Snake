"""
Freeze SC-02 baseline metrics.

Outputs:
  docs/baseline/sc-02-baseline.json
  docs/baseline/sc-02-baseline.md
  docs/baseline/sc-02-final-frame.png
"""

import hashlib
import importlib.util
import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = ROOT / "docs" / "baseline"
REPORT_JSON = BASELINE_DIR / "sc-02-baseline.json"
REPORT_MD = BASELINE_DIR / "sc-02-baseline.md"
FINAL_FRAME = BASELINE_DIR / "sc-02-final-frame.png"

HEADLESS_ITERATION = 1
VISUAL_SEED = 1
VISUAL_NOISE = 0.01
INITIAL_FRUITS = 5
TURN_LIMIT = 900
MIN_SCORE_FOR_WINNING = 120


def load_observer():
    observer_path = ROOT / ".github" / "scripts" / "run_and_observe.py"
    spec = importlib.util.spec_from_file_location("run_and_observe", observer_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path):
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args):
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def git_dirty_files():
    try:
        output = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def classify_visual_winner(game):
    labels = ["RLPlayer", "Greedy-B", "Greedy-R", "Greedy-Y"]
    alive = [(i, snake) for i, snake in enumerate(game.snakes) if snake.isAlive]
    if len(alive) == 1:
        return labels[alive[0][0]], "single_alive"
    if len(alive) == 0:
        return "No winner", "all_dead"

    scores = game.getScores()
    max_score = max(scores)
    if max_score < MIN_SCORE_FOR_WINNING:
        return "No winner", "draw_low_score"
    winners = [idx for idx, score in enumerate(scores) if score == max_score]
    if len(winners) == 1:
        return labels[winners[0]], "score_winner"
    return "No winner", "draw_tie"


def run_headless_baseline(observer):
    base_seed = 10000 + HEADLESS_ITERATION
    np.random.seed(base_seed)
    results = [
        observer.run_observed_game(base_seed + idx)
        for idx in range(observer.EVAL_GAMES)
    ]
    metrics, diagnosis = observer.build_observation_report(results)
    return {
        "config": {
            "iteration": HEADLESS_ITERATION,
            "base_seed": base_seed,
            "eval_games": observer.EVAL_GAMES,
            "turn_limit": observer.TURN_LIMIT,
            "initial_fruits": INITIAL_FRUITS,
            "noise": 0.0,
            "rl_epsilon": 0.0,
            "rl_training_enabled": False,
            "opponents": ["Greedy-B", "Greedy-R", "Greedy-Y"],
        },
        "metrics": metrics,
        "diagnosis": diagnosis,
        "games": results,
    }


def run_visual_baseline():
    from GreedyPlayer import GreedyPlayer
    from RLPlayer import RLPlayer
    from SnakeGame import SnakeGame

    random.seed(VISUAL_SEED)
    np.random.seed(VISUAL_SEED)
    game = SnakeGame()
    for _ in range(INITIAL_FRUITS):
        game.addRandomFruit()
    game.setNoise(VISUAL_NOISE)

    players = [
        RLPlayer(0, "G", game, epsilon=0.0, training_enabled=False),
        GreedyPlayer(1, "B", game),
        GreedyPlayer(2, "R", game),
        GreedyPlayer(3, "Y", game),
    ]

    frame_count = 0
    while game.turn < TURN_LIMIT and game.gameIsAlive():
        game.getSnapshot()
        frame_count += 1

        directions = [player.play(None) for player in players]
        for idx, direction in enumerate(directions):
            game.movePlayer(idx, direction)
        game.checkMovements()
        game.update()
        game.turn += 1

    final_board = game.getFinalSnapshot()
    final_image = Image.fromarray((final_board * 255).astype(np.uint8))
    final_image.save(FINAL_FRAME)
    winner, outcome = classify_visual_winner(game)

    scores = game.getScores()
    snakes = []
    for idx, snake in enumerate(game.snakes):
        fruit_score = snake.getFruitScore()
        total_score = snake.getScore()
        snakes.append({
            "player_index": idx,
            "score": total_score,
            "fruit_score": fruit_score,
            "kills": max(0, (total_score - fruit_score) // 30),
            "alive": snake.isAlive,
        })

    return {
        "config": {
            "seed": VISUAL_SEED,
            "noise": VISUAL_NOISE,
            "turn_limit": TURN_LIMIT,
            "initial_fruits": INITIAL_FRUITS,
            "rl_epsilon": 0.0,
            "rl_training_enabled": False,
            "opponents": ["Greedy-B", "Greedy-R", "Greedy-Y"],
        },
        "metrics": {
            "turns": game.turn,
            "frame_count": frame_count,
            "scores": scores,
            "winner": winner,
            "outcome": outcome,
            "final_frame": str(FINAL_FRAME.relative_to(ROOT)).replace("\\", "/"),
            "final_frame_sha256": sha256_file(FINAL_FRAME),
        },
        "snakes": snakes,
    }


def write_markdown(report):
    headless = report["headless"]["metrics"]
    diagnosis = report["headless"]["diagnosis"]
    visual = report["visual"]["metrics"]
    q_table = report["artifacts"]["q_table"]

    lines = [
        "# SC-02 Baseline reproducible",
        "",
        f"- Generated UTC: {report['generated_utc']}",
        f"- Git commit: {report['git']['commit'] or 'unknown'}",
        f"- Dirty files at freeze: {len(report['git']['dirty_files'])}",
        f"- Q-table: `{q_table['path']}`",
        f"- Q-table SHA256: `{q_table['sha256']}`",
        "",
        "## Headless baseline",
        "",
        f"- Eval games: {report['headless']['config']['eval_games']}",
        f"- Base seed: {report['headless']['config']['base_seed']}",
        f"- RL win rate: {headless['rl_win_rate']}%",
        f"- Avg turns: {headless['avg_turns']}",
        f"- Avg RL score: {headless['avg_rl_score']}",
        f"- Avg RL fruit score: {headless['avg_rl_fruit_score']}",
        f"- Avg RL kills: {headless['avg_rl_kills']}",
        f"- Early death rate: {headless['early_death_rate']}%",
        f"- Death causes: `{json.dumps(headless['death_cause_counts'], sort_keys=True)}`",
        f"- Outcomes: `{json.dumps(headless['outcome_counts'], sort_keys=True)}`",
        f"- Diagnosis: `{diagnosis['problem_code']}` ({diagnosis['dominant_cause']})",
        "",
        "## Visual baseline",
        "",
        f"- Seed: {report['visual']['config']['seed']}",
        f"- Noise: {report['visual']['config']['noise']}",
        f"- Turns: {visual['turns']}",
        f"- Frames observed: {visual['frame_count']}",
        f"- Scores: `{json.dumps(visual['scores'])}`",
        f"- Winner: {visual['winner']}",
        f"- Outcome: {visual['outcome']}",
        f"- Final frame: `{visual['final_frame']}`",
        f"- Final frame SHA256: `{visual['final_frame_sha256']}`",
        "",
        "## Reproduce",
        "",
        "```powershell",
        "python .\\tools\\freeze_sc02_baseline.py",
        "```",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main():
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))

    observer = load_observer()
    q_table_path = ROOT / "models" / "q_table_p0.pkl"

    report = {
        "task": "SC-02",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "git": {
            "commit": git_value("rev-parse", "HEAD"),
            "branch": git_value("branch", "--show-current"),
            "dirty_files": git_dirty_files(),
        },
        "artifacts": {
            "q_table": {
                "path": "models/q_table_p0.pkl",
                "sha256": sha256_file(q_table_path),
            },
        },
        "headless": run_headless_baseline(observer),
        "visual": run_visual_baseline(),
    }

    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    write_markdown(report)

    print(f"Wrote {REPORT_JSON.relative_to(ROOT)}")
    print(f"Wrote {REPORT_MD.relative_to(ROOT)}")
    print(f"Wrote {FINAL_FRAME.relative_to(ROOT)}")
    print(json.dumps({
        "headless": report["headless"]["metrics"],
        "visual": report["visual"]["metrics"],
    }, ensure_ascii=True))


if __name__ == "__main__":
    main()
