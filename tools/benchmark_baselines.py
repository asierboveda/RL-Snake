"""
Run a minimal reproducible benchmark for the baseline bot set.

The script writes a JSON report and a short Markdown summary into docs/baseline/.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from AggressivePlayer import AggressivePlayer
from GreedyPlayer import GreedyPlayer
from HybridPlayer import HybridPlayer
from RandomPlayer import RandomPlayer
from snake_env import SnakeEnv
from SurvivalPlayer import SurvivalPlayer
BASELINE_DIR = ROOT / "docs" / "baseline"
JSON_REPORT = BASELINE_DIR / "sc-06-baseline.json"
MD_REPORT = BASELINE_DIR / "sc-06-baseline.md"


PLAYER_CLASSES = {
    "Random": RandomPlayer,
    "Greedy": GreedyPlayer,
    "Survival": SurvivalPlayer,
    "Aggressive": AggressivePlayer,
    "Hybrid": HybridPlayer,
}

MATCHUPS = [
    ("Random vs Random vs Random vs Random", ["Random", "Random", "Random", "Random"]),
    ("Greedy vs Randoms", ["Greedy", "Random", "Random", "Random"]),
    ("Survival vs Randoms", ["Survival", "Random", "Random", "Random"]),
    ("Hybrid vs Greedy vs Survival vs Aggressive", ["Hybrid", "Greedy", "Survival", "Aggressive"]),
]


def make_players(labels):
    players = []
    colors = ("G", "B", "R", "Y")
    for idx, label in enumerate(labels):
        players.append(PLAYER_CLASSES[label](idx, colors[idx], None))
    return players


def display_labels(labels):
    return [f"{label}@{idx + 1}" for idx, label in enumerate(labels)]


def choose_action(player, state):
    if hasattr(player, "play_board_state"):
        return player.play_board_state(state)
    return player.play(state)


def run_episode(labels, seat_labels, seed, initial_fruits, turn_limit):
    env = SnakeEnv(seed=seed, initial_fruits=initial_fruits, turn_limit=turn_limit)
    state = env.reset()
    players = make_players(labels)

    while True:
        actions = {
            idx: choose_action(player, state)
            for idx, player in enumerate(players)
        }
        transition = env.step(actions)
        state = transition.state
        if transition.done:
            break

    alive = [idx for idx, snake in enumerate(state.snakes) if snake.alive]
    if state.winner_id is not None:
        winner = seat_labels[state.winner_id]
    elif len(alive) == 1:
        winner = seat_labels[alive[0]]
    else:
        winner = None

    return {
        "seed": seed,
        "turns": state.turn,
        "winner": winner,
        "winner_id": state.winner_id,
        "terminal_reason": state.terminal_reason or ("turn_limit" if state.game_alive else None),
        "scores": {seat_labels[idx]: snake.score for idx, snake in enumerate(state.snakes)},
        "fruit_scores": {seat_labels[idx]: snake.fruit_score for idx, snake in enumerate(state.snakes)},
        "alive": {seat_labels[idx]: snake.alive for idx, snake in enumerate(state.snakes)},
        "replay_steps": len(env.replay),
    }


def summarize(results, seat_labels):
    turns = [item["turns"] for item in results]
    winner_counts = Counter(item["winner"] or "none" for item in results)
    avg_scores = {
        label: round(statistics.mean(item["scores"][label] for item in results), 2)
        for label in seat_labels
    }
    avg_fruit_scores = {
        label: round(statistics.mean(item["fruit_scores"][label] for item in results), 2)
        for label in seat_labels
    }
    return {
        "games": len(results),
        "avg_turns": round(statistics.mean(turns), 2) if turns else 0.0,
        "min_turns": min(turns) if turns else 0,
        "max_turns": max(turns) if turns else 0,
        "winner_counts": dict(winner_counts),
        "avg_scores": avg_scores,
        "avg_fruit_scores": avg_fruit_scores,
    }


def write_report(report):
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    lines = [
        "# SC-06 Baseline bot benchmark",
        "",
        f"- Generated UTC: {report['generated_utc']}",
        f"- Seed base: {report['seed_base']}",
        f"- Games per matchup: {report['games_per_matchup']}",
        f"- Turn limit: {report['turn_limit']}",
        f"- Initial fruits: {report['initial_fruits']}",
        "",
        "## Common Interface",
        "",
        "- Constructors follow `Player(playerID, color, game=None)`.",
        "- Every bot exposes `play(state)` and `play_board_state(board_state)`.",
        "- `play_board_state()` is the preferred entry point for `SnakeEnv` and other headless runners.",
        "",
        "## Run Bots",
        "",
        "```python",
        "from GreedyPlayer import GreedyPlayer",
        "from SurvivalPlayer import SurvivalPlayer",
        "from AggressivePlayer import AggressivePlayer",
        "from HybridPlayer import HybridPlayer",
        "",
        "player = GreedyPlayer(0, 'G', game)",
        "action = player.play_board_state(board_state)",
        "```",
        "",
        "## Run Benchmark",
        "",
        "```powershell",
        "python .\\tools\\benchmark_baselines.py --games 2 --seed 20260425",
        "```",
        "",
    ]
    for matchup in report["matchups"]:
        lines.extend([
            f"## {matchup['name']}",
            "",
            f"- Labels: {', '.join(matchup['labels'])}",
            f"- Seat labels: {', '.join(matchup['seat_labels'])}",
            f"- Avg turns: {matchup['summary']['avg_turns']}",
            f"- Winner counts: `{json.dumps(matchup['summary']['winner_counts'], sort_keys=True)}`",
            f"- Avg scores: `{json.dumps(matchup['summary']['avg_scores'], sort_keys=True)}`",
            f"- Avg fruit scores: `{json.dumps(matchup['summary']['avg_fruit_scores'], sort_keys=True)}`",
            "",
        ])
    lines.extend([
        "## Limitations",
        "",
        "- Greedy and Aggressive are heuristic policies; they are stable but not optimal.",
        "- Survival maximizes reachable space, so it may ignore fruit in open boards.",
        "- Results are deterministic for the reported seed base and fixed game configuration.",
        "",
    ])
    MD_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run baseline bot benchmark.")
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--turn-limit", type=int, default=900)
    parser.add_argument("--initial-fruits", type=int, default=5)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed_base": args.seed,
        "games_per_matchup": args.games,
        "turn_limit": args.turn_limit,
        "initial_fruits": args.initial_fruits,
        "matchups": [],
    }

    for matchup_index, (name, labels) in enumerate(MATCHUPS):
        seats = display_labels(labels)
        results = []
        for game_index in range(args.games):
            seed = args.seed + matchup_index * 1000 + game_index
            results.append(run_episode(labels, seats, seed, args.initial_fruits, args.turn_limit))
        report["matchups"].append({
            "name": name,
            "labels": labels,
            "seat_labels": seats,
            "summary": summarize(results, seats),
            "results": results,
        })

    write_report(report)

    print(json.dumps({
        "seed_base": report["seed_base"],
        "games_per_matchup": report["games_per_matchup"],
        "matchups": [
            {
                "name": item["name"],
                "summary": item["summary"],
            }
            for item in report["matchups"]
        ],
    }, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
