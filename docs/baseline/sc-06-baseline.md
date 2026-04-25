# SC-06 Baseline bot benchmark

- Generated UTC: 2026-04-25T17:32:34.248916+00:00
- Seed base: 20260425
- Games per matchup: 2
- Turn limit: 900
- Initial fruits: 5

## Common Interface

- Constructors follow `Player(playerID, color, game=None)`.
- Every bot exposes `play(state)` and `play_board_state(board_state)`.
- `play_board_state()` is the preferred entry point for `SnakeEnv` and other headless runners.

## Run Bots

```python
from GreedyPlayer import GreedyPlayer
from SurvivalPlayer import SurvivalPlayer
from AggressivePlayer import AggressivePlayer
from HybridPlayer import HybridPlayer

player = GreedyPlayer(0, 'G', game)
action = player.play_board_state(board_state)
```

## Run Benchmark

```powershell
python .\tools\benchmark_baselines.py --games 2 --seed 20260425
```

## Random vs Random vs Random vs Random

- Labels: Random, Random, Random, Random
- Seat labels: Random@1, Random@2, Random@3, Random@4
- Avg turns: 900
- Winner counts: `{"none": 2}`
- Avg scores: `{"Random@1": 30, "Random@2": 17.5, "Random@3": 15, "Random@4": 37.5}`
- Avg fruit scores: `{"Random@1": 30, "Random@2": 17.5, "Random@3": 15, "Random@4": 37.5}`

## Greedy vs Randoms

- Labels: Greedy, Random, Random, Random
- Seat labels: Greedy@1, Random@2, Random@3, Random@4
- Avg turns: 221.5
- Winner counts: `{"Greedy@1": 2}`
- Avg scores: `{"Greedy@1": 477.5, "Random@2": 7.5, "Random@3": 7.5, "Random@4": 5}`
- Avg fruit scores: `{"Greedy@1": 132.5, "Random@2": 7.5, "Random@3": 7.5, "Random@4": 5}`

## Survival vs Randoms

- Labels: Survival, Random, Random, Random
- Seat labels: Survival@1, Random@2, Random@3, Random@4
- Avg turns: 900
- Winner counts: `{"none": 2}`
- Avg scores: `{"Random@2": 17.5, "Random@3": 37.5, "Random@4": 0, "Survival@1": 0}`
- Avg fruit scores: `{"Random@2": 17.5, "Random@3": 37.5, "Random@4": 0, "Survival@1": 0}`

## Hybrid vs Greedy vs Survival vs Aggressive

- Labels: Hybrid, Greedy, Survival, Aggressive
- Seat labels: Hybrid@1, Greedy@2, Survival@3, Aggressive@4
- Avg turns: 167
- Winner counts: `{"Hybrid@1": 1, "none": 1}`
- Avg scores: `{"Aggressive@4": 0, "Greedy@2": 30, "Hybrid@1": 112.5, "Survival@3": 0}`
- Avg fruit scores: `{"Aggressive@4": 0, "Greedy@2": 30, "Hybrid@1": 97.5, "Survival@3": 0}`

## Limitations

- Greedy and Aggressive are heuristic policies; they are stable but not optimal.
- Survival maximizes reachable space, so it may ignore fruit in open boards.
- Results are deterministic for the reported seed base and fixed game configuration.
