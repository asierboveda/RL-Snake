# SC-02 Baseline reproducible

- Generated UTC: 2026-04-25T09:45:38.379829+00:00
- Git commit: 3edd89b8b3165ad5c670a923dcfc2a513c5397b6
- Dirty files at freeze: 21
- Q-table: `models/q_table_p0.pkl`
- Q-table SHA256: `2770970429104e5e9a941862de7f46d9cbdbfcc3d95901b7070948e1a4893d65`

## Headless baseline

- Eval games: 30
- Base seed: 10001
- RL win rate: 66.7%
- Avg turns: 76.1
- Avg RL score: 21.5
- Avg RL fruit score: 21.5
- Avg RL kills: 0.0
- Early death rate: 23.3%
- Death causes: `{"alive": 20, "enemy": 9, "unknown": 1}`
- Outcomes: `{"all_dead": 3, "single_alive": 27}`
- Diagnosis: `consolidation` (alive)

## Visual baseline

- Seed: 1
- Noise: 0.01
- Turns: 99
- Frames observed: 99
- Scores: `[15, 30, 25, 90]`
- Winner: RLPlayer
- Outcome: single_alive
- Final frame: `docs/baseline/sc-02-final-frame.png`
- Final frame SHA256: `9300f9a715a882108ebda48951ed25ac68c4562f9324908bcbae4480c949a56b`

## Reproduce

```powershell
python .\tools\freeze_sc02_baseline.py
```
