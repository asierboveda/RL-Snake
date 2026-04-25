# Headless Environment Contract

`SnakeEnv` is the reproducible, Gym-like API for training agents without image rendering. It wraps the existing `SnakeGame` engine and exposes only structured state through `BoardState`.

## API

```python
from snake_env import SnakeEnv

env = SnakeEnv(seed=42, initial_fruits=5, turn_limit=900)
state = env.reset()
transition = env.step({0: "S", 1: "S", 2: "N", 3: "N"})
```

## Constructor

- `seed`: optional integer used to seed Python `random` and `numpy`.
- `initial_fruits`: number of fruits spawned during `reset`.
- `turn_limit`: maximum number of turns before `done=True`.

## `reset(seed=None) -> BoardState`

Creates a fresh `SnakeGame`, disables visual noise, spawns initial fruits, clears replay history, and returns turn `0` as a `BoardState`.

Passing a seed to `reset` replaces the environment seed for that episode.

## `step(actions) -> EnvTransition`

`actions` can be either:

- a dictionary `{player_id: action}`;
- a sequence ordered by player id, for example `["S", "S", "N", "N"]`.

Missing actions default to the snake current direction. Invalid actions raise `ValueError`.

`EnvTransition` contains:

- `state`: next `BoardState`;
- `rewards`: per-player score delta minus `100` when a previously alive snake dies;
- `done`: true when the game is over or `turn_limit` is reached;
- `info`: current turn, scores, and legal actions after the step.

## Legal Actions

`legal_actions(player_id)` returns actions that keep the next cell inside the board and outside currently occupied cells. Dead snakes return an empty tuple.

## Replay Format

Every `step` appends one replay entry. `to_replay_dict()` returns:

```json
{
  "seed": 42,
  "initial_fruits": 5,
  "turn_limit": 900,
  "steps": [
    {
      "turn": 1,
      "actions": {"0": "S", "1": "S", "2": "N", "3": "N"},
      "state": {"turn": 1},
      "rewards": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0},
      "done": false
    }
  ]
}
```

The `state` field is the full `BoardState.to_dict()` payload.

## Example Episode

```python
env = SnakeEnv(seed=7, initial_fruits=3, turn_limit=5)
state = env.reset()

while True:
    actions = {
        player_id: legal[0] if legal else "N"
        for player_id in range(4)
        for legal in [env.legal_actions(player_id)]
    }
    transition = env.step(actions)
    if transition.done:
        break

replay = env.to_replay_dict()
```

This loop never reads pixels and does not depend on rendering. Vision integration must produce a compatible `BoardState` separately.

