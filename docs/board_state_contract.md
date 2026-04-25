# BoardState Contract and Official Rules

This document defines the stable state contract for the Snake project. It is the source of truth for vision parsing, RL input, engine validation, and future tests.

## Scope

`BoardState` describes one completed turn of the game in grid coordinates. It does not contain pixels, model tensors, rendering data, or player implementation details.

The executable contract lives in `board_state.py`.

## Constants

- Board size: `44` rows x `44` columns.
- Players: `A`, `B`, `C`, `D`.
- Snake colors: `G`, `B`, `R`, `Y`.
- Directions: `N`, `S`, `E`, `W`.
- Fruit values: `10`, `15`, `20`.
- Fruit kill threshold: `120` fruit points.
- Kill award: `30` score points.

## Coordinate System

All game entities use zero-based board coordinates:

- `row`: `0..43`, increasing downward.
- `col`: `0..43`, increasing rightward.
- HUD pixels and rendered image offsets are outside this contract.

## BoardState Fields

Every turn state must provide:

- `turn`: non-negative integer turn index.
- `rows`: must be `44`.
- `cols`: must be `44`.
- `snakes`: list of `SnakeState`.
- `fruits`: list of `FruitState`.
- `game_alive`: true while more than one snake remains alive.
- `winner_id`: player id when the winner is known, otherwise null.
- `terminal_reason`: null while the game is active, otherwise one of the official terminal reasons.

When `game_alive` is true, `terminal_reason` must be null. When `game_alive` is false, `terminal_reason` must be one of the official victory reasons below. If `winner_id` is not null, it must reference an existing snake in `snakes`.

## SnakeState Fields

Each snake must provide:

- `player_id`: stable non-negative integer id.
- `label`: one of `A`, `B`, `C`, `D`.
- `color`: one of `G`, `B`, `R`, `Y`.
- `alive`: whether the snake can still act.
- `body`: ordered cells from head to tail.
- `head`: derived from `body[0]`.
- `score`: total score from fruit and kills.
- `fruit_score`: score earned only from fruit.
- `is_hunter`: derived as `fruit_score >= 120`.

Each body cell is `(row, col, direction)`.

## FruitState Fields

Each fruit must provide:

- `row`: board row.
- `col`: board column.
- `value`: one of `10`, `15`, `20`.
- `time_left`: non-negative integer turns before expiry.

## Collision Rules

Collision means any occupied cell from one snake overlaps any occupied cell from another snake.

When two snakes collide:

- If both have the same `fruit_score`, both die and no kill points are awarded.
- If neither snake has `fruit_score >= 120`, both die and no kill points are awarded.
- Otherwise, the snake with the higher `fruit_score` kills the lower `fruit_score` snake.
- A kill awards `30` total score points to the killer.
- Kill points do not increase `fruit_score`.

The threshold is inclusive: `fruit_score == 120` is hunter-eligible.

## Non-Collision Death Rules

A snake dies if:

- Its head or body is outside the board.
- Its head overlaps its own body.

These deaths do not award kill points.

## Fruit Rules

- New fruit may spawn only on empty cells.
- Fruit expires when `time_left` reaches zero.
- Eating fruit increases both `score` and `fruit_score` by the fruit value.
- Eating fruit grows the snake by one body cell for that move.

## Victory Rules

Winner determination follows this order:

- If exactly one snake is alive, that snake wins.
- If no snakes are alive, there is no winner with reason `all_dead`.
- If multiple snakes are alive and the maximum total `score` is below `120`, there is no winner with reason `too_few_points`.
- If multiple snakes are alive and exactly one snake has the maximum total `score` with at least `120`, that snake wins with reason `score_threshold`.
- If multiple alive snakes tie for maximum total `score`, there is no winner with reason `draw`.

## Serialization

`BoardState.to_dict()` returns a plain Python dictionary with only primitive values, tuples, lists, booleans, and nulls. Consumers may JSON-encode it after converting tuples to lists if strict JSON shape is required.
