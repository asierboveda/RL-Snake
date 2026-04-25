# RL Observation v1

`rl_observation_v1` define la entrada del agente RL desde `BoardState`, sin imagen, OCR ni acceso al render.

## Contrato de entrada revisado

La observacion usa solo campos publicos de `BoardState`, `SnakeState` y `FruitState`:

- `BoardState.turn`, `rows`, `cols`, `snakes`, `fruits`, `game_alive`, `winner_id`, `terminal_reason`.
- `SnakeState.player_id`, `label`, `color`, `alive`, `body`, `score`, `fruit_score`.
- `SnakeState.head`, derivado de `body[0]`.
- `SnakeState.is_hunter`, derivado de `fruit_score >= 120`.
- `FruitState.row`, `col`, `value`, `time_left`.

No se ha modificado el contrato actual. La direccion de movimiento existe por segmento como tercer valor de cada celda `(row, col, direction)`, aunque `rl_observation_v1` solo la usa indirectamente al leer la cabeza.

## Salida

`build_observation(board, player_id)` devuelve:

```python
{
    "spatial": np.ndarray,   # float32, shape (10, H, W)
    "features": np.ndarray,  # float32, shape (23,)
}
```

Donde `H = BoardState.rows` y `W = BoardState.cols`.

La definicion versionada esta en `FEATURE_SET_V1`:

- `version`: `rl_observation_v1`
- `score_scale`: `120.0`
- `turn_limit`: `900`

## Canales espaciales

Todos los canales tienen rango `[0.0, 1.0]`.

| Canal | Nombre | Significado |
| --- | --- | --- |
| 0 | `own_body` | Cuerpo propio sin incluir cabeza. |
| 1 | `own_head` | Cabeza propia. |
| 2 | `enemy_bodies` | Cuerpos enemigos vivos sin incluir cabeza. |
| 3 | `enemy_heads` | Cabezas enemigas vivas. |
| 4 | `fruits` | Celdas con fruta activa. |
| 5 | `walls_or_bounds` | Perimetro del tablero. El contrato no tiene celdas fuera de tablero, por eso se marca el borde interno. |
| 6 | `dangerous_enemies` | Celdas de enemigos no atacables de forma segura. |
| 7 | `attackable_enemies` | Celdas de enemigos atacables por score propio. |
| 8 | `immediate_danger` | Celdas ocupadas y vecindad Manhattan de enemigos peligrosos. |
| 9 | `free_navigable` | Celdas no ocupadas y no marcadas como peligro inmediato. |

Un enemigo es atacable si:

```python
own.fruit_score >= 120 and own.fruit_score > enemy.fruit_score
```

Todo enemigo vivo que no cumpla esa condicion se trata como peligroso. Es una decision conservadora: antes del umbral de 120, una colision puede matar a ambas serpientes, asi que no debe verse como una accion ofensiva segura.

## Features escalares

El vector mantiene orden estable por `player_id`: los rivales se ordenan ascendentemente y ocupan tres slots fijos. Si hay menos de tres rivales, el slot se rellena con cero salvo distancias globales, que usan `1.0` cuando no hay objetivo.

| Feature | Rango esperado | Significado |
| --- | --- | --- |
| `own_fruit_score_norm` | `score / 120` | Puntuacion de fruta propia normalizada por umbral de kill. |
| `rival_i_fruit_score_norm` | `score / 120` | Puntuacion de fruta del rival `i`. |
| `rival_i_score_delta_norm` | `(own - rival) / 120` | Ventaja o desventaja de fruta contra rival `i`. |
| `own_can_kill` | `0` o `1` | `1` si la serpiente propia ha alcanzado 120 fruta. |
| `rival_i_can_kill` | `0` o `1` | `1` si el rival `i` ha alcanzado 120 fruta. |
| `own_length_norm` | `len(body) / max(H, W)` | Longitud propia normalizada. |
| `rival_i_length_norm` | `len(body) / max(H, W)` | Longitud del rival `i`. |
| `rival_i_alive` | `0` o `1` | Estado vivo/muerto del rival `i`. |
| `nearest_fruit_distance_norm` | `dist / (H + W)` | Distancia Manhattan a la fruta mas cercana. |
| `nearest_dangerous_enemy_distance_norm` | `dist / (H + W)` | Distancia Manhattan a la cabeza peligrosa mas cercana. |
| `nearest_attackable_enemy_distance_norm` | `dist / (H + W)` | Distancia Manhattan a la cabeza atacable mas cercana. |
| `alive_snake_count_norm` | `alive / total` | Proporcion de serpientes vivas. |
| `turn_norm` | `min(turn / 900, 1)` | Turno normalizado por limite de episodio. |

## Decisiones de diseno

- La observacion es determinista: no depende de random, reloj ni imagen.
- No se entrena ningun modelo en esta tarjeta.
- No se acopla a `RLPlayer`; esta API esta preparada para un agente posterior tipo DQN/PPO/self-play.
- La fruta se marca binaria en el canal espacial. El valor de fruta puede anadirse en una version posterior si el reward o la politica lo necesita.
- Las serpientes muertas no se colocan en canales enemigos, pero sus slots escalares conservan `alive = 0` para mantener forma fija.

## Tests

La cobertura principal esta en `tests/test_rl_observation.py`:

- Shape y dtype de `spatial` y `features`.
- Marcado de cabeza, cuerpo, fruta, perimetro y celdas libres.
- Clasificacion de enemigos peligrosos frente a atacables respetando el umbral de 120.
- Orden estable de features escalares por slots de rival.
- Rechazo explicito de `player_id` inexistente o serpiente propia muerta.
