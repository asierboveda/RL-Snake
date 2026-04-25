# VI-05 VisionParser contract

`VisionParser` convierte una captura del juego en una estructura consumible por RL (`BoardState`) sin usar el estado interno del motor.

## Interfaz

```python
from vision_parser import VisionParser

parser = VisionParser()
result = parser.parse(image)
```

`parse(image)` devuelve `VisionParseResult` con:

- `board_state`: `BoardState` canonico (`rows`, `cols`, `turn`, `snakes`, `fruits`, estado terminal).
- `confidence`: score global de parseo (`0.0..1.0`).
- `component_confidence`: confianza por componente (`hud`, `snakes`, `fruits`, `consistency`).
- `errors`: inconsistencias duras detectadas.
- `warnings`: inconsistencias blandas (por ejemplo fruta sobre serpiente en el frame).
- `metadata`: geometria, umbrales de detectores y politica de estimacion.
- `components`: salida cruda de HUD/serpientes/frutas para trazabilidad.

## Pipeline

1. `detect_grid_geometry(image)` fija marco y conversion pixel-celda.
2. `detect_hud(image)` obtiene turno y scores por color.
3. `detect_snakes(image)` reconstruye segmentos por jugador.
4. `detect_fruits(image)` obtiene frutas activas y valor.
5. Se arma `BoardState` normalizado para RL.

## Mapeo a BoardState

- Jugadores canonicos:
  - `A/G` -> `player_id=0`
  - `B/B` -> `player_id=1`
  - `C/R` -> `player_id=2`
  - `D/Y` -> `player_id=3`
- `score` viene del HUD.
- `body` viene de la cadena ordenada de vision (head -> tail).
- `alive` es `True` si hay cuerpo detectado para ese color.
- Frutas se convierten a `FruitState(row, col, value, time_left=0)`.

## Regla del umbral 120 (anti-falsos hunters)

En vision no hay historial exacto de fruta vs kills, por eso `fruit_score` no puede recuperarse de forma exacta desde un frame.

El parser aplica una politica conservadora:

- `fruit_score_lower_bound = min(score_total, 10 * (len(body) - 1))`
- `fruit_score` en `BoardState` se fija a ese lower bound.

Con esto una serpiente **nunca** se marca como hunter (`fruit_score >= 120`) sin evidencia estructural suficiente, evitando violar la regla de que por debajo de 120 no deberia matar validamente.

Los bounds por jugador se exponen en `metadata["fruit_score_bounds"]`.

## Validaciones de consistencia

El parser reporta:

- celdas de serpientes solapadas entre jugadores (`errors`);
- frutas duplicadas en la misma celda (`errors`);
- fruta sobre celda ocupada por serpiente (`warnings`);
- score positivo sin cuerpo detectado (`warnings`).

## Compatibilidad con RL

La salida `board_state` de `VisionParseResult` es directa para `RLPlayer.play_board_state(...)` sin acoplarse al entorno headless.

## Validacion contra dataset (VI-06)

La evaluacion oficial del parser visual se ejecuta con:

```powershell
python .\tools\validate_vision_parser.py
```

Este comando genera:

- `docs/vision_validation_report.md` (reporte humano con metricas y top-20 casos dificiles comentados).
- `datasets/vision_captures/vision_validation_report.json` (salida estructurada para QA/CI).
