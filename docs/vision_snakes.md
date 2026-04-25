# VI-03 Deteccion de serpientes por imagen

Este modulo implementa deteccion de serpientes sin leer estado interno del motor.

## Objetivo

- Separar sprites por jugador (`G`, `B`, `R`, `Y`).
- Identificar tipo de segmento (`snake_head`, `snake_body`, `snake_tail`).
- Leer direccion del sprite (`N`, `E`, `S`, `W`, `NE`, `NW`, `SE`, `SW`, `EN`, `ES`, `WN`, `WS`).
- Reconstruir una cadena por jugador para producir datos comparables con `BoardState`.

## Implementacion

Archivo: `vision_snakes.py`.

Estrategia:

1. Se recorre cada celda jugable `44x44` con la geometria de `vision_grid.py`.
2. Se compara el recorte `17x17` contra plantillas de serpiente en `input/17`:
   - `snakehead_<COLOR>_<DIR>.png`
   - `snake_<COLOR>_<DIR>.png`
   - `snaketail_<COLOR>_<DIR>.png`
3. Se exige:
   - error MSE menor que umbral fijo (`SNAKE_MATCH_MAX_ERROR`);
   - ventaja frente a plantillas no-serpiente (frutas/bomba) con margen minimo.
4. Cada deteccion incluye `confidence` basada en separacion de error frente a no-serpiente.
5. Se agrupa por jugador y se reconstruye una cadena (head -> body -> tail) por vecindad Manhattan.

## Errores tipicos y mitigacion

| Caso | Riesgo | Mitigacion |
| --- | --- | --- |
| Fruta parecida a serpiente | Falso positivo | Comparar contra plantillas no-serpiente y exigir margen |
| Ruido visual | Bajada de confianza | Umbral de error + `confidence` por segmento |
| Segmentos desconectados en captura parcial | Orden inestable | Reconstruccion por vecindad; fallback determinista por distancia |
| Mismo jugador con varias piezas cercanas | Ambiguedad de cadena | Prioridad de clase (`head`, `body`, `tail`) y orden fijo |

## Ejemplos visuales (dataset base)

| Captura | Jugador | Cadena reconstruida (row, col, dir) |
| --- | --- | --- |
| `snake000.png` | `G` | `(10, 10, N)` |
| `snake012.png` | `B` | `(1, 37, E) -> (1, 36, E)` |
| `snake025.png` | `R` | `(31, 14, N) -> (32, 14, N) -> (33, 14, N)` |

## Salida principal

`detect_snakes(image)` devuelve:

- `segments`: lista plana de segmentos detectados con clase, jugador, celda, bbox, direccion y confidence.
- `players`: reconstruccion por jugador con `head`, `body`, `tail` y `board_body` (lista ordenada `row,col,direction`).

Esto prepara la base para VI-05 (parser completo imagen -> BoardState).
