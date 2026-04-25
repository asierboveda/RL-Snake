# SC-03 Dataset etiquetado de vision

Este documento define el dataset base para validar el parser de imagen del juego Snake Battle Royale. El objetivo es que las tareas de percepcion puedan convertir capturas PNG en una estructura equivalente a `BoardState` sin leer estado interno del motor.

## Capturas seleccionadas

Las capturas disponibles estan en `output/` y todas tienen resolucion `793x901`. Se selecciona un subconjunto pequeno y representativo para empezar:

| id | archivo | fase | split | criterio |
| --- | --- | --- | --- | --- |
| `snake000` | `output/snake000.png` | inicio | train | partida temprana con pocas entidades y HUD limpio |
| `snake012` | `output/snake012.png` | media | val | mas turnos, mas ruido visual y posibles frutas activas |
| `snake025` | `output/snake025.png` | caos | test | estado tardio para probar oclusiones, serpientes largas y lectura del HUD |

El manifiesto versionado esta en `datasets/vision_captures/manifest.json`.

## Geometria fija de captura

La geometria sale del render de `SnakeGame.getSnapshot()`:

| campo | valor |
| --- | --- |
| imagen | `width=793`, `height=901` |
| tablero logico | `44x44` celdas |
| panel HUD | `6` filas logicas superiores |
| factor celda | `17` pixeles utiles |
| paso celda | `18` pixeles |
| origen celda tablero | `x = 1 + col * 18`, `y = 1 + (row + 6) * 18` |
| bbox celda | `[x, y, 17, 17]` |

La region global del HUD es `[0, 0, 793, 109]`. La region global del tablero empieza en `y=109` y ocupa el resto de la imagen.

La implementacion ejecutable de esta geometria esta en `vision_grid.py`. Expone `detect_grid_geometry(image)`, `GridGeometry.cell_bbox(row, col)`, `GridGeometry.cell_center(row, col)` y `GridGeometry.pixel_to_cell(x, y)`.

## Detector HUD

La implementacion ejecutable para el panel superior esta en `vision_hud.py`. Expone `detect_hud(image)` y devuelve:

- `turn`: contador de turno leido del centro del HUD.
- `scores`: diccionario `{ "G": int, "B": int, "R": int, "Y": int }`.
- `turn_bbox`: region del contador de turno.
- `score_bboxes`: regiones de puntuacion por color.

El detector usa matching de plantillas contra los mismos sprites de digitos que usa `SnakeGame.drawScore`, por lo que queda desacoplado del estado interno del motor y solo depende de la captura renderizada.

## Formato de etiqueta

Cada captura debe tener un JSON de anotacion que cumpla `datasets/vision_captures/labels.schema.json`. La etiqueta separa cuatro niveles:

- `geometry`: parametros de imagen y conversion pixel-celda.
- `hud`: contador de turno y puntuaciones por jugador.
- `board`: region del tablero y limites detectables.
- `objects`: frutas y segmentos de serpiente con coordenadas de celda y bbox en pixeles.

Las clases canonicas son:

| tipo | clases |
| --- | --- |
| fruta | `fruit_10`, `fruit_15`, `fruit_20` |
| serpiente | `snake_head`, `snake_body`, `snake_tail` |
| jugador/color | `G`, `B`, `R`, `Y` |
| direccion | `N`, `E`, `S`, `W`, `NE`, `NW`, `SE`, `SW`, `EN`, `ES`, `WN`, `WS` |
| HUD | `turn_counter`, `score_G`, `score_B`, `score_R`, `score_Y` |

Las coordenadas de celdas usan `row` y `col` cero-indexados en el tablero de `44x44`. Las cajas usan formato `[x, y, width, height]` en pixeles de la captura original.

## Criterio de aceptacion

- El dataset inicial contiene las tres fases obligatorias: inicio, media y caos.
- El manifiesto referencia imagenes existentes y una ruta de anotacion por imagen.
- El esquema JSON fija clases, coordenadas y campos obligatorios para que VA-02 pueda medir precision por HUD, frutas y serpientes.
- Las anotaciones se mantienen desacopladas del motor: no dependen de `SnakeGame`, `Snake`, `Fruit`, `RLPlayer` ni `trainRL`.

## Validacion automatizada

El contrato minimo del dataset se valida con:

```powershell
python -m unittest tests.test_vision_dataset_contract
python -m unittest tests.test_vision_grid
python -m unittest tests.test_vision_hud
```

Las pruebas comprueban fases/splits, existencia de rutas, geometria fija de captura, correspondencia entre celda logica y bbox en pixeles, conversion pixel-a-celda para el area jugable y lectura de turno/puntuaciones del HUD contra las capturas anotadas.

## Ampliacion prevista

Cuando haya mas capturas, mantener el mismo esquema y equilibrar por:

- numero de frutas visibles por valor;
- longitud de serpientes;
- presencia de choques o muertes;
- puntuaciones por debajo y por encima de 120 puntos de fruta;
- ruido visual si `setNoise()` se usa en futuras capturas.
