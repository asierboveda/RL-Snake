# VI-07 Politica de confianza y fallback visual

Esta capa decide si un `VisionParseResult` puede usarse para RL o si debe activarse fallback operativo.

## Objetivo operacional

Entre `VisionParser` y el agente RL:

1. detectar estados visuales peligrosos;
2. aceptar solo estados suficientemente fiables;
3. degradar de forma segura cuando hay dudas;
4. evitar acciones suicidas por percepcion corrupta.

## API implementada

Archivo: `vision_fallback.py`

- `VisionFallbackPolicy.evaluate(parse_result, snake_id, last_action)` -> `VisionFallbackDecision`
- `choose_safe_action(board_state, snake_id, preferred_action)` -> accion segura (`N/S/E/W`)

## Modos de decision

| modo | cuando aplica | salida |
| --- | --- | --- |
| `trust` | parse limpio y confianza alta | usar `board_state` actual |
| `conservative` | calidad intermedia, sin fallos criticos | usar `board_state` actual en modo conservador |
| `reuse_last_reliable` | fallo critico y existe estado fiable previo | reutilizar ultimo `BoardState` fiable |
| `safe_action` | fallo critico sin estado fiable previo | no usar board actual, forzar accion segura |
| `drop_frame` | fallo critico extremo y sin accion segura util | descartar frame y pedir siguiente |

## Tabla de umbrales (calibrada con VI-06)

Referencia de VI-06: `mean_parser_confidence ~= 0.999257`, errores/warnings agregados `0`.

| regla | umbral por defecto | comportamiento esperado |
| --- | --- | --- |
| confianza para confiar (`trust_confidence_min`) | `>= 0.998` | `trust` si no hay otras alertas |
| confianza minima antes de rechazo (`reject_confidence_max`) | `< 0.995` | modo critico (`reuse_last_reliable` o `safe_action`) |
| confianza minima serpientes para trust | `>= 0.998` | evita confiar con cabeza/cuerpo dudosos |
| confianza minima frutas para trust | `>= 0.998` | evita trampas por fruta falsa |
| warnings para trust | `0` | cualquier warning degrada a conservador |
| warnings maximos tolerables | `<= 2` | mas de 2 warnings se considera critico |
| salto maximo de turno en conservador | `3` frames | si se supera, modo conservador |
| salto maximo de score por turno | `60` puntos/turno | si se supera, critico (lectura HUD sospechosa) |

## Fallos criticos detectados

- `parse_result.errors` no vacio.
- confianza global por debajo del umbral de rechazo.
- confianza de consistencia demasiado baja.
- warning critico de cuerpo propio no detectado con score positivo.
- serpiente propia ausente.
- regresion de turno.
- salto de score propio fisicamente implausible.

## Estrategia de fallback

1. **Si hay ultimo estado fiable:** usar `reuse_last_reliable`.
2. **Si no hay historial fiable:** calcular `safe_action` usando ocupacion actual y limites.
3. Mantener `last_action` si sigue siendo segura; si no, elegir accion con mayor separacion a cabezas enemigas.
4. Solicitar siguiente frame (`request_next_frame=True`) en todo modo critico.

## Integracion recomendada

```python
parse_result = vision_parser.parse(frame)
decision = fallback_policy.evaluate(parse_result, snake_id=my_id, last_action=last_action)

if decision.force_safe_action:
    action = decision.safe_action
else:
    board_for_rl = decision.board_state
    action = rl_player.play_board_state(board_for_rl)
```

## Cobertura de pruebas

`tests/test_vision_fallback.py` valida:

- modo trust;
- modo conservador;
- reuse de estado fiable tras fallo critico;
- safe action sin historial fiable;
- deteccion de salto de score anomalo;
- mantenimiento de direccion previa cuando sigue siendo segura.
