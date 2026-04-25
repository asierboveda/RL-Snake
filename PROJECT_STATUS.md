# Project Status

## Estado actual

El repositorio esta preparado como entrega presentable de un proyecto RL/vision para Snake battle royale.

Funciona:

- motor headless;
- contrato `BoardState`;
- bots baseline;
- PPO headless;
- evaluacion contra bots;
- replay/demo;
- vision parser hacia `BoardState`;
- TacticalPlanner BFS en codigo;
- documentacion tecnica principal.

## Mejor modelo documentado

```text
models/ppo_headless_v2/best_model/best_model.zip
```

Motivo: tiene evaluacion disponible en `logs/ppo_headless_v2/evaluation.json`.

## Metric clave disponible

Contra Survival bots:

- reward medio: `66.00`
- score medio: `44.00`
- fruta media: `44.00`
- supervivencia media: `900.00`
- invalid action rate: `0.062`
- win rate: `0.00`
- kills: `0.00`

## Pendiente

- Evaluar formalmente `models/ppo_headless_v4/best_model/best_model.zip`.
- Generar `logs/ppo_headless_v4/evaluation.json`.
- Comparar v2/v4 con las mismas seeds y episodios.
- Mejorar win rate y kills.
- Integrar demo visual end-to-end imagen -> BoardState -> PPO.

## Demo recomendada

```bash
python play_ppo_demo.py --model-path models/ppo_headless_v2/best_model/best_model.zip --seed 2004 --bot-kind random --out logs/ppo_headless_v2/demo_replay.json
```

## Limitaciones

- No hay victorias medidas.
- Las kills medias son 0 en los resultados disponibles.
- v4 no debe venderse como mejora hasta evaluarlo.
- Las metricas existentes tienen pocos episodios en v2 (`10` por rival).
