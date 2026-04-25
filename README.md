# RL Snake Battle Royale

Proyecto de Reinforcement Learning y vision por computador sobre un juego tipo battle royale con 4 serpientes en un tablero discreto. Cada serpiente se mueve por una malla, come fruta, acumula score, sobrevive y puede eliminar rivales cuando cumple la regla critica de combate: tener al menos 120 puntos de fruta y ventaja frente al rival.

El repositorio esta organizado alrededor de un contrato central: `BoardState`. El motor headless genera estados estructurados, la vision reconstruye esos estados desde imagen y el agente PPO consume observaciones derivadas de `BoardState`.

## Que contiene

- Motor de juego headless: `SnakeGame.py`, `Snake.py`, `Fruit.py`, `snake_env.py`
- Contrato de estado: `board_state.py`
- Bots baseline: `baseline_bots.py`
- RL PPO headless: `ppo_env.py`, `train_ppo.py`, `evaluate_ppo.py`, `play_ppo_demo.py`
- Observacion RL: `rl_observation.py`
- Reward: `rl_reward.py`
- Planner tactico BFS: `tactical_planner.py`
- Vision imagen -> BoardState: `vision_grid.py`, `vision_hud.py`, `vision_snakes.py`, `vision_fruits.py`, `vision_parser.py`
- Tests: `tests/`
- Resultados y replays: `logs/`
- Modelos entrenados: `models/`

## Reglas resumidas

- Hay 4 serpientes en un tablero `44x44`.
- Cada turno cada serpiente elige una direccion.
- Las frutas dan puntos y aumentan el `fruit_score`.
- Una serpiente solo puede matar correctamente si su `fruit_score >= 120` y supera al rival.
- Las kills suman score de kill, pero no cuentan como fruta.
- El juego termina cuando queda una serpiente viva, cuando todas mueren o por condiciones de score/empate.

Mas detalle en [docs/game_rules.md](docs/game_rules.md).

## Arquitectura

```text
Imagen del juego
  -> VisionParser
  -> BoardState
  -> RL Observation
  -> PPO Agent
  -> Action
```

```text
SnakeEnv headless
  -> BoardState
  -> Observation + TacticalPlanner
  -> PPO
  -> Reward
  -> Evaluation vs Bots
```

El flujo de entrenamiento no usa imagen. La vision existe como pipeline separado para convertir capturas del juego a `BoardState`, pero PPO se entrena en headless para que sea reproducible y rapido.

## RL actual

El algoritmo elegido es PPO con Stable-Baselines3. No se comparan algoritmos ni se usa DQN, Q-Learning, self-play, Optuna, MCTS o minimax.

Evolucion documentada:

- `ppo_headless`: PPO inicial headless.
- `ppo_headless_v2`: mejor modelo medido actualmente, con mejores metricas disponibles.
- `ppo_headless_v3`: acciones relativas preparadas en el wrapper.
- `ppo_headless_v4`: TacticalPlanner/BFS disponible; existe modelo, pero falta evaluacion formal registrada en `evaluation.json`.

Mas detalle en [docs/rl_pipeline.md](docs/rl_pipeline.md) y [docs/results.md](docs/results.md).

## Resultados principales disponibles

Mejor modelo medido por logs existentes:

```text
models/ppo_headless_v2/best_model/best_model.zip
```

Resumen contra bots Survival, segun `logs/ppo_headless_v2/evaluation.json`:

| Modelo | Rival | Reward medio | Score medio | Fruta media | Supervivencia | Win rate | Invalid action rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| PPO v2 | Survival | 66.00 | 44.00 | 44.00 | 900.00 | 0.00 | 0.062 |

No se debe afirmar victoria competitiva: el win rate medido sigue siendo 0. El valor actual del proyecto esta en la arquitectura, reproducibilidad, mejora progresiva del agente y demo/replay.

## Comandos

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar tests:

```bash
python -m unittest discover -s tests
```

Entrenar PPO:

```bash
python train_ppo.py --timesteps 100000 --seed 42 --bot-kind random
```

Evaluar:

```bash
python evaluate_ppo.py --model-path models/ppo_headless_v2/best_model/best_model.zip --episodes 50
```

Demo/replay:

```bash
python play_ppo_demo.py --model-path models/ppo_headless_v2/best_model/best_model.zip --seed 2004 --bot-kind random
```

En esta maquina se ha usado tambien el Python bundled de Codex cuando Anaconda falla cargando Torch (`c10.dll`). Ver [docs/how_to_run.md](docs/how_to_run.md).

## Documentacion

- [docs/architecture.md](docs/architecture.md)
- [docs/game_rules.md](docs/game_rules.md)
- [docs/rl_pipeline.md](docs/rl_pipeline.md)
- [docs/results.md](docs/results.md)
- [docs/how_to_run.md](docs/how_to_run.md)
- [docs/agents_orchestration.md](docs/agents_orchestration.md)
- [PROJECT_STATUS.md](PROJECT_STATUS.md)
- [PRESENTATION_NOTES.md](PRESENTATION_NOTES.md)

## Limitaciones

- El agente aun no gana partidas de forma consistente.
- Las metricas disponibles son parciales y deben leerse por version.
- v4 tiene artefacto de modelo, pero falta evaluacion formal versionada.
- El pipeline de vision esta separado del entrenamiento PPO; la integracion end-to-end completa queda como siguiente paso.
