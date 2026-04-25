# Arquitectura del sistema

Este proyecto separa el juego en tres capas: motor, estado estructurado y agentes.

## Componentes

| Componente | Archivos | Responsabilidad |
|---|---|---|
| Motor | `SnakeGame.py`, `Snake.py`, `Fruit.py` | Ejecutar reglas, movimiento, frutas, colisiones y score |
| Entorno headless | `snake_env.py` | Adaptar el motor a episodios reproducibles `reset/step` |
| Estado comun | `board_state.py` | Contrato estructurado entre motor, vision y RL |
| Bots baseline | `baseline_bots.py` | Rivales Random, Greedy, Survival, Aggressive, Hybrid |
| PPO | `ppo_env.py`, `train_ppo.py` | Entrenamiento headless con Stable-Baselines3 |
| Observacion | `rl_observation.py` | Vector fijo para PPO a partir de `BoardState` |
| Reward | `rl_reward.py` | Reward principal y shaping pequeno |
| Tactica | `tactical_planner.py` | BFS/pathfinding hacia fruta y rivales atacables |
| Vision | `vision_*.py` | Reconstruir `BoardState` desde imagen |

## Contrato central: BoardState

`BoardState` contiene:

- turno;
- dimensiones del tablero;
- serpientes vivas/muertas;
- cuerpo, cabeza, direccion y color;
- score total y `fruit_score`;
- frutas activas;
- estado terminal y ganador.

Esto permite que RL no dependa de variables privadas del motor y que la vision pueda integrarse produciendo el mismo formato.

## Flujo de entrenamiento

```text
SnakeEnv headless
  -> BoardState
  -> rl_observation.build_observation
  -> TacticalPlanner BFS features
  -> PPO policy
  -> relative action
  -> absolute direction
  -> SnakeGame
  -> Reward
```

## Flujo con vision

```text
Frame / imagen del juego
  -> vision_grid / vision_hud / vision_snakes / vision_fruits
  -> vision_parser
  -> BoardState
  -> misma observacion RL
  -> PPO Agent
```

La separacion permite entrenar rapido en headless y reservar vision para inferencia/validacion.
