# Orquestacion de agentes

El proyecto se desarrollo con metodologia asistida por IA, usando agentes para dividir trabajo y acelerar iteraciones. La coordinacion se mantuvo alrededor de `BoardState` como contrato tecnico comun.

## Enfoque

La estrategia fue separar tareas por responsabilidad:

- reglas del motor;
- contrato de estado;
- entorno headless;
- bots baseline;
- observacion RL;
- reward;
- entrenamiento PPO;
- evaluacion;
- vision;
- documentacion y limpieza.

Cada bloque se valido con tests o artefactos medibles cuando fue posible.

## Roles de agentes

| Area | Contribucion |
|---|---|
| Reglas | Formalizacion de colisiones, kills, umbral de 120 y score |
| Scope/base | `BoardState`, dataset/contratos, entorno headless inicial |
| Baselines | Bots Random, Greedy, Survival, Aggressive e Hybrid |
| Vision | Parser imagen -> `BoardState`, HUD, snakes, frutas y fallback |
| RL inicial | PPO headless, wrapper Gymnasium, reward basica |
| Debugging RL | Revision de acciones invalidas, muerte temprana y metricas |
| Mejora tactica | Acciones relativas y BFS sobre `BoardState` |
| Documentacion | Preparacion de README, docs y estado de entrega |

## Decisiones importantes

- Mantener vision y RL separados.
- Entrenar PPO en headless, no desde imagen.
- Usar bots baseline como rivales reproducibles.
- Evitar comparativas grandes de algoritmos.
- No introducir MCTS/minimax/self-play en esta entrega.
- Medir con metricas de juego, no solo reward.

## Validacion

Se usaron tests unitarios sobre:

- contrato `BoardState`;
- reglas criticas;
- entorno headless;
- observacion RL;
- bots baseline;
- vision;
- wrapper PPO;
- planner tactico.

Tambien se generaron logs de evaluacion en `logs/`.

## Lectura honesta

Los agentes ayudaron a acelerar implementacion y documentacion, pero el resultado debe evaluarse por codigo, tests y metricas presentes en el repositorio. Donde no hay resultados medidos, la documentacion lo marca como no disponible.
