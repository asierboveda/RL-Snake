# Pipeline RL

## Algoritmo

El agente usa PPO con Stable-Baselines3. Se eligio PPO porque es robusto, facil de entrenar con un `Gymnasium Env` y suficiente para una demo headless reproducible.

No se usan DQN, Q-Learning, Optuna, self-play, MCTS, minimax ni curriculum en la entrega actual.

## Entorno

`PPOHeadlessSnakeEnv` controla la serpiente `0`. Las otras serpientes son bots baseline:

- `RandomPlayer`
- `GreedyPlayer`
- `SurvivalPlayer`

El motor sigue recibiendo acciones absolutas `N/S/E/W`; el wrapper puede exponer acciones relativas:

```text
0 = FORWARD
1 = LEFT
2 = RIGHT
```

## Observacion

La observacion sale de `rl_observation.py` y se basa en `BoardState`.

Incluye:

- scores normalizados;
- estado vivo/muerto de rivales;
- distancia a fruta;
- distancia a enemigos peligrosos/atacables;
- direccion actual;
- peligro relativo;
- features tacticas de `TacticalPlanner`.

La version tactica actual tiene shape `(51,)`.

## TacticalPlanner

`tactical_planner.py` usa BFS sobre el grid para calcular:

- si `FORWARD/LEFT/RIGHT` son seguros;
- distancia a fruta alcanzable;
- primera accion recomendada hacia fruta;
- espacio libre alcanzable tras moverse;
- ataque disponible si `fruit_score >= 120`;
- distancia y primera accion hacia rival atacable;
- riesgo de enemigo fuerte;
- riesgo de head-to-head.

Esto no reemplaza a PPO. Solo le da informacion tactica mas compacta.

## Reward

La reward mantiene los componentes principales:

- fruta positiva;
- supervivencia pequena;
- kill valida positiva;
- victoria positiva;
- muerte negativa;
- accion invalida negativa.

Tambien existe shaping pequeno por acercarse o alejarse de fruta segura. La idea es orientar sin tapar la recompensa principal.

## Evaluacion

La evaluacion se hace contra:

- Random bots;
- Greedy bots;
- Survival bots.

Metricas relevantes:

- `mean_reward`;
- `mean_score`;
- `mean_fruit_score`;
- `mean_survival_turns`;
- `early_death_rate`;
- `invalid_action_rate`;
- `win_rate`;
- `mean_kills`.

No se debe declarar exito solo por reward. En Snake importan score, fruta, supervivencia y acciones invalidas.
