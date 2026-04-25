# PPO headless demo

El alcance actual usa un unico algoritmo: PPO headless con Stable-Baselines3.
No usa imagen, vision, DQN, Q-Learning, MCTS, minimax, self-play, Optuna, curriculum ni ablations.

El repositorio contiene dos lineas reproducibles:

- v3: PPO headless basico en `models/ppo_headless/`.
- v4: PPO headless tactico con `TacticalPlanner` en `models/ppo_headless_v4/`.

La v4 esta preparada para entrenar/evaluar, pero no debe presentarse como mejora hasta ejecutar entrenamiento y comparar metricas de juego.

## Observacion

`PPOHeadlessSnakeEnv` usa `rl_observation.build_observation(...).features`.
La observacion es vectorial, fija y compatible con `MlpPolicy`:

- Nombre runtime: `rl_observation_v1_features`
- Shape actual: `(51,)`
- Tipo: `float32`
- Sanitizada con `np.nan_to_num`
- Acciones PPO: `0=FORWARD`, `1=LEFT`, `2=RIGHT`

Las features tacticas vienen de `tactical_planner.py` y se calculan con BFS sobre `BoardState`:

- seguridad de forward/left/right
- distancia y primera accion hacia fruta segura
- espacio libre alcanzable tras mover
- ataque solo si `fruit_score >= 120` y rival mas debil
- riesgos de enemigo fuerte y head-to-head

## Reward

Usa `rl_reward.compute_reward` con `RewardConfig` por defecto:

- fruta positiva
- supervivencia pequena
- kill valida positiva
- victoria positiva
- muerte negativa
- penalizacion adicional por accion ilegal del wrapper
- shaping pequeno por acercarse/alejarse de fruta

## Bots

Por defecto entrena contra tres `RandomPlayer`.
Tambien se puede lanzar contra `GreedyPlayer` o `SurvivalPlayer` con `--bot-kind greedy` o `--bot-kind survival`.

## Comandos

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Entrenar PPO:

```bash
python train_ppo.py --timesteps 300000 --seed 42 --bot-kind random --model-path models/ppo_headless_v4/ppo_snake --log-dir logs/ppo_headless_v4
```

Entrenamiento minimo si hay poco tiempo:

```bash
python train_ppo.py --timesteps 100000 --seed 42 --bot-kind random --model-path models/ppo_headless_v4/ppo_snake --log-dir logs/ppo_headless_v4
```

Evaluar contra Random, Greedy y Survival:

```bash
python evaluate_ppo.py --model-path models/ppo_headless_v4/best_model/best_model.zip --episodes 50 --out logs/ppo_headless_v4/evaluation.json
```

Comparar v3 vs v4:

```bash
python compare_ppo_versions.py --v3-model models/ppo_headless/ppo_snake.zip --v4-model models/ppo_headless_v4/best_model/best_model.zip --episodes 50
```

Generar demo/replay reproducible:

```bash
python play_ppo_demo.py --model-path models/ppo_headless_v4/best_model/best_model.zip --seed 2026 --bot-kind random --out logs/ppo_headless_v4/demo_replay.json
```

## Artefactos

- Modelo v3: `models/ppo_headless/ppo_snake.zip`
- Modelo v4 esperado: `models/ppo_headless_v4/ppo_snake.zip`
- Best model v4 esperado: `models/ppo_headless_v4/best_model/best_model.zip`
- Metadata v4: `models/ppo_headless_v4/training_metadata.json`
- Logs v4: `logs/ppo_headless_v4/`
- Evaluacion v4: `logs/ppo_headless_v4/evaluation.json`
- Comparacion: `logs/ppo_headless_v4/v3_vs_v4_comparison.json`
- Replay demo: `logs/ppo_headless_v4/demo_replay.json`

## Limitaciones

Este setup prioriza demo funcional sobre rendimiento maximo. Usa observacion vectorial, reward basica con shaping pequeno y un unico entorno sin vectorizacion. El criterio de exito no es solo reward: hay que revisar score medio, fruta, supervivencia, early death, invalid actions, win rate y kills.
