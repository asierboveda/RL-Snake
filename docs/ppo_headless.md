# PPO headless demo

El alcance actual usa un unico algoritmo: PPO headless con Stable-Baselines3.
No usa imagen, vision, DQN, Q-Learning, self-play, Optuna, curriculum ni ablations.

## Observacion

`PPOHeadlessSnakeEnv` usa `rl_observation.build_observation(...).features`.
La observacion es vectorial, fija y compatible con `MlpPolicy`:

- Nombre: `rl_observation_v1_features`
- Shape: `(23,)`
- Tipo: `float32`
- Sanitizada con `np.nan_to_num`

## Reward

Usa `rl_reward.compute_reward` con `RewardConfig` por defecto:

- fruta positiva
- supervivencia pequena
- kill valida positiva
- victoria positiva
- muerte negativa
- penalizacion adicional por accion ilegal del wrapper

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
python train_ppo.py --timesteps 100000 --seed 42 --bot-kind random --model-path models/ppo_headless/ppo_snake
```

Evaluar contra bots y contra una politica Random inicial:

```bash
python evaluate_ppo.py --model-path models/ppo_headless/ppo_snake.zip --episodes 50 --bot-kind random --out logs/ppo_headless/evaluation.json
```

Generar demo/replay reproducible:

```bash
python play_ppo_demo.py --model-path models/ppo_headless/ppo_snake.zip --seed 2026 --bot-kind random --out logs/ppo_headless/demo_replay.json
```

## Artefactos

- Modelo: `models/ppo_headless/ppo_snake.zip`
- Metadata: `models/ppo_headless/training_metadata.json`
- Logs: `logs/ppo_headless/`
- Evaluacion: `logs/ppo_headless/evaluation.json`
- Replay demo: `logs/ppo_headless/demo_replay.json`

## Limitaciones

Este setup prioriza demo funcional sobre rendimiento maximo. Usa una observacion vectorial simple, reward basica y un unico entorno sin vectorizacion. Si el entrenamiento corto no mejora win rate, revisar primero supervivencia, score medio o fruta comida antes de cambiar algoritmo.
