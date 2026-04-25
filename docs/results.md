# Resultados

Este documento resume solo metricas encontradas en archivos de logs. No se inventan resultados.

## Artefactos encontrados

| Version | Modelo | Evaluacion |
|---|---|---|
| PPO inicial (`ppo_headless`) | `models/ppo_headless/ppo_snake.zip` | `logs/ppo_headless/evaluation.json` |
| PPO v2 | `models/ppo_headless_v2/best_model/best_model.zip` | `logs/ppo_headless_v2/evaluation.json` |
| PPO v4 tactico | `models/ppo_headless_v4/best_model/best_model.zip` | No disponible |

## PPO inicial

Evaluacion contra Random, 50 episodios:

| Politica | Reward medio | Score medio | Fruta media | Supervivencia | Kills | Win rate |
|---|---:|---:|---:|---:|---:|---:|
| PPO | -77.85 | 1.90 | 1.90 | 34.34 | 0.00 | 0.00 |
| Random baseline | -92.50 | 4.70 | 4.70 | 240.58 | 0.00 | 0.00 |

Lectura: PPO inicial mejora reward frente a la politica random interna, pero no mejora score ni supervivencia.

## PPO v2

Evaluacion existente: 10 episodios por rival.

| Rival | Politica | Reward medio | Score medio | Fruta media | Supervivencia | Kills | Win rate | Invalid action rate | Early death rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random | PPO | -59.63 | 16.00 | 16.00 | 550.30 | 0.00 | 0.00 | 0.103 | 0.10 |
| Random | Random baseline | -267.32 | 32.00 | 32.00 | 882.90 | 0.00 | 0.00 | 0.235 | 0.00 |
| Greedy | PPO | -25.22 | 13.00 | 13.00 | 194.80 | 0.00 | 0.00 | 0.045 | 0.10 |
| Greedy | Random baseline | -43.59 | 0.00 | 0.00 | 189.10 | 0.00 | 0.00 | 0.033 | 0.00 |
| Survival | PPO | 66.00 | 44.00 | 44.00 | 900.00 | 0.00 | 0.00 | 0.062 | 0.00 |
| Survival | Random baseline | -199.20 | 35.50 | 35.50 | 900.00 | 0.00 | 0.00 | 0.200 | 0.00 |

Lectura:

- PPO v2 muestra mejor reward frente a baseline random en los tres rivales.
- Contra Survival tambien mejora score/fruta.
- El win rate sigue en 0.
- No hay kills medias.
- Hay acciones invalidas, pero en Random/Survival son menores que el baseline random registrado.

## PPO v3/v4

La rama de codigo contiene:

- acciones relativas en `ppo_env.py`;
- features tacticas en `rl_observation.py`;
- BFS en `tactical_planner.py`;
- modelo v4 en `models/ppo_headless_v4/best_model/best_model.zip`.

No se ha encontrado `logs/ppo_headless_v4/evaluation.json` en esta revision documental. Por tanto, no se declara mejora v4 todavia.

## Mejor modelo actual documentado

Por metricas disponibles, el mejor candidato presentable es:

```text
models/ppo_headless_v2/best_model/best_model.zip
```

El modelo v4 puede ser prometedor, pero necesita evaluacion formal con `evaluate_ppo.py` antes de presentarlo como superior.
