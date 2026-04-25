# Presentation Notes

## 1. Problema

Queremos entrenar un agente para jugar Snake battle royale con 4 serpientes. El juego no es solo comer fruta: hay supervivencia, colisiones, score y una regla de combate con umbral de 120 puntos.

## 2. Solucion

La solucion separa motor, estado y agentes:

- motor headless para entrenar rapido;
- `BoardState` como contrato;
- bots baseline como rivales;
- PPO como agente RL;
- TacticalPlanner para dar informacion de pathfinding;
- vision como pipeline separado hacia `BoardState`.

## 3. Arquitectura

```text
SnakeEnv headless
  -> BoardState
  -> Observation + TacticalPlanner
  -> PPO
  -> Reward
  -> Bots baseline
```

```text
Imagen
  -> VisionParser
  -> BoardState
  -> PPO
```

## 4. Por que PPO

PPO es estable, disponible en Stable-Baselines3 y permite iterar rapido sin comparar muchos algoritmos. La entrega prioriza una demo funcional y reproducible.

## 5. Evolucion del agente

- PPO inicial: wrapper headless y reward basica.
- PPO v2: mejor observacion/reward, metricas mejores contra varios bots.
- Acciones relativas: el agente decide `FORWARD/LEFT/RIGHT` en vez de `N/S/E/W`.
- TacticalPlanner: BFS para fruta, seguridad y ataque si `fruit_score >= 120`.

## 6. Resultados

Mejor modelo documentado:

```text
models/ppo_headless_v2/best_model/best_model.zip
```

Resultado destacable contra Survival:

- score medio `44`;
- fruta media `44`;
- supervivencia media `900`;
- reward medio positivo `66`;
- invalid action rate `0.062`;
- win rate todavia `0`.

## 7. Demo

Comando recomendado:

```bash
python play_ppo_demo.py --model-path models/ppo_headless_v2/best_model/best_model.zip --seed 2004 --bot-kind random
```

## 8. Limitaciones

- No hay win rate positivo documentado.
- No hay kills medias.
- v4 tactico existe como codigo/modelo, pero falta evaluacion formal.
- La integracion vision -> PPO completa queda como trabajo posterior.

## 9. Conclusion

El proyecto demuestra una arquitectura RL reproducible, extensible y medible. La parte mas importante no es solo el modelo entrenado, sino el contrato `BoardState`, la separacion headless/vision y la capacidad de comparar mejoras con logs reales.
