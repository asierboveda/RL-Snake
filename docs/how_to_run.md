# Como ejecutar el proyecto

## Dependencias

```bash
pip install -r requirements.txt
```

En esta maquina, Anaconda puede fallar importando Torch con `torch/lib/c10.dll`. Si ocurre, usar el Python bundled de Codex:

```powershell
& 'C:\Users\asier\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s tests
```

## Tests

```bash
python -m unittest discover -s tests
```

## Entrenamiento PPO

Entrenamiento corto/reproducible:

```bash
python train_ppo.py --timesteps 100000 --seed 42 --bot-kind random
```

Entrenamiento mas largo:

```bash
python train_ppo.py --timesteps 300000 --seed 42 --bot-kind random
```

## Evaluacion

Evaluar modelo v2 documentado:

```bash
python evaluate_ppo.py --model-path models/ppo_headless_v2/best_model/best_model.zip --episodes 50
```

Evaluar modelo v4 si se quiere validar formalmente:

```bash
python evaluate_ppo.py --model-path models/ppo_headless_v4/best_model/best_model.zip --episodes 50 --out logs/ppo_headless_v4/evaluation.json
```

## Comparacion v2/v4

Si existen ambos modelos y evaluaciones:

```bash
python compare_ppo_versions.py --v3-model models/ppo_headless_v2/best_model/best_model.zip --v4-model models/ppo_headless_v4/best_model/best_model.zip --episodes 50
```

## Demo/replay

Modelo recomendado actualmente:

```bash
python play_ppo_demo.py --model-path models/ppo_headless_v2/best_model/best_model.zip --seed 2004 --bot-kind random --out logs/ppo_headless_v2/demo_replay.json
```

Modelo tactico v4, pendiente de evaluacion formal:

```bash
python play_ppo_demo.py --model-path models/ppo_headless_v4/best_model/best_model.zip --seed 2004 --bot-kind random --out logs/ppo_headless_v4/demo_replay.json
```

## Notas

- No entrenar desde imagen.
- No tocar vision para entrenar PPO.
- No declarar un modelo como mejor sin `evaluation.json`.
- Conservar logs y metadata para reproducibilidad.
