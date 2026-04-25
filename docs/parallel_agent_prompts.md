# Parallel Agent Prompts: Scope + Vision

Base project: `RL-Snake`

Use two separate Codex sessions:

- Scope workspace: `C:\Users\asier\Documents\8ºCD\PROCESAMIENTO DE IMAGEN\ProyectoSnake\generador-scope`
- Vision workspace: `C:\Users\asier\Documents\8ºCD\PROCESAMIENTO DE IMAGEN\ProyectoSnake\generador-vision`

Before launching each agent, approve only the intended Kanvas tasks in Obsidian by changing them from purple to red. Keep ownership strict.

## Current Kanvas Entry Points

- Scope starts from `SC-04 Crear entorno headless reproducible tipo Gym`.
- Vision starts by reviewing `VI-02 Detectar HUD y contadores por color`, then proceeds to `VI-03` and `VI-04` when you approve them.
- `DO-00` is also ready, but keep it separate unless you explicitly want the Scope agent to own documentation gates.

## Scope Agent Prompt

```text
You are working in the Scope branch/worktree:
C:\Users\asier\Documents\8ºCD\PROCESAMIENTO DE IMAGEN\ProyectoSnake\generador-scope

Branch ownership:
- Branch: feature/scope-foundation
- Own these Kanvas tasks: SC-04, SC-05, SC-06, SC-07, VA-01, VA-02 when unlocked.
- Own these files/modules: SnakeGame.py, Snake.py, Fruit.py, GreedyPlayer.py, RandomPlayer.py, board_state.py, future environment/headless adapter modules, tests for engine and baselines, docs for scope decisions.
- Do not modify vision modules (`vision_*.py`, datasets/vision_captures) unless you stop and ask first.
- Do not modify RL training logic in RLPlayer.py beyond interface compatibility unless you stop and ask first.

Goal:
Create a clean headless environment and scope foundation for reliable RL training and later vision integration.

Required workflow:
1. Run:
   python .\canvas-tool.py .\Project.canvas status
   python .\canvas-tool.py .\Project.canvas ready
2. Work only on ready Scope/Validation tasks.
3. Start each task with:
   python .\canvas-tool.py .\Project.canvas start TASK_ID
4. Implement with focused tests.
5. Run:
   python -m unittest discover -s tests -v
6. Finish completed tasks with:
   python .\canvas-tool.py .\Project.canvas finish TASK_ID
7. Commit locally with focused messages.
8. Do not push until I explicitly approve review.

Acceptance criteria:
- There is a deterministic headless environment API.
- Critical game rules are testable without rendering.
- Baseline non-RL bots can be evaluated reproducibly.
- Project.canvas reflects task progress.
```

## Vision Agent Prompt

```text
You are working in the Vision branch/worktree:
C:\Users\asier\Documents\8ºCD\PROCESAMIENTO DE IMAGEN\ProyectoSnake\generador-vision

Branch ownership:
- Branch: feature/vision-pipeline
- Own these Kanvas tasks: VI-02, VI-03, VI-04, VI-05, VI-06, VI-07, VA-03 when unlocked.
- Own these files/modules: vision_grid.py, vision_hud.py, future vision_snakes.py, vision_fruits.py, vision_parser.py, datasets/vision_captures, docs/vision_dataset.md, tests/test_vision_*.py.
- Do not modify SnakeGame.py, RLPlayer.py, trainRL.py, reward logic, or baseline bots unless you stop and ask first.

Goal:
Build the image perception pipeline from screenshot to VisionState/BoardState-compatible output.

Required workflow:
1. Run:
   python .\canvas-tool.py .\Project.canvas status
   python .\canvas-tool.py .\Project.canvas ready
2. If VI-02 is still in Review, audit it first. Fix if needed, keep it in Review when done.
3. Work only on ready/approved Vision tasks.
4. Start each task with:
   python .\canvas-tool.py .\Project.canvas start TASK_ID
5. Implement with deterministic tests on saved PNG captures.
6. Run:
   python -m unittest discover -s tests -v
7. Finish completed tasks with:
   python .\canvas-tool.py .\Project.canvas finish TASK_ID
8. Commit locally with focused messages.
9. Do not push until I explicitly approve review.

Acceptance criteria:
- Vision code reads screenshots, not game internals.
- Detection outputs stable grid/HUD/snake/fruit data with confidence where applicable.
- Tests cover labeled captures from `datasets/vision_captures`.
- Project.canvas reflects task progress.
```

## Push And Merge After Review

Push Scope:

```powershell
cd "C:\Users\asier\Documents\8ºCD\PROCESAMIENTO DE IMAGEN\ProyectoSnake\generador-scope"
git push -u origin feature/scope-foundation
```

Push Vision:

```powershell
cd "C:\Users\asier\Documents\8ºCD\PROCESAMIENTO DE IMAGEN\ProyectoSnake\generador-vision"
git push -u origin feature/vision-pipeline
```

Merge order:

1. Merge Scope first if it changes shared engine/BoardState contracts.
2. Update Vision from `main` after Scope is merged.
3. Resolve conflicts in `Project.canvas`, `board_state.py`, and tests.
4. Run `python -m unittest discover -s tests -v`.
5. Merge Vision.

