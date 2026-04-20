import json
import os
import re
import sys
from typing import Any, Dict, List


def _as_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _slug(text: str, fallback: str = "rl-improvement") -> str:
    if not text:
        return fallback
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug[:48] if slug else fallback


def _load_metrics() -> Dict[str, Any]:
    raw = os.environ.get("METRICS_JSON", "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _implementation_focus(metrics: Dict[str, Any]) -> str:
    early_deaths = float(metrics.get("early_death_rate", 0.0))
    win_rate = float(metrics.get("rl_win_rate", 0.0))
    avg_fruit = float(metrics.get("avg_rl_fruit_score", 0.0))
    death_counts = metrics.get("death_cause_counts", {}) if isinstance(metrics, dict) else {}
    wall_deaths = float(death_counts.get("wall", 0))
    self_deaths = float(death_counts.get("self", 0))
    enemy_deaths = float(death_counts.get("enemy", 0))

    if wall_deaths + self_deaths >= enemy_deaths and early_deaths >= 25.0:
        return (
            "Prioriza supervivencia local: action masking estricto para muro/cuerpo, "
            "evitar callejones y reforzar política anti-suicidio."
        )
    if enemy_deaths > wall_deaths + self_deaths:
        return (
            "Prioriza combate seguro: evaluar riesgo head-to-head, distancia a enemigos "
            "y no entrar en rangos de colisión desfavorables."
        )
    if win_rate < 20.0:
        return (
            "Prioriza política: ajustar reward shaping y features de estado "
            "(ventaja de fruit_score, presión enemiga, dirección actual)."
        )
    if avg_fruit < 40.0:
        return (
            "Prioriza farmeo temprano: mejorar navegación a fruta y reducir "
            "movimientos que alejan del objetivo sin seguridad."
        )
    return (
        "Prioriza consolidación: mantener seguridad y afinar balance "
        "exploración/explotación con checkpoints estables."
    )


def _build_tasks(
    objective: str,
    iteration: int,
    max_iterations: int,
    base_branch: str,
    metrics: Dict[str, Any],
) -> List[Dict[str, str]]:
    run_slug = _slug(objective)
    runner_branch = f"ai/runner-{run_slug}-it{iteration}"
    implementer_branch = f"ai/impl-{run_slug}-it{iteration}"
    focus = _implementation_focus(metrics)

    runner_instruction = (
        "Ejecuta ciclo de entrenamiento/evaluación headless en el workflow runner "
        "y emite métricas parseables en JSON al final, incluyendo causas de muerte y empates. "
        "Mantén salida de progreso visible en logs durante el entrenamiento. "
        f"Iteración {iteration}/{max_iterations}. Objetivo: {objective}"
    )
    implementer_instruction = (
        "Implementa mejoras concretas de RL basadas en el feedback del manager y runner. "
        "Realiza cambios atómicos, deja trazabilidad en docs/ai-traces y crea commit con "
        "mensaje claro en la rama asignada. "
        f"Foco actual: {focus} Iteración {iteration}/{max_iterations}. Objetivo: {objective}"
    )

    tasks = [
        {
            "worker_role": "implementer",
            "branch_name": implementer_branch,
            "base_branch": base_branch,
            "instructions": implementer_instruction,
        },
        {
            "worker_role": "runner",
            "branch_name": runner_branch,
            "base_branch": base_branch,
            "instructions": runner_instruction,
        },
    ]
    return tasks


def main() -> None:
    objective = os.environ.get("OBJECTIVE", "").strip() or os.environ.get("ISSUE_BODY", "").strip()
    if not objective:
        objective = "Improve RL Snake agent with manager-worker loop"

    iteration = _as_int(os.environ.get("ITERATION", "1"), 1)
    max_iterations = _as_int(os.environ.get("MAX_ITERATIONS", "3"), 3)
    base_branch = os.environ.get("BASE_BRANCH", "main").strip() or "main"

    metrics = _load_metrics()
    should_continue = iteration <= max_iterations

    payload: Dict[str, Any] = {
        "objective": objective,
        "iteration": iteration,
        "max_iterations": max_iterations,
        "continue": should_continue,
        "metrics": metrics,
        "tasks": [],
    }

    if should_continue:
        payload["tasks"] = _build_tasks(objective, iteration, max_iterations, base_branch, metrics)

    print(json.dumps(payload, ensure_ascii=True))


if __name__ == "__main__":
    main()
