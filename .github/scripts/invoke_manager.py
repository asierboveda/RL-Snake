"""
invoke_manager.py
-----------------
Manager del sistema multi-agente. Lee el reporte del Observer (runner worker)
y genera instrucciones tecnicas concretas para el Implementer worker.

FLUJO DE DATOS:
  Runner (run_and_observe.py)
      -> runner-report.json (metrics + diagnosis)
      -> variable RUNNER_REPORT_JSON en GitHub Actions
      -> este script
      -> payload JSON con instrucciones para implementer

PRINCIPIO DE DECISION: el manager NO adivina. Usa el 'problem_code' y los
'findings' del observer para decidir exactamente que metodo tocar y como.
Cada diagnostico tiene instrucciones tecnicas especificas y acotadas.
"""

import json
import os
import re
import sys
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _slug(text: str, fallback: str = "rl-improvement") -> str:
    """Convierte texto libre en un slug valido para nombres de rama git."""
    if not text:
        return fallback
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug[:48] if slug else fallback


def _load_report() -> Dict[str, Any]:
    """
    Lee el reporte completo del Observer desde la variable de entorno.
    El reporte tiene dos secciones: 'metrics' y 'diagnosis'.
    En la primera iteracion no hay reporte previo; se usan defaults.
    """
    raw = os.environ.get("RUNNER_REPORT_JSON", "").strip()
    if not raw or raw == "{}":
        # Primera iteracion: sin datos. El observer no ha corrido aun.
        # Defaults que empujan hacia diagnostico de supervivencia basica.
        return {
            "metrics": {
                "rl_win_rate":        0.0,
                "avg_turns":          0.0,
                "avg_rl_score":       0.0,
                "avg_rl_fruit_score": 0.0,
                "avg_rl_kills":       0.0,
                "early_death_rate":   100.0,
                "death_cause_counts": {},
                "outcome_counts":     {},
            },
            "diagnosis": {
                "problem_code":         "avoidable_wall_self_deaths",
                "dominant_cause":       "wall",
                "avoidable_death_pct":  100.0,
                "trapped_death_pct":    0.0,
                "avg_wall_dist_death":  0.5,
                "avg_enemy_dist_death": 10.0,
                "hunter_deaths":        0,
                "farmer_deaths":        0,
                "findings":             ["Primera iteracion: sin datos del observer."],
            }
        }
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[MANAGER] WARN: No se pudo parsear RUNNER_REPORT_JSON: {e}", file=sys.stderr)
        return {}


# ---------------------------------------------------------------------------
# Instrucciones tecnicas por diagnostico
# Cada clave es un 'problem_code' del observer.
# El valor son instrucciones exactas para el implementer: que metodo tocar,
# que argumento cambiar, que NO tocar. Cuanto mas preciso, menos alucina el LLM.
# ---------------------------------------------------------------------------

IMPLEMENTATION_PLAN = {

    "avoidable_wall_self_deaths": {
        "summary": "El agente muere por muro/cuerpo teniendo alternativas seguras disponibles.",
        "instructions": (
            "PROBLEMA DIAGNOSTICADO: muertes evitables por muro o cuerpo propio "
            "(el agente tenia acciones seguras pero eligio una peligrosa). "
            "CAMBIOS REQUERIDOS en generador/RLPlayer.py: "
            "(1) En get_state(): añadir 4 valores de distancia normalizada a los bordes: "
            "    north_dist=head[0]/game.rSize, south_dist=(game.rSize-1-head[0])/game.rSize, "
            "    east_dist=(game.cSize-1-head[1])/game.cSize, west_dist=head[1]/game.cSize. "
            "    Discretizar en 3 niveles: 0=peligro(dist<=2), 1=precaucion(3-5), 2=libre(>5). "
            "    Incorporar estas 4 variables al tuple que retorna get_state(). "
            "(2) En update_q_table(): añadir penalizacion extra de -10 cuando la accion "
            "    reduce la distancia minima al muro a <= 2 celdas, aunque no sea muerte inmediata. "
            "    Esto se calcula antes de llamar a la formula Q estándar. "
            "(3) En play(): aumentar la penalizacion de muerte de -100 a -150. "
            "NO toques alpha, gamma, find_goal ni la logica de frutas en este ciclo."
        ),
    },

    "trapped_no_safe_action": {
        "summary": "El agente se mete en callejones sin salida (ninguna accion segura disponible).",
        "instructions": (
            "PROBLEMA DIAGNOSTICADO: el agente muere atrapado sin ninguna accion segura. "
            "Esto indica que la politica no anticipa el espacio libre futuro. "
            "CAMBIOS REQUERIDOS en generador/RLPlayer.py: "
            "(1) Crear nuevo metodo count_reachable_cells(self, from_pos, max_depth=8): "
            "    Implementar un BFS simple desde 'from_pos' que cuente cuantas celdas "
            "    son alcanzables en max_depth pasos sin chocar con paredes ni serpientes. "
            "    Retornar ese conteo como entero. "
            "(2) En get_state(): incorporar el conteo de celdas alcanzables desde la cabeza "
            "    discretizado en 3 niveles: 0=critico(<10), 1=limitado(10-30), 2=libre(>30). "
            "    Añadir este valor al tuple de get_state(). "
            "(3) En get_safe_actions(): filtrar tambien acciones que llevan a zonas con "
            "    menos de 5 celdas alcanzables (usando count_reachable_cells con max_depth=4). "
            "NO toques el sistema de rewards ni la logica de frutas en este ciclo."
        ),
    },

    "enemy_proximity_deaths": {
        "summary": "El agente muere por enemigos cercanos que no evade correctamente.",
        "instructions": (
            "PROBLEMA DIAGNOSTICADO: el agente muere por colision con enemigos aunque estaban "
            "a menos de 4 celdas (detectables con un vistazo). "
            "CAMBIOS REQUERIDOS en generador/RLPlayer.py: "
            "(1) En get_state(): añadir 4 flags booleanos de presencia enemiga en celdas "
            "    adyacentes directas: enemy_N, enemy_S, enemy_E, enemy_W. "
            "    Cada flag es True si la celda en esa direccion esta ocupada por "
            "    la cabeza O el cuerpo de cualquier serpiente enemiga viva. "
            "    Incorporar estos 4 flags al tuple de retorno de get_state(). "
            "(2) En is_dangerous(): incluir las CABEZAS de serpientes enemigas vivas "
            "    como posiciones prohibidas (ademas de sus cuerpos). "
            "    Esto refuerza el action masking para que ningun movimiento acerque "
            "    la cabeza propia a la cabeza enemiga. "
            "(3) En find_goal(): si is_hunter=False AND existe un enemigo a distancia "
            "    Manhattan <= 3, retornar el centro del tablero en lugar de la fruta. "
            "    Comportamiento evasivo prioritario. "
            "NO toques update_q_table ni los rewards en este ciclo."
        ),
    },

    "aggressive_hunter_deaths": {
        "summary": "El agente muere mas en modo cazador que en modo farmer. La agresividad es contraproducente.",
        "instructions": (
            "PROBLEMA DIAGNOSTICADO: el agente muere mas veces cuando tiene fruit_score >= 120 "
            "(modo cazador). Esta siendo demasiado agresivo sin verificar que puede ganar el duelo. "
            "CAMBIOS REQUERIDOS en generador/RLPlayer.py: "
            "(1) En find_goal(): cuando is_hunter=True, antes de retornar la posicion del rival "
            "    mas cercano, verificar que ese rival tiene fruit_score MENOR que el propio. "
            "    Si no hay rival mas debil disponible, retornar la fruta mas cercana como fallback "
            "    (no perseguir rivales iguales o mas fuertes). "
            "(2) En get_state(): añadir un flag 'prey_available' = True si existe un rival vivo "
            "    con fruit_score < propio_fruit_score. Esto permite que la Q-table distinga "
            "    'hunter con presa real' de 'hunter sin presa valida'. "
            "(3) En play(): cuando is_hunter=True y no hay rival mas debil, "
            "    aplicar penalizacion adicional de -5 si la accion acerca al enemigo mas fuerte. "
            "NO toques get_safe_actions ni la logica de entramiento en este ciclo."
        ),
    },

    "survives_but_no_wins": {
        "summary": "El agente sobrevive bien pero no convierte supervivencia en victoria.",
        "instructions": (
            "PROBLEMA DIAGNOSTICADO: el agente sobrevive pero no gana. "
            "La politica es demasiado conservadora: farmea poco y no ataca cuando puede. "
            "CAMBIOS REQUERIDOS en generador/RLPlayer.py: "
            "(1) En play(): aumentar el reward de frutas de +20 a +35 cuando la fruta "
            "    consumida lleva al agente a fruit_score >= 50 (reward extra por hito de puntuacion). "
            "(2) En play(): añadir reward de +10 cuando fruit_score supera 100 por primera vez "
            "    en el episodio (umbral pre-hunter). Usar un flag self.reached_100 = False en __init__. "
            "(3) En play(): añadir reward de +15 al matar un rival (current_score - last_score > 20 "
            "    y la diferencia no es multiplo de las frutas normales, o sea viene de un kill). "
            "    Ya existe logica de kills; solo hay que reforzar la recompensa aqui. "
            "NO toques get_state ni la arquitectura de deteccion de peligros en este ciclo."
        ),
    },

    "consolidation": {
        "summary": "Metricas aceptables. Afinar balance exploracion/explotacion.",
        "instructions": (
            "ESTADO: metricas aceptables sin un problema dominante. "
            "CAMBIOS REQUERIDOS en generador/RLPlayer.py: "
            "(1) En __init__(): añadir parametro 'epsilon_decay=0.995' con valor por defecto. "
            "    Guardar como self.epsilon_decay = epsilon_decay. "
            "(2) Crear metodo decay_epsilon(self): "
            "    'self.epsilon = max(self.epsilon * self.epsilon_decay, 0.05)'. "
            "(3) En end_episode(): llamar a self.decay_epsilon() al final del metodo, "
            "    solo cuando training_enabled=True. "
            "(4) En update_q_table(): si len(self.q_table) > 500, usar self.alpha = 0.15 "
            "    en lugar del valor por defecto (el agente tiene experiencia suficiente). "
            "NO añadas nuevas features de estado en este ciclo."
        ),
    },
}


# ---------------------------------------------------------------------------
# Core del Manager
# ---------------------------------------------------------------------------

def _select_plan(diagnosis: Dict[str, Any]) -> Dict[str, str]:
    """
    Selecciona el plan de implementacion basado en el problem_code del observer.
    Si el codigo no existe en el mapa, usa el diagnostico de fallback.
    """
    code = diagnosis.get("problem_code", "consolidation")
    return IMPLEMENTATION_PLAN.get(code, IMPLEMENTATION_PLAN["consolidation"])


def _build_tasks(
    objective: str,
    iteration: int,
    max_iterations: int,
    base_branch: str,
    report: Dict[str, Any],
) -> List[Dict[str, Any]]:

    metrics   = report.get("metrics", {})
    diagnosis = report.get("diagnosis", {})
    plan      = _select_plan(diagnosis)
    run_slug  = _slug(objective)

    runner_branch      = f"ai/runner-{run_slug}-it{iteration}"
    implementer_branch = f"ai/impl-{run_slug}-it{iteration}"

    # Instruccion para el RUNNER:
    # No toma decisiones de implementacion. Solo entrena + observa + reporta.
    runner_instruction = (
        f"Ejecuta las dos fases: "
        f"(1) ENTRENAMIENTO: 'python trainRL.py --episodes 80' en el directorio generador. "
        f"(2) OBSERVACION: 'python .github/scripts/run_and_observe.py {iteration}' "
        f"    desde el directorio raiz del repo. "
        f"El script run_and_observe.py evalua 30 partidas instrumentadas y genera "
        f"runner-report.json con metricas + diagnostico cualitativo. "
        f"Iteracion {iteration}/{max_iterations}. Objetivo: {objective}"
    )

    # Instruccion para el IMPLEMENTER:
    # Basada en evidencia real del observer, no en suposiciones.
    findings_text = " | ".join(diagnosis.get("findings", ["Sin datos del observer."]))
    implementer_instruction = (
        f"{plan['instructions']} "
        f"EVIDENCIA DEL OBSERVER (ultima evaluacion): {findings_text} "
        f"Iteracion {iteration}/{max_iterations}. Objetivo: {objective}"
    )

    return [
        {
            "worker_role":       "runner",
            "branch_name":       runner_branch,
            "base_branch":       base_branch,
            "instructions":      runner_instruction,
            "problem_diagnosed": diagnosis.get("problem_code", "unknown"),
        },
        {
            "worker_role":       "implementer",
            "branch_name":       implementer_branch,
            "base_branch":       base_branch,
            "instructions":      implementer_instruction,
            "problem_diagnosed": diagnosis.get("problem_code", "unknown"),
        },
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    objective = (
        os.environ.get("OBJECTIVE", "").strip()
        or os.environ.get("ISSUE_BODY", "").strip()
        or "Improve RL Snake agent with manager-worker loop"
    )

    iteration      = _as_int(os.environ.get("ITERATION", "1"), 1)
    max_iterations = _as_int(os.environ.get("MAX_ITERATIONS", "3"), 3)
    base_branch    = os.environ.get("BASE_BRANCH", "main").strip() or "main"

    report         = _load_report()
    should_continue = iteration <= max_iterations

    # Log del estado del manager (visible en Actions logs)
    diagnosis = report.get("diagnosis", {})
    metrics   = report.get("metrics", {})
    if should_continue:
        plan = _select_plan(diagnosis)
        print(f"[MANAGER] iteration={iteration}/{max_iterations}", file=sys.stderr)
        print(f"[MANAGER] problem_code={diagnosis.get('problem_code','?')}", file=sys.stderr)
        print(f"[MANAGER] diagnosis_summary={plan['summary']}", file=sys.stderr)
        print(
            f"[MANAGER] metrics: win_rate={metrics.get('rl_win_rate',0):.1f}% "
            f"early_death={metrics.get('early_death_rate',0):.1f}% "
            f"avg_fruit={metrics.get('avg_rl_fruit_score',0):.1f} "
            f"avoidable_deaths={diagnosis.get('avoidable_death_pct',0):.1f}%",
            file=sys.stderr
        )

    payload = {
        "objective":      objective,
        "iteration":      iteration,
        "max_iterations": max_iterations,
        "continue":       should_continue,
        "metrics":        metrics,
        "diagnosis":      diagnosis,
        "tasks":          [],
    }

    if should_continue:
        payload["tasks"] = _build_tasks(
            objective, iteration, max_iterations, base_branch, report
        )

    print(json.dumps(payload, ensure_ascii=True))


if __name__ == "__main__":
    main()
