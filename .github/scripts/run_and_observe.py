"""
run_and_observe.py
------------------
Script ejecutado por el Runner Worker en GitHub Actions.

Hace dos cosas distintas:
  1. ENTRENAMIENTO: lanza trainRL.py en modo headless para que el agente aprenda.
  2. OBSERVACION: evalua el agente entrenado en N partidas, pero en lugar de
     solo contar muertes, observa el COMPORTAMIENTO en cada turno para diagnosticar
     POR QUE falla el agente.

El resultado es un JSON con dos secciones:
  - metrics: numeros crudos (para comparar iteraciones)
  - diagnosis: narrativa cualitativa (para que el manager tome decisiones)

REGLA CLAVE: El runner NUNCA modifica codigo. Solo observa y reporta.
El manager usa el 'diagnosis' para decidir que debe cambiar el implementer.
"""

import json
import random
import sys
import os
from collections import Counter, defaultdict
from pathlib import Path

# Asegurarse de que los modulos del proyecto (GreedyPlayer, RLPlayer, SnakeGame)
# son importables independientemente de desde donde se ejecuta el script.
# En GitHub Actions el runner hace 'cd generador' antes de invocar este script.
# Localmente puede ejecutarse desde el directorio 'generador' directamente.
_script_dir = Path(__file__).resolve().parent          # .github/scripts/
_repo_root  = _script_dir.parent.parent                # raiz del repo
_game_dir   = _repo_root / "generador"                 # donde estan los .py del juego
for _p in [str(_game_dir), str(_repo_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


import numpy as np

# Importaciones del proyecto
from GreedyPlayer import GreedyPlayer
from RLPlayer import RLPlayer
from SnakeGame import SnakeGame

# ─── Configuracion ────────────────────────────────────────────────────────────

EVAL_GAMES             = 30     # Partidas de evaluacion (epsilon=0, sin entrenamiento)
TURN_LIMIT             = 900    # Maximo de turnos por partida (igual que launcher)
MIN_SCORE_FOR_WINNING  = 120    # Umbral de puntuacion para ganar por score
EARLY_DEATH_TURN       = 60     # Muertes antes de este turno se consideran "tempranas"

# ─── Clasificadores de estado y muerte ───────────────────────────────────────

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def dist_to_nearest_wall(head, game):
    """Distancia minima del agente a cualquier borde del tablero."""
    return min(head[0], game.rSize - 1 - head[0],
               head[1], game.cSize - 1 - head[1])


def dist_to_nearest_enemy_head(head, game, my_id=0):
    """Distancia Manhattan a la cabeza del enemigo mas cercano."""
    dists = []
    for i, s in enumerate(game.snakes):
        if i == my_id or not s.isAlive:
            continue
        dists.append(manhattan(head, s.body[0]))
    return min(dists) if dists else 999


def count_safe_actions(head, game):
    """Cuantas de las 4 acciones no llevan a colision inmediata."""
    moves = {
        'N': [head[0] - 1, head[1]],
        'S': [head[0] + 1, head[1]],
        'E': [head[0], head[1] + 1],
        'W': [head[0], head[1] - 1],
    }
    safe = 0
    for pos in moves.values():
        r, c = pos
        out = r < 0 or r >= game.rSize or c < 0 or c >= game.cSize
        if out:
            continue
        body_hit = any(s.occupies(pos) for s in game.snakes if s.isAlive)
        if not body_hit:
            safe += 1
    return safe


def classify_death(game, rl_snake):
    """
    Clasifica la causa de muerte del RLPlayer con granularidad maxima.

    Returns:
        str: 'wall' | 'self' | 'enemy' | 'trapped' | 'alive' | 'unknown'
    """
    if rl_snake.isAlive:
        return "alive"

    head = rl_snake.body[0]

    # Fuera de tablero
    if head[0] < 0 or head[0] >= game.rSize or head[1] < 0 or head[1] >= game.cSize:
        return "wall"

    # Cuerpo propio
    for piece in rl_snake.body[1:]:
        if head[:2] == piece[:2]:
            return "self"

    # Cuerpo de enemigo
    for i, s in enumerate(game.snakes):
        if i == 0:
            continue
        for piece in s.body:
            if head[:2] == piece[:2]:
                return "enemy"

    return "unknown"


def classify_outcome(game):
    snake_labels = ["RLPlayer", "Greedy-B", "Greedy-R", "Greedy-Y"]
    alive = [(i, s) for i, s in enumerate(game.snakes) if s.isAlive]
    if len(alive) == 1:
        return snake_labels[alive[0][0]], "single_alive"
    if len(alive) == 0:
        return "No winner", "all_dead"
    scores = game.getScores()
    max_score = max(scores)
    if max_score < MIN_SCORE_FOR_WINNING:
        return "No winner", "draw_low_score"
    winners = [i for i, sc in enumerate(scores) if sc == max_score]
    if len(winners) == 1:
        return snake_labels[winners[0]], "score_winner"
    return "No winner", "draw_tie"


# ─── Partida instrumentada ────────────────────────────────────────────────────

def run_observed_game(seed):
    """
    Ejecuta una partida de evaluacion completa e instrumentada.

    En cada turno recoge el contexto del agente para poder diagnosticar
    patrones de comportamiento (no solo el resultado final).
    """
    random.seed(seed)
    game = SnakeGame()
    for _ in range(5):
        game.addRandomFruit()
    game.setNoise(0.0)

    # Evaluacion: sin exploracion aleatoria, sin actualizacion de Q-table
    rl = RLPlayer(0, "G", game, epsilon=0.0, training_enabled=False)
    players = [rl, GreedyPlayer(1, "B", game),
                   GreedyPlayer(2, "R", game),
                   GreedyPlayer(3, "Y", game)]

    # Historico de condiciones por turno (para analisis post-mortem)
    turn_history = []

    while game.turn < TURN_LIMIT and game.gameIsAlive():
        rl_snake = game.snakes[0]

        if rl_snake.isAlive:
            head = rl_snake.body[0]
            safe_n = count_safe_actions(head, game)
            wall_d = dist_to_nearest_wall(head, game)
            enemy_d = dist_to_nearest_enemy_head(head, game)
            is_hunter = rl_snake.getFruitScore() >= 120

            turn_history.append({
                "turn":          game.turn,
                "safe_actions":  safe_n,
                "wall_dist":     wall_d,
                "enemy_dist":    enemy_d,
                "is_hunter":     is_hunter,
                "fruit_score":   rl_snake.getFruitScore(),
            })

        dirs = [p.play(None) for p in players]
        for idx, d in enumerate(dirs):
            game.movePlayer(idx, d)
        game.checkMovements()
        game.update()
        game.turn += 1

    rl.end_episode()

    rl_snake = game.snakes[0]
    winner, outcome = classify_outcome(game)
    cause = classify_death(game, rl_snake)
    fruit_score = rl_snake.getFruitScore()
    total_score = rl_snake.getScore()
    kills = max(0, (total_score - fruit_score) // 30)

    # Contexto del momento de la muerte (ultimo registro del historico)
    death_ctx = turn_history[-1] if turn_history else {}

    return {
        "turns":              game.turn,
        "cause":              cause,
        "winner":             winner,
        "outcome":            outcome,
        "fruit_score":        fruit_score,
        "total_score":        total_score,
        "kills":              kills,
        "rl_alive":           rl_snake.isAlive,
        # Contexto de muerte para diagnostico
        "death_safe_actions": death_ctx.get("safe_actions", -1),
        "death_wall_dist":    death_ctx.get("wall_dist", -1),
        "death_enemy_dist":   death_ctx.get("enemy_dist", -1),
        "death_is_hunter":    death_ctx.get("is_hunter", False),
        "death_fruit_score":  death_ctx.get("fruit_score", 0),
        # Estadisticas generales de la partida
        "avg_wall_dist":      float(np.mean([t["wall_dist"] for t in turn_history])) if turn_history else 0,
        "avg_safe_actions":   float(np.mean([t["safe_actions"] for t in turn_history])) if turn_history else 0,
        "turns_as_hunter":    sum(1 for t in turn_history if t["is_hunter"]),
    }


# ─── Analisis agregado y diagnostico cualitativo ─────────────────────────────

def build_observation_report(results):
    """
    Convierte los resultados crudos en un diagnostico cualitativo
    que el manager puede usar directamente para decidir que cambiar.
    """
    n = len(results)
    deaths = [r for r in results if not r["rl_alive"]]
    n_dead = len(deaths)

    # ── Metricas cuantitativas (para comparar iteraciones) ──────────────────
    wins         = sum(1 for r in results if r["winner"] == "RLPlayer")
    early_deaths = sum(1 for r in results if not r["rl_alive"] and r["turns"] < EARLY_DEATH_TURN)
    cause_counts = Counter(r["cause"] for r in results)
    outcome_counts = Counter(r["outcome"] for r in results)

    metrics = {
        "rl_win_rate":        round((wins / n) * 100, 1),
        "avg_turns":          round(float(np.mean([r["turns"] for r in results])), 1),
        "avg_rl_score":       round(float(np.mean([r["total_score"] for r in results])), 1),
        "avg_rl_fruit_score": round(float(np.mean([r["fruit_score"] for r in results])), 1),
        "avg_rl_kills":       round(float(np.mean([r["kills"] for r in results])), 2),
        "early_death_rate":   round((early_deaths / n) * 100, 1),
        "death_cause_counts": dict(cause_counts),
        "outcome_counts":     dict(outcome_counts),
    }

    # ── Analisis cualitativo de comportamiento ───────────────────────────────

    # 1. ¿Tenia el agente alternativas cuando murio?
    avoidable_deaths = sum(
        1 for r in deaths
        if r["death_safe_actions"] > 0  # habia al menos una accion segura disponible
    )
    trapped_deaths = sum(
        1 for r in deaths
        if r["death_safe_actions"] == 0  # sin salida (inevitable)
    )
    avoidable_pct = (avoidable_deaths / n_dead * 100) if n_dead > 0 else 0
    trapped_pct   = (trapped_deaths   / n_dead * 100) if n_dead > 0 else 0

    # 2. Distancia media a la pared en el momento de la muerte
    wall_dists_at_death = [r["death_wall_dist"] for r in deaths if r["death_wall_dist"] >= 0]
    avg_wall_at_death   = float(np.mean(wall_dists_at_death)) if wall_dists_at_death else 0

    # 3. ¿En que estado (hunter/farmer) murio?
    hunter_deaths = sum(1 for r in deaths if r["death_is_hunter"])
    farmer_deaths = n_dead - hunter_deaths

    # 4. Distancia media al enemigo mas cercano en la muerte
    enemy_dists = [r["death_enemy_dist"] for r in deaths if r["death_enemy_dist"] < 999]
    avg_enemy_at_death = float(np.mean(enemy_dists)) if enemy_dists else 999

    # 5. Wall dist media durante toda la partida (¿juega cerca de bordes?)
    avg_wall_overall = float(np.mean([r["avg_wall_dist"] for r in results]))

    # ── Seleccion del problema dominante con evidencia ───────────────────────

    dominated_cause = cause_counts.most_common(1)[0][0] if cause_counts else "unknown"

    # Construir narrativa de diagnostico
    findings = []

    # Problema 1: muertes evitables -> la Q-table no penaliza bien
    if avoidable_pct >= 60 and dominated_cause in ("wall", "self"):
        findings.append(
            f"CRITICO: El {avoidable_pct:.0f}% de las muertes por '{dominated_cause}' son EVITABLES "
            f"(habia acciones seguras disponibles que el agente ignoro). "
            f"La Q-table no asocia correctamente la proximidad al muro con valor negativo. "
            f"Recomendacion: (1) incluir distancia a muros en el estado del agente, "
            f"(2) aumentar penalizacion por accion que acerca al muro aunque no mate inmediatamente."
        )
    elif dominated_cause in ("wall", "self") and avoidable_pct >= 30:
        findings.append(
            f"MODERADO: Muertes por '{dominated_cause}'. El {avoidable_pct:.0f}% eran evitables. "
            f"Distancia media al muro en el momento de la muerte: {avg_wall_at_death:.1f} celdas. "
            f"El agente se acerca a los bordes innecesariamente. "
            f"Recomendacion: añadir shaping reward negativo proporcional a la proximidad al muro."
        )

    # Problema 2: muertes atrapado -> problema de planificacion mas profundo
    if trapped_pct >= 40:
        findings.append(
            f"CRITICO: El {trapped_pct:.0f}% de las muertes ocurren cuando el agente NO tenia "
            f"NINGUNA accion segura disponible. El agente se mete en callejones sin salida. "
            f"Recomendacion: el estado debe incluir deteccion de 'espacio libre' a 2-3 pasos "
            f"(flood-fill simplificado) para evitar entrar en zonas cerradas."
        )

    # Problema 3: muertes por enemigo con enemigo cerca
    if dominated_cause == "enemy" and avg_enemy_at_death < 4:
        findings.append(
            f"CRITICO: El agente muere por enemigo con distancia media de {avg_enemy_at_death:.1f} celdas. "
            f"El agente detecta al enemigo pero no evade. "
            f"Recomendacion: incluir en el estado flags de presencia enemiga en las 4 celdas adyacentes "
            f"y añadir penalizacion por acercarse a enemigos cuando fruit_score < 120."
        )
    elif dominated_cause == "enemy" and avg_enemy_at_death >= 4:
        findings.append(
            f"MODERADO: Muertes por enemigo pero el enemigo estaba lejos ({avg_enemy_at_death:.1f} celdas). "
            f"Probable colision lateral o temporal (dos serpientes en la misma celda en el mismo turno). "
            f"Recomendacion: mejorar la funcion find_goal para evitar converger en la misma celda "
            f"que un enemigo de forma simultanea."
        )

    # Problema 4: muere mas como hunter que como farmer -> comportamiento agresivo mal calibrado
    if hunter_deaths > farmer_deaths and n_dead > 5:
        findings.append(
            f"OBSERVACION: El agente muere mas veces en modo CAZADOR ({hunter_deaths}) "
            f"que en modo FARMER ({farmer_deaths}). "
            f"El comportamiento agresivo (perseguir rivales) es contraproducente. "
            f"Recomendacion: cuando is_hunter=True, añadir verificacion de que el rival "
            f"objetivo sigue siendo mas debil ANTES de moverse hacia el."
        )

    # Problema 5: win_rate bajo pero no muere pronto -> no sabe convertir ventaja
    if metrics["rl_win_rate"] < 15 and metrics["early_death_rate"] < 25:
        findings.append(
            f"OBSERVACION: El agente sobrevive ({early_death_rate:.0f}% muertes tempranas) "
            f"pero su win_rate es solo {metrics['rl_win_rate']:.0f}%. "
            f"El agente no convierte supervivencia en victoria. "
            f"Recomendacion: reforzar reward shaping when the agent reaches fruit_score >= 100 "
            f"y añadir reward positivo por cada rival eliminado."
        )

    # Si no hay problema critico claro
    if not findings:
        findings.append(
            f"ESTADO ACEPTABLE: No hay un patron de fallo dominante. "
            f"win_rate={metrics['rl_win_rate']:.0f}%, early_death={metrics['early_death_rate']:.0f}%. "
            f"Recomendacion: reducir epsilon para consolidar la politica aprendida."
        )

    # ── Diagnostico final para el manager ───────────────────────────────────
    # Codigo de problema (para heuristica automatica del manager)
    if avoidable_pct >= 60 and dominated_cause in ("wall", "self"):
        problem_code = "avoidable_wall_self_deaths"
    elif trapped_pct >= 40:
        problem_code = "trapped_no_safe_action"
    elif dominated_cause == "enemy" and avg_enemy_at_death < 4:
        problem_code = "enemy_proximity_deaths"
    elif hunter_deaths > farmer_deaths and n_dead > 5:
        problem_code = "aggressive_hunter_deaths"
    elif metrics["rl_win_rate"] < 15 and metrics["early_death_rate"] < 25:
        problem_code = "survives_but_no_wins"
    else:
        problem_code = "consolidation"

    diagnosis = {
        "problem_code":        problem_code,
        "dominant_cause":      dominated_cause,
        "avoidable_death_pct": round(avoidable_pct, 1),
        "trapped_death_pct":   round(trapped_pct, 1),
        "avg_wall_dist_death": round(avg_wall_at_death, 2),
        "avg_enemy_dist_death":round(avg_enemy_at_death, 2),
        "hunter_deaths":       hunter_deaths,
        "farmer_deaths":       farmer_deaths,
        "findings":            findings,
    }

    return metrics, diagnosis


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    iteration = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    base_seed = 10000 + iteration

    print(f"=== OBSERVER: {EVAL_GAMES} partidas de evaluacion (seed base={base_seed}) ===")

    results = []
    for i in range(EVAL_GAMES):
        r = run_observed_game(base_seed + i)
        results.append(r)
        print(
            f"[eval {i+1:>2}/{EVAL_GAMES}] "
            f"turns={r['turns']:>4}  alive={r['rl_alive']}  "
            f"cause={r['cause']:>12}  "
            f"safe_at_death={r['death_safe_actions']}  "
            f"wall_dist={r['death_wall_dist']}  "
            f"fruit={r['fruit_score']:>3}  "
            f"winner={r['winner']}"
        )

    metrics, diagnosis = build_observation_report(results)

    print("\n=== DIAGNOSIS ===")
    print(f"  problem_code   : {diagnosis['problem_code']}")
    print(f"  dominant_cause : {diagnosis['dominant_cause']}")
    print(f"  avoidable%     : {diagnosis['avoidable_death_pct']:.1f}%")
    print(f"  trapped%       : {diagnosis['trapped_death_pct']:.1f}%")
    print(f"  wall_dist@death: {diagnosis['avg_wall_dist_death']:.1f}")
    print(f"  enemy_dist@dth : {diagnosis['avg_enemy_dist_death']:.1f}")
    print(f"  hunter deaths  : {diagnosis['hunter_deaths']}")
    print(f"  farmer deaths  : {diagnosis['farmer_deaths']}")
    print("\n  --- Findings ---")
    for f in diagnosis["findings"]:
        print(f"  * {f}")

    # Payload final para el manager
    report = {"metrics": metrics, "diagnosis": diagnosis}
    Path("runner-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    print("\n=== RUNNER REPORT JSON ===")
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
