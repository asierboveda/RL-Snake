"""
trainRL.py - Orquestador de Entrenamiento Headless para RLPlayer

Ejecuta N episodios (partidas completas) a maxima velocidad, sin renderizado ni
guardado de imagenes. Al terminar, la Q-Table del agente queda persistida en disco
lista para ser usada en modo inferencia desde snakeGameLauncher.py.

Uso:
    python trainRL.py              <- 500 episodios por defecto
    python trainRL.py --episodes 2000
    python trainRL.py --episodes 1000 --seed 42
"""

import argparse
import random

from SnakeGame import SnakeGame
from RLPlayer import RLPlayer
from GreedyPlayer import GreedyPlayer

# --- Configuracion de entrenamiento ------------------------------------------

TURN_LIMIT      = 900   # Maximo de turnos por episodio (igual que el launcher visual)
NOISE_LEVEL     = 0.0   # Sin ruido en entrenamiento (reduce variabilidad innecesaria)
INITIAL_FRUITS  = 5     # Frutas al inicio de cada episodio
SAVE_EVERY      = 50    # Guardar Q-Table cada N episodios
LOG_EVERY       = 10    # Imprimir metricas cada N episodios

# --- Argumentos de linea de comandos -----------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Entrenamiento Headless del RLPlayer')
    parser.add_argument('--episodes', type=int, default=500,
                        help='Numero de episodios de entrenamiento (default: 500)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Semilla aleatoria para reproducibilidad (default: ninguna)')
    return parser.parse_args()

# --- Funcion para crear un episodio nuevo ------------------------------------

def create_episode(epsilon):
    """Inicializa motor de juego y agentes para un episodio nuevo."""
    sg = SnakeGame()
    sg.setNoise(NOISE_LEVEL)

    # Frutas iniciales
    for _ in range(INITIAL_FRUITS):
        sg.addRandomFruit()

    # El RLPlayer aprende (epsilon > 0 = exploracion activa durante entrenamiento)
    # Los oponentes son Greedy para dar presion de aprendizaje realista
    rl_agent  = RLPlayer(0, 'G', sg, epsilon=epsilon)
    opponentB = GreedyPlayer(1, 'B', sg)
    opponentC = GreedyPlayer(2, 'R', sg)
    opponentD = GreedyPlayer(3, 'Y', sg)

    agents = [rl_agent, opponentB, opponentC, opponentD]
    return sg, agents

# --- Bucle principal de Entrenamiento ----------------------------------------

def train(num_episodes, seed):
    if seed is not None:
        random.seed(seed)
        print(f"[Semilla fijada: {seed}]")

    # Epsilon decae linealmente: arranca en 1.0 (todo exploracion) y
    # termina en 0.05 (casi pura explotacion) al final del entrenamiento
    epsilon_start = 1.0
    epsilon_end   = 0.05

    sep = '=' * 52
    print(f"\n{sep}")
    print(f"  Entrenamiento RLPlayer --- {num_episodes} episodios")
    print(f"  Epsilon: {epsilon_start:.2f} -> {epsilon_end:.2f} (decay lineal)")
    print(f"{sep}\n")

    total_wins   = 0
    total_scores = []

    for ep in range(1, num_episodes + 1):

        # Epsilon actual (decay lineal por episodio)
        progress = (ep - 1) / max(num_episodes - 1, 1)
        epsilon  = epsilon_start - progress * (epsilon_start - epsilon_end)

        # Iniciar nuevo episodio
        sg, agents = create_episode(epsilon)
        rl_agent = agents[0]

        # Bucle de turno (HEADLESS: sin imagenes ni matplotlib)
        while sg.turn < TURN_LIMIT and sg.gameIsAlive():

            # Solicitar accion a cada agente (None = estado visual no se usa)
            directions = [agent.play(None) for agent in agents]

            # Enviar movimientos al motor del juego
            for player_id, direction in enumerate(directions):
                sg.movePlayer(player_id, direction)

            sg.checkMovements()
            sg.update()
            sg.turn += 1

        # Notificar al RLPlayer que el episodio termino
        # Si murio, aplica la penalizacion final sobre el ultimo estado
        rl_agent.play(None)

        # Metricas del episodio
        rl_score = sg.snakes[0].getScore()
        rl_alive = sg.snakes[0].isAlive
        total_scores.append(rl_score)
        if rl_alive:
            total_wins += 1

        # Guardado periodico
        if ep % SAVE_EVERY == 0:
            rl_agent.save_q_table()

        # Log periodico
        if ep % LOG_EVERY == 0:
            avg_score = sum(total_scores[-LOG_EVERY:]) / LOG_EVERY
            win_rate  = total_wins / ep * 100
            q_size    = len(rl_agent.q_table)
            print(
                f"[Ep {ep:>5}/{num_episodes}]  "
                f"eps={epsilon:.3f}  "
                f"score_avg={avg_score:>6.1f}  "
                f"win%={win_rate:>5.1f}  "
                f"Q-states={q_size:>6}"
            )

    # Guardado final garantizado
    rl_agent.save_q_table()
    print(f"\n{sep}")
    print(f"  Entrenamiento completado.")
    print(f"  Victorias totales : {total_wins}/{num_episodes} ({total_wins/num_episodes*100:.1f}%)")
    print(f"  Score medio global: {sum(total_scores)/len(total_scores):.1f}")
    print(f"  Estados Q aprendidos: {len(rl_agent.q_table)}")
    print(f"  Q-Table guardada en : {rl_agent.q_file}")
    print(f"{sep}\n")


# --- Entry point -------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args()
    train(num_episodes=args.episodes, seed=args.seed)
