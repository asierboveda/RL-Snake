# ROL
Actúa como un Ingeniero de Machine Learning Staff / Experto en Deep Reinforcement Learning.
Tu objetivo es diseñar, implementar y optimizar un agente que juegue de forma perfecta a un entorno personalizado de Snake utilizando técnicas de Reinforcement Learning.

# CONTEXTO DEL ENTORNO (SNAKE MULTIJUGADOR)
El entorno del juego (`SnakeGame.py`) es un grid donde interactúan múltiples serpientes.
Las clases de jugadores actuales (`GreedyPlayer.py`, `RandomPlayer.py`) implementan una interfaz básica:
- Se inicializan con `__init__(self, playerID, color, game)`.
- Deben implementar el método `play(self, im)` que recibe un snapshot visual del tablero (`im`) y debe retornar un string con la dirección: `'N'`, `'S'`, `'E'`, o `'W'`.
- El objeto `self.game` (instancia de `SnakeGame`) contiene el estado exacto: `self.game.snakes`, `self.game.fruits`, etc.

## REGLAS CRÍTICAS DE RECOMPENSA (REWARD SHAPING)
Para entrenar o definir la lógica matemática del agente, debes tener en cuenta la siguiente heurística de recompensas del entorno:
1. **Comer Fruta (+)**: Aumenta los puntos base y los `fruit_points`. Es prioritario en fases tempranas.
2. **Supervivencia (+)**: Mantenerse vivo suma valor. Chocar contra muros o contra sí mismo es una muerte instantánea (penalización masiva).
3. **Battle Royale (+/-)**: Si dos serpientes chocan:
   - Si ambas tienen `< 120` puntos de fruta, mueren ambas.
   - Si chocan y una tiene `> 120` puntos y la otra menos (o empate), la lógica determina quién sobrevive en base a `getFruitScore()`. Matar otorga `+30` puntos.
   - Estrategia óptima: Huir de rivales si `fruit_points < 120`. Perseguir rivales más débiles si `fruit_points >= 120`.

# INSTRUCCIONES DE IMPLEMENTACIÓN
Debes generar un nuevo componente llamado `RLPlayer.py` que contenga una clase `RLPlayer`.

1. **Extracción de Estado**: En el método `play`, no confíes solo en la imagen plana. Extrae un vector de estado de `self.game` (posición de la cabeza, distancia a frutas, distancia a paredes, posiciones enemigas, y un flag binario de si somos "cazadores" `score >= 120`).
2. **Arquitectura del Agente**:
   - Implementa la estructura de un agente Q-Learning tabular avanzado o una red neuronal pequeña usando `PyTorch` (Deep Q-Network).
   - Si usas DQN, proporciona la definición de la red neuronal (clase que herede de `nn.Module`).
3. **Inferencia Robusta**: El método `play(self, im)` debe ejecutar el `forward` de la red o la consulta a la Q-Table para decidir la mejor acción.
4. **Evitación Crítica de Colisiones**: Implementa una capa final heurística (Action Masking) que prohíba a la red elegir una acción que resulte en suicidio instantáneo (muros o propio cuerpo) independientemente de la recomendación de la red.
5. **Código de Entrenamiento (Opcional pero recomendado)**: Incluye un bloque comentado al final del archivo o métodos separados que muestren cómo se debe instanciar el loop de entrenamiento de episodios (cálculo de recompensas, propagación hacia atrás del error).

# RESTRICCIONES DE CÓDIGO
- Usa sintaxis compatible con Python 3.9+.
- Importa solo librerías estándar o comunes en ML (`numpy`, `torch`, `random`).
- Tu salida debe ser estrictamente código fuente, sin charla introductoria. Genera el código completo de `RLPlayer.py`.