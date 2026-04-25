# Reglas Críticas y Casos del Motor de Juego (Battle Royale)

Este documento detalla la matriz de resolución de eventos del motor headless (`SnakeGame.py`) para garantizar un entorno RL determinista y congruente con las reglas oficiales del Snake Battle Royale.

## 1. Fases de Ejecución (Movimiento Simultáneo)
El motor opera por turnos discretos y asume movimiento simultáneo. La correcta resolución ocurre en el siguiente orden estricto (implementado en `SnakeGame.checkMovements()`):

1. **Identificación de Supervivientes**: Solo las serpientes que estaban vivas al comienzo del turno actual pueden causar colisiones. Las serpientes muertas en turnos previos "desaparecen" como obstáculos.
2. **Evaluación de Muros y Auto-choques**: Cualquier serpiente viva que se salga de los límites o se choque consigo misma muere inmediatamente.
3. **Colisiones Cruzadas**: Se evalúan posibles cruces (head-to-head o head-to-body) entre cualquier par de serpientes vivas. Las consecuencias se calculan en base a los puntos, recolectando recompensas y marcando a las víctimas.
4. **Aplicación Simultánea de Efectos**: Se aplican las muertes por paredes y las muertes/recompensas por combates de manera simultánea.
5. **Consumo de Frutas**: **Solo las serpientes que han logrado sobrevivir a todas las colisiones de este turno** pueden finalmente reclamar y digerir frutas, otorgando la puntuación final por recolección de fruta y evitando el consumo concurrente injusto de la misma pieza.

## 2. Resolución de Colisiones entre Jugadores
Se produce una colisión de combate cuando **cualquier parte del cuerpo** de la Serpiente A se solapa con la Serpiente B (incluyendo choques cabeza-cabeza y cabeza-cuerpo). No se distingue entre chocar contra la cabeza o el cuerpo de otro jugador. Las reglas para resolver quién vive y quién muere se basan estrictamente en la cantidad de "puntos de fruta" (`fruitScore`) acumulados:

| Caso | Puntos de A | Puntos de B | Resultado (A) | Resultado (B) | Recompensa Otorgada |
|---|---|---|---|---|---|
| Ambos < 120 (Debilidad) | < 120 | < 120 | Muere | Muere | 0 |
| Empate de fuerzas | Igual a B | Igual a A | Muere | Muere | 0 |
| A es Cazador Mayor | >= 120 y A > B | < A | Sobrevive | Muere | A recibe +30 ptos (fruta) |
| B es Cazador Mayor | < B | >= 120 y B > A | Muere | Sobrevive | B recibe +30 ptos (fruta) |
| Ambos Cazadores (A > B)| >= 120 (ej: 150) | >= 120 (ej: 130) | Sobrevive | Muere | A recibe +30 ptos (fruta) |
| Ambos Cazadores, Empate| >= 120 (ej: 150) | >= 120 (ej: 150) | Muere | Muere | 0 |

**Notas Clave:**
- **Requisito de 120 puntos:** No basta con tener más puntos que el rival. Se debe tener un mínimo de 120 puntos de fruta acumulados para poder asesinar a otra serpiente y salir victorioso (Cazador). Si ninguno cumple este requisito de fuerza, la colisión resulta fatal para ambos por debilidad.
- **Mordiscos al cuerpo:** A diferencia de juegos de Snake tradicionales, morder el cuerpo de otra serpiente desencadena el mismo sistema de combate. Si eres un Cazador Mayor, morder el cuerpo de otra serpiente te permite matarla sin morir en el intento.

## 3. Puntuación y Recompensas
- **Puntuación de Frutas:** Consumir una fruta otorga 10, 15 o 20 puntos tanto al Score Total (`score`) como a los Puntos de Fruta (`fruit_score`).
- **Puntos de Kill:** Sobrevivir y ganar una colisión otorga un "Kill Award" de +30 puntos. Estos puntos *también* se añaden tanto al Score Total como a los Puntos de Fruta, incrementando tu fuerza para futuros combates (las reglas especifican: "Una serpiente que mata también gana puntos de fruta").

## 4. Gestión de Cadáveres ("Serpiente muerta")
Una vez muerta, el estado de la serpiente cambia a `isAlive = False`. En el tablero de juego y la matriz visual del motor headless:
- Las serpientes muertas dejan de ser dibujadas y representadas físicamente.
- No ocupan posiciones (`occupied_cells` solo rastrea piezas de serpientes vivas).
- No bloquean a otras serpientes en turnos posteriores (desaparecen). 
- Continúan apareciendo en el contrato `BoardState` emitido por el entorno para propósitos analíticos y de RL (`alive=False`), pero sin interferir en la física del juego.
