# Implementer Trace - Iteracion 2
**Branch:** ai/impl-improve-rl-snake-agent-with-manager-worker-loop-it2
**Manager run:** 24690469870

## Objetivo
Improve RL Snake agent with manager-worker loop

## Instrucciones del manager
PROBLEMA DIAGNOSTICADO: muertes evitables por muro o cuerpo propio (el agente tenia acciones seguras pero eligio una peligrosa). CAMBIOS REQUERIDOS en RLPlayer.py: (1) En get_state(): añadir 4 valores de distancia normalizada a los bordes:     north_dist=head[0]/game.rSize, south_dist=(game.rSize-1-head[0])/game.rSize,     east_dist=(game.cSize-1-head[1])/game.cSize, west_dist=head[1]/game.cSize.     Discretizar en 3 niveles: 0=peligro(dist<=2), 1=precaucion(3-5), 2=libre(>5).     Incorporar estas 4 variables al tuple que retorna get_state(). (2) En update_q_table(): añadir penalizacion extra de -10 cuando la accion     reduce la distancia minima al muro a <= 2 celdas, aunque no sea muerte inmediata.     Esto se calcula antes de llamar a la formula Q estándar. (3) En play(): aumentar la penalizacion de muerte de -100 a -150. NO toques alpha, gamma, find_goal ni la logica de frutas en este ciclo. EVIDENCIA DEL OBSERVER (ultima evaluacion): Primera iteracion o sin datos del observer. Iteracion 2/3. Objetivo: Improve RL Snake agent with manager-worker loop

## Archivos modificados
- RLPlayer.py
