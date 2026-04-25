# Reglas del juego

## Tablero

El juego se ejecuta sobre una malla discreta de `44x44`. Cada serpiente ocupa una o mas celdas y tiene una direccion actual: `N`, `S`, `E` o `W`.

## Movimiento

En cada turno las serpientes reciben una direccion. El motor actual sigue usando direcciones absolutas, aunque el wrapper PPO puede traducir acciones relativas (`FORWARD`, `LEFT`, `RIGHT`) a `N/S/E/W`.

Una serpiente muere si:

- sale del tablero;
- choca consigo misma;
- choca con otra serpiente en una situacion que no puede ganar.

## Fruta y score

Las frutas tienen valor `10`, `15` o `20`. Al comer fruta:

- aumenta el score total;
- aumenta el `fruit_score`;
- la serpiente crece.

El `fruit_score` es importante porque determina si una serpiente puede matar.

## Combate y umbral de 120

La regla critica del proyecto es:

```text
Una serpiente solo puede matar correctamente si fruit_score >= 120
y tiene mas fruit_score que el rival.
```

Casos:

- Si ambas tienen el mismo `fruit_score`, mueren ambas.
- Si ninguna tiene `fruit_score >= 120`, mueren ambas al chocar.
- Si una supera el umbral y tiene mas fruta que la otra, mata al rival.
- La kill suma score de kill, pero no aumenta `fruit_score`.

Esta separacion evita que una serpiente se vuelva "cazadora" por kills sin haber comido fruta.

## Fin de partida

El entorno puede terminar por:

- una unica serpiente viva;
- todas muertas;
- condicion de score;
- empate;
- limite de turnos del entorno headless.

Los casos estan formalizados en `board_state.py`, `snake_env.py` y documentos de reglas como `docs/rules_cases.md`.
