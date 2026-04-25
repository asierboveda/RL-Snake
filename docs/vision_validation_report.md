# VI-06 Validacion del VisionParser contra dataset

## Metricas agregadas

| metrica | valor |
| --- | --- |
| `captures` | `3` |
| `turn_accuracy` | `1.000000` |
| `score_exact_rate` | `1.000000` |
| `snake_cell_precision` | `1.000000` |
| `snake_cell_recall` | `1.000000` |
| `snake_cell_f1` | `1.000000` |
| `snake_head_accuracy` | `1.000000` |
| `fruit_precision` | `1.000000` |
| `fruit_recall` | `1.000000` |
| `fruit_f1` | `1.000000` |
| `mean_parser_confidence` | `0.999257` |
| `total_errors` | `0` |
| `total_warnings` | `0` |

## Metricas por captura

| captura | turn_ok | score_exact | snake_precision | snake_recall | head_accuracy | fruit_precision | fruit_recall | parser_confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `snake000` | `True` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.999238` |
| `snake012` | `True` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.999235` |
| `snake025` | `True` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.999299` |

## Casos dificiles comentados (top 20)

| # | captura | componente | entidad | celda | confianza | match | comentario |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `snake025` | `snake` | `B:snake_tail` | `(0, 35)` | `0.998674` | `True` | segmento serpiente: coincide con ground truth; zona de borde, alta densidad local |
| 2 | `snake025` | `snake` | `B:snake_head` | `(0, 33)` | `0.998802` | `True` | segmento serpiente: coincide con ground truth; zona de borde, alta densidad local |
| 3 | `snake025` | `snake` | `B:snake_body` | `(0, 34)` | `0.999005` | `True` | segmento serpiente: coincide con ground truth; zona de borde, alta densidad local |
| 4 | `snake012` | `fruit` | `fruit_15` | `(0, 28)` | `0.998552` | `True` | fruta: coincide con ground truth; zona de borde |
| 5 | `snake012` | `fruit` | `fruit_10` | `(24, 13)` | `0.998598` | `True` | fruta: coincide con ground truth; alta densidad local |
| 6 | `snake012` | `snake` | `Y:snake_head` | `(28, 30)` | `0.998678` | `True` | segmento serpiente: coincide con ground truth; alta densidad local |
| 7 | `snake012` | `snake` | `Y:snake_tail` | `(27, 30)` | `0.998758` | `True` | segmento serpiente: coincide con ground truth; alta densidad local |
| 8 | `snake025` | `snake` | `Y:snake_tail` | `(32, 24)` | `0.998770` | `True` | segmento serpiente: coincide con ground truth; alta densidad local |
| 9 | `snake025` | `snake` | `Y:snake_head` | `(31, 24)` | `0.998817` | `True` | segmento serpiente: coincide con ground truth; alta densidad local |
| 10 | `snake012` | `snake` | `B:snake_tail` | `(1, 36)` | `0.998837` | `True` | segmento serpiente: coincide con ground truth; alta densidad local |
| 11 | `snake012` | `snake` | `B:snake_head` | `(1, 37)` | `0.998859` | `True` | segmento serpiente: coincide con ground truth; alta densidad local |
| 12 | `snake025` | `snake` | `R:snake_head` | `(31, 14)` | `0.999126` | `True` | segmento serpiente: coincide con ground truth; alta densidad local |
| 13 | `snake025` | `snake` | `R:snake_tail` | `(33, 14)` | `0.999356` | `True` | segmento serpiente: coincide con ground truth; alta densidad local |
| 14 | `snake025` | `snake` | `R:snake_body` | `(32, 14)` | `0.999382` | `True` | segmento serpiente: coincide con ground truth; alta densidad local |
| 15 | `snake000` | `fruit` | `fruit_15` | `(31, 1)` | `0.998377` | `True` | fruta: coincide con ground truth; caso de referencia estable |
| 16 | `snake012` | `fruit` | `fruit_15` | `(31, 1)` | `0.998538` | `True` | fruta: coincide con ground truth; caso de referencia estable |
| 17 | `snake000` | `fruit` | `fruit_10` | `(24, 13)` | `0.998574` | `True` | fruta: coincide con ground truth; caso de referencia estable |
| 18 | `snake000` | `fruit` | `fruit_15` | `(16, 7)` | `0.998580` | `True` | fruta: coincide con ground truth; caso de referencia estable |
| 19 | `snake012` | `fruit` | `fruit_15` | `(16, 7)` | `0.998583` | `True` | fruta: coincide con ground truth; caso de referencia estable |
| 20 | `snake012` | `fruit` | `fruit_15` | `(33, 14)` | `0.998584` | `True` | fruta: coincide con ground truth; caso de referencia estable |
