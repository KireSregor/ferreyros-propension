# Revisión temporal anidada

Objetivo operativo: estimar compra por visita con información disponible antes del resultado; cada nueva visita supone que los resultados de visitas anteriores ya se conocen. El objetivo deseado es precisión y recall simultáneamente superiores al 60%; no se garantiza alcanzarlo.

Para cada fold exterior, Optuna seleccionó parámetros, grupo de variables y umbral en tres validaciones internas anteriores. Fold exterior 405–512: validaciones internas 151–220, 221–296 y 297–404. Fold exterior 513–620: 221–296, 297–404 y 405–512. En cada validación interna se entrena desde el día 1 hasta el día previo. El objetivo de Optuna es el promedio del mínimo entre precisión y recall en los tres folds interiores. Veinte pruebas por fold exterior; 122 entrenamientos en total, 65,2 segundos. El candidato cambia entre folds porque se vuelve a seleccionar con la historia disponible en cada corte.

| Train | Validación exterior | Precisión train | Recall train | Precisión val | Recall val | F1 val |
|---|---|---:|---:|---:|---:|---:|
| 1–404 | 405–512 | 50,4% | 59,5% | 47,1% | 63,0% | 53,9% |
| 1–512 | 513–620 | 49,1% | 54,0% | 47,5% | 60,1% | 53,1% |

El modelo anterior de 3 hojas obtuvo, respectivamente, 50,8%/58,8% y 50,4%/56,8% en precisión/recall de validación. Sus variables y enfoque se eligieron tras examinar los períodos, así que no es una comparación estrictamente independiente. La nueva búsqueda sacrifica precisión para ganar recall y no mejora simultáneamente ambas métricas. Baseline de la comparación anterior: 50,9%/52,7% y 48,1%/48,1%, con su propio umbral.

Los cortes exteriores no influyen directamente en la búsqueda interna correspondiente, pero todos los períodos han sido revisados anteriormente en este proyecto. La validación exterior no es una prueba independiente de todo el proceso iterativo. El período 621–730 no se consultó en esta ejecución y también fue examinado anteriormente. Las métricas no prueban imposibilidad, ausencia de sobreajuste ni desempeño futuro.

Archivos `trials_404.csv`, `trials_512.csv`, `configuration.json` y `metrics.csv` contienen la búsqueda; reproducir con `python nested_review.py`.
