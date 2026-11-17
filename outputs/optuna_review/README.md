# Revisión competitiva y búsqueda Optuna

La revisión detectó que el grupo llamado historial de la prueba anterior incluía derivados de precios y promociones. En esta prueba se separaron explícitamente historial, historial con ofertas e historial con ofertas y perfil. El historial se actualiza con resultados observados en visitas previas, incluso dentro de cada período de validación: supone que esos resultados están disponibles antes de la siguiente visita.

Se alteraron la compra, marca y cantidad de todas las visitas desde el día 512 y se recalcularon las variables. No cambió ninguna celda de variables en visitas del día 512 o anteriores. Es una comprobación de fuga temporal, no una prueba exhaustiva de todas las fuentes de fuga.

Optuna ejecutó 30 pruebas TPE; cada prueba entrenó dos modelos: días 1–296 para elegir el umbral en días 297–404, y días 1–404 para evaluar el candidato en 405–512. El objetivo fue maximizar el menor valor entre precisión y recall en 405–512. El mejor candidato se volvió a entrenar con días 1–512 y se contrastó en 513–620 con el umbral fijo. Se añadieron tres entrenamientos para la tabla final: 63 ajustes en total. Duración medida: 24 segundos. No se usaron días posteriores al 620 en este script.

| Train | Validación | Precisión train | Recall train | Precisión val | Recall val | F1 val |
|---|---|---:|---:|---:|---:|---:|
| 1–296 | 297–404, calibración | 49,4% | 48,4% | 51,3% | 51,3% | 51,3% |
| 1–404 | 405–512, selección Optuna | 50,4% | 48,1% | 54,3% | 54,2% | 54,3% |
| 1–512 | 513–620, contraste | 51,9% | 48,7% | 49,3% | 55,4% | 52,2% |

El umbral es 28,99%. El fold de selección es optimista porque eligió el mejor de 30 candidatos. El período de contraste ya fue revisado en investigaciones previas, de modo que tampoco constituye evaluación independiente. En la última validación el modelo de la comparación anterior, con 3 hojas, obtuvo 50,4% de precisión y 56,8% de recall. No hay evidencia suficiente para sustituirlo por este candidato. El test 621–730 no se consultó durante esta revisión.

Los resultados y parámetros completos están en `metrics.csv`, `trials.csv` y `configuration.json`. Reproducir con `python optuna_review.py`.
