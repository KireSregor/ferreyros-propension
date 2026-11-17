# Resultado: no se consiguió una mejora del equilibrio frente al histórico

La revisión implementó y ejecutó el protocolo anidado, las cuatro familias clásicas, TabM/MLP, nuevas variables, ablaciones, semillas, calibración y ensambles. **Ningún procedimiento completo alcanza más de 60% en precisión y recall simultáneamente en los tres folds.** No se evaluó el test.

## Comparación principal

Promedios simples de las tres validaciones exteriores. El criterio congelado es la media de min(precisión, recall) calculado dentro de cada fold; no equivale al mínimo de las dos medias.

| Procedimiento | Precisión | Recall | F1 | AUC | Criterio de equilibrio |
|---|---:|---:|---:|---:|---:|
| Histórico | 50.0% | 50.5% | 50.2% | 0.724 | 49.3% |
| LightGBM de referencia | 52.4% | 53.8% | 52.8% | 0.756 | 49.3% |
| XGBoost: mejor familia nueva observada | 50.0% | 57.6% | 53.1% | 0.756 | 48.6% |

El histórico conserva el primer puesto según el criterio pactado. La diferencia frente a la referencia es mínima y no establece superioridad futura. XGBoost mejora F1 y AUC frente al histórico, pero su desequilibrio en el segundo fold impide declarar cumplido el objetivo. No se cambia el criterio después de observar este resultado.

La referencia utiliza los hiperparámetros del LightGBM anterior bajo el preprocesamiento y ajuste interno de umbral corregidos. Es una comparación controlada de esa configuración, no una reproducción literal de una antigua cifra obtenida con otro procedimiento.

## XGBoost: entrenamiento y validación por fold

| Fold | Train / validación | Datos | Visitas | Precisión | Recall | F1 | AUC |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | 1–296 / 297–404 | Train | 25,754 | 52.6% | 47.6% | 50.0% | 0.753 |
| 1 | 1–296 / 297–404 | Validación | 8,409 | 54.2% | 50.2% | 52.1% | 0.741 |
| 2 | 1–404 / 405–512 | Train | 34,163 | 51.4% | 56.9% | 54.0% | 0.775 |
| 2 | 1–404 / 405–512 | Validación | 8,364 | 46.6% | 65.0% | 54.3% | 0.774 |
| 3 | 1–512 / 513–620 | Train | 42,527 | 50.4% | 58.5% | 54.1% | 0.775 |
| 3 | 1–512 / 513–620 | Validación | 8,029 | 49.1% | 57.6% | 53.0% | 0.754 |

Train describe el ajuste sobre sus propias visitas; validación ocurre después, con modelo y umbral fijos y el historial actualizado según el contrato. Las brechas de AUC son moderadas; esto no demuestra ausencia de sobreajuste. El problema también incluye transferencia del umbral y capacidad predictiva insuficiente para la meta exigida.

## Qué indican las comprobaciones

- **El historial domina la señal.** Las permutaciones y las doce ablaciones no muestran una mejora grande y estable al añadir perfil, ofertas y nuevas tendencias. Las importancias son descriptivas y pueden ocultar redundancia.
- **El umbral pierde equilibrio.** En el segundo fold, XGBoost obtiene precisión 46,6% y recall 65,0%. Un diagnóstico posterior sobre su curva fija alcanza un equilibrio máximo de 54,7%; no llega a 60/60. Ese valor usa respuestas exteriores, es optimista y no se utiliza como umbral ni resultado del modelo.
- **No basta con mover el umbral de estos modelos.** La referencia y XGBoost no tienen ningún punto 60/60 en sus curvas exteriores observadas. Esto limita esas predicciones concretas; no prueba un límite irrebatible de los datos o de otros modelos.
- **Más complejidad no aportó una mejora defendible.** El blend neuronal/árboles combina probabilidades muy correlacionadas (0,95–0,98) y no supera el criterio principal. La calibración y el ajuste con tendencia previa de tienda tampoco resuelven el problema.
- **Sí hay mejora en otras propiedades.** XGBoost mejora F1 y AUC frente al histórico en los tres folds; los intervalos descriptivos de bootstrap por cliente son positivos para esas diferencias. No corrigen dependencia tienda/día ni la selección reiterada del proyecto.

## Trabajo efectivo y cobertura

- 370 pruebas Optuna completas en estudios principales: 310 clásicas y 60 neuronales; las redes completaron 20 por corte.
- 8 variantes neuronales adicionales de capacidad. Algunas familias clásicas terminaron por presupuesto antes de sus 40 pruebas objetivo; no se presenta la búsqueda como exhaustiva.
- 2,382 ajustes contabilizados: 2,314 principales, 60 diagnósticos y 8 del piloto GPU. Los ajustes clásicos se infieren de las trazas; los neuronales se cuentan mediante registros. Se excluyen pruebas sintéticas y la prueba de memorización pequeña.
- 15,721 épocas neuronales ejecutadas y registradas, contando búsquedas, reajustes y diagnósticos incluidos en el inventario.
- Cierre observado de la sesión de entrenamiento: 107.9 minutos, dentro de los 120 autorizados. Incluye trabajos solapados, coordinación y pausas; no es suma de tiempos por modelo ni tiempo activo continuo de GPU. El informe y la verificación posteriores no entrenan modelos.
- 23 pruebas del código aprobadas y revisión estática sin incidencias. EDA y notebook de resultados ejecutados; el último comprueba que el scoring reproduce predicciones guardadas.

## Recomendación

Conservar el histórico como referencia operativa para el objetivo de equilibrio y mantener el LightGBM de referencia y XGBoost como alternativas probabilísticas documentadas. No añadir un ensamble al flujo final ni sustituir automáticamente el artefacto vigente por una mejora incierta. Antes de otra búsqueda, concretar el coste de falsas alarmas y compras omitidas, revisar el punto real de intervención y obtener señales adicionales disponibles antes de la visita o datos nuevos para confirmar resultados. No hay evidencia para prometer 60/60 con más pruebas del mismo tipo.

Los períodos de desarrollo ya se examinaron repetidamente. El test 621–730, consultado históricamente, permaneció fuera de este encargo; una confirmación plenamente independiente requiere datos nuevos. No se publicaron cambios ni se generaron PDF, slides o TXT finales.

**Detalle:** `metrics.csv`, `leaderboard.csv`, `gaps.csv`, `bootstrap_vs_historical.csv`, `threshold_transfer_diagnostic.csv`, `experiment_inventory.csv` y `runtime_summary.json`. Auditoría: `AUDIT.md`; variables: `FEATURES.md`; metodología y skills: `PROTOCOL.md`; errores: `DIAGNOSTICS.md` y `LEADER_EXPLANATION.md`.
