# Comparación temporal de modelos y ensambles

Se comparan regresión logística, LightGBM, XGBoost, CatBoost, TabM, un promedio de los cuatro primeros y un promedio de los cinco. Todos predicen compra por visita; se supone que los resultados de visitas anteriores están disponibles antes de cada nueva visita. El historial puede actualizarse dentro de un período de validación con resultados previos observados, nunca con el resultado de la visita actual.

## Diseño

| Fold exterior | Entrenamiento | Validación | Validaciones internas para escoger parámetros y umbral |
|---|---|---|---|
| 1 | 1–404 | 405–512 | 221–296 y 297–404 |
| 2 | 1–512 | 513–620 | 297–404 y 405–512 |

En cada validación interna el entrenamiento comienza en el día 1 y termina el día anterior a la validación. Para cada familia clásica se hicieron ocho pruebas Optuna por fold exterior (64 pruebas en total); cada prueba se puntuó con el menor valor entre precisión y recall en las predicciones internas concatenadas, eligiendo allí también el umbral. El umbral y los parámetros se congelan antes de evaluar el fold exterior. Los mismos datos internos se usan para seleccionar tanto parámetros como umbral; su puntuación es optimista. Los folds exteriores no intervinieron en la búsqueda concreta correspondiente, pero ya fueron examinados en iteraciones anteriores del proyecto: no son una evaluación independiente del proyecto completo.

LightGBM, XGBoost y CatBoost seleccionan la cantidad de árboles con parada temprana de 35 rondas, utilizando los últimos 40 días del entrenamiento de cada ajuste. Luego se reentrena con todo el período de entrenamiento y la cantidad de árboles elegida. La regresión logística emplea imputación y escalado ajustados en cada entrenamiento; las variables categóricas se tratan aquí como códigos numéricos, una simplificación que podría mejorar con codificación apropiada. CatBoost también recibe esos códigos como numéricos. Los cuatro modelos usan las mismas variables y una ponderación temporal ajustada por Optuna. Ocho pruebas por estudio son una exploración acotada, no garantía de óptimo.

TabM usó una configuración fija, ponderación temporal de vida media 180 días y selección interna de épocas por log loss (paciencia 10, máximo 80), seguida de reajuste. Seleccionó 2–7 épocas y no recibió la misma búsqueda Optuna; la comparación neuronal es preliminar. El ensamble promedia probabilidades con pesos iguales; su umbral se escoge sobre predicciones internas. No se optimizaron pesos ni se implementó stacking. No se calculó métrica de train de los ensambles porque no se guardaron las predicciones de train de cada componente.

## Validación exterior: precisión / recall / F1 / AUC

| Modelo | Fold 1: días 405–512 | Fold 2: días 513–620 |
|---|---|---|
| Baseline histórico | 50,9% / 52,7% / 51,8% / 0,743 | 48,9% / 47,7% / 48,3% / 0,707 |
| Mejor anterior, LightGBM 3 hojas | 50,8% / 58,8% / 54,5% / 0,777 | 50,4% / 56,8% / 53,4% / 0,752 |
| Regresión logística | 52,5% / 57,0% / 54,6% / 0,768 | 49,9% / 55,2% / 52,4% / 0,741 |
| LightGBM | 46,8% / 62,8% / 53,7% / 0,772 | 51,7% / 53,6% / 52,6% / 0,748 |
| XGBoost | 47,7% / 64,6% / 54,9% / 0,774 | 51,0% / 55,0% / 52,9% / 0,750 |
| CatBoost | 51,9% / 57,2% / 54,4% / 0,771 | 48,7% / 58,7% / 53,2% / 0,745 |
| TabM | 48,8% / 59,9% / 53,8% / 0,767 | 51,3% / 52,9% / 52,1% / 0,737 |
| Ensamble cuatro modelos | 49,6% / 61,0% / 54,7% / 0,774 | 50,6% / 56,8% / 53,5% / 0,749 |
| Ensamble cinco modelos | 49,5% / 60,5% / 54,5% / 0,773 | 50,4% / 55,6% / 52,9% / 0,748 |

El mejor anterior se eligió tras revisar estos períodos y empleó otro protocolo; es referencia histórica, no comparación confirmatoria equivalente. El baseline tiene su propio umbral, obtenido en los folds internos; sus cifras cambian respecto a anteriores por esa decisión. Ningún método supera 51% en precisión y recall simultáneamente en ambos folds. Ninguno demuestra ausencia de sobreajuste. El período 621–730 no se consultó en esta ejecución, aunque ya se había examinado en el proyecto.

## Tiempos y reproducibilidad

Cuatro procesos simultáneos, dos hilos por modelo de árboles, 65,2 segundos de reloj para los estudios clásicos y su ensamble. Tiempos por tarea, con solapamiento: logística 5,9/7,5 s; LightGBM 18,4/24,1 s; XGBoost 36,5/32,1 s; CatBoost 34,2/28,9 s (folds 1/2). TabM tardó 10,6 s en su proceso y se ejecutó solapado con otras tareas. Los tiempos de tareas no deben sumarse como tiempo de espera; no se hizo benchmark contra ejecución secuencial.

`comparison.csv` contiene precisión, recall, F1, AUC, AP y log loss de train y validación cuando corresponde. Los CSV por modelo conservan predicciones y pruebas; los JSON contienen parámetros, umbrales, iteraciones y tiempos. Reproducir: `python competitive_benchmark.py --trials 8 --workers 4`, `python competitive_tabm.py`, `python competitive_summary.py`.
