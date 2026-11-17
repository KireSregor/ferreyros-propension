# Prueba con cortes 480 / 580 / 730

Entrenamiento días 1–480: 39.987 visitas. Validación 481–580: 7.673. Evaluación 581–730: 11.033. Se interpretó «81–580» como «481–580» para evitar solapamiento.

Hiperparámetros LightGBM acordados, sin búsqueda nueva. Se incluyen todos los días de entrenamiento, manteniendo ponderación temporal con vida media 137,82 días. Parada interna con los últimos 45 días del entrenamiento. Umbral 0,287218 elegido para equilibrar precisión y recall en validación, congelado para el periodo posterior. Para evaluar se reajustó en 1–580.

Validación: precisión 52,80%, recall 52,82%, F1 0,5281, AUC 0,7378. Evaluación posterior: precisión 45,91%, recall 51,85%, F1 0,4870, AUC 0,7205.

Esta validación se usó para elegir el umbral: su 52,8% en ambas métricas es optimista como estimación final. Los periodos posteriores ya habían sido inspeccionados en otras pruebas; no es un test independiente. No demuestra que estos cortes sean mejores ni ausencia de sobreajuste.

El artefacto de esta prueba se guarda separado del modelo acordado. `metrics.csv` contiene además la tasa real de compra y el entrenamiento del reajuste final. Reproducción: `python final_balance_search.py --split480`.
