# Revisión de propensión: resultados de desarrollo

## Comparación

Ranking: promedio simple por fold de min(precisión, recall); desempate F1 y menor log loss. Solo candidatos con todos los folds completados se consideran comparables para el ranking completo. El umbral se elige internamente; las tablas exteriores no participan en esa búsqueda. Un procedimiento puede escoger diferentes configuraciones en distintos cortes.

| model | folds | balance_mean | balance_min | precision_mean | recall_mean | f1_mean | auc_mean | folds_60_60 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| historical | 3 | 0.4933 | 0.4766 | 0.5003 | 0.5050 | 0.5025 | 0.7236 | 0 |
| reference | 3 | 0.4927 | 0.4626 | 0.5239 | 0.5378 | 0.5279 | 0.7563 | 0 |
| xgboost | 3 | 0.4863 | 0.4662 | 0.4999 | 0.5759 | 0.5315 | 0.7564 | 0 |
| catboost | 3 | 0.4831 | 0.4377 | 0.5329 | 0.5332 | 0.5277 | 0.7528 | 0 |
| chronological_weighted | 3 | 0.4808 | 0.4564 | 0.5197 | 0.5544 | 0.5309 | 0.7535 | 0 |
| store_adjusted_rank2 | 3 | 0.4773 | 0.4377 | 0.5271 | 0.5223 | 0.5196 | 0.7492 | 0 |
| seedmean_rank2 | 3 | 0.4762 | 0.4307 | 0.5300 | 0.5364 | 0.5264 | 0.7506 | 0 |
| ablation_history | 3 | 0.4722 | 0.4355 | 0.5245 | 0.5386 | 0.5245 | 0.7497 | 0 |
| calibrated_reference | 3 | 0.4701 | 0.4209 | 0.5316 | 0.5322 | 0.5237 | 0.7539 | 0 |
| neural_tree_blend | 3 | 0.4700 | 0.4138 | 0.5366 | 0.5188 | 0.5195 | 0.7510 | 0 |
| neural | 3 | 0.4668 | 0.4081 | 0.5365 | 0.4995 | 0.5098 | 0.7461 | 0 |
| chronological_equal | 3 | 0.4647 | 0.4246 | 0.4991 | 0.5780 | 0.5253 | 0.7551 | 0 |
| chronological_stack | 3 | 0.4642 | 0.4223 | 0.4966 | 0.5846 | 0.5265 | 0.7550 | 0 |
| neural_wide | 3 | 0.4642 | 0.4331 | 0.5221 | 0.4875 | 0.5013 | 0.7429 | 0 |
| lightgbm | 3 | 0.4630 | 0.4262 | 0.5204 | 0.5567 | 0.5276 | 0.7554 | 0 |
| ablation_history_offers | 3 | 0.4520 | 0.3961 | 0.5272 | 0.5420 | 0.5202 | 0.7525 | 0 |
| logistic | 3 | 0.4512 | 0.3961 | 0.5264 | 0.5417 | 0.5197 | 0.7491 | 0 |
| store_adjusted_rank1 | 3 | 0.4498 | 0.3961 | 0.5250 | 0.5366 | 0.5168 | 0.7522 | 0 |
| seedmean_rank1 | 3 | 0.4482 | 0.3961 | 0.5234 | 0.5460 | 0.5196 | 0.7526 | 0 |
| ablation_all | 3 | 0.4462 | 0.3903 | 0.5254 | 0.5427 | 0.5182 | 0.7519 | 0 |
| ablation_enhanced | 3 | 0.4202 | 0.3856 | 0.4945 | 0.5923 | 0.5105 | 0.7518 | 0 |
| constant | 3 | 0.2606 | 0.2529 | 0.2606 | 1.0000 | 0.4134 | 0.5000 | 0 |

Las métricas están en escala 0–1. `metrics.csv` contiene train y validación por fold, `gaps.csv` las diferencias train menos validación y `figures/` su comparación visual. Histórico y referencia fija se evalúan con los mismos períodos y sus propios umbrales internos. No se selecciona automáticamente un artefacto final.

`figures/neural_curves.png` muestra pérdidas del entrenamiento y de la ventana interna de parada, con épocas elegidas y ejecutadas. Estas curvas no utilizan etiquetas del fold exterior. `parameter_importances.json` describe la importancia de hiperparámetros para las puntuaciones internas de Optuna cuando hay al menos diez pruebas completadas; no es importancia de variables ni una interpretación causal.

`figures/selected_candidate_diagnostics.png` destaca el procedimiento completo con mayor equilibrio observado, sin declararlo ganador confirmado. Compara sus curvas precisión–recall con histórico y referencia, marca la decisión con el umbral interno ya fijado y muestra matrices de confusión en visitas. La zona verde indica precisión y recall de al menos 60%; recorrer la curva es diagnóstico, no selección de otro umbral.

## Incertidumbre y errores

`bootstrap_vs_historical.csv` usa 200 remuestras pareadas de clientes, semilla 42, para diferencias de precisión, recall, F1 y AUC frente al histórico. Los intervalos percentiles de 95% describen variación entre clientes en los períodos observados; no corrigen dependencia tienda/día ni selección repetida de modelos. No constituyen garantía de mejora futura.

`error_segments.csv` muestra TP/FP/FN/TN, clientes, visitas y métricas por recencia, frecuencia, edad, género, promoción y bloque de 30 días. Segmentos con menos de 100 visitas o 20 clientes se marcan de bajo soporte. Son asociaciones diagnósticas, no causas demostradas. Las segmentaciones se superponen y no deben sumarse entre dimensiones.

`calibration_by_model.csv` compara probabilidad media y tasa observada en diez intervalos fijos de probabilidad, con visitas y clientes como soporte. `threshold_transfer_diagnostic.csv` compara el equilibrio desplegado con el mejor equilibrio alcanzable al recorrer la curva de esas mismas predicciones exteriores. Este segundo valor es un diagnóstico oráculo posterior al resultado, optimista por construcción: no elige modelos ni umbrales, no se usa para entrenar y no demuestra un límite de los datos. No se exporta el umbral oráculo. Ambos diagnósticos utilizan etiquetas exteriores después de la evaluación; cualquier hipótesis motivada por ellos debe etiquetarse como exploratoria.

## Supervisión y límites

Skills aplicadas en la revisión: negocio y EDA para contrato/disponibilidad y diagnóstico; competitive-data-science y machine-learning para variables/modelos/ensambles; scikit-learn para preprocesamiento local; tuning para búsquedas reanudables; guía neuronal para curvas y parada; clean-code/deslop para legibilidad; prompt-engineering para criterios explícitos. Las skills orientan decisiones, no certifican resultados.

Los días 621–730 no se consultan por este informe. Fueron examinados históricamente; tampoco los folds de desarrollo son evidencia completamente independiente de la historia del proyecto. No se afirma ausencia de sobreajuste ni imposibilidad de alcanzar 60/60. Las medias exteriores resumen evaluaciones temporales, no validan el artefacto reajustado final.

`times.csv` recoge duración y pruebas por tarea; `runtime_summary.json` conserva los registros de reloj global disponibles (blend_runtime.json, calibration_runtime.json, classical_runtime.json, ensembles_initial_runtime.json, ensembles_runtime.json, neural_completion_runtime.json, neural_initial_runtime.json, neural_runtime.json, neural_wide_runtime.json). Tareas paralelas se solapan: sus tiempos no se suman como tiempo de reloj. Los contadores de pruebas no equivalen al número de entrenamientos; consultar configuraciones y trazas para árboles, épocas y reajustes.

`experiment_inventory.csv` y `inventory_summary.json` separan pruebas completas, ajustes inferidos de modelos clásicos y ajustes neuronales registrados, incluyendo épocas y memoria máxima. El trabajo compartido de ensambles se cuenta una vez; el piloto GPU se presenta aparte. No incluyen entrenamientos sintéticos y no pueden reconstruir ajustes parciales fallidos sin registro.
