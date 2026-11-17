# Diagnóstico de variables y ablaciones controladas

Solo desarrollo hasta el día 620; no se consultan etiquetas ni artefactos de test. En cada fold se elige la familia clásica por su puntuación interna, nunca por su validación exterior.

## Importancia por permutación

La muestra exterior contiene como máximo 2.000 visitas por fold. Se conserva el umbral seleccionado internamente. Las caídas positivas indican empeoramiento al permutar; se reportan precisión, recall, F1 y AUC. Los grupos se permutan conjuntamente, con tres repeticiones; cada variable individual tiene una repetición exploratoria. Este diagnóstico usa etiquetas exteriores y no se utiliza para seleccionar modelos ni parámetros.

La permutación puede crear combinaciones poco plausibles y la correlación entre variables puede ocultar importancia. No mide causalidad ni importancia universal. Los resultados individuales son orientativos y no sustituyen una ablación temporal.

- Fold con corte 296, logistic: customer_history, caída media de AUC +0.2471.
- Fold con corte 296, logistic: offers, caída media de AUC +0.0070.
- Fold con corte 404, logistic: customer_history, caída media de AUC +0.2861.
- Fold con corte 404, logistic: offers, caída media de AUC +0.0218.
- Fold con corte 404, logistic: profile, caída media de AUC +0.0002.
- Fold con corte 512, lightgbm: customer_history, caída media de AUC +0.2061.
- Fold con corte 512, lightgbm: offers, caída media de AUC +0.0054.
- Fold con corte 512, lightgbm: profile, caída media de AUC +0.0000.
- Fold con corte 512, lightgbm: store_trend_and_new, caída media de AUC +0.0023.

## Ablaciones

Se fija la familia y los hiperparámetros ganadores internos de cada fold; únicamente cambia el conjunto de variables. El umbral se recalibra con los mismos folds internos y la cantidad de árboles se vuelve a seleccionar mediante la misma parada interna. Por ello se comparan procedimientos completos y no un efecto aislado del umbral. El presupuesto se comprueba entre tareas; una tarea iniciada puede terminar después del límite.

## Cota de la curva precisión–recall: oráculo diagnóstico

`oracle_curve_diagnostic.csv` recorre todos los umbrales sobre las predicciones exteriores del candidato elegido solo mediante puntuación interna. Compara su umbral desplegado con el máximo de min(precisión, recall) alcanzable en esa curva específica. El oráculo usa las respuestas exteriores y es optimista: NO es una mejora lograda por el modelo, NO se usa para escoger modelos ni cambiar umbrales y NO es una cota de todos los modelos posibles. Una brecha orienta sobre transferencia del umbral; una curva limitada sugiere investigar el ranking. Ninguno de estos resultados identifica por sí solo una causa.

- Corte 296, logistic: equilibrio desplegado 39.6%; oráculo diagnóstico 52.1%; brecha 12.5%. No es una métrica de despliegue.
- Corte 404, logistic: equilibrio desplegado 46.9%; oráculo diagnóstico 53.7%; brecha 6.8%. No es una métrica de despliegue.
- Corte 512, lightgbm: equilibrio desplegado 48.1%; oráculo diagnóstico 52.0%; brecha 3.9%. No es una métrica de despliegue.
## Diagnóstico de calibración

`calibration_diagnostic.csv` agrupa la misma muestra exterior por deciles de probabilidad predicha y contrasta probabilidad media con compras observadas, junto a su soporte. Los empates de score se reparten por orden para mantener grupos de tamaño comparable. No se ajusta ningún calibrador con las etiquetas exteriores. Las visitas de un cliente no son independientes: no deben aplicarse intervalos binomiales ingenuos ni atribuir causalidad a las diferencias observadas.

- Corte 296, ablation_history: precisión 59.2%, recall 43.6%, F1 50.2%, AUC 0.739.
- Corte 404, ablation_history: precisión 49.4%, recall 59.4%, F1 53.9%, AUC 0.761.
- Corte 512, ablation_history: precisión 48.8%, recall 58.6%, F1 53.2%, AUC 0.750.
- Corte 296, ablation_history_offers: precisión 62.2%, recall 39.6%, F1 48.4%, AUC 0.740.
- Corte 404, ablation_history_offers: precisión 47.3%, recall 65.0%, F1 54.7%, AUC 0.766.
- Corte 512, ablation_history_offers: precisión 48.7%, recall 58.0%, F1 53.0%, AUC 0.751.
- Corte 296, ablation_all: precisión 62.8%, recall 39.0%, F1 48.1%, AUC 0.740.
- Corte 404, ablation_all: precisión 46.9%, recall 64.3%, F1 54.2%, AUC 0.765.
- Corte 512, ablation_all: precisión 47.9%, recall 59.5%, F1 53.1%, AUC 0.750.
- Corte 296, ablation_enhanced: precisión 61.7%, recall 39.4%, F1 48.1%, AUC 0.737.
- Corte 404, ablation_enhanced: precisión 38.6%, recall 78.3%, F1 51.7%, AUC 0.766.
- Corte 512, ablation_enhanced: precisión 48.1%, recall 59.9%, F1 53.4%, AUC 0.752.

Los estados JSON indican coberturas incompletas por tiempo. Estos períodos ya se habían revisado en el proyecto: las conclusiones siguen siendo exploratorias. Ninguna caída o mejora identifica por sí sola la causa de un error.
