# Protocolo de la revisión aprobada

El objetivo es identificar compra antes del resultado de una visita conocida. La evaluación supone disponibilidad del perfil, ofertas actuales y resultados de visitas anteriores. El historial se actualiza causalmente entre visitas; el estimador y el umbral no se actualizan dentro de la validación exterior. No se confunde este escenario con pronosticar simultáneamente todo el período futuro.

## Datos y evaluación

Desarrollo: días 1–620; 50.556 visitas de 500 clientes. El lector descarta las filas posteriores al 620 antes de validar target, claves y ofertas. Los días 621–730 y sus artefactos guardados no intervienen en esta revisión. Se reconoce que fueron examinados históricamente.

Los cortes exteriores e interiores están fijados en `review_config.yml`; `splits.csv` conserva límites y denominadores exactos. En cada ajuste, imputación, escalado y codificación se aprenden solamente con su entrenamiento. Logística recibe one-hot y escalado; CatBoost recibe categorías nativas; boosting restante recibe categorías one-hot. El constructor `review_v1` y el esquema ordenado se incluyen en los artefactos; el scoring rechaza versiones incompatibles.

El objetivo es el promedio simple entre folds internos de min(precisión, recall), con un umbral compartido. Los desempates son F1 y menor log loss. No se usa el resultado exterior para elegir parámetros o umbral del ajuste correspondiente. Entrenar con log loss, vigilar esa pérdida para parar y seleccionar mediante precisión/recall son decisiones distintas.

Se guardan modelos de cada fold, no solo sus métricas. El modelo de referencia fija reproduce los parámetros del LightGBM anterior, pero recalcula su umbral dentro del nuevo protocolo: sus cifras no tienen por qué coincidir con las históricas. Las comparaciones antiguas no se mezclan como si fueran réplicas actuales.

## Ingeniería y búsquedas

Las nuevas hipótesis son tasas de compra de tienda de días anteriores, cambios de ritmo individual, rachas, asociación histórica a exposición promocional y precios actuales relativos a la mediana anterior. Las señales de tienda excluyen todo el día actual. Las asociaciones promocionales no identifican efectos causales.

Grupos separados: historial; historial y ofertas; todos los originales; todos más las nuevas señales. El conjunto de referencia histórica se conserva con un nombre explícito. Las ablaciones mantienen familia y parámetros para distinguir efecto de variables de efecto del tuning.

Estudios Optuna persistentes, TPE con diez pruebas aleatorias iniciales, semilla 42 y huellas de desarrollo/configuración/código. Objetivos iniciales por fold: 20 pruebas logísticas, 40 por familia de boosting y 20 neuronales, con límites temporales y parada por estancamiento. Las cantidades efectivas se publican. No se poda por el resultado de un fold temprano: la heterogeneidad temporal y el umbral compartido harían poco fiable esa poda. Se usa early stopping interno por iteración y parada entre pruebas.

Los árboles seleccionan iteraciones en los últimos 40 días de su entrenamiento y se reajustan con todo ese entrenamiento. Las redes registran pérdidas, gradientes, épocas ejecutadas y seleccionadas, parámetros y memoria. El diagnóstico de memorización de 128 ejemplos comprobó capacidad de optimización; no demuestra generalización. No se introdujo Lightning como dependencia: se aplicaron sus guías de supervisión al flujo PyTorch existente.

## Semillas y ensambles

Las dos familias clásicas con mejor puntuación interna de cada corte se repiten con semillas 42, 137 y 915. Para ensambles cronológicos, los parámetros base se eligen solo entre las diez propuestas aleatorias iniciales y usando el primer fold interno; se excluyen propuestas TPE posteriores porque ya incorporan resultados de folds posteriores. Los combinadores se aprenden en el segundo fold interno y el umbral en el tercero. El combinador logístico usa C=0,01 fijado de antemano. Los parámetros base quedan congelados antes de ambos períodos. Se compara promedio igual, pesos convexos y stacking sin usar etiquetas exteriores.

## Supervisión y fuentes

| Skill | Aplicación concreta |
|---|---|
| product-business-analysis | Unidad visita, momento de decisión y diferencia entre propensión e impacto incremental. |
| exploratory-data-analysis | EDA inicial separado de diagnósticos posteriores, denominadores y soporte. |
| competitive-data-science | Ablaciones, diversidad de familias, semillas y ensambles con predicciones fuera de muestra. |
| machine-learning | Separar pérdida, selección y decisión; diagnosticar memorización y brechas temporales. |
| scikit-learn | Transformaciones dentro del ajuste y manejo de categorías desconocidas. |
| ml-hyperparameter-tuning | Espacios logarítmicos, estudios persistentes, límites y trazas de pruebas. |
| pytorch-lightning | Registro de curvas, checkpoint de mejor época y parada supervisada. |
| clean-code / deslop | Funciones reutilizables, configuración central y limpieza limitada a cambios de esta revisión. |
| prompt-engineering-patterns | Encargo con restricciones, criterios de selección y evidencia de cumplimiento explícitos. |

Referencias primarias: [validación anidada en scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html), [categorías nativas de CatBoost](https://catboost.ai/docs/en/features/categorical-features), [implementación y tuning oficiales de TabM](https://github.com/yandex-research/tabm). Estas fuentes sustentan el procedimiento; no garantizan métricas en esta base.

Los intervalos por bootstrap de cliente son descriptivos y no corrigen toda la dependencia tienda/día ni la selección repetida en el proyecto. No se afirma ausencia de sobreajuste por observar una brecha pequeña. Cualquier ajuste motivado por errores exteriores será identificado como exploratorio.

## Extensiones medidas y alcance

La comprobación de capacidad compara TabM con k=32 y anchuras 256/512 frente a MLP de anchura 512, con selección interna y cobertura efectiva registrada. La búsqueda neuronal inicial se reanuda para completar veinte pruebas por corte cuando queda presupuesto, conservando sus selecciones anteriores. Las pérdidas por época publicadas son descriptivas sin ponderación; el optimizador sí usa pesos temporales normalizados.

Se probó CatBoost en GPU con una configuración muestreada idéntica a la primera del estudio original. El trial interno tardó 151,2 s en GPU frente a 61,9 s en la ejecución anterior; la prueba GPU completa, incluidos reajustes exteriores, tardó 183,8 s. Los backends pueden elegir cantidades distintas de árboles y se ejecutaron en momentos diferentes: no es una medición aislada de hardware. No se amplió esa rama porque no mostró mayor throughput en esta comprobación. Evidencia: `gpu_pilot_comparison.json`; el código opcional permanece en `review_cat_gpu.py` y la prueba reproducible se conserva en `.tmp/cat_gpu_pilot`. No se presenta como candidato evaluado en los tres folds.

El blend neuronal/árboles elige bases, pesos 25/50/75% y umbral solo con las predicciones internas. Sus puntuaciones internas son optimistas por compartir selección; a diferencia del stacking cronológico, no se afirma validación meta temporal independiente. Su desempeño se mide en los folds exteriores.
