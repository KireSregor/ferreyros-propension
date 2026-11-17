# Modelo acordado: LightGBM TPE

Artefacto: `model.joblib`. Métricas: `metrics.csv`. Configuración y columnas: `configuration.json`. Umbral fijo 0,283338, elegido en validaciones de los folds 1–2. No se ajustó con el periodo final.

El entrenamiento usa ventanas de 270 días y ponderación temporal con vida media 137,82 días. La validación 513–620 corresponde a entrenamiento 243–512. Para puntuar 621–730 se reajustó en 351–620; sus últimos 45 días eligieron internamente los árboles antes del reajuste completo. El candidato llegó al límite de 1.000 árboles.

El entrenamiento final contiene 20.621 visitas; la validación 8.029 y la evaluación final 8.137. La fila train 243–512 corresponde al modelo que produce la validación; la fila final_train 351–620 corresponde al que produce el test. No son el mismo ajuste.

**Limitación:** tanto validación como evaluación final fueron consultadas en investigaciones anteriores. No constituyen confirmación independiente. No se cumplió precisión/recall >51% y el rendimiento posterior se deteriora. La elección del usuario no elimina esta limitación.

Reproducir: `python final_balance_search.py --finalize`. El script conserva exactamente el preprocesamiento usado, incluidos historiales causales y precios relativos a los 60 días anteriores. El artefacto requiere esas mismas columnas y no es compatible con el helper original `score_visits`. El modelo original en `outputs/model.joblib` pertenece al flujo anterior y no es el modelo acordado de esta iteración.

La investigación descartada se consolidó en `outputs/research_archive.zip` y se retiró de las carpetas activas. Datos originales, skills y código principal se preservan.
