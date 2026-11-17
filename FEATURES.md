# Las 17 variables del modelo depurado

El constructor genera candidatos y dependencias históricas, pero LightGBM recibe solo estas 17 columnas. El ranking SHAP y las correlaciones se aprenden con días 1–120, antes de las validaciones internas. Entre las primeras 25 variables del ranking se elimina la que tenga correlación Spearman absoluta ≥0,90 con otra conservada.

Se compararon 10, 15 y 17 variables con tuning propio en los tres folds internos del último corte, frente al modelo de 25. Se eligió el menor conjunto cuya log loss interna no empeorara más de 0,001 respecto al mejor. Esta es una elección retrospectiva de desarrollo; no se afirma independencia de los primeros folds ni significancia estadística por una prueba de hipótesis. No intervino el test.

| Variable | Definición antes del resultado actual |
|---|---|
| `tasa_ultimas_10_visitas` | Media de compra/no compra en las últimas 10 visitas anteriores. |
| `precio_relativo_4` | Precio actual de marca 4 dividido entre el mínimo actual de las cinco marcas. |
| `intervalo_compra_medio_previo` | Media de intervalos entre compras reales anteriores. |
| `dias_desde_visita` | Días desde la visita anterior del cliente. |
| `month_recent_smoothed` | Tasa de compras en últimos 30 días, suavizada hacia los 90 anteriores con fuerza 5. |
| `unidades_por_dia_desde_compra` | Cantidad de la última compra real dividida entre días desde ella más 1. |
| `precio_relativo_1` | Precio actual de marca 1 dividido entre el mínimo de las cinco marcas. |
| `month_visits_31_60` | Visitas entre 31 y 60 días anteriores: ventana relativa, no mes calendario. |
| `ingreso_anual` | Ingreso suministrado; no se inventa moneda. |
| `dias_desde_compra` | Días desde la última compra real anterior. |
| `tienda_tasa_60` | Compras/visitas de tienda en los 60 días anteriores, suavizado `(compras+1)/(visitas+4)`, excluyendo el día actual. |
| `precio_relativo_5` | Precio actual de marca 5 dividido entre el mínimo de las cinco marcas. |
| `recencia_compra_relativa` | Días desde compra divididos entre intervalo medio previo de compra. |
| `promo_marca_1` | Promoción de marca 1 conocida antes de la compra. |
| `precio_vs_anterior60_5` | Precio actual de marca 5 dividido entre su mediana en los 60 días previos; mínimo 5 precios diarios anteriores. |
| `intervalo_visita_medio_previo` | Media de intervalos entre visitas anteriores. |
| `month_rate_change` | Tasa de compra de los últimos 30 días menos la de los 30 anteriores. |

No se usan ID, fecha absoluta, tamaño de ciudad ni marca/cantidad del resultado actual como predictores. Las cantidades y marcas actuales solo pueden alimentar visitas posteriores. Los campos originales de última compra no se interpretan como última compra real sin reconstruir su historia.

Las ventanas excluyen el día actual y usan el historial observado. Sin suficiente historial se conservan faltantes y se imputan dentro del train de cada ajuste; no se inventan compras. Los ceros de precio ponderado sin preferencias, cuando actúan como candidatos del piloto, no son precios reales cero.

La correlación alta señala redundancia potencial, no inutilidad universal. La versión de 17 sacrifica algo de F1 respecto a 25 en el último período de desarrollo a cambio de simplicidad; su log loss interna quedó dentro de la tolerancia fijada. Los esquemas y parámetros efectivos se guardan en `outputs/lightgbm/execution.json`, campo `model_schema`.
