# Las 17 variables de LightGBM

Cada fila representa una visita. El modelo combina el historial del cliente y de la tienda hasta el día anterior con las ofertas y el perfil disponibles en esa visita.

Los días 1–120 sirvieron para ordenar los candidatos por contribución SHAP. Entre los primeros 25, se descartó cada candidato que ya tenía una correlación absoluta de Spearman de al menos 0,90 con una variable conservada. Se compararon conjuntos de 10, 15, 17 y 25 variables en las tres validaciones internas del último corte, ajustando cada conjunto por separado. Se eligió el conjunto más pequeño cuya *log loss* quedó a 0,001 o menos de la mejor. La selección se realizó con datos de desarrollo.

| Variable | Definición antes del resultado actual |
|---|---|
| `tasa_ultimas_10_visitas` | Media de compra/no compra en las últimas 10 visitas anteriores. |
| `precio_relativo_4` | Precio actual de marca 4 dividido entre el mínimo actual de las cinco marcas. |
| `intervalo_compra_medio_previo` | Media de intervalos entre compras reales anteriores. |
| `dias_desde_visita` | Días desde la visita anterior del cliente. |
| `month_recent_smoothed` | Proporción de visitas con compra en los últimos 30 días, suavizada con el historial de los 90 días anteriores a esa ventana. |
| `unidades_por_dia_desde_compra` | Cantidad de la última compra real dividida entre días desde ella más 1. |
| `precio_relativo_1` | Precio actual de marca 1 dividido entre el mínimo de las cinco marcas. |
| `month_visits_31_60` | Visitas del cliente entre 31 y 60 días antes de la visita actual. |
| `ingreso_anual` | Ingreso anual del cliente en la unidad registrada en la base. |
| `dias_desde_compra` | Días desde la última compra real anterior. |
| `tienda_tasa_60` | Tasa de compra de la tienda en los 60 días anteriores, calculada como (compras + 1) / (visitas + 4). |
| `precio_relativo_5` | Precio actual de marca 5 dividido entre el mínimo de las cinco marcas. |
| `recencia_compra_relativa` | Días desde compra divididos entre intervalo medio previo de compra. |
| `promo_marca_1` | Promoción de marca 1 conocida antes de la compra. |
| `precio_vs_anterior60_5` | Precio actual de marca 5 dividido por su mediana en los 60 días previos. Requiere al menos cinco precios diarios anteriores. |
| `intervalo_visita_medio_previo` | Media de intervalos entre visitas anteriores. |
| `month_rate_change` | Tasa de compra de los últimos 30 días menos la de los 30 anteriores. |

**Cómo se calcula la tasa mensual.** Al número de compras de los últimos 30 días se le suman cinco veces la tasa de los 90 días anteriores a esa ventana. El resultado se divide entre las visitas de los últimos 30 días más cinco. La tasa de referencia de 90 días se calcula como (compras + 1) / (visitas + 4). Así, cuando hay pocas visitas recientes, el historial anterior tiene mayor peso.

El identificador del cliente y el día sirven para reconstruir el historial. La marca y la cantidad de la visita se conocen después de comprar y pueden alimentar visitas posteriores. El diccionario de origen indica excluir el tamaño de la ciudad.

Cuando una variable carece de historial suficiente, el modelo aplica la imputación aprendida en el entrenamiento de ese corte. El campo `model_schema` de `outputs/lightgbm/execution.json` registra las columnas y los parámetros utilizados.
