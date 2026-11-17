# Diccionario de los 106 predictores de la revisión

Inventario: grupo `enhanced` de `manifest.json`. Fórmulas comprobadas contra `data.py` y `feature_research.py`; versión del constructor: `review_v1`. Los modelos pueden seleccionar un subconjunto: las columnas efectivas están en cada `result.json` y artefacto versionado. Este documento describe variables, no garantiza su utilidad ni les atribuye efectos causales.

**Unidad:** una visita de un cliente en el día relativo `t`. Salvo perfil y ofertas actuales, los conteos y resultados son estrictamente anteriores a esa visita. `V` es el número de visitas previas; `C`, el número de compras previas; `r=(C+1)/(V+4)`. Una compra es una visita con incidencia 1, no una unidad física. En ventanas de `w` días se incluye `t−w` y se excluye `t`: intervalos relativos, no meses de calendario.

## 1. Perfil y ofertas actuales: 16 variables

Se supone que se conocen antes del resultado de la visita. En desarrollo, el perfil es constante dentro del cliente y precios/promociones son comunes a todos los clientes del mismo día. Sus asociaciones mezclan diferencias entre personas, ofertas y tiempo.

| Variables | Definición y disponibilidad | Hipótesis |
|---|---|---|
| `genero` | Código de género suministrado; categoría, sin inventar etiquetas. | Diferencias de propensión entre perfiles. |
| `estado_civil` | Código de estado civil suministrado; categoría. | Diferencias de contexto de consumo. |
| `edad` | Edad observada, numérica; no se actualiza artificialmente con los días. | Diferencias de patrones de compra según edad. |
| `nivel_educacion` | Código suministrado; tratado como categoría. | Información complementaria del perfil. |
| `ingreso_anual` | Ingreso anual suministrado; no se inventa moneda ni unidad adicional. | Diferencias de capacidad o hábitos de compra. |
| `ocupacion` | Código de ocupación suministrado; categoría. | Diferencias de comportamiento entre grupos. |
| `precio_marca_1`, `precio_marca_2`, `precio_marca_3`, `precio_marca_4`, `precio_marca_5` | Precio actual de cada marca 1–5; no precio de la marca que se terminará comprando. | Condiciones comerciales disponibles al decidir. |
| `promo_marca_1`, `promo_marca_2`, `promo_marca_3`, `promo_marca_4`, `promo_marca_5` | Indicador 0/1 de promoción actual por marca. | Asociación entre exposición comercial y compra. |

## 2. Historial acumulado y última compra real: 8 variables

Disponibles a partir de visitas observadas anteriores. La última compra real se reconstruye filtrando visitas con compra y desplazando una visita; no se confía en el nombre de los campos originales.

| Variable | Fórmula o definición | Hipótesis |
|---|---|---|
| `visitas_previas` | `V`: número de visitas anteriores del cliente. | Exposición y profundidad del historial. |
| `compras_previas` | `C`: suma de incidencias anteriores. | Experiencia compradora acumulada. |
| `tasa_previa` | `(C+1)/(V+4)`; prior fijo de media 25% y fuerza 4 visitas. En primera visita vale 0,25. | Propensión individual suavizada; baseline histórico. |
| `dias_desde_visita` | Día actual menos día de la visita anterior. | Ritmo de retorno. |
| `marca_previa` | Marca de la última visita anterior que sí tuvo compra. Se codifica como categoría; sin compra anterior se usa código −1. | Persistencia de preferencias. |
| `cantidad_previa` | Unidades de la última compra anterior real. | Tamaño de abastecimiento previo. |
| `dias_desde_compra` | Día actual menos día de la última compra anterior. | Necesidad de reposición o pérdida de actividad. |
| `cantidad_media_previa` | Unidades compradas antes de la visita divididas entre `C`; falta si `C=0`. | Tamaño habitual de compra. |

## 3. Ventanas recientes de días: 13 variables

`Vw` y `Cw` cuentan visitas y compras anteriores en `[t−w,t)`. En todos los casos la tasa suavizada es `(Cw+1)/(Vw+4)`, no una media calculada con el resultado actual.

| Ventana | Conteo de visitas | Conteo de compras | Tasa |
|---|---|---|---|
| 7 días | `visitas_7_dias` | No se expone por separado | `tasa_7_dias` |
| 14 días | `visitas_14_dias` | No se expone por separado | `tasa_14_dias` |
| 30 días | `visitas_30` | `compras_30` | `tasa_30` |
| 60 días | `visitas_60` | `compras_60` | `tasa_60` |
| 90 días | `visitas_90` | `compras_90` | `tasa_90` |

Hipótesis: distinguir cambios recientes de la propensión acumulada y separar una tasa extrema con pocas visitas de un comportamiento sostenido. Estas ventanas se solapan; no deben sumarse entre sí.

## 4. Preferencias y contexto de ofertas: 13 variables

| Variables | Fórmula o definición | Hipótesis |
|---|---|---|
| `preferencia_1`, `preferencia_2`, `preferencia_3`, `preferencia_4`, `preferencia_5` | Compras anteriores de cada marca divididas entre `C`; sin compras previas son faltantes. | Fidelidad y adecuación de las ofertas a cada cliente. |
| `precio_min` | Mínimo de los cinco precios actuales. | Nivel de precio accesible. |
| `rango_precio` | Máximo precio actual menos mínimo. | Dispersión de alternativas. |
| `n_promociones` | Suma de los cinco indicadores actuales; número de marcas promocionadas, no de clientes expuestos. | Intensidad promocional del día. |
| `precio_relativo_1`, `precio_relativo_2`, `precio_relativo_3`, `precio_relativo_4`, `precio_relativo_5` | Precio actual de cada marca dividido entre `precio_min`. | Posicionamiento entre marcas dentro del día. |

## 5. Secuencia de visitas e intervalos: 19 variables

| Variables | Fórmula o definición | Hipótesis |
|---|---|---|
| `antiguedad_observada` | Día actual menos día de la primera visita observada del cliente. No equivale necesariamente a antigüedad real en la tienda. | Madurez del historial disponible. |
| `compra_visita_lag_1`, `compra_visita_lag_2`, `compra_visita_lag_3` | Incidencia en la primera, segunda y tercera visita anterior. Falta cuando no existe esa visita. | Dependencia de corto plazo y persistencia. |
| `tasa_ultimas_3_visitas`, `tasa_ultimas_5_visitas`, `tasa_ultimas_10_visitas`, `tasa_ultimas_20_visitas` | Media de incidencia en hasta las últimas 3/5/10/20 visitas anteriores, sin suavizado y con al menos una observación. | Cambios medidos por oportunidades de visita, no por días. |
| `tasa_ewm_5`, `tasa_ewm_20` | Media exponencial de incidencias anteriores con `span=5` o `20`, `adjust=False`; peso de actualización `2/(span+1)`. | Adaptación gradual al comportamiento reciente. |
| `intervalo_visita_medio_previo` | Media de los intervalos entre visitas ya completados antes de la visita actual; no incluye el intervalo actual. | Ritmo habitual de retorno. |
| `intervalo_visita_relativo` | `dias_desde_visita / intervalo_visita_medio_previo`. | Retorno más rápido o lento que el habitual. |
| `intervalo_compra_medio_previo` | Media de intervalos entre compras reales anteriores ya completados; no utiliza si la visita actual comprará. | Ciclo de reposición personal. |
| `recencia_compra_relativa` | `dias_desde_compra / intervalo_compra_medio_previo`. | Grado de retraso frente al ciclo habitual. |
| `unidades_por_dia_desde_compra` | `cantidad_previa / (dias_desde_compra+1)`. Es un indicador construido, no consumo diario observado. | Relación entre tamaño y recencia del abastecimiento. |
| `visitas_desde_compra` | Diferencia entre el índice de visita actual y el índice de la última compra anterior. La visita inmediatamente posterior a comprar vale 1. | Número de oportunidades transcurridas desde comprar. |
| `precio_segun_preferencia` | Suma de `preferencia_m × precio_marca_m`. Las preferencias faltantes se sustituyen por 0 solo para esta suma. | Precio actual relevante para la mezcla histórica del cliente. |
| `promo_segun_preferencia` | Suma de `preferencia_m × promo_marca_m`; mismas convenciones de faltantes. | Promoción ajustada a la mezcla histórica. |
| `precio_promedio` | Media simple de los cinco precios actuales. | Nivel comercial general del día. |

En primera visita, intervalos, lags y recencias pueden ser faltantes. Los indicadores ponderados por preferencias valen 0 cuando no hay compras anteriores: ese 0 no significa precio cero ni ausencia demostrada de interés.

## 6. Bloques disjuntos de 30 días y tendencia del cliente: 19 variables

Los nombres `month_*` son heredados del código: **no representan meses de calendario**. Para cada visita, los bloques son edades relativas de sus visitas anteriores. Las tasas de bloque no están suavizadas y faltan cuando no hay visitas.

| Antigüedad de la visita anterior | Visitas (`Vb`) | Compras (`Cb`) | Tasa (`Rb=Cb/Vb`) |
|---|---|---|---|
| 1–30 días | `month_visits_1_30` | `month_purchases_1_30` | `month_rate_1_30` |
| 31–60 días | `month_visits_31_60` | `month_purchases_31_60` | `month_rate_31_60` |
| 61–90 días | `month_visits_61_90` | `month_purchases_61_90` | `month_rate_61_90` |
| 91–120 días | `month_visits_91_120` | `month_purchases_91_120` | `month_rate_91_120` |

Numerando los bloques anteriores de 0 (más reciente) a 3:

| Variable | Fórmula | Hipótesis |
|---|---|---|
| `month_rate_change` | `R0−R1`. | Aceleración o caída reciente de propensión. |
| `month_rate_acceleration` | `R0−2×R1+R2`. | Cambio de la tendencia, sensible a poco soporte. |
| `month_visit_change` | `V0−V1`. | Cambio de actividad del cliente. |
| `month_purchase_change` | `C0−C1`. | Cambio de compras observadas. |
| `month_change_support` | `min(V0,V1)`. | Soporte mínimo de la comparación de los dos últimos bloques. |
| `month_recent_smoothed` | Sea `q=(C1+C2+C3+1)/(V1+V2+V3+4)`; valor `(C0+5×q)/(V0+5)`. | Tasa reciente suavizada hacia los 90 días anteriores a ese bloque. |
| `month_recent_vs_prior90` | `month_recent_smoothed−q`. | Cambio reciente respecto al comportamiento anterior. |

## 7. Nuevas variables de tienda, ofertas y rachas: 18 variables

Las señales de tienda requieren el historial anterior de **toda la tienda**, no solo el de los clientes a puntuar. Ninguna tasa de tienda incluye resultados del día actual.

| Variables | Fórmula o definición | Hipótesis |
|---|---|---|
| `tienda_tasa_7`, `tienda_tasa_30`, `tienda_tasa_60` | Para los 7/30/60 días calendario relativos previos, sumar compras `Ct` y visitas `Vt` de tienda: `(Ct+1)/(Vt+4)`. Se desplaza un día antes del rolling. El primer día queda faltante por ausencia de historia de tienda. | Contexto general que las tasas personales no capturan. |
| `tienda_cambio_tasa` | `tienda_tasa_7−tienda_tasa_60`. | Cambio reciente de conversión de tienda. |
| `precio_vs_anterior60_1`, `precio_vs_anterior60_2`, `precio_vs_anterior60_3`, `precio_vs_anterior60_4`, `precio_vs_anterior60_5` | Precio actual por marca dividido entre su mediana en los 60 días anteriores; exige al menos 5 precios diarios previos observados. | Precio alto o bajo frente a la referencia temporal de esa marca. |
| `tasa_exposicion_promo_0`, `tasa_exposicion_promo_1` | Para visitas previas del cliente sin/con al menos una marca en promoción: `(compras_expuestas+4×r)/(visitas_expuestas+4)`. | Asociación histórica personal con el contexto promocional, suavizada hacia su propensión. |
| `soporte_exposicion_promo_0`, `soporte_exposicion_promo_1` | Número de visitas anteriores del cliente sin/con alguna promoción. | Fiabilidad de las tasas condicionadas. |
| `asociacion_promo_previa` | `tasa_exposicion_promo_1−tasa_exposicion_promo_0`. No es uplift ni efecto causal. | Diferencia descriptiva de respuesta entre contextos. |
| `ritmo_visitas_reciente` | `visitas_30 / ((visitas_90−visitas_30)/2+1)`. Compara últimos 30 días con promedio de visitas por bloque de 30 días en los 60 anteriores, con 1 en el denominador. | Activación o pérdida de ritmo de retorno. |
| `compra_previa_visita` | Incidencia de visita anterior, imputando 0 cuando no existe. Coincide con el primer lag salvo por ese faltante. | Estado reciente explícito para el constructor de rachas. |
| `racha_compras` | Longitud de la secuencia consecutiva de unos en `compra_previa_visita`, por cliente, hasta la visita anterior. | Persistencia de visitas con compra. |
| `racha_no_compras` | Longitud de la secuencia consecutiva de ceros en ese indicador. La primera visita vale 1 por la imputación del historial inexistente a 0. | Persistencia de no compra; primera observación sintética requiere cautela. |

## Disponibilidad, exclusiones y cautelas

- Excluidos como predictores: `id` (solo clave), día absoluto, resultado actual, marca y cantidad actuales, los dos campos originales de “última compra”, `tamanio_ciudad`, fecha reconstruida de última compra y ciclos artificiales de 7/30 días. No se inventaron nombres de marca ni días de semana.
- Marca y cantidad actuales son válidas para reconstruir **historia anterior** después de que se observa su resultado; no para predecir su propia visita. La definición de `marca_previa` y `cantidad_previa` usa la última compra real, mientras que los campos originales describen otra cosa.
- Los conteos anteriores empiezan en 0. Ausencia de compra previa deja recencias/cantidades/intervalos faltantes, salvo las convenciones explícitas anteriores. El constructor convierte infinitos a faltantes. Imputación, escalado y codificación aprendidos se ajustan dentro del entrenamiento de cada fold.
- Los grupos se comparan por ablaciones. `history` contiene historial individual; `history_offers` agrega ofertas; `all` agrega perfil; `enhanced` agrega las señales nuevas. `reference` reproduce el conjunto del LightGBM anterior y sí contiene algunos derivados de ofertas: no debe llamarse “solo historial”. Los nombres exactos por grupo están en `manifest.json`.
- Muchas variables son redundantes o correlacionadas. Una importancia pequeña por permutación no prueba irrelevancia; una asociación grande no prueba causalidad. Las tendencias de tienda pueden reflejar mezcla de clientes, tiempo u ofertas. Las rachas con historia faltante y los cocientes de soporte escaso deben interpretarse junto a sus conteos.
