# EDA inicial y diagnóstico temporal de desarrollo

## Contrato y límites

Se estima compra antes del resultado de una visita conocida. Perfil y ofertas actuales se consideran disponibles; los resultados de visitas anteriores se conocen antes de cada nueva visita. Las variables históricas se actualizan durante validación únicamente con resultados anteriores. Las asociaciones no identifican efectos causales.
Días 621–730 excluidos antes de validaciones y construcción de variables. No se consultaron artefactos de test.

## EDA inicial: días 1–296

- 25,754 visitas de 500 clientes; 6,124 visitas con compra (23.8%).
- Un cliente registra una mediana de 50 visitas. La mediana de su conversión individual es 19.0%; no debe confundirse con la conversión agregada de visitas.
- La conversión de los 43 bloques de siete días varía entre 11.2% y 39.8%. El último bloque tiene dos días: comparar sus volúmenes brutos exige considerar su menor duración.
- El mapa recencia/frecuencia cuenta visitas, no clientes únicos. Un cliente puede aparecer en distintas celdas; las celdas con poco soporte no justifican conclusiones firmes.
- Las tablas de promociones y edad incluyen visitas, compras y clientes únicos; las tasas se calculan por visita. Las mismas personas pueden aparecer en ambos grupos de exposición promocional.

## Calidad y semántica en desarrollo

- 50,556 visitas, 500 clientes y 24 columnas originales; 0 celdas faltantes y 0 filas duplicadas.
- Se construyen 106 variables; se comparan 5 grupos de variables. Cambios de campos de perfil dentro del cliente: 0.
- `ultima_marca_comprada` coincide con la marca de la visita anterior; `ultima_cantidad_comprada` coincide con la incidencia de compra de la visita anterior. La comprobación cuantitativa está en `semantica_historial.csv`.

## Diagnóstico temporal: no es EDA inicial

Las siguientes tasas describen los períodos exteriores de desarrollo. Ya examinados históricamente, sirven para investigar estabilidad; no son una confirmación independiente.

- Días 1–296: 25,754 visitas, 500 clientes y conversión 23.8%.
- Días 297–404: 8,409 visitas, 500 clientes y conversión 26.8%.
- Días 405–512: 8,364 visitas, 500 clientes y conversión 25.3%.
- Días 513–620: 8,029 visitas, 500 clientes y conversión 26.1%.

## Lecturas y siguientes hipótesis

Investigar tendencia de tienda estrictamente rezagada, cambios de frecuencia del cliente y diferencias en respuesta histórica a ofertas. Contrastarlas mediante ablaciones dentro de validación temporal; estas visuales no demuestran causalidad ni explican por sí solas los errores del modelo.

Visuales: `01_semanas_iniciales.png`, `02_clientes_iniciales.png`, `03_historial_inicial.png`, `04_asociaciones_iniciales.png` y `05_diagnostico_desarrollo.png`. Tablas CSV conservan denominadores y límites de períodos.
