# Auditoría y cambios de esta revisión

## Qué se corrigió

| Hallazgo | Cambio aplicado y comprobación |
|---|---|
| Validar el CSV completo antes de recortar desarrollo podía consultar el período reservado. | La lectura por bloques filtra hasta el día 620 antes de validar o resumir. Una prueba sintética coloca datos inválidos después del corte y comprueba que no intervienen. |
| Elegir un umbral sobre todas las visitas internas juntas no equivale a optimizar el promedio simple entre folds. | Se optimiza exactamente la media de min(precisión, recall) de los tres folds internos. Un contraejemplo verifica la diferencia frente al cálculo agrupado. |
| Los códigos de categorías podían interpretarse como cantidades ordenadas. | Preprocesamiento aprendido dentro de cada ajuste; categorías nativas en CatBoost y codificación explícita en los demás procedimientos. Se comprueban categorías desconocidas. |
| Elegir bases para stacking entre propuestas TPE posteriores podía incorporar indirectamente información del futuro período del combinador. | Sus configuraciones se escogen en el primer fold interno y solo entre las diez propuestas aleatorias iniciales. El segundo período aprende el combinador y el tercero fija su umbral. |
| Resultados almacenados podían confundirse entre versiones. | Huellas de datos, configuración y código; constructores versionados, esquema ordenado y comprobación de reproducción del scoring. |
| Una reanudación o copia de respaldo podía distorsionar tiempos y contar ajustes dos veces. | Se conservan las fases de ejecución y se deduplican copias idénticas de registros de entrenamiento; las búsquedas acumuladas se cuentan una sola vez. |

No se atribuyen retrospectivamente todas las diferencias históricas a estos hallazgos. Los resultados actuales se comparan bajo un procedimiento común.

## Qué significa la información

- Una fila es una visita; el objetivo es comprar en esa visita, antes de conocer su desenlace.
- Los campos suministrados como última marca y última cantidad describen la visita anterior según las comprobaciones realizadas. El segundo coincide con su incidencia de compra; su nombre no describe unidades de la última compra real.
- Marca y cantidad actuales son resultados y se excluyen del modelo. Las ofertas del día se consideran conocidas al puntuar.
- El historial se actualiza con resultados de visitas anteriores, también durante validación. Las estadísticas de tienda excluyen todo el día actual. Este contrato no equivale a pronosticar un período completo sin recibir nuevos resultados.
- Perfil y ofertas no garantizan señal predictiva adicional. Su utilidad se contrasta mediante ablaciones y permutaciones; una asociación no demuestra un efecto comercial causal.

## Evidencia disponible

`splits.csv` contiene días, visitas, clientes y tasas de cada partición; `manifest.json` identifica grupos de predictores y `feature_audit.csv` sus faltantes y cardinalidad. Los estudios y registros conservan parámetros, semillas, iteraciones seleccionadas, ajustes neuronales, épocas y tiempos. `PROTOCOL.md` detalla selección y decisiones guiadas por las skills.

Pasaron **23 pruebas** del código. Además, el scoring de un artefacto se contrasta con sus predicciones guardadas dentro del notebook de resultados. Estos controles verifican comportamiento del código; no certifican rendimiento futuro ni ausencia de sobreajuste.

Los días 621–730 no se consultaron en esta revisión. Fueron examinados históricamente; una confirmación independiente requeriría datos nuevos.
