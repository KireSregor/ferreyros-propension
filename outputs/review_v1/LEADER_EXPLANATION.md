# Explicación descriptiva del líder aprendido

Primer procedimiento completo del ranking: `historical`. Procedimiento aprendido explicado con artefactos de los tres folds: `reference`.

La elección del líder es retrospectiva a la tabla de desarrollo; esta explicación no constituye validación independiente. Si gana una referencia no aprendida, se explica el mejor candidato aprendido, sin presentarlo como ganador general.

Se permutan conjuntamente historial, perfil, ofertas y variables nuevas cuando están presentes, con tres repeticiones sobre un máximo de 1.500 visitas por fold y el umbral original fijo. Cada caída positiva indica pérdida de la métrica respecto al mismo modelo sin permutación. Los cambios son descriptivos, no causales; variables correlacionadas y combinaciones artificiales pueden alterar su interpretación. No se usa este análisis para ajustar parámetros ni umbrales. No se consultó test.

- Corte 296, customer_history: caída media de precisión +26.9%, recall +23.6%, F1 +25.2%, AUC +0.2401.
- Corte 296, offers: caída media de precisión -0.9%, recall -1.0%, F1 -1.0%, AUC +0.0066.
- Corte 404, customer_history: caída media de precisión +25.7%, recall +30.3%, F1 +27.8%, AUC +0.2698.
- Corte 404, offers: caída media de precisión +1.7%, recall +3.2%, F1 +2.3%, AUC +0.0203.
- Corte 512, customer_history: caída media de precisión +22.8%, recall +24.5%, F1 +23.6%, AUC +0.2427.
- Corte 512, offers: caída media de precisión +1.1%, recall -1.1%, F1 +0.1%, AUC +0.0080.

Tiempo: 4.4 segundos; tareas incompletas por presupuesto: 0.
