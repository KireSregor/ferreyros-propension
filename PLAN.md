# Plan vigente: revisión integral de propensión

Este documento sustituye la planificación histórica. La implementación vigente se configura en `review_config.yml` y se ejecuta mediante `run_review.py`.

**Estado: revisión implementada y ejecutada.** Se completaron los tres folds exteriores, 370 pruebas Optuna principales, ocho variantes adicionales de capacidad y los diagnósticos documentados. Los dos notebooks están ejecutados. El objetivo 60/60 no se alcanzó y no se declara un nuevo ganador frente al histórico bajo el criterio congelado. Conclusión: `outputs/review_v1/RESULTS.md`.

## Decisiones confirmadas

- Desarrollo 1–620; test 621–730 fuera de la revisión, incluyendo sus artefactos guardados.
- Tres folds exteriores: train 1–296 / val 297–404; train 1–404 / val 405–512; train 1–512 / val 513–620.
- Tres validaciones internas anteriores a cada fold exterior; ver límites y registros en `outputs/review_v1/splits.csv`.
- Selección por promedio del menor valor entre precisión y recall; desempates F1 y log loss. Meta aspiracional: ambas >60%.
- Hasta dos horas de entrenamiento de reloj, con estudios reanudables y paradas entre pruebas.
- Perfil y ofertas de la visita disponibles antes del desenlace; historial actualizado causalmente con visitas previas.

## Trabajo y evidencia

1. Auditoría del PDF, diccionario, datos, código y procedimientos; aplicación documentada de skills y revisión independiente.
2. Filtrado de desarrollo antes de validar; constructor causal versionado y scoring compatible.
3. EDA inicial 1–296 separado de diagnósticos posteriores; soporte, unidades y asociaciones explícitos.
4. Baselines constante/histórico y referencia LightGBM fija; logística, LightGBM, XGBoost y CatBoost con preprocesamiento adecuado y tuning interno.
5. TabM/MLP con diagnóstico de aprendizaje, curvas, parada temprana y reajuste; comprobación adicional de capacidad.
6. Semillas, ensambles cronológicos y ajuste causal de odds con tasa anterior de tienda, elegidos internamente.
7. Ablaciones, explicaciones, calibración descriptiva, errores y bootstrap por cliente. Cualquier límite o cobertura incompleta debe publicarse.
8. Tablas train/val por fold y promedios, tiempos y pruebas reales; notebooks ejecutables sin test y recomendación basada en evidencia.

No se promete una métrica, ausencia de sobreajuste ni imposibilidad de mejora. Los períodos de desarrollo fueron revisados históricamente. No se publica GitHub, no se generan PDF/slides/TXT finales y no se realizan borrados amplios.

Estado detallado y resultados efectivos: `outputs/review_v1/README.md`. Contrato, fuentes y decisiones de las skills: `outputs/review_v1/PROTOCOL.md`.
