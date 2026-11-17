# Propensión de compra por visita

> **Rama de investigación:** conserva experimentos, comparaciones y diagnósticos. No es la entrega simplificada de LightGBM. El código, los parámetros y las tablas se versionan; entornos, cachés, modelos binarios y bases de búsqueda excluidas por `.gitignore` permanecen fuera de Git.
>
> Después de cerrar la revisión de desarrollo, el usuario autorizó evaluar seis artefactos entrenados hasta el día 512 en los días 621–730. Esa evaluación, sin reajuste ni cambio de umbral, está en `outputs/review_test_512/`; se reproduce con `evaluate_review_test.py` si están disponibles los artefactos locales. Las referencias siguientes a test reservado describen el alcance de la revisión anterior, no una prohibición vigente ni un test todavía intacto.

La revisión vigente usa **días 1–620** y tres folds temporales exteriores. Los días **621–730 están reservados**: no se consultan ni se evalúan en este flujo. Ya se habían examinado históricamente; no se presentan como un test intacto.

## Recorrido recomendado

1. `notebooks/01_eda.ipynb`: datos, negocio, calidad y análisis visual inicial.
2. `notebooks/02_modelado_resultados.ipynb`: protocolo, resultados por fold, baseline, incertidumbre y errores.
3. `outputs/review_v1/README.md`: síntesis de la revisión ejecutada.
4. `outputs/review_v1/PROTOCOL.md`: contrato de predicción, selección, skills y fuentes.

La conclusión breve queda en `outputs/review_v1/RESULTS.md`; las correcciones auditadas en `AUDIT.md` y el diccionario de predictores en `FEATURES.md`, dentro de esa misma carpeta.

Las métricas de entrenamiento y validación se muestran juntas: precisión, recall, F1, AUC, AP y log loss. El criterio de búsqueda es el promedio entre folds internos de **min(precisión, recall)**; F1 desempata. La aspiración 60/60 no es una garantía.

## Reproducir con el entorno existente

```powershell
.venv\Scripts\python.exe run_review.py audit
.venv\Scripts\python.exe -m ferreyros_propension.review_eda
.venv\Scripts\python.exe run_review.py classical
.venv\Scripts\python.exe run_review.py neural
.venv\Scripts\python.exe run_review.py wide
.venv\Scripts\python.exe run_review.py neural-resume
.venv\Scripts\python.exe run_review.py extras
.venv\Scripts\python.exe run_review.py calibrate
.venv\Scripts\python.exe run_review.py neural-seeds
.venv\Scripts\python.exe run_review.py blend
.venv\Scripts\python.exe run_review.py report
.venv\Scripts\python.exe run_review.py leader
.venv\Scripts\python.exe -m pytest tests/test_core.py tests/test_review.py -q
```

`review_config.yml` es la configuración central de esta revisión. Los estudios guardan estado SQLite, parámetros y predicciones. Se rechaza reutilizar resultados con otra huella de datos, configuración o código. La reanudación conserva pruebas completadas; no garantiza la misma trayectoria de propuestas futuras que una ejecución sin interrupciones.

`neural-resume` completa el objetivo inicial de pruebas si queda presupuesto global. Conserva la selección previa y continúa los mismos estudios; no cambia el criterio ni consulta el test. Ejecutar las etapas neuronales secuencialmente para compartir una sola GPU.

El reloj compartido queda en `outputs/review_v1/training_clock.json`. Para otra sesión de pruebas pendientes con un nuevo presupuesto autorizado, usar `--new-budget`; no borra estudios ni cambia sus parámetros. La entrada rechaza configuraciones que amplíen desarrollo más allá del día 620.

Cuatro tareas clásicas pueden ejecutarse a la vez; una tarea neuronal utiliza la GPU. Los tiempos por tarea se solapan y no equivalen a tiempo de reloj sumado. Las búsquedas tienen presupuesto y paradas explícitas; completar una búsqueda no prueba un óptimo global.

## Uso y scoring

Se predice antes del resultado de una visita conocida. Las ofertas del día y el perfil están disponibles; compras de visitas anteriores alimentan el historial. El modelo y su umbral permanecen fijos durante cada validación exterior.

Los nuevos artefactos guardan `feature_builder_version=review_v1`, columnas ordenadas, umbral y fin del entrenamiento. Usar `ferreyros_propension.review.score_review_visits(bundle, history, visits)` con una sola fecha futura e historial completo anterior de la tienda. Las señales globales requieren el historial de toda la tienda, no solo de los clientes puntuados. El resultado entrega cliente, día, probabilidad y decisión binaria.

El antiguo `score_visits` y los scripts de investigaciones anteriores corresponden a otros constructores/protocolos. Se conservan como historia del proyecto y no son la entrada vigente. No mezclar sus artefactos ni tablas con esta revisión. No se sobrescribe automáticamente un modelo anterior como ganador.

## Entregables y límites

El PDF original pide EDA, metodología, procesamiento, modelo/ajuste/explicabilidad y captura de valor; sugiere notebook funcional. El ZIP/repositorio, documento de código, documentación PDF y presentación PDF se prepararán cuando el usuario lo decida. Esta revisión no publica ni genera esos entregables finales.

Las asociaciones y las métricas predictivas no estiman el efecto incremental de una intervención comercial. Para medir valor se necesitan costes, márgenes y una prueba controlada. Los resultados de desarrollo, ya examinados repetidamente, siguen siendo exploratorios.
