# Propensión de compra por visita

LightGBM asigna un puntaje a cada visita según la probabilidad de que el cliente compre en la categoría observada. Comercial puede ordenar por puntaje las visitas del día y atenderlas hasta cubrir su capacidad.

En los días 621–730 se observaron 2 050 compras en 8 137 visitas. Al ordenar todo ese período, las 813 visitas de mayor puntaje reunieron 494 compras. Esto equivale al **24,1% de las compras del período** y a una **conversión del 60,8% en ese grupo**. Un piloto con la misma capacidad en cada grupo permitiría medir si aplicar la priorización diaria mejora el margen.

## Ejecutar el notebook

Desde la raíz del proyecto, con Python 3.12 en Windows

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.lock
.venv\Scripts\python.exe run_notebook.py
```

El comando ejecuta el [notebook](notebooks/propension_lightgbm.ipynb) desde el CSV original con un kernel nuevo y guarda las salidas en las celdas. También puede abrirse en Jupyter o VS Code, seleccionar el entorno creado y ejecutar las celdas en orden. La primera celda de código permite ajustar la configuración de la búsqueda.

## Recorrido

| Pregunta | Dónde encontrar la respuesta |
|---|---|
| ¿Qué se conoce al llegar a la tienda y qué señales aporta el historial? | Secciones 2–3 y [diccionario de predictores](FEATURES.md) |
| ¿Por qué LightGBM y cómo se eligieron sus parámetros? | Secciones 4–5. Tres validaciones temporales y comparación previa de algoritmos |
| ¿Qué variables pesan más y qué tan bien ordena las visitas nuevas? | Secciones 6–7. Contribuciones SHAP, comparación con dos reglas simples, deciles y decisión binaria |
| ¿Cómo se puntúa una visita y cómo se mediría el beneficio? | Secciones 8–9. Ejemplo diario y diseño del piloto |

## Entrenamiento y evaluación

El análisis exploratorio usa los días 1–296. El historial de cada visita incorpora las compras ya registradas. Los indicadores de tienda utilizan días anteriores. La selección de las 17 variables se documenta en [FEATURES.md](FEATURES.md).

| Corte | Entrenamiento | Validación |
|---|---|---|
| 1 | Días 1–296 | Días 297–404 |
| 2 | Días 1–404 | Días 405–512 |
| 3 | Días 1–512 | Días 513–620 |

En cada corte, Optuna minimiza la *log loss* media de tres validaciones internas. La búsqueda acaba después de al menos 50 pruebas completadas y 30 sin una mejora de 0,0001. El F1 interno fija el umbral para clasificar compra o ausencia de compra. El ordenamiento usa el puntaje continuo y se evalúa con ROC AUC, KS y Gini.

El modelo del tercer corte se aplica también al test de los días 621–730 con los parámetros y el umbral ya fijados. Este período se consultó durante el desarrollo. La comparación comercial mediante el piloto requiere observaciones posteriores.

## Datos, código y resultados

| Ubicación | Contenido |
|---|---|
| `Dataset/` | CSV original y diccionario de los campos de entrada |
| `src/data.py`, `src/feature_research.py` | Controles e historial disponible antes de cada visita |
| `src/training.py`, `src/workflow.py` | Cortes temporales, búsqueda, entrenamiento y predicción |
| `src/eda.py`, `src/reporting.py`, `src/plotting.py` | Análisis exploratorio, gráficas y tablas |
| `outputs/lightgbm/` | Se crea al ejecutar. Contiene el modelo (`model.joblib`), las métricas (`metrics.csv`), los deciles (`deciles.csv`) y la configuración, los parámetros y los tiempos (`execution.json`) |
