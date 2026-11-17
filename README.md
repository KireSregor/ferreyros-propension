# Propensión de compra · LightGBM

Abrir [el notebook](notebooks/propension_lightgbm.ipynb) y ejecutar las celdas en orden. Reconstruye EDA, selección de 17 variables, ajuste con Optuna, evaluación y scoring desde el CSV. No carga modelos ni parámetros ganadores.

## Ejecutar

Python 3.12, desde la raíz del proyecto:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.lock
.venv\Scripts\python.exe run_notebook.py
```

También se puede ejecutar desde VS Code/Jupyter seleccionando ese entorno. No es necesario instalar el proyecto como paquete. La primera celda contiene los parámetros de ejecución; `src/training.py` define los cortes temporales y el espacio de búsqueda.

## Estructura

```text
Dataset/                         Datos originales y diccionario
notebooks/propension_lightgbm.ipynb  Recorrido y resultados paso a paso
src/
  data.py                        Lectura, controles y variables base
  feature_research.py            Historial y variables candidatas
  training.py                    LightGBM, selección, Optuna y scoring
  workflow.py                    Ejecución de folds y exportación
  eda.py                         Gráficas exploratorias desde los datos
  reporting.py                   Tablas, métricas y explicabilidad
  plotting.py                    Estilo visual compartido
run_notebook.py                   Ejecución con kernel limpio
FEATURES.md                      Diccionario de los 17 predictores
requirements.lock                Dependencias fijadas
```

Los módulos están directamente en `src`; `__init__.py` permite imports relativos. El notebook muestra decisiones, configuración y resultados; los módulos contienen la implementación y las gráficas.

## Protocolo

- Una fila representa una visita. Se estima compra/no compra antes de conocer el desenlace.
- EDA: días 1–296. Train/validación: 1–296/297–404; 1–404/405–512; 1–512/513–620.
- Cada trial evalúa tres folds internos anteriores. Parámetros por log loss media; umbral por F1 medio interno.
- Optuna sin timeout ni máximo de trials: mínimo 50 completados y paciencia 30 sin mejora relevante de 0,0001. Los folds internos se ejecutan en paralelo y los trials secuencialmente.
- Ranking y correlaciones aprendidos con días 1–120. El tamaño de 17 se seleccionó retrospectivamente en desarrollo: ver [criterios y variables](FEATURES.md).
- Modelo final entrenado hasta 512. El notebook muestra la evaluación final en test 621–730 sin reajuste. Ese período se revisó durante el desarrollo; se recomienda confirmar el resultado con nuevas observaciones.
- El historial se actualiza con visitas anteriores conocidas; las variables de tienda excluyen el día actual. No se predice todo un período sin recibir nuevas observaciones.

## Resultados y trazabilidad

El notebook presenta precisión, recall, F1, ROC AUC, AP y log loss frente al histórico, por fold y con promedio de validación. Train es diagnóstico; una brecha pequeña no prueba ausencia de sobreajuste.

Solo se escriben tres archivos en `outputs/lightgbm/`: `model.joblib`, `metrics.csv` y `execution.json`. El último guarda configuración, parámetros, trials, umbrales, tiempos y entorno de esta ejecución. Se regeneran al ejecutar el notebook. Los respaldos y la investigación no son dependencias de ejecución.

## Requisitos del PDF

| Requisito | Dónde se encuentra |
|---|---|
| EDA: patrones, anomalías y tendencias | Sección 2 del notebook |
| Metodología y limpieza | Secciones 2–4; controles e imputación dentro de train |
| Modelo, fine tuning, selección y explicabilidad | Secciones 4–6; comparación con histórico y SHAP |
| Impacto por áreas y captura de valor | Sección 9; piloto y KPIs propuestos |
| Operatividad y notebook paso a paso | Sección 8; scoring verificado y ejecución completa |

La comparación histórica entre algoritmos se conserva en la rama `investigacion`; este recorrido reproduce únicamente LightGBM. La propensión no demuestra impacto incremental ni permite inventar ROI.

Los entregables Word/TXT, documentación PDF y presentación PDF (máximo siete slides) se distribuyen por separado y se excluyen de Git mediante `entregables/`. La comparación histórica y las correlaciones están incluidas en el notebook; sus resultados históricos están identificados y enlazados a la revisión archivada.
