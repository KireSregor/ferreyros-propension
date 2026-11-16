# Propensión de compra por visita

Solución para priorizar atención entre visitantes. No estima el efecto causal de una oferta. La categoría/producto, moneda y etiquetas de códigos no están documentados en la base.

## Qué revisar

1. [EDA](notebooks/01_eda.ipynb): negocio, calidad, perfil, recurrencia, marcas, precios, promociones e hipótesis.
2. [Modelado y resultados](notebooks/02_modelado_resultados.ipynb): comparación temporal, optimización, calibración, explicación, limitaciones y valor.
3. [Código](src/ferreyros_propension): datos, modelado, evaluación y gráficos, en cuatro módulos.

Los notebooks conservan sus salidas para revisión. Los PDF, slides y TXT finales quedan pendientes de revisión con el usuario.

## Instalar

Python 3.12. Crear y activar un entorno virtual y ejecutar desde la raíz:

```powershell
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
```

Seleccionar ese entorno como kernel. En una copia limpia: ejecutar EDA, ejecutar el comando de experimentos de la siguiente sección y después abrir el notebook de modelado. Con resultados ya generados, basta ejecutar los notebooks en orden. Ambos detectan la raíz si se abren desde `notebooks/`. `Dataset/compras_data.csv` nunca se sobrescribe.

## Reproducir experimentos

```powershell
python -m ferreyros_propension.modeling
```

Los presupuestos y cortes están en `config.yml`. Optuna conserva búsquedas en SQLite; MLflow registra experimentos. Una ejecución con la misma versión reutiliza estudios completos; cambios de código/configuración crean estudios nuevos para no mezclar resultados. La trayectoria exacta de Optuna puede variar entre una búsqueda continua y reanudada.

El segundo notebook carga los resultados por defecto. La celda `RERUN_EXPERIMENTS` permite regenerarlos explícitamente. La explicación y algunos diagnósticos pueden recalcularse al ejecutar todas las celdas.

Para inspeccionar MLflow:

```powershell
mlflow ui --backend-store-uri sqlite:///outputs/tracking/mlflow.db --host 127.0.0.1
```

No hace falta abrir esa interfaz para leer los notebooks. `outputs/tables/` contiene exportaciones para revisión; `outputs/figures/`, figuras; `outputs/optuna.db`, estado del optimizador; `outputs/tracking/`, registro MLflow. No publicar automáticamente datos ni bases de experimentos con identificadores de clientes.

## Metodología y límites

- EDA interpretativo y auditoría descriptiva: días 1–296. La lectura valida la estructura del archivo completo. El EDA ampliado anticipa la propuesta 85/15; el modelado guardado conserva sus cortes originales.
- Cuatro ventanas expansivas: entrenamiento hasta 200/296/392/488 y validación hasta 296/392/488/584.
- Benchmark 585–730: **previamente consultado** en el prototipo, no test intacto.
- Familias: logística, Random Forest, HistGradientBoosting, XGBoost, LightGBM y CatBoost. Referencias constante e histórica.
- Selección por AP media; también dispersión, AUC, Brier, log loss y capacidad comercial.
- Históricos estrictamente anteriores y preprocesamiento aprendido dentro del fold. El escenario es futuro de clientes conocidos.
- Número de iteraciones optimizado explícitamente: **sin early stopping ni pruning** en esta ejecución, para completar todos los folds con un protocolo uniforme.
- Calibración seleccionada cronológicamente en desarrollo, ablaciones sin retuning y diagnóstico anidado reducido de finalistas. Las familias ya observaron desarrollo: el diagnóstico no elimina todo sesgo de selección.
- SHAP explica el estimador base antes de calibración; no demuestra causalidad.

## Scoring

```python
import joblib
from ferreyros_propension.data import score_visits

bundle = joblib.load("outputs/model.joblib")
predictions = score_visits(bundle, history, visits)
```

`history` contiene resultados observados en días estrictamente anteriores. `visits` contiene un solo día, identificadores y condiciones disponibles antes de comprar; no necesita resultado, marca o cantidad actuales. Los históricos se calculan internamente. La salida incluye cliente, día, probabilidad y versión. Cargar solo artefactos de origen confiable.

Para el siguiente día, incorporar al historial los resultados realmente observados. No hay API, cloud ni monitoreo desplegado.

## Verificar

```powershell
python -m pytest -q
ruff check src tests notebooks
ruff format --check src tests notebooks
```

Las pruebas cubren invariancia ante eventos futuros, exclusión del resultado actual, ventanas de calendario, scoring, nombres inválidos y ranking constante. El segundo notebook verifica la equivalencia entre modelo persistido y predicciones guardadas.

Las cifras definitivas se consultan en las tablas y notebooks ejecutados, sin mantener manualmente resultados duplicados. El alcance y las decisiones están en [el plan](PLAN.md).

## Diseño de las visualizaciones

Las figuras y tablas utilizan una adaptación editorial propia: fondo blanco, azul profundo, acento azul, titulares con el hallazgo, cuadrícula reducida y notas de población y fuente. Las tipografías disponibles DejaVu Serif y Sans aportan contraste sin incorporar fuentes de marca. Las paletas son decisiones del proyecto, no especificaciones oficiales de McKinsey. En SHAP, el color y los signos conservan el significado de las contribuciones.

Referencias consultadas: [identidad visual de McKinsey](https://www.mckinsey.com/about-us/new-at-mckinsey-blog/meet-the-new-mckinsey-look/), [caso de diseño de Wolff Olins](https://wolffolins.com/work/mckinsey) y [Vizro, repositorio público de McKinsey](https://github.com/mckinsey/vizro), con [temas y colores documentados](https://vizro.readthedocs.io/en/stable/pages/user-guides/themes/).

Vizro ofrece temas reutilizables para Plotly y dashboards. Aquí se conservan Matplotlib y Seaborn para mantener el flujo existente; el estilo se centraliza en `plotting.py` y se reutiliza también en las explicaciones y tablas de los notebooks. No se utilizan logotipos ni se presenta el resultado como material oficial de la firma.
