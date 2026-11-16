# Plan de trabajo: propensión de compra

**Estado:** reconstrucción implementada y verificada. Ambos notebooks ejecutados íntegramente sin errores; seis pruebas y controles Ruff aprobados. Lista para revisión conjunta. No incluye generar PDF, slides ni TXT finales.

## Tareas solicitadas por la prueba

| Tarea | Cobertura en este plan |
|---|---|
| **1. Análisis Exploratorio de Datos (EDA)** | Sección 6: patrones, anomalías y tendencias. |
| **2. Metodología** | Secciones 2–5 y 8: objetivo, flujo y validación. |
| **3. Procesamiento y limpieza** | Sección 7: reglas y preparación de variables. |
| **4. Modelo de propensión de compra** | Secciones 9–11: entrenamiento, ajuste, selección y explicación. |
| **5. Impacto en el negocio y captura de valor** | Sección 12: uso, indicadores y evaluación comercial. |

## 1. Objetivo y criterios de calidad

Producir un análisis defendible, código modular y evidencia de generalización temporal. La calidad se juzga por la validez de la evaluación, profundidad de los hallazgos, claridad de la documentación y reproducibilidad, además de las métricas.

Adoptaremos la disciplina de experimentación de una competencia de datos tabulares, conservando la pregunta comercial y las restricciones temporales del caso. No se promete una métrica determinada ni que un algoritmo específico gane.

## 2. Bases metodológicas

- **CRISP-DM:** organiza el trabajo de negocio, comprensión de datos, preparación, modelado, evaluación y uso de la solución. El EDA pertenece principalmente a comprensión de datos y puede revisitarse tras la limpieza.
- **Python modular y scikit-learn:** funciones reutilizables para el flujo, con `Pipeline` y `ColumnTransformer` para las transformaciones aprendidas que correspondan. Retiramos Kedro: para este alcance no necesitamos su catálogo ni registro de pipelines, y tampoco construiremos un orquestador propio.
- **Optuna:** gestionará búsquedas de hiperparámetros, resultados y reanudación de experimentos.
- **MLflow Tracking:** registrará parámetros, métricas y artefactos de los experimentos. Las tablas comparativas se obtendrán de esos registros; no mantendremos un sistema paralelo de tracking en CSV.
- **Skills del proyecto:** negocio para definir decisiones, análisis para producir evidencia, ciencia de datos para evaluación, código limpio para implementación y síntesis para comunicar conclusiones. La eficiencia evitará repetir trabajo; no recortará el alcance necesario.

El detalle de notebooks, ventanas temporales y presupuestos de búsqueda que sigue es una propuesta específica para esta prueba, no una receta oficial de McKinsey.

## 3. Flujo de trabajo

```text
Definición del negocio y contrato de predicción
                  ↓
Inventario de datos + protocolo de evaluación
                  ↓
EDA inicial: estructura, calidad, población y target
                  ↓
Limpieza documentada → EDA de datos limpios e hipótesis
                  ↓
Variables históricas + preprocesamiento dentro de cada fold
                  ↓
Benchmark de modelos → Optuna → comparación temporal
                  ↓
Calibración y política de decisión → evaluación y explicación
                  ↓
Captura de valor + scoring reproducible + documentación de revisión
```

La infraestructura se prepara al inicio para que el análisis sea reproducible; el entrenamiento comienza después de entender y auditar la base. Las revisiones posteriores del EDA se registrarán cuando cambien una decisión.

## 4. Arquitectura propuesta

```text
ferreyros-propension/
├── Dataset/                          # Archivos originales, sin modificar
├── config.yml                        # Cortes, variables y presupuestos
├── src/ferreyros_propension/
│   ├── data.py                       # Auditoría, limpieza e históricos
│   ├── modeling.py                   # Validación, búsqueda y scoring
│   ├── evaluation.py                 # Métricas, calibración y explicación
│   └── plotting.py                   # Gráficos reutilizables
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modelado_resultados.ipynb   # Comparación, explicación y negocio
├── outputs/                          # Resultados generados, no lectura obligatoria
│   ├── tracking/                     # MLflow y almacenamiento de artefactos
│   ├── optuna.db                     # Estado reanudable de las búsquedas
│   ├── figures/
│   └── tables/                       # Exportaciones para presentación
├── tests/test_core.py                # Invariantes temporales y scoring
├── pyproject.toml                    # Paquete y configuración mínima
├── requirements.lock
└── README.md
```

Las funciones de los cuatro módulos resolverán preparación, experimentación, evaluación y scoring. Los notebooks las llamarán directamente en orden; no crearemos un framework de ejecución propio. El paquete se instalará localmente para evitar cambios manuales de rutas de importación. `pyproject.toml` y el lockfile tendrán funciones distintas: definir el paquete y fijar el entorno reproducible.

Los dos notebooks invocarán funciones o leerán artefactos guardados, sin duplicar entrenamiento o transformaciones. El primero explicará datos, calidad, EDA e hipótesis; el segundo metodología de modelado, experimentos, evaluación, explicación y negocio. El README será la guía de ejecución. Los diccionarios extensos podrán guardarse como tablas enlazadas desde el notebook en lugar de documentos duplicados. La carpeta contiene únicamente el recorrido vigente de análisis y modelado.

**Resultado:** ejecución por etapa y de extremo a extremo desde los notebooks, con una sola implementación de cada regla y rutas resueltas desde la raíz del proyecto.

### Regla de simplificación

El objetivo son **dos notebooks y cuatro módulos principales de lógica**, además de configuración mínima y pruebas pertinentes. Es una orientación, no un límite que obligue a crear archivos gigantes. Extraer otro módulo solo cuando facilite entender, probar o reutilizar una responsabilidad real.

Reduciremos piezas y duplicación, conservando EDA interpretado, comparación de familias, Optuna, evaluación temporal, control de fuga, calibración, explicación y reproducibilidad. Cada figura debe responder una pregunta: combinar paneles relacionados y omitir gráficos que repiten la misma conclusión. Cada experimento debe resolver una incertidumbre: comparar familias primero y concentrar búsqueda adicional en candidatos prometedores, dejando registrado el criterio. Los análisis condicionales del plan siguen siendo condicionales.

Preferir herramientas mantenidas cuando reduzcan el costo total de implementar y mantener la solución. MLflow reemplazará el seguimiento propio; Optuna conservará únicamente el estado necesario para optimizar y reanudar. No guardar todas las tablas intermedias ni construir servicios para un scoring batch. Los archivos generados son artefactos consultables, no documentos que el evaluador deba leer uno a uno.

### Alcance de MLflow y reparto de responsabilidades

Usar almacenamiento del proyecto, sin infraestructura cloud, registro de producción ni plugin de integración con Kedro. La interfaz se abrirá solo cuando se quiera inspeccionar experimentos; los notebooks mostrarán tablas y gráficos guardados sin exigir navegar por MLflow.

Registrar un run por candidato o trial, con identificadores del estudio/trial cuando corresponda, configuración, huella de datos, versión de código disponible, cortes, métricas por fold y resúmenes, tiempo y estado. Incluir explícitamente métricas propias y protocolo temporal: el autologging no los deduce. Usar autologging solo si evita código sin duplicar runs, modelos ni resultados.

Guardar modelos y predicciones detalladas de finalistas, no un modelo por fold de cada trial. Mantener la integración como una función breve dentro del módulo de modelado; evitar una capa genérica de tracking. La base de Optuna es el estado del optimizador y MLflow es el registro de experimentos; vincularlos con IDs, sin copiar todo el historial a un tercer sistema. Exportar únicamente tablas necesarias para el notebook o revisión.

## 5. Negocio y contrato analítico

1. Relacionar las cinco tareas del PDF con los componentes de la solución.
2. Explicar qué representa una visita, una compra y una marca. Documentar que producto, moneda y etiquetas de códigos no están definidos.
3. Mantener como hipótesis inicial la predicción de compra durante una visita, antes de su resultado. Distinguir ese objetivo de comprar una marca específica o comprar dentro de un horizonte futuro.
4. Definir consumidor de la predicción, población elegible, instante de scoring, disponibilidad de variables y acción comercial posible.
5. Construir un mapa de valor: visitantes elegibles → capacidad de atención → selección → conversión → margen neto. Separar mejora del ranking de efecto incremental de una intervención.

**Salida:** documento de contexto y supuestos, contrato de predicción y preguntas cuya respuesta cambiaría el alcance. Los vacíos no críticos se tratarán con supuestos explícitos.

## 6. EDA profundo y visual

### 6.1 Inventario y auditoría

Identificar dimensiones, tipos, cardinalidad, fechas relativas, entidades y claves. Revisar faltantes y valores centinela, duplicados completos y por clave, rangos, coherencia entre compra/marca/cantidad, atributos que cambian por cliente y disponibilidad temporal.

Investigar específicamente las discrepancias de última marca y cantidad: comparar con compra anterior, visita anterior y resultado actual. Si los datos permiten identificar su semántica, documentarla; si no, conservar la incertidumbre. No declararlas erróneas solo por diferir de una reconstrucción.

**Visuales:** resumen de calidad, faltantes por variable si existen y ejemplos de anomalías. Si no hay faltantes, bastará una tabla; no crear un heatmap vacío por decoración.

### 6.2 Target y desbalance

Medir compras/no compras con conteos y porcentajes, conversión por periodo y distribución de conversión por cliente. Explicar qué significa acertar siempre la clase mayoritaria y por qué accuracy aislada es insuficiente.

**Visuales:** barras del target, evolución con volumen de observaciones y distribución de tasas por cliente. Evaluar estabilidad de prevalencia antes de proponer pesos o remuestreo. Un reparto 75/25 no obliga a balancear artificialmente.

### 6.3 Perfil y representatividad de clientes

Analizar edad e ingreso mediante histogramas, ECDF y percentiles; usar escala logarítmica para ingreso solo si mejora la lectura. Mostrar composición por códigos demográficos y concentración de visitas por cliente.

**Distinción clave:** contar personas únicas para describir edad de los clientes; contar visitas para describir la exposición comercial. Si un atributo cambia en el tiempo, definir el momento de referencia en lugar de elegir una fila arbitraria.

Evaluar cobertura de rangos de edad y su asociación con conversión. La edad no tiene que estar uniformemente distribuida: hablaremos de concentración o representatividad, no de un defecto de balance por defecto. Sin población externa de referencia no afirmaremos representatividad de mercado.

**Visuales:** distribuciones por cliente, comparación cliente/visita y tasas por rangos con soporte e incertidumbre. Conservar códigos desconocidos sin inventar etiquetas.

### 6.4 Comportamiento e historial

Estudiar frecuencia de visitas, tiempo entre visitas y compras, clientes con distinta recurrencia, volumen comprado y concentración. Diferenciar ausencia de compra observada de ausencia de historial previo al inicio de la base.

**Visuales:** distribuciones de intervalos, concentración acumulada, relación frecuencia/conversión y mapas de actividad por cliente o cohortes cuando permitan una lectura clara. Las cohortes se definirán por primera aparición observada, no como fecha de adquisición demostrada.

### 6.5 Marcas, precios y promociones

Analizar participación por visitas con compra y por unidades, preferencia/repetición de marca, transiciones entre compras, distribución y evolución de precios, diferencias de precio entre alternativas y frecuencia de promociones simultáneas.

**Visuales:** participación con denominador explícito, matriz de transición normalizada con conteos, series de precios/promociones y conversión por condiciones comerciales. Comparar patrones por periodo y segmentos para detectar asociaciones que cambian al controlar composición.

Estas gráficas no demostrarán causalidad ni elasticidad de demanda. No calcular una correlación entre el código numérico de marca y compra como si la marca fuera una magnitud.

### 6.6 Relación entre variables y target

Usar correlaciones entre variables numéricas con significado, análisis de redundancia y tasas por categorías con suficiente soporte. Explorar interacciones justificadas, por ejemplo recencia × frecuencia y diferencia de precios × promoción.

**Visuales:** correlación numérica seleccionada, tasas y volúmenes por segmento y superficies agrupadas de interacciones donde haya cobertura. Evitar un pairplot masivo que no responda preguntas.

### 6.7 Conclusiones del EDA

Cada bloque concluirá con **evidencia → interpretación → limitación → decisión o hipótesis**. Mantener un registro de hallazgos con población, periodo, soporte y figura correspondiente. Los hallazgos podrán justificar nuevas variables, exclusiones o una evaluación adicional; no se inventarán para llenar secciones.

El EDA que guíe decisiones comenzará en el tramo inicial de desarrollo. La auditoría estructural puede abarcar todo el archivo; el periodo final ya consultado quedará fuera de la exploración para selección. En ampliaciones temporales distinguiremos análisis exploratorio de diagnóstico de validación.

**Estándar gráfico:** estilo uniforme, etiquetas comprensibles, unidades, denominadores, tamaños de muestra, títulos respaldados por evidencia y anotaciones útiles. Guardar figuras legibles y tablas fuente; revisar cortes, escalas y sobreposición. Usar gráficos interactivos solo donde faciliten explorar.

## 7. Limpieza y construcción de variables

### Limpieza

Definir reglas desde la auditoría. Registrar qué cambió y cuántas filas afectó cada regla. Conservar datos originales. Tratar los extremos según significado; no eliminar ingresos altos o compradores frecuentes automáticamente.

### Variables candidatas

| Grupo | Ejemplos | Control necesario |
|---|---|---|
| Recencia | Días desde visita y compra anteriores | Solo eventos anteriores al scoring |
| Frecuencia | Visitas/compras históricas y ventanas de 30/60/90 días | Ventanas por calendario, no por cantidad de filas |
| Conversión | Tasa histórica suavizada y tasas recientes | Excluir target actual y estimar priors solo en train |
| Intensidad | Cantidades previas y medias históricas | No usar cantidad actual |
| Preferencia | Última marca, proporciones históricas por marca | Categorías y arranque sin historial explícitos |
| Precio relativo | Mínimo, rango, diferencia y razón entre marcas | Disponibilidad previa y denominadores válidos |
| Promoción | Número de marcas promocionadas e interacción con preferencia | No interpretar como efecto causal |
| Perfil/tiempo | Edad, ingreso, códigos y tendencia temporal | No inventar meses o días de semana sin fecha inicial |

Las ventanas y combinaciones son candidatas, no una obligación de incluirlas todas. Evaluaremos su aporte por grupos mediante ablación.

### Preprocesamiento

Separar transformaciones determinísticas de transformaciones aprendidas. Ajustar imputación, escalado, codificación, selección y cualquier remuestreo únicamente dentro del entrenamiento de cada fold. Regresión logística usará un tratamiento adecuado a sus supuestos; CatBoost podrá trabajar con categorías nativas. No imponer un mismo escalado a todos los algoritmos.

Excluir del predictor principal ID, tamaño de ciudad y resultados actuales. Tratar atributos demográficos como información cuyo aporte y estabilidad se examinan; comparar un modelo sin ellos.

**Salida:** diccionario de variables con fórmula, disponibilidad, manejo de faltantes y razones de inclusión; transformación histórica reutilizable en entrenamiento y scoring.

## 8. Validación cruzada temporal y protección frente a fuga

Las visitas son irregulares y hay clientes repetidos. No usaremos K-fold aleatorio por filas ni aplicaremos `TimeSeriesSplit` al orden actual por cliente. Construiremos folds por día calendario, manteniendo todas las visitas de un día en la misma partición.

Propuesta inicial de ventanas expansivas dentro de los días 1–584:

| Fold | Entrenamiento | Validación futura |
|---|---|---|
| 1 | Días 1–200 | 201–296 |
| 2 | Días 1–296 | 297–392 |
| 3 | Días 1–392 | 393–488 |
| 4 | Días 1–488 | 489–584 |

Confirmaremos soporte de clases y clientes antes de congelar estos cortes, sin elegir ventanas por cuál produce mejores resultados. El histórico puede actualizarse con resultados ya observados en visitas anteriores, siempre que ese sea el protocolo operacional; lo probaremos explícitamente.

El objetivo principal evalúa futuro de clientes conocidos. El desempeño en clientes nuevos se informará por separado si existe soporte; una prueba por clientes excluidos correspondería a otro escenario, no a reemplazar automáticamente el temporal.

**Limitación heredada:** los días 585–730 ya se usaron como test y sus métricas se discutieron. Se conservarán como benchmark temporal previamente consultado y no participarán en tuning. No basta renombrar o recortar ese periodo para obtener un test virgen. Una evaluación externa verdaderamente nueva necesita datos no utilizados.

Para limitar optimismo de la búsqueda, los finalistas tendrán evaluación temporal anidada: búsqueda en subventanas internas previas al corte y puntuación en la ventana externa posterior. Esa evaluación cubre el procedimiento de ajuste; no elimina el sesgo que pudiera introducir una exploración previa de la misma base. Se documentará esa diferencia.

Decisión de esta implementación: se optimiza el número de iteraciones explícitamente y se completan todos los folds, sin early stopping ni pruning. Así no se consultan etiquetas del fold externo para escoger iteraciones. Si se incorpora early stopping en una iteración futura, deberá usar una cola interna de entrenamiento. Si las etiquetas requieren un horizonte futuro o llegan con retraso, añadiremos separación temporal según ese retraso; no un gap arbitrario.

## 9. Benchmark y optimización de modelos

### Familias que compararemos

| Candidato | Función en el estudio |
|---|---|
| Probabilidad constante | Referencia mínima de clasificación y calibración |
| Tasa histórica suavizada | Referencia comercial fuerte basada en recurrencia |
| Regresión logística regularizada | Referencia interpretable y control de complejidad |
| Random Forest | Comparación de árboles agregados mediante bagging |
| HistGradientBoosting | Continuidad con el prototipo y comparación técnica |
| XGBoost | Candidato de boosting con regularización y muestreo |
| LightGBM | Candidato eficiente con control de hojas y soporte mínimo |
| CatBoost | Candidato con tratamiento nativo de variables categóricas |

Todos usarán la misma población y folds, con preprocesamiento apropiado a cada familia. No se seleccionará un modelo por prestigio o por una única métrica favorable en test.

### Optuna

- Primero ejecutar configuraciones base y medir costo por fold, para validar el flujo y presupuestar la búsqueda.
- Propuesta inicial: 30 trials por XGBoost, LightGBM y CatBoost; 15–20 para Random Forest y HistGradientBoosting; búsqueda pequeña de regularización para logística. Registrar los que completan, fallan o son podados.
- Muestreo TPE con semilla registrada y búsqueda en escala logarítmica cuando corresponda. Esta ejecución optimiza iteraciones explícitas y no poda trials: cada comparación usa los cuatro folds. Revalidar finalistas en todos los folds.
- Optimizar AP media entre folds y registrar dispersión, peor fold, métricas secundarias y tiempo. No usar métricas internas diferentes como si fueran AP idéntica; calcular una definición común de evaluación.
- Ajustar profundidad/hojas, learning rate, número de árboles, soporte mínimo, regularización y muestreo con espacios coherentes por familia. Los rangos concretos irán en configuración y se justificarán tras el benchmark de costo.
- Comparar sin pesos y con alternativas razonables para desbalance; no aplicar SMOTE por defecto a un panel temporal. Revisar el impacto de pesos sobre calibración.
- Guardar estudios en almacenamiento persistente y reanudable y evitar repetir trials ya completados. Extender la búsqueda solo si la trayectoria de mejora y la estabilidad justifican el tiempo adicional.

**Salida:** experimentos trazables en MLflow y búsquedas reanudables en Optuna; tabla comparativa generada desde el registro y predicciones fuera de entrenamiento de finalistas. La evaluación anidada de finalistas tendrá presupuesto interno acotado y registrado por separado.

## 10. Selección, calibración y evaluación

### Métricas

AP será la métrica primaria inicial para priorización con positivos minoritarios. Reportaremos ROC AUC, log loss y Brier, además de precisión, recall, lift y ganancias a capacidades de 5 %, 10 % y 20 %. Mostrar prevalencia por fold porque AP depende de ella.

Si se requiere una decisión binaria, elegir el umbral con datos de desarrollo según costo o capacidad, y presentar matriz de confusión junto al criterio de selección. No asumir 0,5 como umbral óptimo.

### Calibración

Comparar probabilidades originales y calibradas, empezando por sigmoide; considerar isotónica solo con soporte suficiente. Ajustar el calibrador con predicciones temporales fuera de entrenamiento o un tramo interno independiente, y evaluar en un periodo posterior no usado para calibrar. Conservar la versión original si calibrar empeora generalización.

### Robustez

- Comparar resultados por periodo, frecuencia, historial disponible y segmentos con soporte suficiente.
- Hacer ablaciones: sin historial, sin precios/promociones y sin demografía; medir aportes de grupos de variables.
- Estimar incertidumbre y diferencias emparejadas mediante bootstrap por cliente en un periodo fijo, conservando sus visitas. Distinguir esa incertidumbre de cambios entre periodos; no tratar cada visita como independiente.
- Evaluar capacidad por día cuando se simule atención diaria. Para campañas por cliente, deduplicar y definir cadencia; no llamar clientes al top de filas de visitas.
- Revisar sensibilidad a semillas de los finalistas y señales de sobreajuste.

**Visuales:** comparación por fold, curvas precision–recall con referencia de prevalencia, ROC, lift/ganancias, calibración, distribución de scores y errores por segmento. Matriz de confusión solo con umbral documentado.

Un blend simple de finalistas se considerará únicamente si sus errores son complementarios; pesos aprendidos en predicciones de desarrollo y evaluación temporal posterior. Stacking queda condicionado a una ganancia verificable que justifique mayor complejidad.

## 11. Explicabilidad e interpretación comercial

Calcular SHAP para el modelo compatible seleccionado con una muestra documentada y representativa del periodo evaluado. Incluir resumen global, relaciones para variables importantes y ejemplos de predicciones altas, bajas y errores, con criterio de selección explícito.

Complementar con importancia por permutación y ablaciones por grupos. Explicar la unidad del resultado de SHAP (por ejemplo, log-odds si corresponde); no presentar automáticamente contribuciones como puntos porcentuales. Considerar dependencia entre variables y usar datos de referencia de entrenamiento cuando el método lo requiera.

Cada interpretación debe distinguir dependencia del modelo, asociación observada y causalidad. Las explicaciones no demuestran que intervenir una variable cambie la compra.

## 12. Valor de negocio y operación mínima

Traducir capacidad y ranking en una propuesta de atención, con restricciones y ejemplos de uso. Preparar escenarios de margen, costo y efecto incremental como supuestos editables, nunca como ROI demostrado.

Diseñar un piloto con control y asignación por cliente para evitar contaminación entre visitas, comparando política propuesta y actual a presupuesto comparable. Definir métricas comerciales, adopción, cobertura y horizonte de seguimiento.

Implementar scoring batch que reciba visitas elegibles y su historial previo, construya las mismas variables y devuelva ID, instante, probabilidad y versión. No obligar al usuario a preparar manualmente las columnas derivadas.

Registrar versiones de datos/código/modelo, esquema y configuración. Definir fallbacks y monitoreo de calidad, distribución de scores y desempeño con etiquetas maduras. El despliegue cloud no está incluido en esta reconstrucción.

## 13. Calidad de código, pruebas y documentación

- Funciones legibles con responsabilidades claras, docstrings para contratos y supuestos, tipos en interfaces relevantes y parámetros fuera del código.
- Entorno reproducible y dependencias verificadas. Formato y lint de código; sin estado oculto entre notebooks.
- Pruebas de ausencia de eventos futuros en históricos, exclusión del resultado actual, límites de ventanas, consistencia de joins y separación temporal.
- Prueba de invariancia: agregar eventos futuros no debe cambiar variables de eventos pasados.
- Pruebas de categorías desconocidas, clientes sin historial y coherencia entrenamiento/scoring; persistir y cargar el modelo debe conservar predicciones.
- Ejecución integral desde cero y verificación visual de figuras. Los tests se centrarán en comportamiento y riesgos reales, no en repetir la implementación.
- Documentar negocio, diccionario, auditoría, hallazgos, protocolo de evaluación, experimentos, decisiones, limitaciones y modo de ejecución en Markdown y notebooks.

## 14. Secuencia de implementación y resultados revisables

| Orden | Trabajo | Criterio de cierre |
|---|---|---|
| 1 | Contrato, módulos Python y protocolo temporal | Entradas/salidas y evaluación definidos; sin entrenar aún |
| 2 | Auditoría y EDA elaborado | Gráficos interpretados, incidencias y preguntas prioritarias documentadas |
| 3 | Limpieza y variables | Transformaciones auditables y pruebas temporales correctas |
| 4 | Benchmark y Optuna | Comparación de familias con registros por fold y presupuesto consumido |
| 5 | Validación de finalistas, calibración y explicación | Selección justificada, límites y estabilidad visibles |
| 6 | Negocio, scoring y revisión integral | Proyecto reproducible y documentación lista para revisión conjunta |

No estimaremos una duración exacta antes de medir el costo de los entrenamientos. Los resultados conservarán trazabilidad para poder iterar sin ejecutar nuevamente toda la búsqueda.

## 15. Qué recibirá el usuario antes de los entregables finales

Proyecto Python ejecutable, dos notebooks con salidas y narrativa, cuatro módulos principales de lógica, reportes de EDA, experimentos registrados en MLflow, búsquedas Optuna, evaluación temporal, modelo y scoring, pruebas relevantes e instrucciones de reproducción. El README resumirá qué se comprobó, qué se decidió y qué queda abierto, enlazando el detalle de los notebooks.

Los cinco requerimientos de la prueba quedan cubiertos: EDA en sección 6; metodología en 2–5 y 8; limpieza en 7; desarrollo, ajuste y explicación en 9–11; impacto y captura de valor en 12. Los PDF, slides y TXT se prepararán únicamente después de la revisión del usuario.

## 16. Referencias técnicas verificadas

Consultadas el 15 de noviembre de 2026. Las fuentes fundamentan los métodos; la organización específica y los presupuestos son decisiones propuestas para este proyecto.

- [IBM: CRISP-DM](https://www.ibm.com/docs/en/spss-modeler/saas?topic=dm-crisp-help-overview).
- [MLflow: registro de experimentos](https://mlflow.org/docs/latest/ml/tracking/).
- [Optuna: optimización eficiente y pruning](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html).
- [scikit-learn: TimeSeriesSplit y supuesto de espaciado](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
- [XGBoost: parámetros](https://xgboost.readthedocs.io/en/stable/parameter.html).
- [LightGBM: ajuste de parámetros](https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html).
- [CatBoost: ajuste de parámetros](https://catboost.ai/docs/en/concepts/parameter-tuning).
- [scikit-learn: calibración](https://scikit-learn.org/stable/modules/calibration.html).
- [SHAP: TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html).

## 17. Registro de ejecución y decisiones

- Seis familias, **130 trials principales**: 10 logística, 15 Random Forest, 15 HistGradientBoosting y 30 por XGBoost, LightGBM y CatBoost. Cada trial usa cuatro ventanas temporales.
- Diagnóstico anidado reducido: dos familias finalistas, tres ventanas externas, cinco trials internos por ventana (**30 trials adicionales**). La elección previa de familias limita su independencia, como se explica en el notebook.
- CatBoost seleccionado por AP media de desarrollo. Se conservan modelos base y OOF de los dos finalistas; el bundle principal incluye columnas, versión y política de calibración.
- Calibración sigmoide comparada y descartada porque no mejora log loss de desarrollo. No se fuerza calibración por completar una lista de técnicas.
- Ablaciones de historial, demografía y condiciones comerciales; dos semillas adicionales; incertidumbre mediante 300 remuestreos de clientes emparejados.
- Ranking global y cupos diarios se evalúan por separado. No se infiere retorno incremental ni se atribuye causalidad a SHAP o a promociones.
- Sin early stopping ni pruning: número de iteraciones optimizado explícitamente. No se construye un ensemble sin evidencia adicional de mejora que justifique la complejidad.
- Los campos suministrados de historial tienen semántica distinta a sus nombres: marca de visita anterior e incidencia de compra anterior, ambas coincidencias del 100 %. Se documentan, sin eliminar observaciones.
- Código refactorizado, formateado y verificado con Ruff; pruebas de comportamiento en `tests/test_core.py`. Notebooks principales: `01_eda.ipynb` y `02_modelado_resultados.ipynb`.
- Figuras y tablas con una adaptación editorial inspirada en publicaciones de McKinsey, centralizada en `plotting.py`. Las referencias, incluido el repositorio público Vizro, están en el README. La renovación visual conserva el modelo y las métricas.

Los resultados detallados se consultan en las tablas generadas y los notebooks, evitando mantener cifras manualmente en documentos duplicados. La revisión con el usuario y los entregables finales continúan pendientes.

