# Último intento: auditoría por grupos de variables

**Corrección:** el grupo denominado «historial» en esta prueba también incluye variables derivadas de precios y promociones (por ejemplo, precio mínimo y precio según preferencia). Por ello, la comparación original no aísla el efecto de esas variables. La prueba corregida está en `outputs/optuna_review`.

Ejecución: `python final_audit.py`. Tiempo medido: 44,5 segundos.
12 variantes de LightGBM, 4 ajustes temporales cada una: 48 entrenamientos.
Baseline histórico evaluado en los mismos cuatro períodos.

Se compararon historial, historial con ofertas, historial con perfil e historial con ambos; cada grupo con 3, 7 y 15 hojas. Parámetros fijos: 400 árboles, learning rate 0,025, mínimo 300 observaciones por hoja, L2=100 y L1=5. Ponderación temporal con vida media de 180 días. Esta es una comparación controlada, no una búsqueda exhaustiva ni Optuna.

Cada candidato fija su umbral exclusivamente en días 297–404, equilibrando precisión y recall. Ese período es calibración y sus métricas son optimistas. Los siguientes períodos mantienen ese umbral. Se reajusta el modelo usando todos los días anteriores a cada evaluación.

## Resultado ilustrativo: historial, 7 hojas

| Entrenamiento | Evaluación | Precisión train | Recall train | Precisión evaluación | Recall evaluación | Baseline precisión | Baseline recall |
|---|---|---:|---:|---:|---:|---:|---:|
| 1–296 | 297–404 (calibración) | 52,7% | 55,3% | 52,2% | 52,2% | 51,3% | 51,5% |
| 1–404 | 405–512 | 52,9% | 55,4% | 46,3% | 65,7% | 50,9% | 52,7% |
| 1–512 | 513–620 | 52,2% | 56,3% | 51,1% | 56,5% | 48,1% | 48,1% |
| 1–620 | 621–730 | 51,9% | 56,5% | 45,9% | 51,4% | 44,0% | 46,0% |

En 513–620 el modelo supera al baseline en precisión y recall. La mejora no garantiza superar 51% en ambos en otros períodos. En 405–512 aparece un intercambio importante entre precisión y recall, y en 621–730 cae el rendimiento de ambos métodos. La caída conjunta es compatible con cambios temporales, pero no identifica una causa por sí sola.

Agregar perfil u ofertas no mejora consistentemente esta comparación. Esto no prueba que sean inútiles ni que exista un techo irrebasable. Ningún resultado constituye prueba de imposibilidad ni garantía de ausencia de sobreajuste. Los períodos fueron inspeccionados en investigaciones anteriores: toda esta evaluación es exploratoria.

`metrics.csv` conserva métricas de entrenamiento y evaluación de todos los candidatos. `predictions.csv` conserva cliente, día, etiqueta y probabilidad. El artefacto anterior se conserva; esta auditoría no lo sustituye automáticamente.
