---
name: data-science-framework
description: Diseñar y ejecutar proyectos de ciencia de datos mediante CRISP-DM, evaluación válida y criterios de valor. Usar para planes de modelado y selección de modelos; no exige infraestructura de producción.
---

# Metodología de ciencia de datos

Usar CRISP-DM como ciclo flexible: comprensión del negocio, comprensión de datos, preparación, modelado, evaluación y despliegue. Integrar consideraciones de valor de negocio sin presentar esta adaptación como metodología oficial de una consultora.

## Decisiones antes de entrenar

- Especificar objetivo, unidad, población, fecha de corte, horizonte y momento en que se conocerá la etiqueta. Definir qué datos existirán en inferencia y cómo se obtiene la población a puntuar.
- Elegir split según el uso: futuro de clientes conocidos requiere separación temporal; clientes nuevos requieren separación por entidad. Si se necesitan ambos, diseñar la combinación y declarar qué generalización evalúa cada prueba.
- En etiquetas con ventana futura, excluir observaciones sin seguimiento completo y evitar que ventanas de entrenamiento invadan el periodo de evaluación. No escoger porcentajes de split por costumbre si comprometen cobertura o positivos suficientes.
- Definir métrica primaria, referencia simple, restricciones y criterio de selección antes del tuning. Para clasificación desbalanceada considerar average precision, precisión o recall a capacidad fija y lift; evaluar calibración si las probabilidades alimentarán decisiones económicas. No confundir average precision con toda definición de área PR.

## Ejecución proporcional

- Preparar transformaciones reproducibles. Ajustar imputación, codificación, selección, escalado, clustering y muestreo solo dentro del entrenamiento de cada fold, preferentemente en pipelines.
- Crear históricos con operaciones que excluyan el evento actual. En evaluación temporal, reproducir la información disponible en cada predicción; si se actualiza el historial entre eventos, explicitar ese protocolo.
- Comparar una referencia trivial, un modelo interpretable y un candidato no lineal apropiado al tamaño de la base. Ampliar familias solo si los resultados justifican el costo. Usar clustering únicamente con una pregunta concreta.
- Acotar la búsqueda de hiperparámetros y registrar configuración, semilla, split y resultados. Escoger umbral y calibración en entrenamiento o validación; conservar el test para la estimación final.
- Revisar errores y estabilidad en segmentos con soporte suficiente. Explicar el modelo con métodos compatibles y describir sus límites: importancia de variables no demuestra causalidad y variables correlacionadas complican la atribución.
- Evaluar utilidad frente a la práctica actual, costo de operar y viabilidad del proceso. Si el modelo complejo no aporta mejora material, preferir la alternativa simple.

## Entrega y cierre

Entregar plan cuando se haya pedido plan; implementar solo el alcance autorizado. En implementación, producir ejecución reproducible, comparación de modelos, evaluación final y limitaciones. No reutilizar repetidamente el test para elegir variantes; si sucede, reconocer que dejó de ser una evaluación independiente.

## Fuentes

Consultadas el 15 de noviembre de 2026. Los criterios operativos específicos son adaptaciones propias.

- [IBM: CRISP-DM](https://www.ibm.com/docs/en/spss-modeler/saas?topic=dm-crisp-help-overview): fases iterativas y adaptables al problema.
- [scikit-learn: errores comunes](https://scikit-learn.org/stable/common_pitfalls.html): aislamiento de entrenamiento y evaluación, consistencia del preprocesamiento y pipelines.
- [BCG: valor y adopción de IA](https://www.bcg.com/publications/2022/5-rules-for-fixing-ai-and-machine-learning-for-your-business): conectar el trabajo técnico con cambios de negocio.
