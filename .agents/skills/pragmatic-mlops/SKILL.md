---
name: pragmatic-mlops
description: Diseñar reproducibilidad, scoring, versionado y operación de modelos con complejidad proporcional. Usar cuando se pida MLOps o preparar un modelo para uso recurrente; no desplegar infraestructura por defecto en un EDA.
---

# MLOps proporcional al uso

Adaptación de las guías de Google Cloud. Elegir capacidades por necesidad operativa, no por una lista de herramientas. Una prueba técnica local puede necesitar reproducibilidad y scoring batch sin servicios cloud.

## Elegir alcance

- Para entrega local: dependencias declaradas, configuración, semillas cuando apliquen, comando o notebook reproducible, pipeline de preparación y modelo persistido, métricas y ejemplo de inferencia.
- Para scoring recurrente: agregar validación de entradas, versionado, registros de ejecución, ejecución idempotente y monitoreo acorde con la frecuencia de uso.
- Para operación automatizada: considerar CI/CD, registro de modelos, orquestación y despliegue controlado cuando el equipo, la frecuencia y los riesgos lo justifiquen. No activar servicios de pago ni publicar modelos solo porque se esté diseñando una arquitectura.

## Contratos y trazabilidad

- Registrar versión de código, referencia o huella del dataset, fecha de corte, esquema, parámetros, entorno y métricas. Reconocer límites de reproducibilidad numérica entre plataformas.
- Guardar preprocesamiento y modelo como una unidad lógica. Definir nombres, tipos, unidades, categorías desconocidas y tratamiento de faltantes; verificar consistencia entre entrenamiento y scoring.
- Separar entrenamiento de inferencia. Producir probabilidades asociadas a la entidad y al instante correcto; conservar la versión del modelo y, si existe, de la regla de decisión.
- No cargar artefactos serializados de origen no confiable. Proteger datos e identificadores y no incluir secretos en el repositorio ni logs.

## Operación cuando corresponda

- Verificar primero esquema y calidad de entradas. Monitorear distribución de scores, cobertura, latencia o frescura según el modo de servicio.
- Medir desempeño cuando maduren las etiquetas; definir cómo se vinculan a las predicciones. Distinguir cambios de distribución de deterioro predictivo: drift por sí solo no demuestra que el modelo falló.
- Definir responsable, frecuencia y respuesta ante alertas, más un fallback y rollback. Los umbrales deben derivarse de una referencia y del riesgo, no inventarse para completar una tabla.
- Reentrenar ante una necesidad comprobada y promover candidatos tras evaluación comparable. No sustituir automáticamente un modelo en uso solo porque llegó nueva data.

## Salida

Proponer la arquitectura mínima con flujo de datos, artefactos, ejecución, monitoreo y responsabilidades. Distinguir componentes implementados, propuestos y fuera de alcance. No declarar producción un notebook local.

## Fuente

Consultada el 15 de noviembre de 2026. Los criterios mínimos de esta skill son adaptaciones propias.

- [Google Cloud: MLOps, entrega continua y automatización](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning?hl=en): progresión desde operación manual hacia automatización de pipelines y CI/CD, con validación y monitoreo.
