---
name: efficient-context-and-tools
description: Reducir consumo innecesario de tokens, lecturas y llamadas a herramientas sin sacrificar calidad. Usar al investigar, inspeccionar archivos o ejecutar tareas con varias herramientas, especialmente para evitar revisiones repetidas e instalaciones innecesarias.
---

# Uso eficiente de contexto y herramientas

Conservar la calidad del resultado reduciendo trabajo que no cambia una decisión, no resuelve una incertidumbre relevante y no verifica un riesgo concreto. Esta es una preferencia de trabajo del usuario, no una autorización para omitir comprobaciones necesarias.

## Antes de llamar una herramienta

- Identificar qué pregunta resolverá la llamada y cómo su respuesta afectará el siguiente paso. Si no hay una utilidad concreta, omitirla.
- Reutilizar resultados comprobados y documentos ya leídos mientras sigan vigentes y disponibles en el contexto. Volver a consultar únicamente ante cambios, pérdida de información relevante o dudas materiales.
- Elegir la herramienta más directa que esté disponible y sea apropiada. Para un archivo sencillo, empezar con una lectura o extracción existente; no preparar un entorno completo por defecto.
- Acotar búsquedas a rutas y patrones relevantes. Pedir muestras, secciones o resúmenes suficientes en lugar de volcar archivos extensos, registros completos, binarios o respuestas de APIs sin filtrar.
- Agrupar consultas independientes cuando reduzca llamadas sin generar salida innecesaria. Mantener secuenciales las decisiones que dependan del resultado anterior.

## Dependencias y alternativas

- Antes de instalar software, comprobar brevemente si una herramienta disponible ya resuelve la necesidad. Reutilizar el inventario conocido de la sesión; evitar búsquedas recursivas amplias para localizar ejecutables.
- Instalar solo cuando la capacidad faltante sea necesaria y la alternativa disponible sea insuficiente o menos fiable. Explicar el propósito si hace falta autorización y respetar los permisos del entorno.
- Si un intento falla, interpretar el error antes de repetir. No encadenar instalaciones o herramientas distintas sin una hipótesis que justifique el siguiente intento.
- No construir extractores propios complejos para evitar una dependencia adecuada. Una alternativa ligera solo sirve si preserva la información necesaria y permite reconocer sus límites.
- En PDFs, una extracción de texto fiable puede bastar para preguntas de contenido. Revisar visualmente cuando tablas, diagramas, orden de lectura o formato puedan cambiar la interpretación; no dar por completo un texto truncado o ilegible.

## Verificación proporcional

- Comprobar los requisitos y los fallos plausibles que podrían invalidar la entrega. Priorizar exactitud de cifras, temporalidad, integridad de datos y ejecución cuando correspondan.
- No repetir una comprobación que ya pasó si no cambió su entrada, implementación ni una dependencia relevante.
- No ampliar a auditorías, benchmarks, suites de pruebas o búsquedas exhaustivas sin una señal concreta o un requisito del usuario.
- Detenerse cuando el resultado solicitado esté completo y las incertidumbres materiales estén resueltas o claramente declaradas. Ahorrar tokens no significa finalizar con trabajo necesario pendiente.

## Comunicación

- Informar resultados, decisiones y límites relevantes; omitir transcripciones de comandos y detalles rutinarios que no ayuden al usuario.
- Ajustar la extensión al pedido. Un plan general necesita etapas y resultados; no requiere desarrollar de antemano todos los detalles técnicos.
- Mantener evidencia suficiente para sostener conclusiones. No inventar resultados, esconder fallos ni omitir fuentes necesarias para acortar la respuesta.

## Caso de referencia

Ante una solicitud de leer un PDF, comprobar primero las capacidades conocidas de lectura. No iniciar automáticamente instalación de Python y paquetes. Si falta una capacidad indispensable, explicar esa necesidad concreta; si ya existe una vía fiable y autorizada, utilizarla directamente.
