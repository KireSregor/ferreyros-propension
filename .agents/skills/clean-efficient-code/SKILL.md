---
name: clean-efficient-code
description: Implementar y revisar código claro, documentado y proporcional al problema, evitando exploración repetida, sobreingeniería y pruebas sin valor. Usar al modificar scripts, notebooks o módulos del proyecto.
---

# Código limpio y trabajo eficiente

Aplicar criterios de Google Engineering Practices y convenciones de Python con alcance proporcional al cambio. Las reglas de eficiencia y selección de pruebas de esta skill son preferencias del proyecto, no una transcripción de la política de Google.

## Antes de editar

- Identificar el resultado observable, localizar archivos relevantes y leer las instrucciones aplicables. Reutilizar conocimiento comprobado de la sesión; volver a revisar solo si cambió el estado o hay incertidumbre material.
- Preferir búsquedas dirigidas con rg, lecturas acotadas y herramientas ya disponibles. Antes de instalar una dependencia o construir un extractor, comprobar alternativas existentes apropiadas. No cambiar configuración global para resolver una tarea local.
- Elegir la solución más pequeña que cumpla el comportamiento solicitado. No crear capas, clases, configuraciones, frameworks ni herramientas por necesidades hipotéticas.

## Implementación

- Priorizar lectura y mantenimiento sobre cantidad de líneas. No comprimir condiciones, retornos y diccionarios largos en una misma línea ni encadenar instrucciones con punto y coma. Simplificar arquitectura no significa comprimir sintaxis.
- Mantener formato y lint con Ruff cuando esté configurado. El formateador complementa, no sustituye, la revisión de nombres, responsabilidades, contratos y decisiones de modelado.
- Usar nombres que expresen el dominio y funciones con una responsabilidad comprensible. Extraer lógica compartida cuando exista repetición real o una frontera clara; evitar funciones triviales que solo fragmentan la lectura.
- Separar lectura y escritura de datos, transformación y presentación cuando facilite reutilización o pruebas. Mantener rutas relativas al proyecto y parámetros relevantes explícitos.
- En notebooks, explicar decisiones y resultados en Markdown. Mantener ejecución de principio a fin sin variables ocultas ni dependencia del orden manual de celdas. Mover a módulos solo la lógica que se beneficie de ello.
- Documentar contratos no obvios: entradas, salidas, unidades, supuestos temporales, efectos y errores relevantes. Los comentarios explican por qué; evitar narrar cada línea o mantener comentarios desactualizados.
- Manejar fallos esperables con contexto suficiente. No ocultar excepciones, sustituir resultados fallidos por datos falsos ni registrar información sensible.
- Mantener dependencias y configuración explícitas; seguir el estilo existente antes de introducir otro formateador. Optimizar rendimiento cuando haya una medición o un cuello de botella claro.

## Verificación y criterio de parada

- Ejecutar las comprobaciones existentes que cubran lo cambiado. Para cambios de lógica, verificar comportamientos observables y riesgos reales: temporalidad, joins, unidades, cálculo de métricas o inferencia, según aplique.
- No crear pruebas que copien la implementación ni suites nuevas para cambios de texto reversibles. Para una solución analítica completa, verificar la ejecución desde un entorno o kernel limpio cuando sea viable.
- Tras pasar los controles pertinentes, detenerse. Repetir o ampliar solo ante cambios nuevos, fallos o una preocupación concreta sin resolver.
- Informar qué cambió, cómo se verificó y qué no pudo comprobarse. No afirmar éxito si un comando falló o no se ejecutó.

## Fuentes

Consultadas el 15 de noviembre de 2026.

- [Google: criterios de revisión](https://google.github.io/eng-practices/review/reviewer/looking-for.html): diseño, comportamiento, complejidad, nombres y documentación.
- [Google: cambios pequeños](https://github.com/google/eng-practices/blob/master/review/developer/small-cls.md): cambios enfocados que puedan comprenderse y revisarse.
- [Python: PEP 8](https://peps.python.org/pep-0008/): legibilidad y coherencia con el proyecto.
