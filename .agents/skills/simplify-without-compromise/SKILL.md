---
name: simplify-without-compromise
description: Simplificar arquitectura, planes y entregables eliminando piezas que no aportan valor sin reducir rigor, comportamiento ni trazabilidad. Usar al diseñar o refactorizar proyectos con demasiados archivos, capas, pasos o documentos.
---

# Simplificar preservando calidad

Preferencia del usuario: cuestionar lo necesario, eliminar lo superfluo y simplificar antes de optimizar o automatizar. Es una guía propia; no atribuirla como metodología oficial de SpaceX ni de otra organización.

## Aplicación

1. Identificar lo que debe preservarse: resultado solicitado, interfaces, evidencia, reproducibilidad y controles que detectan errores materiales. No confundir una costumbre de implementación con un requisito.
2. Para cada archivo, capa, dependencia, gráfico, experimento o documento, preguntar qué responsabilidad cumple y qué se perdería al quitarlo. Si no hay una consecuencia útil identificable, proponer omitirlo o consolidarlo.
3. Eliminar duplicación y responsabilidades vacías antes de acelerar su ejecución. Mantener una sola fuente para transformaciones, métricas, configuración y cifras reportadas.
4. Agrupar responsabilidades relacionadas cuando puedan comprenderse y probarse juntas. Separarlas cuando tengan interfaces o ciclos de cambio distintos. No comprimir toda la solución en un archivo enorme para alcanzar un número arbitrario de archivos.
5. Optimizar o automatizar únicamente el flujo que sigue siendo necesario. Reutilizar artefactos comprobados; no introducir otro sistema de seguimiento si las herramientas existentes cubren la necesidad.
6. Verificar los invariantes afectados por la simplificación y explicar el beneficio concreto: menos duplicación, navegación más clara o menos mantenimiento. No repetir comprobaciones ajenas al cambio.

## Elegir herramientas por costo total

Comparar una herramienta mantenida con el código propio que reemplazaría: implementación, configuración, dependencias, ejecución, documentación y mantenimiento. Menos dependencias no implica automáticamente menos complejidad. Preferir una integración pequeña cuando evita crear almacenamiento, seguimiento, validación u orquestación propios. Rechazar capas adicionales que dupliquen herramientas existentes o no resuelvan una necesidad concreta. No prometer ahorro de código sin verificar la integración; revisar la decisión si aparecen costos inesperados.

## En ciencia de datos

- Reducir archivos no debe eliminar separación temporal, prevención de fuga, ajuste dentro de folds, evaluación relevante ni procedencia de resultados.
- Los notebooks cuentan el análisis; los módulos ejecutan lógica reutilizable. Los resultados extensos pueden guardarse como artefactos enlazados, sin llenar el recorrido de lectura con logs o tablas de todos los trials.
- Un gráfico debe responder una pregunta y permitir una conclusión respaldada. Consolidar paneles complementarios y retirar repeticiones; no sustituir análisis faltante por estética.
- Comparar candidatos con un protocolo común y asignar búsqueda adicional según evidencia. No entrenar variantes indefinidamente por disponibilidad de cómputo ni descartar familias requeridas sin evaluación o explicación.
- Documentar decisiones y supuestos donde se utilizan, con una guía breve que enlace el detalle. No copiar las mismas conclusiones a múltiples documentos mantenidos manualmente.

## Límites

La simplicidad se mide por facilidad de entender, ejecutar, modificar y verificar, no solo por líneas de código. Conservar una pieza cuando quitarla trasladaría complejidad oculta al usuario o introduciría un riesgo concreto. Antes de retirar archivos existentes, revisar quién los utiliza y preservar trabajo del usuario; una propuesta de simplificación no autoriza eliminaciones indiscriminadas.

Esta skill complementa la eficiencia de contexto y herramientas: aquí se simplifica el producto y su arquitectura, no solo la interacción del agente.
