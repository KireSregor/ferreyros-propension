---
name: data-analysis
description: Explorar tablas y explicar tamaño, estructura, calidad y patrones con implicaciones de negocio. Usar para EDA y diagnóstico de datos; no sustituye el diseño ni la evaluación de modelos.
---

# Análisis de datos orientado a decisiones

Combinar exploración abierta del NIST con preguntas comerciales priorizadas. Las hipótesis orientan el trabajo, pero no deben ocultar patrones inesperados ni producir confirmación selectiva.

## Flujo

1. Leer esquema, diccionario y una muestra antes de cargar o transformar todo. Medir filas, columnas, tamaño, entidades únicas, rango temporal y granularidad. Distinguir número de registros de tamaño efectivo de la población.
2. Construir un diccionario práctico: significado, tipo, unidad, códigos, valores especiales y disponibilidad temporal. No asignar significados a códigos demográficos ni moneda a importes sin respaldo.
3. Revisar calidad que pueda cambiar conclusiones: faltantes reales y centinelas, duplicados según la clave propuesta, rangos, coherencia entre campos, orden temporal y estabilidad de atributos. Documentar anomalías antes de corregirlas; conservar datos originales.
4. Describir el resultado y su evolución. En paneles de clientes, mostrar tanto distribución por visita como por cliente. Diferenciar promedios ponderados por visitas de promedios entre personas.
5. Analizar patrones comerciales relevantes: frecuencia, recurrencia, conversión, marcas, cantidades, precios y promociones. Cada gráfico debe responder una pregunta y especificar población, denominador, periodo, unidades y tamaño de muestra.
6. Examinar explicaciones alternativas y soporte. No interpretar códigos nominales como magnitudes, correlación como causalidad o segmentos pequeños como conclusiones estables. Mostrar incertidumbre cuando afecte la decisión; para inferencia, respetar dependencia temporal y agrupación por cliente.
7. Consolidar hallazgos como evidencia, interpretación, límite e implicación. Separar observaciones descriptivas de hipótesis que necesitarán validación futura.

## Disciplina de trabajo

- Hacer un perfil base una vez y reutilizarlo mientras no cambien datos ni filtros.
- No generar todos los cruces ni gráficos por defecto. Profundizar donde una anomalía o pregunta pueda cambiar preparación, evaluación o acción comercial.
- Si habrá modelado, reservar el test antes de explorar relaciones que guíen selección de variables. Usar entrenamiento para esas decisiones; no optimizar mirando el periodo reservado.
- Detener la exploración cuando estén claras la población, la calidad relevante, los principales patrones y las decisiones pendientes. Registrar preguntas secundarias sin investigarlas automáticamente.

## Salida

Resumen de la base, diccionario ampliado cuando sea necesario, incidencias de calidad y hallazgos priorizados. En propensión por visita, recordar que marca y cantidad de la compra actual describen el resultado y pueden revelar el target. Verificar el significado de los ceros del historial antes de tratarlos como datos faltantes.

## Fuentes

Consultadas el 15 de noviembre de 2026. La selección comercial y las reglas de eficiencia son adaptaciones propias.

- [NIST: What is EDA?](https://www.itl.nist.gov/div898/handbook/eda/section1/eda11.htm): explorar estructura, anomalías y supuestos con evidencia gráfica y cuantitativa.
- [IBM: Data Understanding Overview](https://www.ibm.com/docs/en/spss-modeler/19.0.0?topic=understanding-data-overview): describir y evaluar los datos antes de prepararlos.
- [scikit-learn: errores comunes](https://scikit-learn.org/stable/common_pitfalls.html): prevenir contaminación del conjunto de evaluación.
