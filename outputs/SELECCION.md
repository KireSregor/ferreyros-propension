# Selección y estado de validación

Con el requisito actualizado de precisión superior al 51%, el candidato de investigación es el ensamblado cronológico: **53,1% de precisión, 48,6% de recall y F1 0,507**, frente a 48,9%, 47,7% y 0,483 de la referencia histórica en las mismas visitas. La mejora de precisión es de 4,2 puntos porcentuales; no se ha establecido su significación estadística.

El umbral 0,343916 se eligió con días 405–512 y se fijó antes de comprobar días 513–620. En selección obtuvo aproximadamente 55% en ambas métricas. El deterioro posterior, especialmente en recall, impide certificar ausencia de sobreajuste. El periodo posterior ya se había inspeccionado en otras investigaciones, por lo que tampoco es un test independiente.

**Estado: candidato para revisión, no aprobado como modelo final sin sobreajuste.** El archivo del modelo principal sigue correspondiendo al flujo original. No se presenta la elección retrospectiva como confirmación independiente.

Los resultados y configuraciones quedan en `research_selection.json`. La investigación descartada, incluido el código necesario para reproducir el candidato, se conserva comprimida en `research_archive.zip`: extraer en una carpeta separada y seguir las instrucciones archivadas. Las carpetas de experimentos y sus scripts sueltos se retiraron del proyecto activo.
