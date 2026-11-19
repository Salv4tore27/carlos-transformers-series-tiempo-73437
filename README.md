# Atención temporal y Transformer para pronóstico de series

Proyecto para **R2-A3-S12 Transformers (Moodle 73437)**. Compara una atención temporal aditiva y un Transformer causal para pronosticar el siguiente punto de una serie sintética.

## Contenido

- `Carlos_Transformers_Series_Tiempo.ipynb`: cuaderno para Google Colab.
- `experimento_transformers.py`: preparación de datos, definición de modelos, entrenamiento, evaluación y figuras.
- `main.tex`: informe en LaTeX, con formato de trabajo estudiantil basado en APA 7.
- `resultados/resumen.json` y `resultados/metricas_resumen.csv`: resumen de la corrida.
- `resultados/serie_particion.png` y `resultados/comparacion_rmse.png`: figuras incluidas en el informe.

## Cuaderno en Colab

El cuaderno ejecutado está disponible en [Google Colab](https://colab.research.google.com/drive/162pCXJDeUpa_q_hST-6FBUT8WlL5rwSN). La señal la genera el propio código: es sintética y no representa mediciones de una empresa o servicio.

## Reproducción local

```bash
python -m pip install -r requirements.txt
python experimento_transformers.py
```

La ejecución crea `metricas_semillas.csv`, `predicciones_prueba.csv` y `pesos_atencion.png` en `resultados/`, además de las otras figuras y el resumen. Se fijan las semillas 13, 27 y 41; la partición es cronológica y la normalización se ajusta con el tramo de entrenamiento.

## Lectura de resultados

La comparación informa MAE y RMSE en las unidades de la señal sintética. La persistencia (usar el último valor observado) superó a los modelos entrenables en esta prueba; por tanto, el resultado no se presenta como una victoria del Transformer. El alcance y las limitaciones se explican en el informe.
