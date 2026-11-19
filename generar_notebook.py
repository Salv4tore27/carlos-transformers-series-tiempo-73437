"""Actualiza el notebook para incluir el código completo y ejecutable."""
import json
from pathlib import Path

base = Path(__file__).resolve().parent
script = (base / "experimento_transformers.py").read_text(encoding="utf-8")
script = script.split("\n\nif __name__ == \"__main__\":")[0]

def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}

cells = [
    markdown("""# R2-A3-S12 Transformers: atención y pronóstico de series de tiempo

**Autor:** Carlos Mario Salvatore Ocampo Aguilar  
**Curso:** Deep Learning — Conceptos  

El cuaderno implementa una atención temporal aditiva y un Transformer causal para pronosticar el siguiente punto de una serie sintética. La serie se genera aquí con una semilla fija; no representa mediciones reales. Se separa el tiempo en entrenamiento, validación y prueba, se entrena con TensorFlow/Keras y se contrasta con persistencia.

Ejecuta las celdas de arriba abajo. La última celda deja métricas, predicciones y figuras en `resultados/`.
"""),
    code("%pip -q install 'tensorflow>=2.20,<2.23' pandas matplotlib numpy\n"),
    code(script),
    code("main()\n"),
    code("""from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image

resumen = json.loads(Path('resultados/resumen.json').read_text(encoding='utf-8'))
display(pd.DataFrame([
    {'Modelo': 'Persistencia', **resumen['Persistencia']},
    {'Modelo': 'Atención aditiva', 'MAE': resumen['Atención aditiva']['MAE_media'], 'RMSE': resumen['Atención aditiva']['RMSE_media']},
    {'Modelo': 'Transformer', 'MAE': resumen['Transformer']['MAE_media'], 'RMSE': resumen['Transformer']['RMSE_media']},
]))
for nombre in ['serie_particion.png', 'comparacion_rmse.png', 'pesos_atencion.png']:
    display(Image(filename=str(Path('resultados') / nombre)))
"""),
]
notebook = {
    "cells": cells,
    "metadata": {"colab": {"provenance": []},
                 "kernelspec": {"display_name": "Python 3", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
(base / "Carlos_Transformers_Series_Tiempo.ipynb").write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
)
