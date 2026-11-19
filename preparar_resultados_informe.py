"""Registra los agregados observados en Colab y genera las figuras del informe."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

base = Path(__file__).resolve().parent
out = base / "resultados"
out.mkdir(exist_ok=True)

# Valores copiados del resumen que produjo el cuaderno ejecutado en Colab
# (TensorFlow 2.21.0; semillas 13, 27 y 41).
resumen = {
    "Atención aditiva": {
        "MAE_media": 0.7215237816174825,
        "MAE_desviacion": 0.0015910912880660506,
        "RMSE_media": 0.8003073334693909,
        "RMSE_desviacion": 0.0046461913702561366,
    },
    "Transformer": {
        "MAE_media": 0.4238954285780589,
        "MAE_desviacion": 0.03188942321721586,
        "RMSE_media": 0.5096572240193685,
        "RMSE_desviacion": 0.04811328930252961,
    },
    "Persistencia": {"MAE": 0.26986271142959595, "RMSE": 0.33454057574272156},
    "configuracion": {
        "muestra": "serie sintética reproducible; no representa mediciones reales",
        "n_observaciones": 720,
        "ventana": 36,
        "corte_entrenamiento": 576,
        "n_prueba": 144,
        "semillas": [13, 27, 41],
        "normalizacion": "media y desviación del tramo previo al corte",
        "tensorflow": "2.21.0",
    },
}
(out / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

pd.DataFrame([
    {"modelo": "Persistencia", "mae_media": resumen["Persistencia"]["MAE"], "rmse_media": resumen["Persistencia"]["RMSE"], "repeticiones": 1},
    {"modelo": "Atención aditiva", "mae_media": resumen["Atención aditiva"]["MAE_media"], "mae_desviacion": resumen["Atención aditiva"]["MAE_desviacion"], "rmse_media": resumen["Atención aditiva"]["RMSE_media"], "rmse_desviacion": resumen["Atención aditiva"]["RMSE_desviacion"], "repeticiones": 3},
    {"modelo": "Transformer", "mae_media": resumen["Transformer"]["MAE_media"], "mae_desviacion": resumen["Transformer"]["MAE_desviacion"], "rmse_media": resumen["Transformer"]["RMSE_media"], "rmse_desviacion": resumen["Transformer"]["RMSE_desviacion"], "repeticiones": 3},
]).to_csv(out / "metricas_resumen.csv", index=False)

rng = np.random.default_rng(2026)
t = np.arange(720, dtype=np.float32)
values = (0.004 * t + np.sin(2 * np.pi * t / 24)
          + 0.32 * np.sin(2 * np.pi * t / 8 + 0.4)
          + 0.18 * np.sin(2 * np.pi * t / 72)
          + rng.normal(0, 0.18, size=720))
cut = 576
plt.figure(figsize=(9.4, 4.3))
plt.plot(t[:cut], values[:cut], color="#466b91", lw=1.0, label="Entrenamiento")
plt.plot(t[cut:], values[cut:], color="#df9861", lw=1.2, label="Prueba")
plt.axvline(cut, color="#333333", ls="--", lw=1, label="Corte temporal")
plt.xlabel("Paso temporal sintético")
plt.ylabel("Índice de demanda")
plt.title("Partición cronológica de la serie")
plt.legend(frameon=False, ncol=3)
plt.tight_layout()
plt.savefig(out / "serie_particion.png", dpi=180)
plt.close()

names = ["Persistencia", "Atención aditiva", "Transformer"]
means = [resumen[n]["RMSE"] if n == "Persistencia" else resumen[n]["RMSE_media"] for n in names]
errors = [0, resumen["Atención aditiva"]["RMSE_desviacion"], resumen["Transformer"]["RMSE_desviacion"]]
plt.figure(figsize=(8.2, 4.4))
plt.bar(names, means, yerr=errors, capsize=4,
        color=["#91a1ae", "#4f81bd", "#e07a5f"], width=.62)
plt.ylabel("RMSE (unidades sintéticas)")
plt.title("Error en el tramo de prueba")
plt.xticks(rotation=10)
plt.grid(axis="y", alpha=.22)
plt.tight_layout()
plt.savefig(out / "comparacion_rmse.png", dpi=180)
plt.close()

print("Figuras y agregados guardados en", out)
