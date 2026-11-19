"""Comparación reproducible de atención aditiva y Transformer para pronóstico.

La serie es sintética: sirve para estudiar el procedimiento sin atribuir los
resultados a una medición real. Ejecute este archivo en Google Colab o con
Python 3.10+ y TensorFlow 2.15+.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUT = ROOT / "resultados"
OUT.mkdir(parents=True, exist_ok=True)
WINDOW = 36
N_POINTS = 720
SEEDS = (13, 27, 41)


def make_series(n: int = N_POINTS, seed: int = 2026) -> np.ndarray:
    """Genera una señal de demanda didáctica con tendencia, ciclos y ruido."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float32)
    signal = (
        0.004 * t
        + 1.0 * np.sin(2 * np.pi * t / 24)
        + 0.32 * np.sin(2 * np.pi * t / 8 + 0.4)
        + 0.18 * np.sin(2 * np.pi * t / 72)
        + rng.normal(0, 0.18, size=n)
    )
    return signal.astype(np.float32)


def split_windows(values: np.ndarray):
    """Crea ventanas ordenadas; los objetivos de prueba quedan al final."""
    cut = int(len(values) * 0.80)
    mu, sigma = float(values[:cut].mean()), float(values[:cut].std())
    scaled = (values - mu) / (sigma + 1e-8)
    xs, ys, target_index = [], [], []
    for target in range(WINDOW, len(values)):
        xs.append(scaled[target - WINDOW:target])
        ys.append(scaled[target])
        target_index.append(target)
    x = np.asarray(xs, dtype=np.float32)[..., None]
    y = np.asarray(ys, dtype=np.float32)
    target_index = np.asarray(target_index)
    train_ids = np.flatnonzero(target_index < cut)
    test_ids = np.flatnonzero(target_index >= cut)
    # Validación temporal: último 15% del tramo de entrenamiento.
    val_start = int(len(train_ids) * 0.85)
    fit_ids, val_ids = train_ids[:val_start], train_ids[val_start:]
    return (x, y, target_index, fit_ids, val_ids, test_ids,
            mu, sigma, cut, scaled)


class AdditiveTemporalAttention(layers.Layer):
    """Atención aditiva: consulta en el instante final sobre la ventana."""
    def __init__(self, width: int, **kwargs):
        super().__init__(**kwargs)
        self.score = layers.Dense(1, use_bias=False, name="puntuacion")

    def call(self, x):
        query = x[:, -1:, :]
        energies = self.score(x + query)
        weights = tf.nn.softmax(energies, axis=1)
        context = tf.reduce_sum(weights * x, axis=1)
        return context, tf.squeeze(weights, axis=-1)


class PositionEmbedding(layers.Layer):
    """Suma a cada instante una codificación posicional aprendible."""
    def __init__(self, length: int, width: int, **kwargs):
        super().__init__(**kwargs)
        self.position = layers.Embedding(length, width, name="embedding_posicion")
        self.length = length

    def call(self, x):
        positions = tf.range(start=0, limit=self.length, delta=1)
        return x + self.position(positions)[None, :, :]


def build_attention_model():
    inp = keras.Input(shape=(WINDOW, 1), name="ventana_entrada")
    projected = layers.Dense(32, activation="tanh", name="representacion_temporal")(inp)
    context, weights = AdditiveTemporalAttention(32, name="atencion_aditiva")(projected)
    pred = layers.Dense(1, name="pronostico")(context)
    model = keras.Model(inp, pred, name="atencion_aditiva")
    attention = keras.Model(inp, weights, name="pesos_atencion")
    return model, attention


def build_transformer_model():
    inp = keras.Input(shape=(WINDOW, 1), name="ventana_entrada")
    x = layers.Dense(32, name="proyeccion_a_d_model")(inp)
    x = PositionEmbedding(WINDOW, 32, name="codificacion_posicional")(x)
    attn = layers.MultiHeadAttention(
        num_heads=4, key_dim=8, dropout=0.10, name="autoatencion_multicabeza"
    )(x, x, use_causal_mask=True)
    x = layers.LayerNormalization(epsilon=1e-6, name="norma_atencion")(x + attn)
    ff = layers.Dense(64, activation="gelu", name="red_feed_forward_1")(x)
    ff = layers.Dense(32, name="red_feed_forward_2")(ff)
    x = layers.LayerNormalization(epsilon=1e-6, name="norma_feed_forward")(x + ff)
    pred = layers.Dense(1, name="pronostico")(x[:, -1, :])
    return keras.Model(inp, pred, name="transformer_temporal")


def compile_model(model):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def inverse(z, mu, sigma):
    return np.asarray(z).reshape(-1) * sigma + mu


def metricas(real, pred):
    err = np.asarray(real) - np.asarray(pred)
    return {"MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2)))}


def main():
    tf.keras.utils.set_random_seed(2026)
    tf.config.experimental.enable_op_determinism()
    values = make_series()
    (x, y, target_index, fit_ids, val_ids, test_ids,
     mu, sigma, cut, scaled) = split_windows(values)
    x_fit, y_fit = x[fit_ids], y[fit_ids]
    x_val, y_val = x[val_ids], y[val_ids]
    x_test, y_test = x[test_ids], y[test_ids]
    y_true = inverse(y_test, mu, sigma)
    persistence = inverse(x_test[:, -1, 0], mu, sigma)
    all_runs = []
    final_attention_model = final_attention = final_transformer = None

    for seed in SEEDS:
        tf.keras.utils.set_random_seed(seed)
        attention_model, attention_reader = build_attention_model()
        transformer = build_transformer_model()
        run_models = (("Atención aditiva", attention_model),
                      ("Transformer", transformer))
        for model_name, model in run_models:
            compile_model(model)
            stop = keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
            history = model.fit(x_fit, y_fit, validation_data=(x_val, y_val),
                                epochs=35, batch_size=32, shuffle=False,
                                callbacks=[stop], verbose=0)
            p = inverse(model.predict(x_test, verbose=0), mu, sigma)
            row = {"semilla": seed, "modelo": model_name,
                   "epocas": len(history.history["loss"]), **metricas(y_true, p)}
            all_runs.append(row)
        final_attention_model, final_attention, final_transformer = (
            attention_model, attention_reader, transformer
        )

    df = pd.DataFrame(all_runs)
    df.to_csv(OUT / "metricas_semillas.csv", index=False)
    grouped = df.groupby("modelo").agg(["mean", "std"])
    summary = {name: {
        "MAE_media": float(grouped.loc[name, ("MAE", "mean")]),
        "MAE_desviacion": float(grouped.loc[name, ("MAE", "std")]),
        "RMSE_media": float(grouped.loc[name, ("RMSE", "mean")]),
        "RMSE_desviacion": float(grouped.loc[name, ("RMSE", "std")]),
    } for name in grouped.index}
    summary["Persistencia"] = metricas(y_true, persistence)
    summary["configuracion"] = {
        "muestra": "serie sintética reproducible; no representa mediciones reales",
        "n_observaciones": int(len(values)), "ventana": WINDOW,
        "corte_entrenamiento": int(cut), "n_prueba": int(len(test_ids)),
        "semillas": list(SEEDS), "normalizacion": "media y desviación del tramo previo al corte",
        "tensorflow": tf.__version__,
    }
    (OUT / "resumen.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Predicciones e índices para auditar la evaluación temporal.
    pred_attention = inverse(final_attention_model.predict(x_test, verbose=0), mu, sigma)
    pred_transformer = inverse(final_transformer.predict(x_test, verbose=0), mu, sigma)
    idx = target_index[test_ids]
    pd.DataFrame({"indice": idx, "real": y_true, "persistencia": persistence,
                  "atencion_aditiva": pred_attention, "transformer": pred_transformer
                  }).to_csv(OUT / "predicciones_prueba.csv", index=False)

    plt.figure(figsize=(10, 4.6))
    t = np.arange(len(values))
    plt.plot(t[:cut], values[:cut], color="#395b8a", lw=1.0, label="Entrenamiento")
    plt.plot(t[cut:], values[cut:], color="#f0a35e", lw=1.2, label="Prueba")
    plt.axvline(cut, color="#333333", ls="--", lw=1, label="Corte temporal")
    plt.title("Serie sintética y partición cronológica")
    plt.xlabel("Paso temporal (unidad sintética)"); plt.ylabel("Índice de demanda")
    plt.legend(frameon=False, ncol=3); plt.tight_layout()
    plt.savefig(OUT / "serie_particion.png", dpi=180); plt.close()

    plt.figure(figsize=(8, 4.6))
    names = ["Persistencia", "Atención aditiva", "Transformer"]
    vals = [summary[n]["RMSE"] if n == "Persistencia" else summary[n]["RMSE_media"] for n in names]
    colors = ["#9aa7b2", "#4f81bd", "#e07a5f"]
    plt.bar(names, vals, color=colors, width=0.62)
    plt.ylabel("RMSE en escala original"); plt.title("Error de pronóstico en el tramo de prueba")
    plt.xticks(rotation=12); plt.grid(axis="y", alpha=.22); plt.tight_layout()
    plt.savefig(OUT / "comparacion_rmse.png", dpi=180); plt.close()

    weights = final_attention.predict(x_test[:1], verbose=0)[0]
    plt.figure(figsize=(9, 2.7))
    plt.bar(np.arange(1, WINDOW + 1), weights, color="#557a95")
    plt.xlabel("Posición dentro de la ventana (1 = más antigua)")
    plt.ylabel("Peso aprendido"); plt.title("Pesos de atención del primer ejemplo de prueba")
    plt.tight_layout(); plt.savefig(OUT / "pesos_atencion.png", dpi=180); plt.close()

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
