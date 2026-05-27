"""Capa de servicio: carga del modelo y lógica de predicción.

Este módulo es el "puente" entre la API HTTP (``main``) y el modelo de ML
(``src``). Sus responsabilidades:

1. Cargar el modelo UNA SOLA VEZ y mantenerlo en memoria (caché). Cargar el
   ``Pipeline`` desde disco es costoso, así que no queremos hacerlo en cada
   petición.
2. Preparar la entrada EXACTAMENTE igual que en entrenamiento, reutilizando la
   lógica de ``src`` (``normalize_categoricals`` vía ``src.predict``). Esto
   garantiza que la API y el entrenamiento "vean" los datos de la misma forma.
3. Exponer ``predict_one`` y ``predict_many`` con una salida sencilla y estable.

La ruta del modelo es configurable mediante la variable de entorno
``PONTIA_MODEL_PATH`` (por defecto, ``src.config.BEST_MODEL_PATH``). Así se puede
servir otro modelo sin tocar el código.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src import config
from src.predict import load_best_model, predict_dataframe

logger = logging.getLogger(__name__)

# Etiquetas legibles para cada clase (índice = clase predicha).
CLASS_LABELS: list[str] = ["No cancelada", "Cancelada"]

# ROC-AUC del modelo en test (reportado por el pipeline de entrenamiento).
# Se expone en /model-info para que el cliente conozca la calidad del modelo.
MODEL_ROC_AUC: float = 0.9614


def get_model_path() -> Path:
    """Devuelve la ruta del modelo a servir.

    Prioriza la variable de entorno ``PONTIA_MODEL_PATH``; si no está definida,
    usa la ruta por defecto del proyecto (``config.BEST_MODEL_PATH``).
    """
    env_path = os.getenv("PONTIA_MODEL_PATH")
    return Path(env_path) if env_path else config.BEST_MODEL_PATH


@lru_cache(maxsize=1)
def get_model():
    """Carga el modelo una sola vez y lo cachea en memoria.

    ``lru_cache`` garantiza que la carga desde disco ocurra en la primera llamada
    y se reutilice en las siguientes (patrón *singleton* sencillo).
    """
    path = get_model_path()
    logger.info("Cargando modelo para la API desde %s", path)
    return load_best_model(path=path)


def is_model_loaded() -> bool:
    """Indica si el modelo puede cargarse correctamente (para /health)."""
    try:
        get_model()
        return True
    except Exception:  # noqa: BLE001 - el health check no debe propagar errores
        logger.exception("No se pudo cargar el modelo en el health check.")
        return False


def _format_result(prediction: int, probability: float) -> dict:
    """Da forma a la salida unitaria con la etiqueta legible."""
    return {
        "prediction": int(prediction),
        "label": CLASS_LABELS[int(prediction)],
        "probability": float(probability),
    }


def predict_many(bookings: list[dict]) -> list[dict]:
    """Predice la cancelación para una lista de reservas (dicts).

    Construye un ``DataFrame`` con todas las reservas y delega en
    ``src.predict.predict_dataframe``, que aplica el mismo preprocesado que en
    entrenamiento y devuelve clase + probabilidad. Procesar en bloque es más
    eficiente que fila a fila.

    Parameters
    ----------
    bookings:
        Lista de reservas, cada una como diccionario con las 27 características.

    Returns
    -------
    list[dict]
        Una predicción por reserva, en el MISMO orden de entrada, con las claves
        ``prediction``, ``label`` y ``probability``.
    """
    if not bookings:
        return []

    df = pd.DataFrame(bookings)
    resultado = predict_dataframe(df, model=get_model())
    return [
        _format_result(row.prediction, row.probability_canceled)
        for row in resultado.itertuples(index=False)
    ]


def predict_one(booking: dict) -> dict:
    """Predice la cancelación para una única reserva (dict)."""
    return predict_many([booking])[0]
