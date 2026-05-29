"""Capa de servicio: carga del modelo (cacheada) y lógica de predicción.

Puente entre la API HTTP y el modelo. Cadena de carga: si ``MLFLOW_MODEL_URI``
está definida intenta el registry MLflow y, si falla, cae al pickle bundled.
Env vars opcionales: ``MLFLOW_MODEL_URI`` y ``PONTIA_MODEL_PATH``.
Ver docs/arquitectura.md para la justificación completa.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests

from ml_hotel_cancellations import config
from ml_hotel_cancellations.ml.predict import load_best_model, predict_dataframe

from .registry import LoadInfo, load_from_registry

logger = logging.getLogger(__name__)

# Índice = clase predicha. Fuente única en `config` (compartida con train y UI).
CLASS_LABELS: list[str] = config.CLASS_LABELS_SHORT

# Leído del artefacto de métricas (no literal) para que /model-info nunca reporte
# una cifra obsoleta; respaldo conocido si el artefacto falta.
try:
    MODEL_ROC_AUC: float = round(config.best_metric_value("roc_auc"), 4)
except Exception:  # noqa: BLE001 - el artefacto puede faltar en algún despliegue
    logger.warning("No se pudo leer el ROC-AUC del artefacto de métricas; uso respaldo.")
    MODEL_ROC_AUC = 0.9614

# Metadatos del último modelo cargado; los lee /model-info.
_LOAD_INFO: LoadInfo | None = None

# Nombre de clase del estimador → nombre legible para /model-info.
_MODEL_TYPE_NAMES: dict[str, str] = {
    "XGBClassifier": "XGBoost",
    "LogisticRegression": "Regresión logística",
    "DecisionTreeClassifier": "Árbol de decisión",
    "RandomForestClassifier": "Random Forest",
    "KerasMLPClassifier": "Red neuronal (Keras)",
}


def get_model_path() -> Path:
    """Ruta del modelo bundled a servir: ``PONTIA_MODEL_PATH`` o, por defecto, ``config.BEST_MODEL_PATH``."""
    env_path = os.getenv("PONTIA_MODEL_PATH")
    return Path(env_path) if env_path else config.BEST_MODEL_PATH


def get_registry_uri() -> str | None:
    """Devuelve la URI del registry a usar, si está configurada."""
    return os.getenv("MLFLOW_MODEL_URI") or None


def _set_load_info(info: LoadInfo) -> LoadInfo:
    """Registra los metadatos del último modelo cargado y los devuelve."""
    global _LOAD_INFO
    _LOAD_INFO = info
    return info


def get_load_info() -> dict:
    """Metadatos del modelo cargado como ``dict``; fuerza la carga si hace falta.

    Si la carga falla reporta ``source="error"`` en vez de mentir; nunca propaga excepciones.
    """
    try:
        if _LOAD_INFO is None:
            get_model()
    except Exception:  # noqa: BLE001 - no romper /model-info por un fallo de carga
        logger.exception("No se pudo cargar el modelo al consultar get_load_info().")

    if _LOAD_INFO is None or not is_model_loaded():
        return asdict(LoadInfo(source="error"))
    return asdict(_LOAD_INFO)


def _load_bundled():
    """Carga el ``.pkl`` versionado en el repositorio."""
    path = get_model_path()
    logger.info("Cargando modelo bundled (pickle local) desde %s", path)
    pipeline = load_best_model(path=path)
    info = LoadInfo(source="bundled", path=str(path))
    return pipeline, info


@lru_cache(maxsize=1)
def get_model():
    """Carga el modelo una sola vez (singleton vía ``lru_cache``), cadena registry → bundled.

    Si el registry falla, registra el motivo en ``fallback_reason``.
    """
    uri = get_registry_uri()
    if uri:
        try:
            pipeline, info = load_from_registry(uri)
            _set_load_info(info)
            return pipeline
        except (
            requests.RequestException,
            KeyError,
            ValueError,
            RuntimeError,
            OSError,
        ) as exc:
            # Solo las familias esperadas (red/auth, JSON, URI, stage, caché):
            # caemos al bundled. No capturamos `Exception` para no ocultar bugs.
            logger.warning(
                "Carga desde el registry MLflow falló (%s). Usando el "
                "pickle local como respaldo.",
                exc,
            )
            pipeline, info = _load_bundled()
            info.fallback_reason = f"{type(exc).__name__}: {exc}"
            _set_load_info(info)
            return pipeline

    # Camino feliz sin registry configurado.
    pipeline, info = _load_bundled()
    _set_load_info(info)
    return pipeline


def is_model_loaded() -> bool:
    """Indica si el modelo puede cargarse correctamente (para /health)."""
    try:
        get_model()
        return True
    except Exception:  # noqa: BLE001 - el health check no debe propagar errores
        logger.exception("No se pudo cargar el modelo en el health check.")
        return False


def get_model_type() -> str:
    """Nombre legible del modelo, derivado del estimador realmente cargado (no codificado a mano)."""
    estimator = get_model().named_steps["model"]
    class_name = type(estimator).__name__
    return _MODEL_TYPE_NAMES.get(class_name, class_name)


def get_model_info_payload() -> dict:
    """Ensambla todos los campos de ``ModelInfo`` para que el handler sea un simple paso-a-través."""
    load_info = get_load_info()
    return {
        "model_type": get_model_type(),
        "primary_metric": config.PRIMARY_METRIC,
        "roc_auc": MODEL_ROC_AUC,
        "n_features": len(config.FEATURE_COLUMNS),
        "features": {
            "numeric": config.NUMERIC_COLUMNS,
            "categorical": config.CATEGORICAL_COLUMNS,
        },
        "source": load_info.get("source", "bundled"),
        "registry_uri": load_info.get("registry_uri"),
        "version": load_info.get("version"),
        "stage": load_info.get("stage"),
        "run_id": load_info.get("run_id"),
        "fallback_reason": load_info.get("fallback_reason"),
    }


def _format_result(prediction: int, probability: float) -> dict:
    """Da forma a la salida unitaria con la etiqueta legible."""
    return {
        "prediction": int(prediction),
        "label": CLASS_LABELS[int(prediction)],
        "probability": float(probability),
    }


def predict_many(bookings: list[dict]) -> list[dict]:
    """Predice la cancelación para una lista de reservas (dicts), en bloque.

    Devuelve una predicción por reserva en el MISMO orden de entrada.
    """
    if not bookings:
        return []

    df = pd.DataFrame(bookings)
    result = predict_dataframe(df, model=get_model())
    return [
        _format_result(row.prediction, row.probability_canceled)
        for row in result.itertuples(index=False)
    ]


def predict_one(booking: dict) -> dict:
    """Predice la cancelación para una única reserva (dict)."""
    return predict_many([booking])[0]
