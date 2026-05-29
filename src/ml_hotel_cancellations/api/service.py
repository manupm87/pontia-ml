"""Capa de servicio: carga del modelo y lógica de predicción.

Puente entre la API HTTP (``main``) y el modelo de ML. Responsabilidades:

1. Cargar el modelo UNA SOLA VEZ y cachearlo en memoria (cargar el ``Pipeline``
   desde disco es costoso).
2. Predecir reutilizando el mismo preprocesado que en entrenamiento
   (``ml.predict``), para que API y entrenamiento "vean" los datos igual.
3. Exponer ``predict_one`` / ``predict_many`` con una salida estable.

Cadena de carga: si ``MLFLOW_MODEL_URI`` está definida, se intenta el **Model
Registry** de MLflow (ver :mod:`ml_hotel_cancellations.api.registry`); si falla por
cualquier motivo, se cae al pickle local ``models/best_model.pkl``.

Variables de entorno: ``MLFLOW_MODEL_URI`` (opcional, activa el registry) y
``PONTIA_MODEL_PATH`` (opcional, ruta a un ``.pkl`` alternativo, útil en tests).
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

# Etiquetas legibles para cada clase (índice = clase predicha). Fuente única en
# `src.config` para no divergir con el entrenamiento ni con la interfaz.
CLASS_LABELS: list[str] = config.CLASS_LABELS_SHORT

# ROC-AUC del modelo en test, leído del artefacto de métricas del entrenamiento
# (no un literal): así /model-info nunca reporta una cifra obsoleta tras un
# reentrenamiento. Si el artefacto no estuviera disponible, caemos a un valor
# de respaldo conocido para no romper el endpoint.
try:
    MODEL_ROC_AUC: float = round(config.best_metric_value("roc_auc"), 4)
except Exception:  # noqa: BLE001 - el artefacto puede faltar en algún despliegue
    logger.warning("No se pudo leer el ROC-AUC del artefacto de métricas; uso respaldo.")
    MODEL_ROC_AUC = 0.9614

# Metadatos del último ``get_model()`` ejecutado. Los rellena
# ``_set_load_info`` y los lee ``/model-info`` para reportar de dónde sale
# el modelo servido (registry o bundled, qué versión, qué stage, etc.).
_LOAD_INFO: LoadInfo | None = None

# Mapa de nombres de clase del estimador final a nombre "comercial" legible
# para ``/model-info``. Se deriva del estimador REALMENTE cargado en tiempo
# de ejecución, en vez de codificar "XGBoost" a mano.
_MODEL_TYPE_NAMES: dict[str, str] = {
    "XGBClassifier": "XGBoost",
    "LogisticRegression": "Regresión logística",
    "DecisionTreeClassifier": "Árbol de decisión",
    "RandomForestClassifier": "Random Forest",
    "KerasMLPClassifier": "Red neuronal (Keras)",
}


def get_model_path() -> Path:
    """Devuelve la ruta del modelo bundled (.pkl en el repo) a servir.

    Prioriza la variable de entorno ``PONTIA_MODEL_PATH``; si no está definida,
    usa la ruta por defecto del proyecto (``config.BEST_MODEL_PATH``).
    """
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
    """Devuelve los metadatos del modelo actualmente cargado como ``dict``.

    Si nunca se ha cargado, fuerza la carga (lo que rellenará ``_LOAD_INFO``).
    Si la carga falla (``_LOAD_INFO`` sigue vacío o el modelo no está
    operativo), reporta un estado DEGRADADO con ``source="error"`` en vez de
    mentir afirmando que sirve el modelo bundled. Nunca propaga excepciones:
    el handler de ``/model-info`` debe poder responder siempre.
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
    """Carga el modelo una sola vez (singleton via ``lru_cache``).

    Sigue la cadena registry → bundled descrita en la cabecera del módulo.
    Si la carga desde el registry falla, registra el motivo en
    ``_LOAD_INFO["fallback_reason"]`` para que ``/model-info`` lo refleje.
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
            # Familias esperadas en la carga del registry (red/auth, JSON
            # incompleto, URI mal formada, stage inexistente, escritura en
            # caché): caemos al pickle bundled y registramos el motivo. No
            # capturamos ``Exception`` a secas para no enmascarar bugs.
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
    """Deriva el nombre legible del modelo del estimador REALMENTE cargado.

    Inspecciona el paso ``model`` del ``Pipeline`` en tiempo de ejecución en
    vez de codificar "XGBoost" a mano: así ``/model-info`` no miente si algún
    día se sirve otra familia de modelos. Para clases desconocidas devuelve el
    propio nombre de la clase.
    """
    estimator = get_model().named_steps["model"]
    class_name = type(estimator).__name__
    return _MODEL_TYPE_NAMES.get(class_name, class_name)


def get_model_info_payload() -> dict:
    """Ensambla TODOS los metadatos que necesita el modelo ``ModelInfo``.

    Centraliza en la capa de servicio el conocimiento de qué campos componen
    la respuesta de ``/model-info`` (tipo de modelo, métrica, origen, versión,
    etc.), para que el handler de ``main`` sea un simple
    ``ModelInfo(**service.get_model_info_payload())``.
    """
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
