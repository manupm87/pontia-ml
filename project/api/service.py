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

Cadena de carga del modelo (en orden)
-------------------------------------
1. Si ``MLFLOW_MODEL_URI`` está definida, se intenta descargar y cargar
   esa versión desde el **Model Registry** de MLflow (DagsHub). El
   artefacto se cachea en ``/tmp/pontia_models/<hash(uri)>`` para no
   re-descargarlo entre warm-restarts del contenedor.
2. Si el paso 1 falla por cualquier motivo (sin credenciales, red caída,
   token expirado, ``mlflow`` no instalado…), se cae automáticamente al
   pickle local (``models/best_model.pkl``), con un *warning* en logs.

Variables de entorno
--------------------
- ``MLFLOW_MODEL_URI`` (opcional): URI tipo
  ``models:/pontia-cancellations/Production`` (recomendado) o
  ``models:/pontia-cancellations/3`` (versión fija).
- ``PONTIA_MODEL_PATH`` (opcional): ruta absoluta a un ``.pkl`` para
  servirlo en lugar del bundled. Útil sobre todo en tests / despliegues
  custom.
"""

from __future__ import annotations

import hashlib
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

# Directorio raíz para cachear modelos descargados del registry. ``/tmp`` es
# escribible en Render free y sobrevive a warm-restarts (pero NO a cold-starts:
# si Render destruye el contenedor por inactividad, la próxima carga re-descarga
# del registry, que es exactamente lo que queremos en ese caso).
_REGISTRY_CACHE_ROOT: Path = Path(os.getenv("PONTIA_REGISTRY_CACHE", "/tmp/pontia_models"))

# Metadatos del último ``get_model()`` ejecutado. Los rellena
# ``_set_load_info`` y los lee ``/model-info`` para reportar de dónde sale
# el modelo servido (registry o bundled, qué versión, qué stage, etc.).
_LOAD_INFO: dict | None = None


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


def _set_load_info(**kwargs) -> dict:
    """Registra los metadatos del último modelo cargado y devuelve el dict."""
    global _LOAD_INFO
    base: dict = {
        "source": None,
        "registry_uri": None,
        "version": None,
        "stage": None,
        "run_id": None,
        "path": None,
        "fallback_reason": None,
    }
    base.update(kwargs)
    _LOAD_INFO = base
    return base


def get_load_info() -> dict:
    """Devuelve los metadatos del modelo actualmente cargado.

    Si nunca se ha cargado, fuerza la carga (lo que rellenará ``_LOAD_INFO``).
    """
    if _LOAD_INFO is None:
        get_model()
    return _LOAD_INFO or {}


def _registry_cache_dir(uri: str) -> Path:
    """Carpeta donde se cachea una URI concreta del registry."""
    h = hashlib.md5(uri.encode("utf-8")).hexdigest()
    return _REGISTRY_CACHE_ROOT / h


def _resolve_model_version(uri: str) -> dict:
    """Para una URI ``models:/name/<stage|version>``, devuelve metadatos.

    Si la URI no es del tipo ``models:/`` (p. ej. ``runs:/...``), devuelve
    un diccionario vacío. Tolerante a fallos: nunca lanza, solo loguea.
    """
    if not uri.startswith("models:/"):
        return {}
    try:
        from mlflow.tracking import MlflowClient

        # "models:/<name>/<ref>" -> ["<name>", "<ref>"]
        parts = uri[len("models:/"):].split("/", 1)
        if len(parts) != 2:
            return {}
        name, ref = parts
        client = MlflowClient()
        if ref.isdigit():
            mv = client.get_model_version(name, ref)
        else:
            # Stage (Staging / Production / None / Archived) -> última versión.
            versions = client.get_latest_versions(name, stages=[ref])
            mv = versions[0] if versions else None
        if mv is None:
            return {}
        return {
            "name": mv.name,
            "version": int(mv.version),
            "stage": mv.current_stage,
            "run_id": mv.run_id,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("No se pudo resolver la versión del registry (%s).", exc)
        return {}


def _load_from_registry(uri: str):
    """Descarga (si no está cacheado) y carga el modelo del registry MLflow.

    Devuelve la tupla ``(pipeline, info)`` donde ``info`` describe el origen
    para reportarlo en ``/model-info``. Lanza si algo falla (auth, red...).
    """
    import mlflow  # import perezoso

    cache_dir = _registry_cache_dir(uri)
    sentinel = cache_dir / ".downloaded"

    if not sentinel.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Descargando modelo desde el registry: %s -> %s", uri, cache_dir)
        mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=str(cache_dir))
        sentinel.touch()
    else:
        logger.info("Modelo del registry ya cacheado en %s", cache_dir)

    pipeline = mlflow.sklearn.load_model(str(cache_dir))
    meta = _resolve_model_version(uri)
    info = {
        "source": "registry",
        "registry_uri": uri,
        "version": meta.get("version"),
        "stage": meta.get("stage"),
        "run_id": meta.get("run_id"),
        "path": str(cache_dir),
    }
    return pipeline, info


def _load_bundled():
    """Carga el ``.pkl`` versionado en el repositorio."""
    path = get_model_path()
    logger.info("Cargando modelo bundled (pickle local) desde %s", path)
    pipeline = load_best_model(path=path)
    info = {
        "source": "bundled",
        "registry_uri": None,
        "version": None,
        "stage": None,
        "run_id": None,
        "path": str(path),
    }
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
            pipeline, info = _load_from_registry(uri)
            _set_load_info(**info)
            return pipeline
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Carga desde el registry MLflow falló (%s). Usando el "
                "pickle local como respaldo.",
                exc,
            )
            pipeline, info = _load_bundled()
            info["fallback_reason"] = f"{type(exc).__name__}: {exc}"
            _set_load_info(**info)
            return pipeline

    # Camino feliz sin registry configurado.
    pipeline, info = _load_bundled()
    _set_load_info(**info)
    return pipeline


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
