"""Registro de experimentos con MLflow (bonus técnico).

Helpers finos para loguear params/métricas/artefactos a MLflow sin acoplar el
código de modelado: si faltan las env vars (``MLFLOW_TRACKING_URI`` /
``MLFLOW_TRACKING_USERNAME`` / ``MLFLOW_TRACKING_PASSWORD``) o ``mlflow`` no está
instalado, todo el tracking se salta de forma silenciosa (no-op).
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)

# Nombre del experimento por defecto. Se puede sobrescribir desde el
# llamador o con la variable de entorno ``MLFLOW_EXPERIMENT_NAME``.
DEFAULT_EXPERIMENT: str = "pontia-cancellations"

# Flag de estado interno. ``init_tracking`` lo pone a True solo si todo
# (env vars + import de mlflow + ``set_tracking_uri``) salió bien.
_ENABLED: bool = False


def _has_required_env() -> bool:
    """Comprueba si las tres variables MLflow están definidas."""
    return all(
        os.getenv(var)
        for var in (
            "MLFLOW_TRACKING_URI",
            "MLFLOW_TRACKING_USERNAME",
            "MLFLOW_TRACKING_PASSWORD",
        )
    )


def init_tracking(experiment: str = DEFAULT_EXPERIMENT) -> bool:
    """Configura MLflow contra el servidor remoto, si es posible (idempotente).

    Devuelve ``True`` si el tracking quedó activo; ``False`` si se omitió.
    """
    global _ENABLED

    if not _has_required_env():
        logger.info(
            "MLflow desactivado: no se encontraron MLFLOW_TRACKING_URI/USERNAME/"
            "PASSWORD. Los scripts seguirán funcionando sin loguear runs."
        )
        _ENABLED = False
        return False

    try:
        import mlflow  # import perezoso: no es dependencia de runtime
    except ImportError:
        logger.warning(
            "MLflow desactivado: la librería no está instalada. Ejecuta "
            "`pip install -r requirements-train.txt` para activarlo."
        )
        _ENABLED = False
        return False

    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", experiment)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    _ENABLED = True
    logger.info(
        "MLflow activo. tracking_uri=%s experimento=%s",
        tracking_uri,
        experiment_name,
    )
    return True


def tracking_enabled() -> bool:
    """Indica si el tracking quedó configurado tras ``init_tracking``."""
    return _ENABLED


@contextmanager
def start_run(run_name: str | None = None, nested: bool | None = None) -> Iterator[object]:
    """Gestor de contexto sustituto de ``mlflow.start_run``; no-op si el tracking no está activo.

    Con ``nested=None`` se autodetecta: anida si ya hay un run activo, si no abre uno raíz.
    """
    if not _ENABLED:
        yield None
        return

    import mlflow

    if nested is None:
        nested = mlflow.active_run() is not None

    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        yield run


def log_params(params: dict) -> None:
    """Loguea un diccionario de hiperparámetros, si el tracking está activo."""
    if not _ENABLED or not params:
        return
    import mlflow

    # MLflow no acepta valores no-stringificables; los serializamos primero.
    safe = {k: _stringify(v) for k, v in params.items()}
    mlflow.log_params(safe)


def log_metrics(metrics: dict, step: int | None = None) -> None:
    """Loguea métricas (float), si el tracking está activo."""
    if not _ENABLED or not metrics:
        return
    import mlflow

    safe = {k: float(v) for k, v in metrics.items() if v is not None}
    mlflow.log_metrics(safe, step=step)


def set_tags(tags: dict) -> None:
    """Adjunta tags al run actual (etiquetas libres), si está activo."""
    if not _ENABLED or not tags:
        return
    import mlflow

    mlflow.set_tags({k: _stringify(v) for k, v in tags.items()})


def log_artifact(path) -> None:
    """Sube un fichero como artefacto del run, si el tracking está activo."""
    if not _ENABLED:
        return
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        logger.warning("Artefacto no encontrado, no se sube: %s", p)
        return

    import mlflow

    mlflow.log_artifact(str(p))


def log_sklearn_model(pipeline, artifact_path: str = "model") -> None:
    """Loguea un ``Pipeline`` de scikit-learn como artefacto MLflow.

    Acepta también pipelines con XGBoost dentro (los loguea como sklearn,
    porque están envueltos en ``Pipeline``). Si quieres registrar el modelo
    en el *registry*, llama después a ``mlflow.register_model(...)`` con el
    URI ``runs:/<run_id>/<artifact_path>``.
    """
    if not _ENABLED:
        return
    import mlflow.sklearn

    mlflow.sklearn.log_model(pipeline, artifact_path=artifact_path)


def _stringify(value) -> str:
    """Convierte cualquier valor a string para que MLflow lo acepte."""
    if value is None:
        return "null"
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return repr(value)
