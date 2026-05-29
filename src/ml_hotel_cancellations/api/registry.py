"""Cliente del Model Registry de MLflow vía REST (sin la librería ``mlflow``).

Resuelve y descarga el modelo publicado en el **Model Registry** de MLflow
(DagsHub) usando solo ``requests`` + ``joblib``. Razón de no usar la librería
``mlflow``: en Render free (512 MB de RAM) importarla añade ~50-150 MB de
superficie que rebasa el límite; aquí solo necesitamos un par de endpoints REST y
``joblib.load`` (el mismo formato que usa ``mlflow.sklearn``). Tradeoff: si DagsHub
cambia sus endpoints habría que tocar este módulo.

Variables de entorno (solo se usan si ``MLFLOW_MODEL_URI`` está definida):
``MLFLOW_TRACKING_URI`` / ``MLFLOW_TRACKING_USERNAME`` / ``MLFLOW_TRACKING_PASSWORD``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import requests

logger = logging.getLogger(__name__)

# Directorio raíz para cachear modelos descargados del registry. ``/tmp`` es
# escribible en Render free y sobrevive a warm-restarts (pero no a cold-starts:
# si el contenedor se destruye por inactividad, la próxima carga re-descarga).
_REGISTRY_CACHE_ROOT: Path = Path(os.getenv("PONTIA_REGISTRY_CACHE", "/tmp/pontia_models"))


@dataclass
class LoadInfo:
    """Metadatos tipados del origen del modelo cargado (lo reporta ``/model-info``).

    Fuente única de las claves de origen del modelo. Los loaders construyen y
    devuelven instancias de ``LoadInfo``; la capa de servicio las serializa a
    ``dict`` con ``dataclasses.asdict``.
    """

    source: str | None = None
    registry_uri: str | None = None
    version: int | None = None
    stage: str | None = None
    run_id: str | None = None
    path: str | None = None
    fallback_reason: str | None = None


def _registry_cache_dir(uri: str) -> Path:
    """Carpeta donde se cachea una URI concreta del registry."""
    h = hashlib.md5(uri.encode("utf-8")).hexdigest()
    return _REGISTRY_CACHE_ROOT / h


def _require_env(name: str) -> str:
    """Devuelve la variable de entorno ``name`` o lanza un error explicativo.

    Evita el ``KeyError`` opaco de ``os.environ[...]``: si falta una credencial
    de MLflow, el mensaje deja claro POR QUÉ se necesita, de modo que el
    ``fallback_reason`` de ``/model-info`` sea autoexplicativo.
    """
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"La variable de entorno {name} es requerida cuando "
            "MLFLOW_MODEL_URI está definida."
        )
    return value


def _mlflow_auth() -> tuple[str, str]:
    """Devuelve (usuario, token) para HTTP Basic contra DagsHub MLflow."""
    return (
        _require_env("MLFLOW_TRACKING_USERNAME"),
        _require_env("MLFLOW_TRACKING_PASSWORD"),
    )


def _resolve_registry_uri(uri: str) -> dict:
    """Resuelve ``models:/name/<stage|version>`` contra la API REST MLflow.

    Devuelve el dict de metadatos de la versión (``source``, ``version``,
    ``current_stage``, ``run_id``). Lanza ``RuntimeError`` si no hay versión en el
    stage pedido, o propaga ``requests.HTTPError`` si la auth/URL son inválidas.
    """
    if not uri.startswith("models:/"):
        raise ValueError(f"URI no soportada para el registry: {uri!r}")
    parts = uri[len("models:/"):].split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"URI mal formada (esperado models:/<name>/<ref>): {uri!r}")
    name, ref = parts

    tracking_uri = _require_env("MLFLOW_TRACKING_URI").rstrip("/")
    auth = _mlflow_auth()

    if ref.isdigit():
        # Versión concreta (p. ej. models:/pontia-cancellations/3).
        resp = requests.get(
            f"{tracking_uri}/api/2.0/mlflow/model-versions/get",
            params={"name": name, "version": ref},
            auth=auth,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()["model_version"]

    # Stage (Production, Staging, Archived). Pedimos el Registered Model entero y
    # filtramos su `latest_versions` por stage (DagsHub no expone
    # `get-latest-versions` como endpoint REST).
    resp = requests.get(
        f"{tracking_uri}/api/2.0/mlflow/registered-models/get",
        params={"name": name},
        auth=auth,
        timeout=15,
    )
    resp.raise_for_status()
    versions = resp.json().get("registered_model", {}).get("latest_versions", [])
    for mv in versions:
        if mv.get("current_stage") == ref:
            return mv
    raise RuntimeError(
        f"No hay ninguna versión en stage {ref!r} para el modelo {name!r}."
    )


def _download_artifact_file(source_uri: str, dest_pkl: Path) -> None:
    """Descarga ``model.pkl`` del artefacto del run vía ``requests`` (streaming)."""
    if not source_uri.startswith("mlflow-artifacts:"):
        raise ValueError(f"Esquema de artefacto no soportado: {source_uri!r}")
    artifact_path = source_uri.split("mlflow-artifacts:/", 1)[1]
    tracking_uri = _require_env("MLFLOW_TRACKING_URI").rstrip("/")
    url = (
        f"{tracking_uri}/api/2.0/mlflow-artifacts/artifacts/"
        f"{artifact_path}/model.pkl"
    )

    dest_pkl.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Descargando modelo del registry: %s", url)
    with requests.get(url, auth=_mlflow_auth(), timeout=120, stream=True) as resp:
        resp.raise_for_status()
        with open(dest_pkl, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB
                f.write(chunk)


def load_from_registry(uri: str) -> tuple[object, LoadInfo]:
    """Carga ``(pipeline, LoadInfo)`` desde el registry usando HTTP + joblib.

    Si el pickle ya está cacheado (warm restart), lo reutiliza; si no, resuelve el
    URI vía REST, descarga ``model.pkl`` y lo deserializa. Lanza si algo falla
    (auth, red, versión inexistente, etc.); el caller cae al pickle bundled.
    """
    cache_dir = _registry_cache_dir(uri)
    pkl_path = cache_dir / "model.pkl"
    info_path = cache_dir / "info.json"

    if pkl_path.exists() and info_path.exists():
        logger.info("Modelo del registry ya cacheado en %s", cache_dir)
        cached = json.loads(info_path.read_text(encoding="utf-8"))
        return joblib.load(pkl_path), LoadInfo(**cached)

    mv = _resolve_registry_uri(uri)
    _download_artifact_file(mv["source"], pkl_path)
    info = LoadInfo(
        source="registry",
        registry_uri=uri,
        version=int(mv["version"]),
        stage=mv.get("current_stage"),
        run_id=mv.get("run_id"),
        path=str(cache_dir),
    )
    info_path.write_text(json.dumps(asdict(info)), encoding="utf-8")
    return joblib.load(pkl_path), info
