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
   token expirado…), se cae automáticamente al pickle local
   (``models/best_model.pkl``), con un *warning* en logs.

Decisión técnica: la descarga del registry se hace con la **REST API
directa** de MLflow vía ``requests`` (sin la librería ``mlflow``). Razón:
en Render free (512 MB de RAM) la importación de ``mlflow`` o incluso
``mlflow-skinny`` añade ~50-150 MB de superficie de import que nos hace
rebasar el límite. Como solo necesitamos ``GET /api/2.0/mlflow/...`` y
descargar el pickle, ~30 líneas de cliente HTTP son equivalentes y mucho
más ligeras. El ``Pipeline`` se deserializa con ``joblib.load`` (el
mismo formato que usa ``mlflow.sklearn``). Tradeoff: si DagsHub cambia
sus endpoints REST tendríamos que tocar este código.

Variables de entorno
--------------------
- ``MLFLOW_MODEL_URI`` (opcional): URI tipo
  ``models:/pontia-cancellations/Production`` (recomendado) o
  ``models:/pontia-cancellations/3`` (versión fija).
- ``MLFLOW_TRACKING_URI`` / ``USERNAME`` / ``PASSWORD``: credenciales
  DagsHub. Solo se usan si ``MLFLOW_MODEL_URI`` está definida.
- ``PONTIA_MODEL_PATH`` (opcional): ruta absoluta a un ``.pkl`` para
  servirlo en lugar del bundled. Útil sobre todo en tests / despliegues
  custom.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
import requests

from src import config
from src.predict import load_best_model, predict_dataframe

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

# Directorio raíz para cachear modelos descargados del registry. ``/tmp`` es
# escribible en Render free y sobrevive a warm-restarts (pero NO a cold-starts:
# si Render destruye el contenedor por inactividad, la próxima carga re-descarga
# del registry, que es exactamente lo que queremos en ese caso).
_REGISTRY_CACHE_ROOT: Path = Path(os.getenv("PONTIA_REGISTRY_CACHE", "/tmp/pontia_models"))

@dataclass
class LoadInfo:
    """Metadatos tipados del último modelo cargado por ``get_model()``.

    Fuente única de las claves que reporta ``/model-info`` sobre el ORIGEN
    del modelo. Antes este dict se deletreaba a mano en tres sitios (con
    riesgo de claves mal escritas); ahora los loaders construyen y devuelven
    instancias de ``LoadInfo`` y ``get_load_info()`` lo serializa a ``dict``.
    """

    source: str | None = None
    registry_uri: str | None = None
    version: int | None = None
    stage: str | None = None
    run_id: str | None = None
    path: str | None = None
    fallback_reason: str | None = None


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

    Llama directamente a los endpoints HTTP del servidor MLflow (sin la
    librería ``mlflow``). Devuelve el dict de metadatos de la versión, que
    incluye ``source`` (URI del artefacto), ``version``, ``current_stage``
    y ``run_id``.

    Lanza ``RuntimeError`` si no se encuentra una versión en el stage pedido,
    o propaga ``requests.HTTPError`` si la auth/URL son inválidas.
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

    # Stage (Production, Staging, Archived). Pedimos el Registered Model
    # entero y filtramos su `latest_versions` por stage. Lo hacemos así
    # porque DagsHub no implementa `get-latest-versions` como endpoint REST.
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
    """Descarga un fichero concreto del artefacto del run vía ``requests``.

    ``source_uri`` es el campo ``source`` del model_version, con forma
    ``mlflow-artifacts:/<exp_id>/<run_id>/artifacts/model``. La URL de
    descarga es ``{tracking_uri}/api/2.0/mlflow-artifacts/artifacts/<path>/model.pkl``.
    """
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


def _load_from_registry(uri: str):
    """Carga el modelo desde el registry MLflow usando solo HTTP + joblib.

    Si el pickle ya está en la caché ``/tmp/pontia_models/<hash>`` (warm
    restart de Render), lo reutiliza. Si no, resuelve el URI vía REST,
    descarga ``model.pkl`` y lo deserializa con ``joblib.load``.

    Lanza si algo falla (auth, red, versión inexistente, etc.); el caller
    captura la excepción y cae al pickle bundled.
    """
    cache_dir = _registry_cache_dir(uri)
    pkl_path = cache_dir / "model.pkl"
    info_path = cache_dir / "info.json"

    if pkl_path.exists() and info_path.exists():
        logger.info("Modelo del registry ya cacheado en %s", cache_dir)
        cached = json.loads(info_path.read_text(encoding="utf-8"))
        info = LoadInfo(**cached)
        return joblib.load(pkl_path), info

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
            pipeline, info = _load_from_registry(uri)
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
