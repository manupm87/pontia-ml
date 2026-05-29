"""Registro del modelo ganador en el *Model Registry* de MLflow (CLI).

Workaround para DagsHub: su UI de MLflow **oculta los botones "Register
Model" y "Transition Stage"** (es una limitación conocida del fork del
frontend), aunque el registry sí funciona vía API. Este script registra
y promociona el modelo desde Python en su lugar.

Flujo típico
------------
1. Has ejecutado ``python -m ml_hotel_cancellations.ml.train`` con las variables MLflow
   exportadas. En DagsHub aparece un run padre llamado
   ``train_all_models`` que contiene el artefacto ``model/`` (el
   ``Pipeline`` ganador).
2. Ejecutas este script: encuentra el último run con ese nombre, crea
   (o reusa) el ``Registered Model`` ``pontia-cancellations``, le añade
   una versión nueva apuntando a ``runs:/<run_id>/model`` y la
   transiciona al stage indicado (``Production`` por defecto).

Ejemplos::

    # Lo más habitual: registra el último run y lo promociona a Production.
    python -m ml_hotel_cancellations.utils.register_model

    # Registrar un run concreto.
    python -m ml_hotel_cancellations.utils.register_model --run-id abc123def456

    # Solo registrar (no transicionar de stage).
    python -m ml_hotel_cancellations.utils.register_model --stage none

    # Cambiar de nombre.
    python -m ml_hotel_cancellations.utils.register_model --name pontia-cancellations-experimental
"""

from __future__ import annotations

import argparse
import logging

from ml_hotel_cancellations import config
from . import tracking

logger = logging.getLogger(__name__)

# Por defecto buscamos el run padre que crea `src.train`. Los hijos
# (Logistic Regression, XGBoost...) NO se registran como Registered
# Models, solo el padre que contiene el modelo ganador.
DEFAULT_RUN_NAME: str = "train_all_models"
DEFAULT_EXPERIMENT: str = "pontia-cancellations-train"
DEFAULT_MODEL_NAME: str = "pontia-cancellations"
DEFAULT_STAGE: str = "Production"


def _find_latest_run_id(experiment_name: str, run_name: str) -> str:
    """Devuelve el ID del run más reciente cuyo nombre coincide.

    Lanza ``RuntimeError`` con un mensaje útil si no encuentra nada (la
    conversión a ``SystemExit`` se hace solo en ``main()``).
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(
            f"❌ No existe el experimento '{experiment_name}' en "
            f"{mlflow.get_tracking_uri()}. ¿Ejecutaste primero "
            "`python -m ml_hotel_cancellations.ml.train`?"
        )
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(
            f"❌ No se encontraron runs llamados '{run_name}' en el "
            f"experimento '{experiment_name}'. Lanza `python -m ml_hotel_cancellations.ml.train` "
            "para generar uno."
        )
    run = runs[0]
    logger.info(
        "Run encontrado: %s (start_time=%s)",
        run.info.run_id,
        run.info.start_time,
    )
    return run.info.run_id


def register_model(
    run_id: str | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    stage: str | None = DEFAULT_STAGE,
    experiment_name: str = DEFAULT_EXPERIMENT,
    artifact_path: str = "model",
) -> "mlflow.entities.model_registry.ModelVersion":
    """Registra ``runs:/<run_id>/<artifact_path>`` como nueva versión.

    Si ``run_id`` es ``None``, busca el último run llamado
    ``train_all_models`` en ``experiment_name``.

    Returns
    -------
    mlflow.entities.model_registry.ModelVersion
        La versión creada (útil si lo llamas desde otro script).
    """
    if not tracking.init_tracking(experiment_name):
        raise RuntimeError(
            "❌ MLflow no está configurado. Asegúrate de tener las "
            "variables MLFLOW_TRACKING_URI/USERNAME/PASSWORD definidas "
            "(p. ej. con `set -a; source .env; set +a`)."
        )

    import mlflow
    from mlflow.tracking import MlflowClient

    if run_id is None:
        run_id = _find_latest_run_id(experiment_name, DEFAULT_RUN_NAME)

    model_uri = f"runs:/{run_id}/{artifact_path}"
    logger.info("Registrando %s como '%s'...", model_uri, model_name)

    # `register_model` crea el Registered Model si no existe y añade una
    # versión nueva. Si ya existía, simplemente añade la versión siguiente.
    version = mlflow.register_model(model_uri=model_uri, name=model_name)
    logger.info(
        "✅ Registrado '%s' versión %s (status=%s).",
        model_name,
        version.version,
        version.status,
    )

    if stage and stage.lower() != "none":
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage=stage,
            # Archivamos cualquier versión que estuviera en el mismo stage,
            # para que solo haya UNA "Production" a la vez.
            archive_existing_versions=True,
        )
        logger.info(
            "🚀 Versión %s movida a stage '%s' (versiones anteriores en ese "
            "stage quedan archivadas).",
            version.version,
            stage,
        )

    return version


def main() -> None:
    config.configure_logging()
    parser = argparse.ArgumentParser(
        description=(
            "Registra el modelo entrenado en el Model Registry de MLflow "
            "(workaround para la UI de DagsHub, que no expone el botón)."
        ),
    )
    parser.add_argument(
        "--run-id",
        help=(
            "ID del run cuyo artefacto `model/` se registrará. Si se omite, "
            "se usa el último run con nombre 'train_all_models'."
        ),
    )
    parser.add_argument(
        "--name",
        default=DEFAULT_MODEL_NAME,
        help=f"Nombre del Registered Model (por defecto: '{DEFAULT_MODEL_NAME}').",
    )
    parser.add_argument(
        "--stage",
        default=DEFAULT_STAGE,
        help=(
            "Stage destino. Valores válidos: 'Staging', 'Production', "
            "'Archived', o 'none' para registrar sin transicionar. "
            f"Por defecto '{DEFAULT_STAGE}'."
        ),
    )
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        help=f"Nombre del experimento (por defecto '{DEFAULT_EXPERIMENT}').",
    )
    args = parser.parse_args()

    # Las funciones reutilizables lanzan excepciones de dominio (RuntimeError);
    # aquí, en el CLI, las convertimos en SystemExit con el mensaje útil.
    try:
        register_model(
            run_id=args.run_id,
            model_name=args.name,
            stage=args.stage,
            experiment_name=args.experiment,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
