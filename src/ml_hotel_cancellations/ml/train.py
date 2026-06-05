"""Pipeline de entrenamiento: carga datos, entrena los 5 modelos, los evalúa,
elige el mejor y guarda los artefactos (modelos, tabla de métricas y gráficos).

Uso::

    python -m ml_hotel_cancellations.ml.train          # entrena con params por defecto
    python -m ml_hotel_cancellations.ml.train --tune   # optimiza hiperparámetros antes
"""

from __future__ import annotations

import argparse
import logging

import joblib
import pandas as pd

from ml_hotel_cancellations import config
from ml_hotel_cancellations.utils import tracking
from ml_hotel_cancellations.utils.reporting import df_to_markdown
from .data_loader import load_and_prepare
from .evaluate import (
    comparison_table,
    evaluate_models,
    plot_confusion_matrices,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curves,
    select_best,
)
from .models import build_models, save_models, train_models

logger = logging.getLogger(__name__)

# Familia de cada modelo; fuente única en `config` (compartida con tuning).
_MODEL_FAMILY: dict[str, str] = config.MODEL_FAMILY


def _resolve_overrides(tune: bool, X_train, y_train) -> dict | None:
    """Hiperparámetros a usar: los buscados con ``--tune`` o los ya guardados."""
    from .tuning import load_best_params, tune as run_tuning

    if tune:
        logger.info("=== Optimización de hiperparámetros (--tune) ===")
        best_params, _ = run_tuning(X_train, y_train)  # persiste resultados + mejores params
        return best_params

    overrides = load_best_params() or None
    if overrides:
        logger.info("Usando hiperparámetros optimizados de %s", config.BEST_PARAMS_PATH)
    return overrides


def _write_metric_tables(table: pd.DataFrame) -> None:
    """Guarda la tabla comparativa de métricas en CSV y Markdown."""
    table.to_csv(config.METRICS_TABLE_PATH)
    md_path = config.OUTPUTS_DIR / "metricas_modelos.md"
    md_path.write_text(df_to_markdown(table.round(4)) + "\n", encoding="utf-8")
    logger.info("Tabla de métricas guardada: %s", config.METRICS_TABLE_PATH)


def _generate_plots(results: dict, fitted: dict, y_test, best: str) -> None:
    """Genera las visualizaciones exigidas por el enunciado."""
    plot_roc_curves(results, y_test, config.OUTPUTS_DIR / "roc_curves.png")
    plot_confusion_matrices(results, y_test, config.OUTPUTS_DIR / "confusion_matrices.png")
    plot_confusion_matrix(results, best, y_test, config.OUTPUTS_DIR / "confusion_matrix_best.png")
    if "Random Forest" in fitted:  # modelo interpretable
        plot_feature_importance(fitted["Random Forest"], config.OUTPUTS_DIR / "feature_importance.png")


def _log_to_mlflow(
    fitted: dict, table: pd.DataFrame, best: str, *, tuned: bool, input_example=None
) -> str | None:
    """Publica el run del mejor modelo en MLflow y devuelve su ``run_id``.

    Devuelve ``None`` si el tracking no está activo (no-op). ``input_example``
    (unas filas de features crudas) se usa para adjuntar la *signature* del modelo.
    """
    if not tracking.tracking_enabled():
        return None
    run_id: str | None = None
    with tracking.start_run(run_name="train_all_models") as run:
        run_id = run.info.run_id if run is not None else None
        tracking.set_tags({"tuned": tuned, "best_model": best, "primary_metric": config.PRIMARY_METRIC})
        # Params y métricas del modelo ganador (la tabla con todos va como artefacto).
        tracking.log_params(fitted[best].named_steps["model"].get_params())
        tracking.log_metrics({m: float(table.loc[best, m]) for m in config.METRIC_NAMES})
        for artifact in (
            config.METRICS_TABLE_PATH,
            config.OUTPUTS_DIR / "metricas_modelos.md",
            config.OUTPUTS_DIR / "roc_curves.png",
            config.OUTPUTS_DIR / "confusion_matrices.png",
            config.OUTPUTS_DIR / "confusion_matrix_best.png",
            config.OUTPUTS_DIR / "feature_importance.png",
        ):
            tracking.log_artifact(artifact)
        tracking.log_sklearn_model(fitted[best], input_example=input_example, artifact_path="model")
        logger.info("MLflow: experimento publicado (mejor=%s, run_id=%s).", best, run_id)
    return run_id


def _maybe_register_best(run_id: str | None, table: pd.DataFrame, best: str) -> None:
    """Registra el ganador en el Model Registry SOLO si supera al champion actual.

    Pensado para `train --register`: encadena entrenamiento y registro en un único
    paso, pero validado por métrica (no promueve un modelo peor que el de Production).
    """
    if not tracking.tracking_enabled():
        logger.warning(
            "--register ignorado: MLflow no está activo "
            "(faltan MLFLOW_TRACKING_URI/USERNAME/PASSWORD)."
        )
        return
    if run_id is None:
        logger.warning("--register: no se obtuvo run_id; se omite el registro.")
        return

    from ..utils.register_model import register_if_better

    metric = config.PRIMARY_METRIC
    register_if_better(
        run_id=run_id, metric_name=metric, metric_value=float(table.loc[best, metric])
    )


def _print_summary(table: pd.DataFrame, best: str) -> None:
    """Imprime un resumen final claro por consola."""
    print("\n" + "=" * 70)
    print("RESUMEN DE LA COMPARATIVA DE MODELOS")
    print("=" * 70)
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(table.to_string())
    print("-" * 70)
    print(f"Métrica principal de selección : {config.PRIMARY_METRIC}")
    print(f"MEJOR MODELO                   : {best}")
    print(f"Artefactos guardados en        : {config.OUTPUTS_DIR} y {config.MODELS_DIR}")
    print("=" * 70 + "\n")


def run_pipeline(tune: bool = False, register: bool = False) -> tuple[pd.DataFrame, str]:
    """Ejecuta el pipeline completo y devuelve ``(tabla_comparativa, mejor_modelo)``.

    Con ``tune=True`` optimiza hiperparámetros por CV antes de entrenar. Con
    ``register=True`` registra el ganador en el Model Registry si supera al
    champion actual (requiere MLflow configurado).
    """
    config.ensure_directories()
    tracking.init_tracking("pontia-cancellations-train")  # no-op sin credenciales

    # 1) Carga + limpieza + partición estratificada.
    logger.info("=== Carga y preparación de datos ===")
    X_train, X_test, y_train, y_test = load_and_prepare()
    overrides = _resolve_overrides(tune, X_train, y_train)

    # 2) Entrenamiento de los 5 modelos (mismo preprocesador para todos).
    logger.info("=== Entrenamiento de modelos ===")
    models = build_models(overrides=overrides)
    fitted, train_times = train_models(models, X_train, y_train)
    save_models(fitted)

    # 3) Evaluación sobre test, tabla comparativa y selección del mejor.
    logger.info("=== Evaluación y selección ===")
    results = evaluate_models(fitted, X_test, y_test)
    table = comparison_table(results, train_times)
    _write_metric_tables(table)
    best = select_best(results)

    # 4) Visualizaciones + guardado del mejor modelo + logging en MLflow.
    _generate_plots(results, fitted, y_test, best)
    joblib.dump(fitted[best], config.BEST_MODEL_PATH)
    logger.info("Mejor modelo guardado en: %s", config.BEST_MODEL_PATH)
    # Ejemplo de entrada (features crudas) para adjuntar la signature al modelo.
    run_id = _log_to_mlflow(fitted, table, best, tuned=tune, input_example=X_train.head(5))

    # 5) (Opcional) registro validado del ganador en el Model Registry.
    if register:
        _maybe_register_best(run_id, table, best)

    _print_summary(table, best)
    return table, best


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento y comparación de modelos.")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Optimiza hiperparámetros por validación cruzada antes de entrenar (más lento).",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help=(
            "Tras entrenar, registra el modelo ganador en el Model Registry de MLflow "
            "y lo pasa a Production SOLO si su métrica principal iguala o supera a la "
            "del modelo actual en Production (requiere credenciales MLflow)."
        ),
    )
    args = parser.parse_args()

    config.configure_logging()
    run_pipeline(tune=args.tune, register=args.register)


if __name__ == "__main__":
    main()
