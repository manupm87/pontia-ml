"""Script principal: orquesta carga, entrenamiento, evaluación y selección,
y persiste los artefactos (modelos, mejor modelo, tabla de métricas y gráficos).

Uso::

    python -m ml_hotel_cancellations.ml.train
"""

from __future__ import annotations

# Silenciar logs de TensorFlow antes de cualquier importación que lo cargue.
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import logging

import joblib
import pandas as pd

from ml_hotel_cancellations import config
from ml_hotel_cancellations.utils import tracking
from .data_loader import load_and_prepare
from .evaluator import Evaluator
from .model_trainer import ModelTrainer
from ml_hotel_cancellations.utils.reporting import df_to_markdown

logger = logging.getLogger(__name__)


# Familia de cada modelo; fuente única en `config` (compartida con tuning/balancing).
_MODEL_FAMILY: dict[str, str] = config.MODEL_FAMILY


def _resolve_param_overrides(tune: bool, X_train, y_train) -> dict | None:
    """Hiperparámetros a usar: si ``tune``, busca y devuelve los óptimos;
    si no, carga los previamente guardados (o ``None``).
    """
    from .tuning import HyperparameterTuner, load_best_params

    if tune:
        logger.info("=== Fase 1.5: optimización de hiperparámetros (--tune) ===")
        tuner = HyperparameterTuner()
        # `tune()` ya persiste resultados + mejores params (quedan por defecto).
        tuner.tune(X_train, y_train)
        return tuner.best_params_

    param_overrides = load_best_params() or None
    if param_overrides:
        logger.info(
            "Usando hiperparámetros optimizados guardados en %s",
            config.BEST_PARAMS_PATH,
        )
    return param_overrides


def _write_metric_tables(table) -> None:
    """Persiste la tabla comparativa de métricas en CSV y Markdown."""
    table.to_csv(config.METRICS_TABLE_PATH)
    md_path = config.OUTPUTS_DIR / "metricas_modelos.md"
    md_path.write_text(df_to_markdown(table.round(4)) + "\n", encoding="utf-8")
    logger.info("Tabla de métricas guardada: %s", config.METRICS_TABLE_PATH)


def _generate_plots(evaluator: Evaluator, models: dict, best: str) -> None:
    """Genera las visualizaciones exigidas por el enunciado."""
    evaluator.plot_roc_curves(config.OUTPUTS_DIR / "roc_curves.png")
    evaluator.plot_confusion_matrices(config.OUTPUTS_DIR / "confusion_matrices.png")
    evaluator.plot_confusion_matrix(
        best, config.OUTPUTS_DIR / "confusion_matrix_best.png"
    )
    # Importancia de variables del Random Forest (modelo interpretable).
    if "Random Forest" in models:
        evaluator.plot_feature_importance(
            models["Random Forest"], config.OUTPUTS_DIR / "feature_importance.png"
        )


def run_pipeline(tune: bool = False) -> tuple:
    """Ejecuta el pipeline completo y devuelve ``(tabla_comparativa, mejor_modelo)``.

    Con ``tune=True`` optimiza hiperparámetros por CV antes de entrenar.
    """
    config.ensure_directories()
    # MLflow: si no hay credenciales, los `tracking.*` son no-op.
    tracking.init_tracking("pontia-cancellations-train")

    # 1) Carga + limpieza + partición estratificada.
    logger.info("=== Fase 1: carga y preparación de datos ===")
    X_train, X_test, y_train, y_test = load_and_prepare()

    # 1.5) Hiperparámetros: buscados con --tune, o los previamente guardados.
    param_overrides = _resolve_param_overrides(tune, X_train, y_train)

    # 2 + 3) Construcción de pipelines (preprocesado + modelo) y entrenamiento.
    logger.info("=== Fase 2-3: entrenamiento de modelos ===")
    trainer = ModelTrainer(param_overrides=param_overrides)
    models = trainer.train(X_train, y_train)
    trainer.save_models()

    # 4) Evaluación sobre el conjunto de test.
    logger.info("=== Fase 4: evaluación y selección ===")
    evaluator = Evaluator()
    evaluator.evaluate(models, X_test, y_test)

    table = evaluator.comparison_table(trainer.train_times_)
    _write_metric_tables(table)

    best = evaluator.select_best()

    # Visualizaciones exigidas por el enunciado.
    _generate_plots(evaluator, models, best)

    # Persistir el mejor modelo para producción / inferencia.
    joblib.dump(models[best], config.BEST_MODEL_PATH)
    logger.info("Mejor modelo guardado en: %s", config.BEST_MODEL_PATH)

    # MLflow: registra el experimento como un árbol de runs (padre + un child por
    # modelo). No-op si el tracking no está activo.
    _log_training_run(trainer, evaluator, models, table, best, tuned=tune)

    _print_summary(table, best)
    return table, best


def _log_training_run(
    trainer: ModelTrainer,
    evaluator: Evaluator,
    models: dict,
    table,
    best: str,
    *,
    tuned: bool,
) -> None:
    """Publica el experimento de entrenamiento completo en MLflow."""
    if not tracking.tracking_enabled():
        return

    with tracking.start_run(run_name="train_all_models"):
        # Run padre: contexto del experimento entero.
        tracking.set_tags(
            {
                "phase": "training",
                "tuned": tuned,
                "n_models": len(models),
                "best_model": best,
                "primary_metric": config.PRIMARY_METRIC,
                "random_state": config.RANDOM_STATE,
            }
        )
        tracking.log_metrics(
            {
                f"best_{config.PRIMARY_METRIC}": float(
                    table.loc[best, config.PRIMARY_METRIC]
                ),
            }
        )

        # Un child run por modelo, con sus params + métricas + tiempo.
        for name, res in evaluator.results_.items():
            with tracking.start_run(run_name=name, nested=True):
                tracking.set_tags(
                    {
                        "model_family": _MODEL_FAMILY.get(name, "other"),
                        "best": name == best,
                    }
                )
                # Params del estimador (sin el preprocesador).
                model_params = trainer.models_[name].named_steps["model"].get_params()
                tracking.log_params(model_params)
                tracking.log_metrics(res["metrics"])
                tracking.log_metrics(
                    {"train_time_s": float(trainer.train_times_.get(name, 0.0))}
                )

        # Artefactos comunes (las 5 figuras y la tabla CSV/markdown).
        for artifact in (
            config.METRICS_TABLE_PATH,
            config.OUTPUTS_DIR / "metricas_modelos.md",
            config.OUTPUTS_DIR / "roc_curves.png",
            config.OUTPUTS_DIR / "confusion_matrices.png",
            config.OUTPUTS_DIR / "confusion_matrix_best.png",
            config.OUTPUTS_DIR / "feature_importance.png",
        ):
            tracking.log_artifact(artifact)

        # Modelo ganador al run padre como `model/` (registrable luego con
        # `mlflow.register_model(...)`).
        tracking.log_sklearn_model(models[best], artifact_path="model")
        logger.info(
            "MLflow: experimento publicado (mejor=%s, %s=%.4f).",
            best,
            config.PRIMARY_METRIC,
            table.loc[best, config.PRIMARY_METRIC],
        )


def _print_summary(table, best: str) -> None:
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


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline de entrenamiento y comparación de modelos.")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Optimiza hiperparámetros por validación cruzada antes de entrenar (más lento).",
    )
    args = parser.parse_args()

    config.configure_logging()
    run_pipeline(tune=args.tune)


if __name__ == "__main__":
    main()
