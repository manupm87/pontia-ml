"""Script principal: orquesta el flujo completo de entrenamiento.

Ejecuta, de principio a fin, las cuatro fases exigidas por el enunciado:

1. **Carga de datos**        (``data_loader``)
2. **Preprocesamiento**      (``preprocessing``, embebido en cada ``Pipeline``)
3. **Entrenamiento**         (``model_trainer``)
4. **Evaluación y selección**(``evaluator``)

Y persiste todos los artefactos: modelos individuales, mejor modelo, tabla de
métricas y gráficos (curva ROC, matrices de confusión e importancia de variables).

Uso::

    python -m src.train
"""

from __future__ import annotations

# Silenciar logs de TensorFlow antes de cualquier importación que lo cargue.
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import logging

import joblib

from . import config
from .data_loader import load_and_prepare
from .evaluator import Evaluator
from .model_trainer import ModelTrainer

logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    """Configura un formato de logging legible para todo el pipeline."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _df_to_markdown(df) -> str:
    """Convierte un DataFrame a tabla Markdown sin dependencias externas."""
    encabezado = "| Modelo | " + " | ".join(df.columns) + " |"
    separador = "|" + "---|" * (len(df.columns) + 1)
    filas = [
        "| " + idx + " | " + " | ".join(f"{v:.4f}" for v in fila) + " |"
        for idx, fila in zip(df.index, df.to_numpy())
    ]
    return "\n".join([encabezado, separador, *filas])


def run_pipeline() -> tuple:
    """Ejecuta el pipeline completo y devuelve ``(tabla_comparativa, mejor_modelo)``."""
    config.ensure_directories()

    # 1) Carga + limpieza + partición estratificada.
    logger.info("=== Fase 1: carga y preparación de datos ===")
    X_train, X_test, y_train, y_test = load_and_prepare()

    # 2 + 3) Construcción de pipelines (preprocesado + modelo) y entrenamiento.
    logger.info("=== Fase 2-3: entrenamiento de modelos ===")
    trainer = ModelTrainer()
    modelos = trainer.train(X_train, y_train)
    trainer.save_models()

    # 4) Evaluación sobre el conjunto de test.
    logger.info("=== Fase 4: evaluación y selección ===")
    evaluator = Evaluator()
    evaluator.evaluate(modelos, X_test, y_test)

    tabla = evaluator.comparison_table(trainer.train_times_)
    tabla.to_csv(config.METRICS_TABLE_PATH)
    md_path = config.OUTPUTS_DIR / "metricas_modelos.md"
    md_path.write_text(_df_to_markdown(tabla.round(4)) + "\n", encoding="utf-8")
    logger.info("Tabla de métricas guardada: %s", config.METRICS_TABLE_PATH)

    mejor = evaluator.select_best()

    # Visualizaciones exigidas por el enunciado.
    evaluator.plot_roc_curves(config.OUTPUTS_DIR / "roc_curves.png")
    evaluator.plot_confusion_matrices(config.OUTPUTS_DIR / "confusion_matrices.png")
    evaluator.plot_confusion_matrix(
        mejor, config.OUTPUTS_DIR / "confusion_matrix_best.png"
    )
    # Importancia de variables a partir del Random Forest (modelo interpretable).
    if "Random Forest" in modelos:
        evaluator.plot_feature_importance(
            modelos["Random Forest"], config.OUTPUTS_DIR / "feature_importance.png"
        )

    # Persistir el mejor modelo para producción / inferencia.
    joblib.dump(modelos[mejor], config.BEST_MODEL_PATH)
    logger.info("Mejor modelo guardado en: %s", config.BEST_MODEL_PATH)

    _print_summary(tabla, mejor)
    return tabla, mejor


def _print_summary(tabla, mejor: str) -> None:
    """Imprime un resumen final claro por consola."""
    print("\n" + "=" * 70)
    print("RESUMEN DE LA COMPARATIVA DE MODELOS")
    print("=" * 70)
    with __import__("pandas").option_context(
        "display.float_format", lambda v: f"{v:.4f}"
    ):
        print(tabla.to_string())
    print("-" * 70)
    print(f"Métrica principal de selección : {config.PRIMARY_METRIC}")
    print(f"MEJOR MODELO                   : {mejor}")
    print(f"Artefactos guardados en        : {config.OUTPUTS_DIR} y {config.MODELS_DIR}")
    print("=" * 70 + "\n")


def main() -> None:
    configure_logging()
    run_pipeline()


if __name__ == "__main__":
    main()
