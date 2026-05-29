"""Balanceo de clases (bonus técnico).

Compara tres estrategias para tratar el desbalance (~37 % de cancelaciones) en
los modelos clásicos:

- **baseline**: sin balanceo.
- **class_weight**: reponderar la clase minoritaria (``class_weight='balanced'``
  en scikit-learn; ``scale_pos_weight`` en XGBoost).
- **SMOTE**: generar ejemplos sintéticos de la clase minoritaria
  (*imbalanced-learn*), aplicado **solo al entrenamiento**.

Idea clave: como la métrica principal (**ROC-AUC**) es independiente del umbral,
el balanceo apenas la cambia. Su efecto real es **subir el recall** (detectar más
cancelaciones) a costa de algo de precisión. Esta comparación lo evidencia.

Para aislar el efecto del balanceo, se usan los hiperparámetros **base** (no los
optimizados), de modo que la única variable que cambia entre filas es la
estrategia de balanceo.

Uso::

    python -m src.balancing
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml_hotel_cancellations import config
from ml_hotel_cancellations.utils import tracking
from .preprocessing import build_preprocessor, make_pipeline
from ml_hotel_cancellations.utils.reporting import df_to_markdown, save_figure

logger = logging.getLogger(__name__)

ESTRATEGIAS = ["baseline", "class_weight", "SMOTE"]


def _make_estimators(strategy: str, pos_weight: float) -> dict:
    """Crea los 4 modelos clásicos con la estrategia de balanceo indicada."""
    from .model_factory import build_classic_estimators

    # class_weight='balanced' para sklearn; scale_pos_weight para XGBoost.
    cw = "balanced" if strategy == "class_weight" else None
    spw = pos_weight if strategy == "class_weight" else 1.0

    return build_classic_estimators(class_weight=cw, scale_pos_weight=spw)


def _build_pipeline(strategy: str, estimator):
    """Pipeline preprocesado + (SMOTE) + modelo.

    Para SMOTE se usa el ``Pipeline`` de *imbalanced-learn*, que aplica el
    sobremuestreo **solo durante el entrenamiento** (nunca al evaluar), evitando
    así inflar artificialmente las métricas de test.
    """
    if strategy == "SMOTE":
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline

        return ImbPipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("smote", SMOTE(random_state=config.RANDOM_STATE)),
                ("model", estimator),
            ]
        )
    return make_pipeline(estimator)


def compare(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """Entrena cada (modelo × estrategia) y devuelve sus métricas de test."""
    n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
    pos_weight = n_neg / n_pos  # para scale_pos_weight de XGBoost

    filas = []
    for strategy in ESTRATEGIAS:
        for nombre, estimator in _make_estimators(strategy, pos_weight).items():
            pipe = _build_pipeline(strategy, estimator)
            pipe.fit(X_train, y_train)
            proba = pipe.predict_proba(X_test)[:, 1]
            pred = (proba >= config.DECISION_THRESHOLD).astype(int)
            filas.append(
                {
                    "modelo": nombre,
                    "estrategia": strategy,
                    "accuracy": accuracy_score(y_test, pred),
                    "precision": precision_score(y_test, pred),
                    "recall": recall_score(y_test, pred),
                    "f1": f1_score(y_test, pred),
                    "roc_auc": roc_auc_score(y_test, proba),
                }
            )
            logger.info(
                "  %-20s | %-12s | recall=%.3f precision=%.3f roc_auc=%.3f",
                nombre,
                strategy,
                filas[-1]["recall"],
                filas[-1]["precision"],
                filas[-1]["roc_auc"],
            )
    return pd.DataFrame(filas)


def save_results(df: pd.DataFrame) -> None:
    """Guarda la tabla comparativa (Markdown) y un gráfico del efecto en XGBoost."""
    texto = [
        "# Balanceo de clases — comparación de estrategias\n",
        "Desbalance del problema: ~37 % de cancelaciones. Métricas sobre el conjunto "
        "de test, con hiperparámetros base (para aislar el efecto del balanceo).\n",
        df_to_markdown(df.round(3), index_label=None, float_fmt="{:.3f}"),
        "\n**Lectura.** El balanceo (class_weight o SMOTE) **sube el recall** "
        "—se detectan más cancelaciones— a costa de **bajar la precisión**, mientras "
        "que el **ROC-AUC apenas cambia** (es independiente del umbral). Es decir, no "
        "hace al modelo \"mejor\" en capacidad de ordenar, pero sí desplaza el "
        "compromiso hacia detectar más positivos, útil si al hotel le cuesta más una "
        "cancelación no detectada que una falsa alarma.\n",
    ]
    config.BALANCING_RESULTS_PATH.write_text("\n".join(texto) + "\n", encoding="utf-8")
    logger.info("Tabla de balanceo guardada en: %s", config.BALANCING_RESULTS_PATH)

    # Gráfico: efecto en XGBoost (recall, precision, f1, roc_auc por estrategia).
    sub = df[df["modelo"] == "XGBoost"].set_index("estrategia")
    metr = ["recall", "precision", "f1", "roc_auc"]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(metr))
    width = 0.25
    colores = {"baseline": "#9ecae1", "class_weight": "#fdae6b", "SMOTE": "#a1d99b"}
    for i, estrategia in enumerate(ESTRATEGIAS):
        ax.bar(x + (i - 1) * width, sub.loc[estrategia, metr].values, width,
               label=estrategia, color=colores[estrategia], edgecolor="k", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(metr)
    ax.set_ylim(0, 1); ax.set_ylabel("valor (test)")
    ax.set_title("Efecto del balanceo en XGBoost\n(sube recall, baja precisión; ROC-AUC casi igual)")
    ax.legend(title="estrategia"); ax.grid(axis="y", alpha=0.3)
    save_figure(fig, config.BALANCING_PLOT_PATH, bbox_inches="tight")


# Fuente única de verdad en `config` (compartida con train/tuning).
_MODEL_FAMILY: dict[str, str] = config.MODEL_FAMILY

_METRIC_COLS: tuple[str, ...] = ("accuracy", "precision", "recall", "f1", "roc_auc")


def _log_to_mlflow(df: pd.DataFrame) -> None:
    """Publica el experimento de balanceo en MLflow.

    Estructura: un parent run ``balancing_strategies`` con un child run por
    cada combinación (modelo × estrategia) — 12 en total. Los artefactos
    (Markdown + PNG) se suben al run padre.
    """
    if not tracking.tracking_enabled():
        return

    with tracking.start_run(run_name="balancing_strategies"):
        tracking.set_tags(
            {
                "phase": "balancing",
                "n_strategies": len(ESTRATEGIAS),
                "n_models": int(df["modelo"].nunique()),
            }
        )

        for fila in df.itertuples(index=False):
            run_name = f"{fila.modelo} · {fila.estrategia}"
            with tracking.start_run(run_name=run_name, nested=True):
                tracking.set_tags(
                    {
                        "strategy": fila.estrategia,
                        "model": fila.modelo,
                        "model_family": _MODEL_FAMILY.get(fila.modelo, "other"),
                    }
                )
                tracking.log_metrics(
                    {m: float(getattr(fila, m)) for m in _METRIC_COLS}
                )

        # Artefactos comunes al experimento.
        tracking.log_artifact(config.BALANCING_RESULTS_PATH)
        tracking.log_artifact(config.BALANCING_PLOT_PATH)


def main() -> None:
    from .data_loader import load_and_prepare

    config.use_agg_backend()  # CLI: backend no interactivo para guardar el PNG sin pantalla
    config.configure_logging()
    config.ensure_directories()
    tracking.init_tracking("pontia-cancellations-balancing")
    logger.info("=== Balanceo de clases: baseline vs class_weight vs SMOTE ===")
    X_train, X_test, y_train, y_test = load_and_prepare()
    df = compare(X_train, X_test, y_train, y_test)
    save_results(df)
    _log_to_mlflow(df)  # tras `save_results`, los artefactos ya existen en disco
    print("\n" + "=" * 70)
    print("COMPARACIÓN DE ESTRATEGIAS DE BALANCEO (test)")
    print("=" * 70)
    print(df.round(3).to_string(index=False))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
