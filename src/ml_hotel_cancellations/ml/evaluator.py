"""Evaluación de modelos y visualizaciones.

Concentra el cálculo de métricas (accuracy, precision, recall, F1, ROC-AUC), la
construcción de la tabla comparativa, la selección del mejor modelo según la
métrica principal y los gráficos exigidos por el enunciado: matriz de confusión,
curva ROC comparativa y gráfico de importancia de variables.
"""

from __future__ import annotations

import logging

from ml_hotel_cancellations import config

config.use_agg_backend()  # backend no interactivo: permite guardar PNG sin pantalla.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ml_hotel_cancellations.utils.reporting import save_figure

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")


def compute_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    """Calcula el conjunto de métricas de clasificación binaria.

    Qué mide cada métrica (más detalle en ``docs/glosario.md``):

    - **accuracy** (exactitud): porcentaje de aciertos totales. Engaña cuando hay
      desbalance de clases.
    - **precision** (precisión): de las predichas como cancelación, cuántas lo eran
      de verdad → mide las "falsas alarmas".
    - **recall** (sensibilidad): de las cancelaciones reales, cuántas se detectaron
      → mide las cancelaciones que "se escapan".
    - **f1**: media equilibrada (armónica) entre precisión y recall.
    - **roc_auc**: área bajo la curva ROC (de 0.5 = azar a 1 = perfecto). Mide la
      capacidad de **ordenar** las reservas por riesgo, sin depender del umbral. Es
      la métrica principal del proyecto.

    Parameters
    ----------
    y_true:
        Etiquetas reales (lo que pasó de verdad).
    y_pred:
        Etiquetas predichas por el modelo (aplicando el umbral 0.5).
    y_proba:
        Probabilidad estimada de la clase positiva (que la reserva se cancele).

    Returns
    -------
    dict[str, float]
        Diccionario con accuracy, precision, recall, f1 y roc_auc.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


class Evaluator:
    """Evalúa una colección de modelos entrenados y genera las visualizaciones.

    Tras llamar a :meth:`evaluate`, el resto de métodos (tablas, gráficos,
    selección del mejor modelo) operan sobre los resultados almacenados.
    """

    def __init__(
        self,
        class_labels: list[str] | None = None,
        metric_names: list[str] | None = None,
    ):
        # Defaults a `None` y asignación interna para no usar listas mutables de
        # `config` como valores por defecto de los parámetros.
        self.class_labels = class_labels if class_labels is not None else config.CLASS_LABELS
        self.metric_names = metric_names if metric_names is not None else config.METRIC_NAMES
        self.results_: dict[str, dict] = {}
        self.y_test_ = None

    def evaluate(self, models: dict, X_test, y_test) -> dict[str, dict]:
        """Calcula predicciones y métricas de cada modelo sobre el test.

        Returns
        -------
        dict[str, dict]
            ``nombre -> {"y_pred", "y_proba", "metrics"}``.
        """
        self.y_test_ = np.asarray(y_test)
        self.results_ = {}
        for name, model in models.items():
            y_pred = np.asarray(model.predict(X_test)).ravel()
            y_proba = np.asarray(model.predict_proba(X_test))[:, 1]
            metrics = compute_metrics(self.y_test_, y_pred, y_proba)
            self.results_[name] = {
                "y_pred": y_pred,
                "y_proba": y_proba,
                "metrics": metrics,
            }
            logger.info(
                "%-24s | acc=%.4f f1=%.4f roc_auc=%.4f",
                name,
                metrics["accuracy"],
                metrics["f1"],
                metrics["roc_auc"],
            )
        return self.results_

    def comparison_table(self, train_times: dict[str, float] | None = None) -> pd.DataFrame:
        """Construye la tabla comparativa de métricas (ordenada por la principal).

        Parameters
        ----------
        train_times:
            Diccionario opcional ``nombre -> segundos de entrenamiento``.

        Returns
        -------
        pandas.DataFrame
            Filas = modelos, columnas = métricas (+ tiempo de entrenamiento).
        """
        rows = {}
        for name, res in self.results_.items():
            row = {m: res["metrics"][m] for m in self.metric_names}
            if train_times and name in train_times:
                row["train_time_s"] = train_times[name]
            rows[name] = row
        table = pd.DataFrame(rows).T
        table = table.sort_values(by=config.PRIMARY_METRIC, ascending=False)
        return table

    def select_best(self, primary_metric: str = config.PRIMARY_METRIC) -> str:
        """Devuelve el nombre del mejor modelo según la métrica principal."""
        best = max(
            self.results_.items(),
            key=lambda kv: kv[1]["metrics"][primary_metric],
        )[0]
        logger.info(
            "Mejor modelo según %s: %s (%.4f)",
            primary_metric,
            best,
            self.results_[best]["metrics"][primary_metric],
        )
        return best

    # -- Visualizaciones ----------------------------------------------------
    def plot_roc_curves(self, path) -> None:
        """Dibuja la curva ROC de todos los modelos en una misma figura."""
        fig = plt.figure(figsize=(8, 7))
        for name, res in self.results_.items():
            fpr, tpr, _ = roc_curve(self.y_test_, res["y_proba"])
            auc = res["metrics"]["roc_auc"]
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)
        plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Azar (AUC = 0.5)")
        plt.xlabel("Tasa de Falsos Positivos (FPR)")
        plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
        plt.title("Curva ROC comparativa")
        plt.legend(loc="lower right")
        save_figure(fig, path)

    def _plot_cm(self, ax, name: str, res: dict, *, colorbar: bool) -> None:
        """Pinta la matriz de confusión de un modelo sobre el eje ``ax``."""
        cm = confusion_matrix(self.y_test_, res["y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=self.class_labels)
        disp.plot(ax=ax, cmap="Blues", colorbar=colorbar, values_format="d")
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")

    def plot_confusion_matrices(self, path) -> None:
        """Dibuja la matriz de confusión de cada modelo en una cuadrícula."""
        n = len(self.results_)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.2 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for ax, (name, res) in zip(axes, self.results_.items()):
            self._plot_cm(ax, name, res, colorbar=False)
            ax.set_title(name)
        for ax in axes[n:]:  # ocultar ejes sobrantes de la cuadrícula
            ax.axis("off")
        fig.suptitle("Matrices de confusión por modelo", fontsize=14)
        save_figure(fig, path)

    def plot_confusion_matrix(self, name: str, path) -> None:
        """Dibuja la matriz de confusión de un único modelo (el mejor)."""
        fig, ax = plt.subplots(figsize=(6, 5))
        self._plot_cm(ax, name, self.results_[name], colorbar=True)
        ax.set_title(f"Matriz de confusión — {name}")
        save_figure(fig, path)

    @staticmethod
    def plot_feature_importance(model_pipeline, path, top_n: int = 20) -> None:
        """Dibuja la importancia de variables de un modelo basado en árboles.

        Extrae ``feature_importances_`` del estimador final del ``Pipeline`` y
        los nombres de característica del preprocesador ajustado.

        Parameters
        ----------
        model_pipeline:
            ``Pipeline`` entrenado (preprocesador + modelo de árbol).
        path:
            Ruta donde guardar el PNG.
        top_n:
            Número de variables más importantes a mostrar.
        """
        estimator = model_pipeline.named_steps["model"]
        if not hasattr(estimator, "feature_importances_"):
            logger.warning(
                "El modelo no expone feature_importances_; se omite el gráfico."
            )
            return
        names = model_pipeline.named_steps["preprocessor"].get_feature_names_out()
        importances = estimator.feature_importances_
        series = pd.Series(importances, index=names).sort_values(ascending=False)
        top = series.head(top_n).iloc[::-1]  # invertir para barh de mayor a menor

        fig = plt.figure(figsize=(9, 0.42 * len(top) + 1.5))
        sns.barplot(x=top.values, y=top.index, color="#2c7fb8")
        plt.xlabel("Importancia")
        plt.ylabel("Variable")
        plt.title(f"Top {len(top)} variables más importantes")
        save_figure(fig, path)
