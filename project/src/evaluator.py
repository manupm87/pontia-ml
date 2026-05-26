"""Evaluación de modelos y visualizaciones.

Concentra el cálculo de métricas (accuracy, precision, recall, F1, ROC-AUC), la
construcción de la tabla comparativa, la selección del mejor modelo según la
métrica principal y los gráficos exigidos por el enunciado: matriz de confusión,
curva ROC comparativa y gráfico de importancia de variables.
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")  # backend no interactivo: permite guardar PNG sin pantalla.

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

from . import config

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")


def compute_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    """Calcula el conjunto de métricas de clasificación binaria.

    Parameters
    ----------
    y_true:
        Etiquetas reales.
    y_pred:
        Etiquetas predichas (umbral 0.5).
    y_proba:
        Probabilidad estimada de la clase positiva (cancelación).

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
        class_labels: list[str] = config.CLASS_LABELS,
        metric_names: list[str] = config.METRIC_NAMES,
    ):
        self.class_labels = class_labels
        self.metric_names = metric_names
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
        for nombre, modelo in models.items():
            y_pred = np.asarray(modelo.predict(X_test)).ravel()
            y_proba = np.asarray(modelo.predict_proba(X_test))[:, 1]
            metrics = compute_metrics(self.y_test_, y_pred, y_proba)
            self.results_[nombre] = {
                "y_pred": y_pred,
                "y_proba": y_proba,
                "metrics": metrics,
            }
            logger.info(
                "%-24s | acc=%.4f f1=%.4f roc_auc=%.4f",
                nombre,
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
        filas = {}
        for nombre, res in self.results_.items():
            fila = {m: res["metrics"][m] for m in self.metric_names}
            if train_times and nombre in train_times:
                fila["train_time_s"] = train_times[nombre]
            filas[nombre] = fila
        tabla = pd.DataFrame(filas).T
        tabla = tabla.sort_values(by=config.PRIMARY_METRIC, ascending=False)
        return tabla

    def select_best(self, primary_metric: str = config.PRIMARY_METRIC) -> str:
        """Devuelve el nombre del mejor modelo según la métrica principal."""
        mejor = max(
            self.results_.items(),
            key=lambda kv: kv[1]["metrics"][primary_metric],
        )[0]
        logger.info(
            "Mejor modelo según %s: %s (%.4f)",
            primary_metric,
            mejor,
            self.results_[mejor]["metrics"][primary_metric],
        )
        return mejor

    # -- Visualizaciones ----------------------------------------------------
    def plot_roc_curves(self, path) -> None:
        """Dibuja la curva ROC de todos los modelos en una misma figura."""
        plt.figure(figsize=(8, 7))
        for nombre, res in self.results_.items():
            fpr, tpr, _ = roc_curve(self.y_test_, res["y_proba"])
            auc = res["metrics"]["roc_auc"]
            plt.plot(fpr, tpr, label=f"{nombre} (AUC = {auc:.3f})", linewidth=2)
        plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Azar (AUC = 0.5)")
        plt.xlabel("Tasa de Falsos Positivos (FPR)")
        plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
        plt.title("Curva ROC comparativa")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(path, dpi=120)
        plt.close()
        logger.info("Gráfico guardado: %s", path)

    def plot_confusion_matrices(self, path) -> None:
        """Dibuja la matriz de confusión de cada modelo en una cuadrícula."""
        n = len(self.results_)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.2 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for ax, (nombre, res) in zip(axes, self.results_.items()):
            cm = confusion_matrix(self.y_test_, res["y_pred"])
            disp = ConfusionMatrixDisplay(cm, display_labels=self.class_labels)
            disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
            ax.set_title(nombre)
            ax.set_xlabel("Predicción")
            ax.set_ylabel("Real")
        for ax in axes[n:]:  # ocultar ejes sobrantes de la cuadrícula
            ax.axis("off")
        fig.suptitle("Matrices de confusión por modelo", fontsize=14)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        logger.info("Gráfico guardado: %s", path)

    def plot_confusion_matrix(self, name: str, path) -> None:
        """Dibuja la matriz de confusión de un único modelo (el mejor)."""
        res = self.results_[name]
        cm = confusion_matrix(self.y_test_, res["y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=self.class_labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
        ax.set_title(f"Matriz de confusión — {name}")
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        logger.info("Gráfico guardado: %s", path)

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
        estimador = model_pipeline.named_steps["model"]
        if not hasattr(estimador, "feature_importances_"):
            logger.warning(
                "El modelo no expone feature_importances_; se omite el gráfico."
            )
            return
        nombres = model_pipeline.named_steps["preprocessor"].get_feature_names_out()
        importancias = estimador.feature_importances_
        serie = pd.Series(importancias, index=nombres).sort_values(ascending=False)
        top = serie.head(top_n).iloc[::-1]  # invertir para barh de mayor a menor

        plt.figure(figsize=(9, 0.42 * len(top) + 1.5))
        sns.barplot(x=top.values, y=top.index, color="#2c7fb8")
        plt.xlabel("Importancia")
        plt.ylabel("Variable")
        plt.title(f"Top {len(top)} variables más importantes")
        plt.tight_layout()
        plt.savefig(path, dpi=120)
        plt.close()
        logger.info("Gráfico guardado: %s", path)
