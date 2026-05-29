"""Tests de `src.evaluator` (métricas, tabla comparativa y selección)."""

from __future__ import annotations

import numpy as np

from ml_hotel_cancellations import config
from ml_hotel_cancellations.ml.evaluator import Evaluator, compute_metrics


class _FakeModel:
    """Modelo mínimo con predict/predict_proba a partir de probabilidades fijas."""

    def __init__(self, proba: np.ndarray):
        self._proba = np.asarray(proba, dtype=float)

    def predict(self, X):  # noqa: N803 - firma sklearn
        return (self._proba >= 0.5).astype(int)

    def predict_proba(self, X):  # noqa: N803
        return np.column_stack([1 - self._proba, self._proba])


def test_compute_metrics_perfect_classifier() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}


def test_compute_metrics_keys_match_config() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.3, 0.6, 0.4, 0.7])
    metrics = compute_metrics(y_true, (y_proba >= 0.5).astype(int), y_proba)
    assert set(metrics) == set(config.METRIC_NAMES)


def test_evaluator_selects_best_by_primary_metric() -> None:
    y_test = np.array([0, 0, 1, 1])
    models = {
        "bueno": _FakeModel([0.1, 0.2, 0.8, 0.9]),   # roc_auc = 1.0
        "malo": _FakeModel([0.6, 0.6, 0.4, 0.4]),    # roc_auc = 0.0
    }
    ev = Evaluator()
    ev.evaluate(models, X_test=np.zeros((4, 1)), y_test=y_test)
    assert ev.select_best() == "bueno"


def test_comparison_table_sorted_by_primary_metric() -> None:
    y_test = np.array([0, 0, 1, 1])
    models = {
        "malo": _FakeModel([0.6, 0.6, 0.4, 0.4]),
        "bueno": _FakeModel([0.1, 0.2, 0.8, 0.9]),
    }
    ev = Evaluator()
    ev.evaluate(models, X_test=np.zeros((4, 1)), y_test=y_test)
    tabla = ev.comparison_table()
    # Ordenada de mejor a peor según la métrica principal.
    assert list(tabla.index)[0] == "bueno"
    assert list(tabla.columns)[: len(config.METRIC_NAMES)] == config.METRIC_NAMES


def test_comparison_table_includes_train_times() -> None:
    y_test = np.array([0, 1])
    models = {"m": _FakeModel([0.2, 0.8])}
    ev = Evaluator()
    ev.evaluate(models, X_test=np.zeros((2, 1)), y_test=y_test)
    tabla = ev.comparison_table(train_times={"m": 1.23})
    assert "train_time_s" in tabla.columns
    assert tabla.loc["m", "train_time_s"] == 1.23
