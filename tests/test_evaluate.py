"""Tests de `ml.evaluate` (métricas, tabla comparativa y selección)."""

from __future__ import annotations

import numpy as np

from ml_hotel_cancellations.ml.evaluate import (
    comparison_table,
    compute_metrics,
    evaluate_models,
    select_best,
)


class _DummyModel:
    """Modelo de juguete: probas fijas para test determinista."""

    def __init__(self, proba):
        self._proba = np.asarray(proba)

    def predict(self, X):
        return (self._proba >= 0.5).astype(int)

    def predict_proba(self, X):
        return np.column_stack([1 - self._proba, self._proba])


def test_compute_metrics_perfect_classifier() -> None:
    """Un clasificador perfecto da métricas = 1.0."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.8])
    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert all(abs(v - 1.0) < 1e-9 for v in metrics.values())


def test_compute_metrics_keys() -> None:
    """`compute_metrics` devuelve exactamente las cinco métricas esperadas."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_proba = np.array([0.2, 0.7, 0.6, 0.9])
    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}


def test_select_best_by_primary_metric() -> None:
    """`select_best` elige el modelo con mayor métrica principal (ROC-AUC)."""
    y_test = np.array([0, 1, 0, 1, 1])
    good = _DummyModel([0.1, 0.9, 0.2, 0.8, 0.7])  # perfecto
    bad = _DummyModel([0.9, 0.1, 0.8, 0.2, 0.3])  # invertido
    results = evaluate_models({"bueno": good, "malo": bad}, X_test=None, y_test=y_test)
    assert select_best(results) == "bueno"


def test_comparison_table_sorted() -> None:
    """La tabla comparativa queda ordenada por la métrica principal (desc)."""
    y_test = np.array([0, 1, 0, 1, 1])
    good = _DummyModel([0.1, 0.9, 0.2, 0.8, 0.7])
    bad = _DummyModel([0.9, 0.1, 0.8, 0.2, 0.3])
    results = evaluate_models({"bueno": good, "malo": bad}, X_test=None, y_test=y_test)
    table = comparison_table(results)
    assert list(table.index)[0] == "bueno"
