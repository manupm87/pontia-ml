"""Utilidades compartidas por los notebooks de modelo.

Centralizar aquí la lógica repetida (evaluación y gráficos comunes) mantiene los
notebooks **breves y con la misma forma** sin necesidad de generarlos: cada
notebook simplemente llama a estas funciones. La visualización específica de cada
modelo está en :mod:`src.model_viz`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluar(pipeline, X_test, y_test, etiqueta: str = "Modelo") -> float:
    """Imprime las métricas de test del pipeline y devuelve su ROC-AUC."""
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metricas = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    print(f"{etiqueta:12s} | " + " | ".join(f"{k}={v:.4f}" for k, v in metricas.items()))
    return metricas["roc_auc"]


def plot_confusion_roc(pipeline, X_test, y_test) -> None:
    """Dibuja, lado a lado, la matriz de confusión y la curva ROC del modelo."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ConfusionMatrixDisplay.from_estimator(
        pipeline, X_test, y_test, ax=ax1,
        display_labels=["No cancela", "Cancela"], cmap="Blues", colorbar=False,
    )
    ax1.set_title("Matriz de confusión")
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax2)
    ax2.plot([0, 1], [0, 1], "--", color="gray", alpha=0.7)
    ax2.set_title("Curva ROC")
    plt.tight_layout()
    plt.show()
