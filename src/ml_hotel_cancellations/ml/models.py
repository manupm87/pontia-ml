"""Catálogo de modelos y entrenamiento.

Define los cinco modelos a comparar (4 clásicos + una red neuronal `MLPClassifier`
de scikit-learn), cada uno metido en un ``Pipeline`` con el preprocesador, y los
entrena con un bucle simple. Mismo preprocesador para todos -> comparación justa y
sin *data leakage*. Al ser todo scikit-learn, los modelos se serializan con
``joblib`` sin trucos.
"""

from __future__ import annotations

import logging
import re
import time

import joblib

from ml_hotel_cancellations import config
from .preprocessing import make_pipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Los cuatro modelos clásicos
# ---------------------------------------------------------------------------
def build_classic_estimators(
    *,
    overrides: dict[str, dict] | None = None,
    n_jobs: dict[str, int] | None = None,
) -> dict:
    """Crea los cuatro estimadores clásicos con los hiperparámetros de ``config``.

    ``overrides`` cambia params por modelo (lo usa `tuning`) y ``n_jobs`` ajusta el
    paralelismo por modelo (para no competir con la búsqueda CV).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    overrides = overrides or {}
    n_jobs = n_jobs or {}

    def params(name: str, base: dict) -> dict:
        """Combina los params base con los overrides (y n_jobs) de ese modelo."""
        merged = {**base, **overrides.get(name, {})}
        if name in n_jobs:
            merged["n_jobs"] = n_jobs[name]
        return merged

    return {
        "Logistic Regression": LogisticRegression(
            **params("Logistic Regression", config.LOGISTIC_REGRESSION_PARAMS)
        ),
        "Decision Tree": DecisionTreeClassifier(
            **params("Decision Tree", config.DECISION_TREE_PARAMS)
        ),
        "Random Forest": RandomForestClassifier(
            **params("Random Forest", config.RANDOM_FOREST_PARAMS)
        ),
        "XGBoost": XGBClassifier(**params("XGBoost", config.XGBOOST_PARAMS)),
    }


# ---------------------------------------------------------------------------
# Catálogo completo y entrenamiento
# ---------------------------------------------------------------------------
def build_models(overrides: dict[str, dict] | None = None) -> dict:
    """Devuelve los 5 modelos a comparar como ``{nombre: Pipeline}`` (sin entrenar).

    Cada modelo va en un ``Pipeline`` con su propio preprocesador, de modo que el
    objeto guardado sabe preprocesar reservas en crudo en inferencia.
    """
    from sklearn.neural_network import MLPClassifier

    estimators = build_classic_estimators(overrides=overrides)
    nn_params = {**config.NN_PARAMS, **(overrides or {}).get("Neural Network (MLP)", {})}
    estimators["Neural Network (MLP)"] = MLPClassifier(**nn_params)
    return {name: make_pipeline(est) for name, est in estimators.items()}


def train_models(models: dict, X_train, y_train) -> tuple[dict, dict]:
    """Entrena cada modelo y devuelve ``(modelos_entrenados, tiempos_en_segundos)``."""
    fitted: dict = {}
    train_times: dict = {}
    for name, pipeline in models.items():
        logger.info("Entrenando %s...", name)
        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_times[name] = time.perf_counter() - start
        fitted[name] = pipeline
        logger.info("  -> %s entrenado en %.1f s", name, train_times[name])
    return fitted, train_times


def save_models(fitted: dict, directory=config.MODELS_DIR) -> None:
    """Guarda cada Pipeline entrenado en disco (un ``.pkl`` por modelo)."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, pipeline in fitted.items():
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        path = directory / f"{slug}.pkl"
        joblib.dump(pipeline, path)
        logger.info("Modelo guardado: %s", path)
