"""Fábrica única de los estimadores clásicos del proyecto.

Construye ``LogisticRegression`` / ``DecisionTree`` / ``RandomForest`` /
``XGBClassifier`` a partir de los hiperparámetros de ``config``. Es la fuente
única que reutilizan ``model_trainer``, ``tuning``, ``balancing`` y
``visualization_2d``, evitando que el catálogo de modelos se reescriba (y diverja)
en cuatro sitios.
"""

from __future__ import annotations

from ml_hotel_cancellations import config

# Nombres canónicos de los cuatro modelos clásicos (la red neuronal se construye
# aparte porque su naturaleza y dependencias difieren).
CLASSIC_MODEL_NAMES: tuple[str, ...] = (
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "XGBoost",
)


def build_classic_estimators(
    *,
    overrides: dict[str, dict] | None = None,
    n_jobs: dict[str, int] | None = None,
    class_weight: str | None = None,
    scale_pos_weight: float | None = None,
) -> dict:
    """Crea los cuatro estimadores clásicos con los hiperparámetros de ``config``.

    Parameters
    ----------
    overrides:
        Mapa ``nombre -> dict`` con hiperparámetros que sobrescriben los base
        (p. ej. los hallados por la optimización con ``src.tuning``).
    n_jobs:
        Mapa ``nombre -> n_jobs`` para forzar el paralelismo de modelos concretos
        (p. ej. ``n_jobs=1`` en Random Forest / XGBoost durante la búsqueda CV,
        para no competir con el paralelismo de la propia búsqueda).
    class_weight:
        Si se indica (``"balanced"``), se añade a los estimadores de scikit-learn
        (no a XGBoost). Lo usa el experimento de balanceo.
    scale_pos_weight:
        Si se indica, se añade a ``XGBClassifier`` (reponderación de la clase
        positiva). Lo usa el experimento de balanceo.

    Returns
    -------
    dict
        ``nombre -> estimador`` sin ajustar, para los cuatro modelos clásicos.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    overrides = overrides or {}
    n_jobs = n_jobs or {}

    def params(name: str, base: dict) -> dict:
        """Combina los parámetros base con los overrides para ese modelo."""
        merged = {**base, **overrides.get(name, {})}
        if name in n_jobs:
            merged["n_jobs"] = n_jobs[name]
        return merged

    sklearn_cw = {"class_weight": class_weight} if class_weight is not None else {}

    lr_params = params("Logistic Regression", config.LOGISTIC_REGRESSION_PARAMS)
    dt_params = params("Decision Tree", config.DECISION_TREE_PARAMS)
    rf_params = params("Random Forest", config.RANDOM_FOREST_PARAMS)

    xgb_params = params("XGBoost", config.XGBOOST_PARAMS)
    if scale_pos_weight is not None:
        xgb_params = {**xgb_params, "scale_pos_weight": scale_pos_weight}

    return {
        "Logistic Regression": LogisticRegression(**lr_params, **sklearn_cw),
        "Decision Tree": DecisionTreeClassifier(**dt_params, **sklearn_cw),
        "Random Forest": RandomForestClassifier(**rf_params, **sklearn_cw),
        "XGBoost": XGBClassifier(**xgb_params),
    }
