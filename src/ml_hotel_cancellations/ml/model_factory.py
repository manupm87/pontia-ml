"""Fábrica única de los estimadores clásicos (Logistic/Tree/Forest/XGBoost).

Construye los modelos desde ``config``; reutilizada por model_trainer, tuning,
balancing y visualization_2d para no duplicar el catálogo.
"""

from __future__ import annotations

from ml_hotel_cancellations import config

# Nombres canónicos de los cuatro clásicos (la red neuronal se construye aparte).
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

    ``overrides`` y ``n_jobs`` ajustan params/paralelismo por modelo;
    ``class_weight`` (sklearn) y ``scale_pos_weight`` (XGBoost) los usa el balanceo.
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
