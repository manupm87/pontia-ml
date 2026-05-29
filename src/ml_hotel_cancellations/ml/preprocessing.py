"""Construcción del preprocesador de características.

Un ``ColumnTransformer`` con ramas distintas para numéricas y categóricas, apto
para insertarse en un ``Pipeline`` (evita *data leakage*) y persistirse con el modelo.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_hotel_cancellations import config


def build_preprocessor() -> ColumnTransformer:
    """Crea el ``ColumnTransformer`` con dos ramas: numéricas (imputación por
    mediana + estandarización) y categóricas (imputación "Unknown" + one-hot con
    cardinalidad acotada). Devuelve el preprocesador sin ajustar.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # OneHotEncoder tolera categorías no vistas y acota la cardinalidad (ver §4.5).
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    max_categories=config.MAX_OHE_CATEGORIES,
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, config.NUMERIC_COLUMNS),
            ("cat", categorical_transformer, config.CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def make_pipeline(estimator) -> Pipeline:
    """Envuelve un estimador con un preprocesador nuevo en un ``Pipeline``.

    Fuente única del patrón preprocesador + modelo (reutilizado por
    model_trainer/tuning/balancing).
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


def get_feature_names(fitted_preprocessor: ColumnTransformer) -> list[str]:
    """Nombres de las características tras el preprocesado (preprocesador ya ajustado)."""
    return list(fitted_preprocessor.get_feature_names_out())
