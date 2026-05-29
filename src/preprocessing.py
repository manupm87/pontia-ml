"""Construcción del preprocesador de características.

Se define un único :class:`~sklearn.compose.ColumnTransformer` que aplica
transformaciones distintas a las variables numéricas y categóricas. Encapsular el
preprocesado en un objeto de scikit-learn permite reutilizarlo dentro de un
``Pipeline`` (evitando *data leakage* entre train y test) y persistirlo junto al
modelo para la inferencia.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def build_preprocessor() -> ColumnTransformer:
    """Crea el ``ColumnTransformer`` con las dos ramas de preprocesado.

    - **Numéricas**: imputación por mediana (robusta a outliers como ``adr``) y
      estandarización (media 0, desviación 1), necesaria sobre todo para la
      regresión logística y la red neuronal.
    - **Categóricas**: imputación por constante (``"Unknown"``) y codificación
      *one-hot*. Para variables de alta cardinalidad (``country``, ``agent``...)
      se limita el número de categorías con ``max_categories``; las menos
      frecuentes se agrupan automáticamente en una categoría "infrequent", lo que
      controla la dimensionalidad y reduce el sobreajuste.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Preprocesador sin ajustar, listo para insertarse en un ``Pipeline``.
    """
    # Mapeo con `recursos/` (ver informe §4.5): SimpleImputer ≈ `.fillna(mediana)`,
    # pero aprende la mediana en train y la reaplica en test/predicción.
    # StandardScaler: la misma herramienta de clase (allí solo en KNN/SVM).
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # SimpleImputer(constant) ≈ `.fillna("Unknown")`; OneHotEncoder ≈
    # `pd.get_dummies(X)` de `recursos/`, pero tolera categorías no vistas en
    # predicción y limita la cardinalidad con `max_categories` (ver informe §4.5).
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

    # ColumnTransformer: aplica cada rama por tipo de columna (en `recursos/` esto
    # se haría a mano con pandas). Equivalencias completas en informe §4.5.
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

    Fuente única del patrón ``Pipeline([("preprocessor", build_preprocessor()),
    ("model", estimator)])`` que antes se reimplementaba en model_trainer/tuning/
    balancing.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


def get_feature_names(fitted_preprocessor: ColumnTransformer) -> list[str]:
    """Devuelve los nombres de las características tras el preprocesado.

    Útil para etiquetar los gráficos de importancia de variables.

    Parameters
    ----------
    fitted_preprocessor:
        ``ColumnTransformer`` ya ajustado (tras ``fit``).

    Returns
    -------
    list[str]
        Lista de nombres de columna resultantes del preprocesado.
    """
    return list(fitted_preprocessor.get_feature_names_out())
