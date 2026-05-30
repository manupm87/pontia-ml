"""Preprocesado de características como pasos de un ``Pipeline`` (fit-on-train).

Aplica, en orden, las decisiones del EDA (ver notebooks/playground/02):
1. :class:`FeatureBuilder` — features derivadas ``has_company``/``has_agent``/``noches``
   (ausencia informativa, EDA §5) y normaliza los IDs de alta cardinalidad.
2. :class:`RareCategoryGrouper` — reducción **supervisada** de cardinalidad de
   ``agent``/``country``/``company`` (EDA §13): conserva las categorías con señal
   extrema y soporte, el resto -> ``"Otros"``. Se ajusta **solo con train** (usa ``y``).
3. ``ColumnTransformer`` — imputación + escalado de numéricas y *one-hot* de categóricas.

Todo vive en el ``Pipeline`` para que se ajuste sin fuga y se persista con el modelo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_hotel_cancellations import config


# ---------------------------------------------------------------------------
# Helpers de normalización de valores categóricos
# ---------------------------------------------------------------------------
# Valores que tratamos como "ausente" (vienen como NaN en el crudo o como texto
# en la inferencia, p. ej. la UI manda "no_company" para una reserva sin empresa).
_ABSENT_VALUES = {"", "nan", "none", "no_company", "unknown", "desconocido"}


def _is_absent(value) -> bool:
    """``True`` si el valor representa una ausencia (NaN o texto centinela)."""
    if pd.isna(value):
        return True
    return str(value).strip().lower() in _ABSENT_VALUES


def _key_str(value) -> str:
    """Clave limpia para one-hot: ``9.0`` -> ``"9"``; el texto se queda igual."""
    if isinstance(value, float) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _group_value(value, keep: set, null_label: str) -> str:
    """Mapea un valor a su categoría reducida: nulo -> ``null_label``;
    conservado -> su clave; resto -> ``"Otros"``."""
    if pd.isna(value):
        return null_label
    key = str(value)
    return key if key in keep else "Otros"


# ---------------------------------------------------------------------------
# Paso 1: features derivadas + normalización de IDs
# ---------------------------------------------------------------------------
class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Crea las features derivadas del EDA y normaliza los IDs de alta cardinalidad.

    Es row-wise y no aprende nada (no necesita ``y``), así que sirve igual en
    entrenamiento e inferencia.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # children: nulos -> 0 (apenas 4 en el crudo, EDA §4).
        if "children" in X.columns:
            X["children"] = X["children"].fillna(0).astype(int)

        # noches = estancia total (EDA §11.5).
        if {"stays_in_week_nights", "stays_in_weekend_nights"} <= set(X.columns):
            X["noches"] = (
                X["stays_in_week_nights"].fillna(0)
                + X["stays_in_weekend_nights"].fillna(0)
            )

        # Ausencia informativa (EDA §5): el nulo de company/agent es señal.
        X["has_company"] = (
            (~X["company"].map(_is_absent)).astype(int) if "company" in X.columns else 0
        )
        X["has_agent"] = (
            (~X["agent"].map(_is_absent)).astype(int) if "agent" in X.columns else 0
        )

        # IDs de alta cardinalidad -> clave limpia; ausentes -> NaN (los etiqueta
        # el RareCategoryGrouper con su null_label propio).
        for col in config.HIGH_CARDINALITY_COLUMNS:
            if col in X.columns:
                X[col] = X[col].map(lambda v: np.nan if _is_absent(v) else _key_str(v))

        return X


# ---------------------------------------------------------------------------
# Paso 2: reducción supervisada de cardinalidad (fit-on-train)
# ---------------------------------------------------------------------------
class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Reduce la cardinalidad de columnas categóricas mirando el target (EDA §13).

    En ``fit`` (solo train) aprende, por columna, qué categorías conservar: las que
    tienen soporte (``n >= min_n``) y tasa de cancelación extrema (``> hi_frac*max``
    o ``< lo_frac*max``). En ``transform`` el resto pasa a ``"Otros"`` y los nulos a
    la etiqueta propia de la columna. Sin ``y`` (p. ej. al reusar el preprocesador
    para visualizar) conserva todas las categorías vistas.
    """

    def __init__(self, columns=None, null_labels=None, min_n=None, hi_frac=None, lo_frac=None):
        self.columns = columns
        self.null_labels = null_labels
        self.min_n = min_n
        self.hi_frac = hi_frac
        self.lo_frac = lo_frac

    def _resolved(self):
        """Resuelve los parámetros con sus valores de ``config`` si no se fijaron."""
        return (
            self.columns if self.columns is not None else config.HIGH_CARDINALITY_COLUMNS,
            self.null_labels if self.null_labels is not None else config.RARE_NULL_LABELS,
            self.min_n if self.min_n is not None else config.RARE_MIN_N,
            self.hi_frac if self.hi_frac is not None else config.RARE_HI_FRAC,
            self.lo_frac if self.lo_frac is not None else config.RARE_LO_FRAC,
        )

    def fit(self, X, y=None):
        columns, _, min_n, hi_frac, lo_frac = self._resolved()
        self.keep_ = {}
        for col in columns:
            if col not in X.columns:
                continue
            if y is None:
                # Sin target: conservar todas las categorías vistas (no supervisado).
                self.keep_[col] = set(X[col].dropna().astype(str).unique())
                continue
            stats = (
                pd.DataFrame({"v": X[col].values, "y": np.asarray(y)})
                .dropna(subset=["v"])
                .groupby("v", observed=True)["y"]
                .agg(rate="mean", n="size")
            )
            enough = stats[stats["n"] >= min_n]
            if enough.empty:
                self.keep_[col] = set()
                continue
            top = enough["rate"].max()
            keep = enough[(enough["rate"] > hi_frac * top) | (enough["rate"] < lo_frac * top)]
            self.keep_[col] = set(keep.index.astype(str))
        return self

    def transform(self, X):
        columns, null_labels, *_ = self._resolved()
        X = X.copy()
        for col in columns:
            if col not in X.columns:
                continue
            null_label = null_labels.get(col, "Desconocido")
            keep = self.keep_.get(col, set())
            X[col] = X[col].map(lambda v: _group_value(v, keep, null_label))
        return X


# ---------------------------------------------------------------------------
# Paso 3: ColumnTransformer (imputar + escalar numéricas, one-hot categóricas)
# ---------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    """Crea el ``ColumnTransformer``: numéricas (mediana + estandarización) y
    categóricas (imputación "Unknown" + one-hot tolerante a categorías nuevas).

    Espera el DataFrame ya pasado por ``FeatureBuilder``/``RareCategoryGrouper``
    (usa ``NUMERIC_FEATURES``, que incluye las derivadas). Sin ajustar.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, config.NUMERIC_FEATURES),
            ("cat", categorical_transformer, config.CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _feature_steps() -> list:
    """Pasos de derivación + reducción (compartidos por el pipeline y la transformación)."""
    return [("features", FeatureBuilder()), ("rare", RareCategoryGrouper())]


def build_transform_pipeline() -> Pipeline:
    """Pipeline de solo preprocesado (sin modelo): derivar -> reducir -> ColumnTransformer.

    Útil para quien necesita la matriz de features (p. ej. la visualización 2D);
    ajústalo con ``.fit(X, y)`` para que la reducción de cardinalidad sea supervisada.
    """
    return Pipeline([*_feature_steps(), ("preprocessor", build_preprocessor())])


def make_pipeline(estimator) -> Pipeline:
    """Envuelve un estimador con todo el preprocesado en un ``Pipeline`` plano.

    Fuente única del patrón preprocesado + modelo (reutilizado por models/tuning).
    Plano a propósito: ``named_steps["preprocessor"]`` es el ``ColumnTransformer``
    (para sacar nombres de features e importancias) y ``named_steps["model"]`` el modelo.
    """
    return Pipeline(
        [*_feature_steps(), ("preprocessor", build_preprocessor()), ("model", estimator)]
    )


def get_feature_names(fitted_preprocessor) -> list[str]:
    """Nombres de las características tras el preprocesado (``ColumnTransformer`` ajustado)."""
    ct = fitted_preprocessor
    if hasattr(ct, "named_steps") and "preprocessor" in getattr(ct, "named_steps", {}):
        ct = ct.named_steps["preprocessor"]
    return list(ct.get_feature_names_out())
