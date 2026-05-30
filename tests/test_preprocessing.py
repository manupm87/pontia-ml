"""Tests de `ml.preprocessing` (FeatureBuilder, RareCategoryGrouper, ColumnTransformer)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_hotel_cancellations import config
from ml_hotel_cancellations.ml import data_loader, preprocessing


def _fit_transform(raw_like_df: pd.DataFrame):
    """Ajusta el pipeline de preprocesado completo (con target) y transforma."""
    cleaned = data_loader.clean_data(raw_like_df)
    X, y = data_loader.get_feature_target(cleaned)
    pipe = preprocessing.build_transform_pipeline()
    matrix = pipe.fit_transform(X, y)
    return pipe, X, matrix


def test_preprocessor_outputs_numeric_matrix(raw_like_df: pd.DataFrame) -> None:
    _, X, matrix = _fit_transform(raw_like_df)
    matrix = np.asarray(matrix)
    assert matrix.shape[0] == len(X)
    assert matrix.dtype.kind in "fiu"  # numérico tras el preprocesado
    assert not np.isnan(matrix).any()  # la imputación elimina los NaN


def test_feature_builder_adds_derived_columns() -> None:
    """FeatureBuilder crea has_company/has_agent/noches a partir del crudo."""
    df = pd.DataFrame(
        {
            "company": ["no_company", "40"],
            "agent": [None, "9"],
            "stays_in_week_nights": [3, 2],
            "stays_in_weekend_nights": [1, 0],
            "children": [None, 1],
        }
    )
    out = preprocessing.FeatureBuilder().transform(df)
    assert list(out["has_company"]) == [0, 1]
    assert list(out["has_agent"]) == [0, 1]
    assert list(out["noches"]) == [4, 2]
    assert list(out["children"]) == [0, 1]  # nulo -> 0


def test_rare_grouper_is_supervised_and_groups_others() -> None:
    """El grouper conserva categorías de señal extrema + soporte; el resto -> 'Otros'."""
    n = 300
    # 'risky' siempre cancela, 'safe' nunca; 'mid' tasa media; 'tiny' sin soporte.
    df = pd.DataFrame(
        {
            "agent": (["risky"] * n + ["safe"] * n + ["mid"] * n + ["tiny"] * 5),
            "country": ["x"] * (3 * n + 5),
            "company": ["no_company"] * (3 * n + 5),
        }
    )
    y = np.array([1] * n + [0] * n + [1, 0] * (n // 2) + [1, 0, 1, 0, 1])
    grouper = preprocessing.RareCategoryGrouper().fit(df, y)
    out = grouper.transform(df)
    kept = set(out["agent"].unique())
    assert "risky" in kept and "safe" in kept  # señal extrema conservada
    assert "tiny" in out["agent"].values or "Otros" in out["agent"].values
    assert "tiny" not in kept  # sin soporte (n<100) -> agrupado en 'Otros'


def test_feature_names_include_numeric_and_derived(raw_like_df: pd.DataFrame) -> None:
    pipe, _, _ = _fit_transform(raw_like_df)
    names = preprocessing.get_feature_names(pipe)
    for col in config.NUMERIC_FEATURES:  # entrada + derivadas
        assert col in names
    assert len(names) >= len(config.NUMERIC_FEATURES)


def test_preprocessor_drops_non_feature_columns(raw_like_df: pd.DataFrame) -> None:
    """`remainder='drop'`: columnas como arrival_date_year no aparecen."""
    pipe, _, _ = _fit_transform(raw_like_df)
    names = preprocessing.get_feature_names(pipe)
    assert not any("arrival_date_year" in n for n in names)
    assert not any("required_car_parking_spaces" in n for n in names)


def test_scaler_standardizes_numeric(raw_like_df: pd.DataFrame) -> None:
    """Tras estandarizar, las numéricas tienen media ~0."""
    _, _, matrix = _fit_transform(raw_like_df)
    matrix = np.asarray(matrix)
    n_numeric = len(config.NUMERIC_FEATURES)
    numeric_block = matrix[:, :n_numeric]
    # Columnas constantes (p. ej. has_* en datos sintéticos) quedan en 0 -> media 0.
    assert np.allclose(numeric_block.mean(axis=0), 0.0, atol=1e-6)
