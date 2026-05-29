"""Tests de `src.preprocessing` (el ColumnTransformer)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config, data_loader, preprocessing


def _fit_transform(raw_like_df: pd.DataFrame):
    cleaned = data_loader.clean_data(raw_like_df)
    X, _ = data_loader.get_feature_target(cleaned)
    pre = preprocessing.build_preprocessor()
    matrix = pre.fit_transform(X)
    return pre, X, matrix


def test_preprocessor_outputs_numeric_matrix(raw_like_df: pd.DataFrame) -> None:
    _, X, matrix = _fit_transform(raw_like_df)
    matrix = np.asarray(matrix)
    assert matrix.shape[0] == len(X)
    assert matrix.dtype.kind in "fiu"  # numérico tras el preprocesado
    assert not np.isnan(matrix).any()  # la imputación elimina los NaN


def test_feature_names_include_numeric_columns(raw_like_df: pd.DataFrame) -> None:
    pre, _, _ = _fit_transform(raw_like_df)
    names = preprocessing.get_feature_names(pre)
    # Las numéricas pasan con su nombre; las categóricas se expanden (one-hot).
    for col in config.NUMERIC_COLUMNS:
        assert col in names
    assert len(names) >= len(config.NUMERIC_COLUMNS)


def test_preprocessor_drops_non_feature_columns(raw_like_df: pd.DataFrame) -> None:
    """`remainder='drop'`: columnas como arrival_date_year no aparecen."""
    pre, _, _ = _fit_transform(raw_like_df)
    names = preprocessing.get_feature_names(pre)
    assert not any("arrival_date_year" in n for n in names)


def test_scaler_standardizes_numeric(raw_like_df: pd.DataFrame) -> None:
    """Tras estandarizar, las numéricas tienen media ~0."""
    pre, _, matrix = _fit_transform(raw_like_df)
    matrix = np.asarray(matrix)
    n_numeric = len(config.NUMERIC_COLUMNS)
    numeric_block = matrix[:, :n_numeric]
    assert np.allclose(numeric_block.mean(axis=0), 0.0, atol=1e-6)
