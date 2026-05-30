"""Tests de `ml.data_loader` (limpieza segura de filas/columnas y partición)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_hotel_cancellations import config
from ml_hotel_cancellations.ml import data_loader


def test_clean_data_drops_leakage_and_non_generalizing_columns(raw_like_df: pd.DataFrame) -> None:
    """Se eliminan leakage, parking (fuga) y año (no generaliza)."""
    cleaned = data_loader.clean_data(raw_like_df)
    for col in config.DROP_COLUMNS:
        assert col not in cleaned.columns


def test_clean_data_keeps_company(raw_like_df: pd.DataFrame) -> None:
    """`company` ya NO se descarta: es una feature de entrada."""
    cleaned = data_loader.clean_data(raw_like_df)
    assert "company" in cleaned.columns


def test_clean_data_removes_rows_without_guests(raw_like_df: pd.DataFrame) -> None:
    """La fila con adults+children+babies == 0 se elimina."""
    cleaned = data_loader.clean_data(raw_like_df)
    guests = cleaned[["adults", "children", "babies"]].fillna(0).sum(axis=1)
    assert (guests == 0).sum() == 0


def test_clean_data_removes_rows_without_nights(raw_like_df: pd.DataFrame) -> None:
    """Las reservas con 0 noches de estancia se eliminan (EDA §8)."""
    cleaned = data_loader.clean_data(raw_like_df)
    nights = cleaned["stays_in_week_nights"] + cleaned["stays_in_weekend_nights"]
    assert (nights == 0).sum() == 0


def test_clean_data_removes_extreme_adr(raw_like_df: pd.DataFrame) -> None:
    """Los `adr` negativos o desorbitados (>=5400) se eliminan (EDA §8)."""
    cleaned = data_loader.clean_data(raw_like_df)
    assert (cleaned["adr"] < 0).sum() == 0
    assert (cleaned["adr"] >= 5400).sum() == 0


def test_clean_data_does_not_mutate_input(raw_like_df: pd.DataFrame) -> None:
    before = raw_like_df.copy()
    data_loader.clean_data(raw_like_df)
    pd.testing.assert_frame_equal(raw_like_df, before)


def test_get_feature_target_splits_target(raw_like_df: pd.DataFrame) -> None:
    cleaned = data_loader.clean_data(raw_like_df)
    X, y = data_loader.get_feature_target(cleaned)
    assert config.TARGET_COLUMN not in X.columns
    assert y.name == config.TARGET_COLUMN
    assert len(X) == len(y)
    assert set(np.unique(y)).issubset({0, 1})


def test_split_data_is_stratified_and_sized(raw_like_df: pd.DataFrame) -> None:
    cleaned = data_loader.clean_data(raw_like_df)
    X, y = data_loader.get_feature_target(cleaned)
    X_train, X_test, y_train, y_test = data_loader.split_data(X, y)

    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)
    # Proporción de test ~ TEST_SIZE.
    assert abs(len(X_test) / len(X) - config.TEST_SIZE) < 0.15
    # Reproducibilidad: misma semilla -> misma partición.
    X_train2, _, _, _ = data_loader.split_data(X, y)
    assert list(X_train.index) == list(X_train2.index)
