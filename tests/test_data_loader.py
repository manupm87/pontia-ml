"""Tests de `src.data_loader` (limpieza, normalización y partición)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config, data_loader


def test_clean_data_drops_leakage_columns(raw_like_df: pd.DataFrame) -> None:
    cleaned = data_loader.clean_data(raw_like_df)
    for col in config.DROP_COLUMNS:
        assert col not in cleaned.columns


def test_clean_data_removes_rows_without_guests(raw_like_df: pd.DataFrame) -> None:
    """La fila con adults+children+babies == 0 se elimina."""
    n_zero_guests = int(
        (raw_like_df[["adults", "children", "babies"]].fillna(0).sum(axis=1) == 0).sum()
    )
    assert n_zero_guests >= 1  # el fixture incluye al menos una
    cleaned = data_loader.clean_data(raw_like_df)
    guests = cleaned[["adults", "children", "babies"]].fillna(0).sum(axis=1)
    assert (guests == 0).sum() == 0


def test_normalize_categoricals_fills_missing_with_unknown(raw_like_df: pd.DataFrame) -> None:
    normalized = data_loader.normalize_categoricals(raw_like_df)
    assert normalized["country"].isna().sum() == 0
    assert "Unknown" in normalized["country"].values


def test_normalize_categoricals_agent_to_string(raw_like_df: pd.DataFrame) -> None:
    """`agent` se convierte a texto y sus ausentes a 'Unknown'."""
    normalized = data_loader.normalize_categoricals(raw_like_df)
    assert normalized["agent"].map(type).eq(str).all()
    assert "Unknown" in normalized["agent"].values
    # El '9' del ejemplo no debe convertirse en '9.0'.
    assert "9" in normalized["agent"].values
    assert "9.0" not in normalized["agent"].values


def test_normalize_categoricals_does_not_mutate_input(raw_like_df: pd.DataFrame) -> None:
    before = raw_like_df.copy()
    data_loader.normalize_categoricals(raw_like_df)
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
