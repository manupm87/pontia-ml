"""Tests de `src.predict` (inferencia con el modelo bundled).

Marcados como `slow` porque cargan el `Pipeline` completo desde disco.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import predict

pytestmark = pytest.mark.slow


def test_predict_dataframe_shape_and_columns(bundled_model, raw_like_df: pd.DataFrame) -> None:
    result = predict.predict_dataframe(raw_like_df, model=bundled_model)
    assert list(result.columns) == ["prediction", "probability_canceled"]
    assert len(result) == len(raw_like_df)


def test_predict_dataframe_values_in_range(bundled_model, raw_like_df: pd.DataFrame) -> None:
    result = predict.predict_dataframe(raw_like_df, model=bundled_model)
    assert set(np.unique(result["prediction"])).issubset({0, 1})
    assert (result["probability_canceled"] >= 0.0).all()
    assert (result["probability_canceled"] <= 1.0).all()


def test_predict_dataframe_preserves_input_index(bundled_model) -> None:
    """La salida conserva el índice de entrada (alineación fila a fila)."""
    from api.schemas import BOOKING_EXAMPLE

    df = pd.DataFrame([BOOKING_EXAMPLE, BOOKING_EXAMPLE], index=["reserva_A", "reserva_B"])
    result = predict.predict_dataframe(df, model=bundled_model)
    assert list(result.index) == ["reserva_A", "reserva_B"]


def test_prepare_for_inference_drops_target(bundled_model) -> None:
    from api.schemas import BOOKING_EXAMPLE
    from src import config

    df = pd.DataFrame([{**BOOKING_EXAMPLE, config.TARGET_COLUMN: 1}])
    prepared = predict.prepare_for_inference(df)
    assert config.TARGET_COLUMN not in prepared.columns


def test_prepare_for_inference_keeps_all_rows(raw_like_df: pd.DataFrame) -> None:
    """A diferencia de clean_data, la inferencia NO descarta filas."""
    prepared = predict.prepare_for_inference(raw_like_df)
    assert len(prepared) == len(raw_like_df)
