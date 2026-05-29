"""Tests adicionales de la API (amplían `api/tests/test_api.py`).

Usan la fixture `api_client` (definida en el conftest raíz) para no recrear el
`TestClient` en cada módulo.
"""

from __future__ import annotations

import pytest

from ml_hotel_cancellations import config


@pytest.mark.slow
def test_predict_batch_large(api_client, booking_example) -> None:
    """Un lote de N reservas devuelve exactamente N predicciones, en orden."""
    payload = {"bookings": [booking_example for _ in range(10)]}
    response = api_client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 10


def test_predict_rejects_out_of_range_week(api_client, booking_example) -> None:
    """`arrival_date_week_number` fuera de [1, 53] -> 422 (validación Pydantic)."""
    booking_example["arrival_date_week_number"] = 99
    response = api_client.post("/predict", json=booking_example)
    assert response.status_code == 422


def test_predict_rejects_negative_lead_time(api_client, booking_example) -> None:
    booking_example["lead_time"] = -5
    response = api_client.post("/predict", json=booking_example)
    assert response.status_code == 422


def test_model_info_features_match_config(api_client) -> None:
    """Las features reportadas coinciden con la configuración del proyecto."""
    body = api_client.get("/model-info").json()
    assert body["features"]["numeric"] == config.NUMERIC_COLUMNS
    assert body["features"]["categorical"] == config.CATEGORICAL_COLUMNS
    assert body["n_features"] == len(config.FEATURE_COLUMNS)
