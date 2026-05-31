"""Tests de la API FastAPI (`/health`, `/predict`, `/predict/batch`, `/model-info`).

Usan el `TestClient` de FastAPI (sin levantar servidor). El modelo se carga una vez.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from ml_hotel_cancellations import config
from ml_hotel_cancellations.api.main import app
from ml_hotel_cancellations.api.schemas import BOOKING_EXAMPLE

client = TestClient(app)


def test_health() -> None:
    """/health responde y reporta el estado del modelo."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_model_info() -> None:
    """/model-info expone metadatos del modelo servido."""
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_type"] == "XGBoost"
    assert body["primary_metric"] == "roc_auc"
    # El ROC-AUC sale del artefacto de métricas (no un literal): comprobamos rango.
    assert 0.5 < body["roc_auc"] <= 1.0
    # 15 numéricas + 11 categóricas = 26 características de entrada.
    assert body["n_features"] == 26
    assert len(body["features"]["numeric"]) == len(config.NUMERIC_COLUMNS) == 15
    assert len(body["features"]["categorical"]) == len(config.CATEGORICAL_COLUMNS) == 11


def test_predict() -> None:
    """/predict devuelve la forma correcta y una probabilidad en [0, 1]."""
    response = client.post("/predict", json=BOOKING_EXAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("No cancelada", "Cancelada")
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_batch() -> None:
    """/predict/batch devuelve una predicción por cada reserva de entrada."""
    payload = {"bookings": [BOOKING_EXAMPLE, BOOKING_EXAMPLE]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == len(payload["bookings"])
