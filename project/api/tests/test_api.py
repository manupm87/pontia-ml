"""Pruebas de integración de la API con ``TestClient``.

``TestClient`` levanta la aplicación FastAPI en memoria (sin necesidad de un
servidor real) y nos permite hacer peticiones HTTP a los endpoints. Comprobamos
el contrato: forma de las respuestas, rangos de probabilidad y validación de
entradas incorrectas.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app
from api.schemas import BOOKING_EXAMPLE

client = TestClient(app)


def test_health() -> None:
    """/health responde 200 con el modelo cargado."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "ok", "model_loaded": True}


def test_root() -> None:
    """/ apunta a la documentación."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["documentacion"] == "/docs"


def test_model_info() -> None:
    """/model-info devuelve los metadatos esperados del modelo."""
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_type"] == "XGBoost"
    assert body["primary_metric"] == "roc_auc"
    assert body["roc_auc"] == 0.9614
    # 16 numéricas + 11 categóricas = 27 características de entrada.
    assert body["n_features"] == 27
    assert len(body["features"]["numeric"]) == 16
    assert len(body["features"]["categorical"]) == 11


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
    predictions = response.json()["predictions"]
    assert len(predictions) == 2
    for pred in predictions:
        assert pred["prediction"] in (0, 1)
        assert 0.0 <= pred["probability"] <= 1.0


def test_predict_validation_error() -> None:
    """Una reserva con un campo obligatorio ausente devuelve 422."""
    incompleta = {k: v for k, v in BOOKING_EXAMPLE.items() if k != "hotel"}
    response = client.post("/predict", json=incompleta)
    assert response.status_code == 422
