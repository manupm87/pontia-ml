"""Aplicación FastAPI que sirve el modelo de cancelaciones de reservas.

Define los endpoints HTTP del contrato de la API. FastAPI genera además, de
forma automática, la documentación interactiva (Swagger UI en ``/docs`` y ReDoc
en ``/redoc``) a partir de los modelos Pydantic de ``schemas``.

Cómo arrancar el servidor (desde la carpeta ``project/``)::

    uvicorn api.main:app --reload

Luego abre http://127.0.0.1:8000/docs para probar los endpoints.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src import config

from . import service
from .schemas import (
    BatchRequest,
    BatchResponse,
    Booking,
    HealthResponse,
    ModelInfo,
    PredictionResponse,
)

app = FastAPI(
    title="API de predicción de cancelaciones de reservas",
    description=(
        "API REST (bonus del proyecto final de ML) que sirve el mejor modelo "
        "entrenado (XGBoost dentro de un Pipeline de scikit-learn) para predecir "
        "si una reserva de hotel será cancelada."
    ),
    version="1.0.0",
)

# CORS abierto: permite que una interfaz web (p. ej. Streamlit en otro puerto u
# origen) consuma esta API desde el navegador sin bloqueos de seguridad.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"], summary="Punto de entrada de la API")
def root() -> dict:
    """Devuelve un mensaje de bienvenida y apunta a la documentación."""
    return {
        "mensaje": "API de predicción de cancelaciones de reservas.",
        "documentacion": "/docs",
    }


@app.get(
    "/health",
    tags=["General"],
    summary="Sondeo de salud del servicio",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    """Comprueba que el servicio responde y que el modelo está cargado."""
    return HealthResponse(status="ok", model_loaded=service.is_model_loaded())


@app.get(
    "/model-info",
    tags=["General"],
    summary="Metadatos del modelo servido",
    response_model=ModelInfo,
)
def model_info() -> ModelInfo:
    """Expone el tipo de modelo, su métrica principal y las características usadas."""
    return ModelInfo(
        model_type="XGBoost",
        primary_metric=config.PRIMARY_METRIC,
        roc_auc=service.MODEL_ROC_AUC,
        n_features=len(config.NUMERIC_COLUMNS) + len(config.CATEGORICAL_COLUMNS),
        features={
            "numeric": config.NUMERIC_COLUMNS,
            "categorical": config.CATEGORICAL_COLUMNS,
        },
    )


@app.post(
    "/predict",
    tags=["Predicción"],
    summary="Predice la cancelación de una reserva",
    response_model=PredictionResponse,
)
def predict(booking: Booking) -> PredictionResponse:
    """Predice si una reserva será cancelada.

    Devuelve la clase (0/1), su etiqueta legible y la probabilidad de cancelación.
    """
    resultado = service.predict_one(booking.model_dump())
    return PredictionResponse(**resultado)


@app.post(
    "/predict/batch",
    tags=["Predicción"],
    summary="Predice la cancelación de varias reservas",
    response_model=BatchResponse,
)
def predict_batch(request: BatchRequest) -> BatchResponse:
    """Predice la cancelación para una lista de reservas (procesado por lotes)."""
    bookings = [b.model_dump() for b in request.bookings]
    resultados = service.predict_many(bookings)
    return BatchResponse(predictions=[PredictionResponse(**r) for r in resultados])
