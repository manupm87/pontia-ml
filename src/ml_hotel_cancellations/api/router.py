"""Rutas de la API: todos los endpoints en un único router."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from . import service
from .schemas import (
    BatchRequest,
    BatchResponse,
    Booking,
    HealthResponse,
    ModelInfo,
    PredictionResponse,
)

router = APIRouter()


def require_model() -> None:
    """Dependencia: corta con 503 si el modelo no está disponible.

    Se aplica a los endpoints que necesitan el modelo cargado (predicción y
    metadatos), de modo que el guard se define una sola vez.
    """
    if not service.is_model_loaded():
        raise HTTPException(status_code=503, detail="Modelo no disponible.")


@router.get("/", tags=["General"], summary="Punto de entrada de la API")
def root() -> dict:
    """Devuelve un mensaje de bienvenida y apunta a la documentación."""
    return {
        "mensaje": "API de predicción de cancelaciones de reservas.",
        "documentacion": "/docs",
    }


@router.get(
    "/health",
    tags=["General"],
    summary="Sondeo de salud del servicio",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    """Comprueba que el servicio responde y que el modelo está cargado."""
    return HealthResponse(status="ok", model_loaded=service.is_model_loaded())


@router.get(
    "/model-info",
    tags=["General"],
    summary="Metadatos del modelo servido",
    response_model=ModelInfo,
    dependencies=[Depends(require_model)],
)
def model_info() -> ModelInfo:
    """Expone tipo de modelo, métrica, características y el origen (registry vs bundled)."""
    return ModelInfo(**service.get_model_info_payload())


@router.post(
    "/predict",
    tags=["Predicción"],
    summary="Predice la cancelación de una reserva",
    response_model=PredictionResponse,
    dependencies=[Depends(require_model)],
)
def predict(booking: Booking) -> PredictionResponse:
    """Predice si una reserva será cancelada.

    Devuelve la clase (0/1), su etiqueta legible y la probabilidad de cancelación.
    """
    result = service.predict_one(booking.model_dump())
    return PredictionResponse(**result)


@router.post(
    "/predict/batch",
    tags=["Predicción"],
    summary="Predice la cancelación de varias reservas",
    response_model=BatchResponse,
    dependencies=[Depends(require_model)],
)
def predict_batch(request: BatchRequest) -> BatchResponse:
    """Predice la cancelación para una lista de reservas (procesado por lotes)."""
    bookings = [b.model_dump() for b in request.bookings]
    results = service.predict_many(bookings)
    return BatchResponse(predictions=[PredictionResponse(**r) for r in results])
