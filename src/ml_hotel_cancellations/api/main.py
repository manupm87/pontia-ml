"""Aplicación FastAPI que sirve el modelo de cancelaciones de reservas.

Arranque: ``uvicorn ml_hotel_cancellations.api.main:app --reload`` → /docs.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .router import router

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

app.include_router(router)
