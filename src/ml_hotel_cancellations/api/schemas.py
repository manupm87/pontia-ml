"""Contratos de datos (modelos Pydantic) de la API: validan la entrada y generan el OpenAPI.

Las 27 características de ``Booking`` = ``NUMERIC_COLUMNS`` (15) +
``CATEGORICAL_COLUMNS`` (12) de ``config``, las mismas columnas crudas que el modelo usa.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ml_hotel_cancellations import config

# Ejemplo para Swagger; fuente única en `config` (compartida con la UI).
BOOKING_EXAMPLE: dict = config.BOOKING_EXAMPLE


class Booking(BaseModel):
    """Una reserva de hotel con las 27 características crudas que espera el ``Pipeline``.

    El pipeline se encarga del preprocesado; aquí solo se reciben los valores tal cual.
    ``agent`` es string (ID de agencia como categoría) y ``children`` es float a propósito.
    """

    # --- Categóricas (12) ---
    hotel: str = Field(..., description="Tipo de hotel: 'City Hotel' o 'Resort Hotel'.")
    arrival_date_month: str = Field(..., description="Mes de llegada (p. ej. 'August').")
    meal: str = Field(..., description="Régimen de comidas (p. ej. 'BB').")
    country: str = Field(..., description="País de origen (código ISO, p. ej. 'PRT').")
    market_segment: str = Field(..., description="Segmento de mercado (p. ej. 'Online TA').")
    distribution_channel: str = Field(..., description="Canal de distribución (p. ej. 'TA/TO').")
    reserved_room_type: str = Field(..., description="Tipo de habitación reservada.")
    assigned_room_type: str = Field(..., description="Tipo de habitación asignada.")
    deposit_type: str = Field(..., description="Tipo de depósito (p. ej. 'No Deposit').")
    customer_type: str = Field(..., description="Tipo de cliente (p. ej. 'Transient').")
    agent: str = Field(..., description="ID de agencia como texto (p. ej. '9' o 'Desconocido').")
    company: str = Field(..., description="ID de empresa como texto; 'no_company' si no hay empresa.")

    # --- Numéricas (15) ---
    lead_time: int = Field(..., ge=0, description="Días de antelación entre reserva y llegada.")
    arrival_date_week_number: int = Field(..., ge=1, le=53, description="Semana del año de llegada.")
    arrival_date_day_of_month: int = Field(..., ge=1, le=31, description="Día del mes de llegada.")
    stays_in_weekend_nights: int = Field(..., ge=0, description="Noches de fin de semana reservadas.")
    stays_in_week_nights: int = Field(..., ge=0, description="Noches entre semana reservadas.")
    adults: int = Field(..., ge=0, description="Número de adultos.")
    children: float = Field(..., ge=0, description="Número de niños (puede ser float).")
    babies: int = Field(..., ge=0, description="Número de bebés.")
    is_repeated_guest: int = Field(..., ge=0, le=1, description="1 si es huésped recurrente, 0 si no.")
    previous_cancellations: int = Field(..., ge=0, description="Cancelaciones previas del cliente.")
    previous_bookings_not_canceled: int = Field(
        ..., ge=0, description="Reservas previas no canceladas del cliente."
    )
    booking_changes: int = Field(..., ge=0, description="Número de cambios hechos a la reserva.")
    days_in_waiting_list: int = Field(..., ge=0, description="Días en lista de espera.")
    adr: float = Field(..., description="Tarifa media diaria (Average Daily Rate).")
    total_of_special_requests: int = Field(..., ge=0, description="Número de peticiones especiales.")

    model_config = ConfigDict(json_schema_extra={"example": BOOKING_EXAMPLE})


class PredictionResponse(BaseModel):
    """Resultado de una predicción para una reserva."""

    prediction: int = Field(..., description="Clase predicha: 0 = no cancela, 1 = cancela.")
    label: str = Field(..., description="Etiqueta legible: 'No cancelada' o 'Cancelada'.")
    probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probabilidad estimada de cancelación P(cancela), en [0, 1]."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"prediction": 1, "label": "Cancelada", "probability": 0.87}
        }
    )


class BatchRequest(BaseModel):
    """Petición de predicción por lotes: una lista de reservas."""

    bookings: list[Booking] = Field(..., description="Lista de reservas a evaluar.")

    model_config = ConfigDict(
        json_schema_extra={"example": {"bookings": [BOOKING_EXAMPLE]}}
    )


class BatchResponse(BaseModel):
    """Respuesta de predicción por lotes: una predicción por reserva, en orden."""

    predictions: list[PredictionResponse] = Field(
        ..., description="Predicciones en el mismo orden que las reservas de entrada."
    )


class HealthResponse(BaseModel):
    """Estado del servicio (sondeo de salud / *health check*)."""

    status: str = Field(..., description="'ok' si el servicio responde.")
    model_loaded: bool = Field(..., description="True si el modelo se cargó correctamente.")


class ModelInfo(BaseModel):
    """Metadatos del modelo servido, incluido su origen (registry MLflow vs pickle bundled)."""

    model_type: str = Field(..., description="Familia del modelo (p. ej. 'XGBoost').")
    primary_metric: str = Field(..., description="Métrica principal de selección (p. ej. 'roc_auc').")
    roc_auc: float = Field(..., description="ROC-AUC obtenido en el conjunto de test.")
    n_features: int = Field(..., description="Número de características de entrada (27).")
    features: dict = Field(
        ..., description="Características divididas en 'numeric' y 'categorical'."
    )

    # --- Origen del modelo (MLOps) ---
    source: str = Field(
        ...,
        description=(
            "Origen del modelo cargado: 'registry' (descargado del Model "
            "Registry MLflow) o 'bundled' (pickle versionado en el repo). "
            "Si la carga desde el registry falla, la API cae a 'bundled' y "
            "rellena 'fallback_reason'."
        ),
    )
    registry_uri: str | None = Field(
        default=None,
        description=(
            "URI del registry cuando source='registry'. Típicamente "
            "'models:/pontia-cancellations/Production'."
        ),
    )
    version: int | None = Field(
        default=None,
        description="Versión registrada del modelo (solo si source='registry').",
    )
    stage: str | None = Field(
        default=None,
        description="Stage del modelo (Staging/Production/Archived/None).",
    )
    run_id: str | None = Field(
        default=None,
        description="ID del run MLflow que generó esa versión, si se conoce.",
    )
    fallback_reason: str | None = Field(
        default=None,
        description=(
            "Solo presente cuando se intentó cargar del registry y falló. "
            "Contiene el tipo y mensaje del error para diagnóstico."
        ),
    )

    # ``model_`` es un prefijo protegido en Pydantic v2; lo permitimos
    # explícitamente para poder usar 'model_type'.
    model_config = ConfigDict(protected_namespaces=())
