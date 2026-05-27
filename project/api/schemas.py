"""Contratos de datos (modelos Pydantic) de la API.

Pydantic valida automáticamente la entrada de cada petición y genera la
documentación interactiva (Swagger/OpenAPI). Aquí definimos:

- ``Booking``           : una reserva con las 27 características de entrada del
                          modelo (las mismas columnas crudas del entrenamiento).
- ``PredictionResponse``: el resultado de una predicción.
- ``BatchRequest`` / ``BatchResponse`` : variantes para predicción por lotes.
- ``HealthResponse``    : estado del servicio.
- ``ModelInfo``         : metadatos del modelo servido.

Nótese que las 27 características = ``NUMERIC_COLUMNS`` (16) + ``CATEGORICAL_COLUMNS``
(11) definidas en ``src.config``. No incluimos ``arrival_date_year``, ``company``,
las columnas de ``reservation_status`` ni el target ``is_canceled``, porque el
modelo no las usa como entrada.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Ejemplo de reserva válido reutilizado en el esquema y en la documentación.
# Sirve para que Swagger muestre un cuerpo de petición listo para "probar".
BOOKING_EXAMPLE: dict = {
    "hotel": "City Hotel",
    "lead_time": 100,
    "arrival_date_month": "August",
    "arrival_date_week_number": 33,
    "arrival_date_day_of_month": 15,
    "stays_in_weekend_nights": 2,
    "stays_in_week_nights": 5,
    "adults": 2,
    "children": 0,
    "babies": 0,
    "meal": "BB",
    "country": "PRT",
    "market_segment": "Online TA",
    "distribution_channel": "TA/TO",
    "is_repeated_guest": 0,
    "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0,
    "reserved_room_type": "A",
    "assigned_room_type": "A",
    "booking_changes": 0,
    "deposit_type": "No Deposit",
    "agent": "9",
    "days_in_waiting_list": 0,
    "customer_type": "Transient",
    "adr": 100.0,
    "required_car_parking_spaces": 0,
    "total_of_special_requests": 1,
}


class Booking(BaseModel):
    """Una reserva de hotel con las 27 características de entrada del modelo.

    El orden y los nombres de los campos coinciden EXACTAMENTE con las columnas
    crudas que el ``Pipeline`` espera. El propio pipeline se encarga del
    preprocesado (codificación de categóricas, escalado, etc.), por lo que aquí
    solo recibimos los valores "tal cual" los daría un sistema de reservas.

    Notas de tipos:
    - ``agent`` es un *string* (un ID de agencia tratado como categoría). Acepta
      por ejemplo ``"9"`` o ``"Unknown"``.
    - ``children`` puede venir como float en el dataset original; lo aceptamos
      como ``float`` para no rechazar entradas válidas.
    """

    # --- Categóricas (11) ---
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
    agent: str = Field(..., description="ID de agencia como texto (p. ej. '9' o 'Unknown').")

    # --- Numéricas (16) ---
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
    required_car_parking_spaces: int = Field(..., ge=0, description="Plazas de parking solicitadas.")
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
    """Metadatos del modelo servido, útiles para una interfaz cliente."""

    model_type: str = Field(..., description="Familia del modelo (p. ej. 'XGBoost').")
    primary_metric: str = Field(..., description="Métrica principal de selección (p. ej. 'roc_auc').")
    roc_auc: float = Field(..., description="ROC-AUC obtenido en el conjunto de test.")
    n_features: int = Field(..., description="Número de características de entrada (27).")
    features: dict = Field(
        ..., description="Características divididas en 'numeric' y 'categorical'."
    )

    # ``model_`` es un prefijo protegido en Pydantic v2; lo permitimos
    # explícitamente para poder usar 'model_type'.
    model_config = ConfigDict(protected_namespaces=())
