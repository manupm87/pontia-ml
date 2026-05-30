"""Esquema y valores por defecto de la reserva (las 27 variables de entrada).

Separa la definición de qué pide el modelo del renderizado del formulario. El
ejemplo por defecto coincide con el contrato de la API, así el formulario abre
con una reserva válida lista para enviar.
"""

from __future__ import annotations

from dataclasses import dataclass

from ml_hotel_cancellations import config

# Reserva de ejemplo (contrato de la API). Nombres de campo EXACTOS.
# Fuente única de verdad en `src.config`, compartida con el esquema de la API.
EXAMPLE_BOOKING: dict = config.BOOKING_EXAMPLE


@dataclass(frozen=True)
class Field:
    """Descripción de un campo del formulario.

    `name` es el nombre exacto que espera la API; `kind` es "categorical",
    "int" o "float"; `min`/`max`/`step` aplican a numéricos; `options_key` es
    la clave en `get_categorical_options()` para las categóricas.
    """

    name: str
    label: str
    kind: str
    help: str = ""
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options_key: str | None = None


# Agrupamos los campos en secciones para que el formulario sea navegable.
FORM_SECTIONS: dict[str, list[Field]] = {
    "Reserva y fechas": [
        Field("hotel", "Tipo de hotel", "categorical",
              "Hotel urbano (City) o vacacional (Resort).", options_key="hotel"),
        Field("lead_time", "Antelación (días)", "int",
              "Días entre la fecha de reserva y la de llegada. Más antelación "
              "suele asociarse a más cancelaciones.", min=0, max=800, step=1),
        Field("arrival_date_month", "Mes de llegada", "categorical",
              "Mes previsto de entrada al hotel.", options_key="arrival_date_month"),
        Field("arrival_date_week_number", "Semana del año", "int",
              "Número de semana ISO de la llegada (1-53).", min=1, max=53, step=1),
        Field("arrival_date_day_of_month", "Día del mes", "int",
              "Día del mes de la llegada (1-31).", min=1, max=31, step=1),
    ],
    "Estancia y ocupantes": [
        Field("stays_in_weekend_nights", "Noches de fin de semana", "int",
              "Noches en sábado/domingo de la estancia.", min=0, max=20, step=1),
        Field("stays_in_week_nights", "Noches entre semana", "int",
              "Noches de lunes a viernes de la estancia.", min=0, max=50, step=1),
        Field("adults", "Adultos", "int", "Número de adultos.", min=0, max=10, step=1),
        Field("children", "Niños", "int", "Número de niños.", min=0, max=10, step=1),
        Field("babies", "Bebés", "int", "Número de bebés.", min=0, max=10, step=1),
        Field("meal", "Régimen de comidas", "categorical",
              "BB=desayuno, HB=media pensión, FB=pensión completa, SC=sin comidas.",
              options_key="meal"),
        Field("total_of_special_requests", "Peticiones especiales", "int",
              "Nº de peticiones especiales (cuna, vistas...). Suele indicar "
              "compromiso del cliente y menos cancelaciones.", min=0, max=10, step=1),
    ],
    "Cliente y canal": [
        Field("country", "País de origen", "categorical",
              "País del cliente (código ISO). Se ofrecen los más frecuentes.",
              options_key="country"),
        Field("market_segment", "Segmento de mercado", "categorical",
              "Canal de captación (TA=agencia de viajes, TO=turoperador...).",
              options_key="market_segment"),
        Field("distribution_channel", "Canal de distribución", "categorical",
              "Vía por la que llegó la reserva.", options_key="distribution_channel"),
        Field("customer_type", "Tipo de cliente", "categorical",
              "Transient=individual, Contract=con contrato, Group=grupo...",
              options_key="customer_type"),
        Field("agent", "Agencia (ID)", "categorical",
              "Identificador de la agencia de viajes. Se ofrecen los más comunes.",
              options_key="agent"),
        Field("company", "Empresa (ID)", "categorical",
              "Identificador de la empresa; 'no_company' si es una reserva particular.",
              options_key="company"),
        Field("is_repeated_guest", "¿Cliente repetidor?", "int",
              "1 si ya se había alojado antes, 0 en caso contrario.",
              min=0, max=1, step=1),
    ],
    "Historial y condiciones": [
        Field("previous_cancellations", "Cancelaciones previas", "int",
              "Reservas que este cliente canceló en el pasado.", min=0, max=50, step=1),
        Field("previous_bookings_not_canceled", "Reservas previas cumplidas", "int",
              "Reservas anteriores que NO canceló.", min=0, max=100, step=1),
        Field("reserved_room_type", "Habitación reservada", "categorical",
              "Tipo de habitación reservada (código anonimizado).",
              options_key="reserved_room_type"),
        Field("assigned_room_type", "Habitación asignada", "categorical",
              "Tipo finalmente asignado. Diferir de la reservada puede influir.",
              options_key="assigned_room_type"),
        Field("booking_changes", "Cambios en la reserva", "int",
              "Nº de modificaciones hechas tras reservar.", min=0, max=30, step=1),
        Field("deposit_type", "Tipo de depósito", "categorical",
              "No Deposit=sin depósito, Non Refund=no reembolsable (muy ligado a "
              "cancelaciones), Refundable=reembolsable.", options_key="deposit_type"),
        Field("days_in_waiting_list", "Días en lista de espera", "int",
              "Días que la reserva estuvo en lista de espera.", min=0, max=400, step=1),
        Field("adr", "Tarifa media diaria (ADR)", "float",
              "Precio medio por noche en euros (Average Daily Rate).",
              min=0.0, max=1000.0, step=1.0),
    ],
}


def all_fields() -> list[Field]:
    """Devuelve la lista plana de los 27 campos (en orden de sección)."""
    fields: list[Field] = []
    for section_fields in FORM_SECTIONS.values():
        fields.extend(section_fields)
    return fields


def build_payload(values: dict) -> dict:
    """Construye el cuerpo JSON para la API casteando cada campo a su tipo
    (int/float/str) según el esquema."""
    payload: dict = {}
    for fld in all_fields():
        raw = values.get(fld.name, EXAMPLE_BOOKING.get(fld.name))
        if fld.kind == "int":
            payload[fld.name] = int(raw)
        elif fld.kind == "float":
            payload[fld.name] = float(raw)
        else:  # categorical -> string
            payload[fld.name] = str(raw)
    return payload
