"""Fixtures compartidas por toda la suite de tests.

Vive en la raíz para que tanto `tests/` como `api/tests/` puedan reutilizar las
mismas piezas: un DataFrame sintético de reservas, el modelo bundled cargado una
sola vez por sesión y un cliente de la API.
"""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from api.schemas import BOOKING_EXAMPLE


def _make_booking(**overrides) -> dict:
    """Devuelve una copia del booking de ejemplo con los campos sobreescritos."""
    booking = copy.deepcopy(BOOKING_EXAMPLE)
    booking.update(overrides)
    return booking


@pytest.fixture
def booking_example() -> dict:
    """Una reserva válida (las 27 características), como dict mutable."""
    return copy.deepcopy(BOOKING_EXAMPLE)


@pytest.fixture
def raw_like_df() -> pd.DataFrame:
    """DataFrame que imita el CSV crudo: 27 features + target + columnas a tirar.

    Incluye a propósito casos que ejercitan la limpieza:
    - columnas de leakage / baja utilidad (`company`, `reservation_status`,
      `reservation_status_date`) que `clean_data` debe eliminar.
    - una fila sin huéspedes (adults+children+babies == 0) que se descarta.
    - un `agent` y una categórica ausentes (NaN) que se normalizan a "Unknown".
    """
    base = _make_booking()

    filas = [
        # Fila normal (cancela).
        {**base, "is_canceled": 1},
        # Fila normal (no cancela), otro hotel.
        {**_make_booking(hotel="Resort Hotel", lead_time=10, adr=80.0), "is_canceled": 0},
        # Fila con agent y country ausentes -> deben pasar a "Unknown".
        {**_make_booking(agent=None, country=None), "is_canceled": 0},
        # Fila SIN huéspedes -> clean_data debe eliminarla.
        {**_make_booking(adults=0, children=0, babies=0), "is_canceled": 0},
        # Más filas para que la partición estratificada tenga material de sobra.
        *[{**_make_booking(lead_time=i * 5), "is_canceled": i % 2} for i in range(16)],
    ]

    df = pd.DataFrame(filas)

    # Columnas crudas que NO son features y que el pipeline de datos elimina.
    df["arrival_date_year"] = 2016
    df["company"] = None
    df["reservation_status"] = ["Canceled" if c else "Check-Out" for c in df["is_canceled"]]
    df["reservation_status_date"] = "2016-08-15"
    return df


@pytest.fixture(scope="session")
def bundled_model():
    """Carga el `models/best_model.pkl` una sola vez por sesión (es costoso)."""
    from src.predict import load_best_model

    return load_best_model()


@pytest.fixture(scope="session")
def api_client():
    """Cliente HTTP en memoria contra la app FastAPI (sin servidor real)."""
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as client:
        yield client
