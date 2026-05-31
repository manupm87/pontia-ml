"""Tests de la lógica de `ui/` que no requiere arrancar Streamlit.

`ui.data` y `ui.booking` se pueden importar sin Streamlit (la caché cae a
`functools.lru_cache`). Para las agregaciones de EDA se inyecta un dataset
sintético monkeypatcheando `ui.data.load_dataset`.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ml_hotel_cancellations.ui import booking
from ml_hotel_cancellations.ui import config as ui_config
from ml_hotel_cancellations.ui import data as ui_data


# ---------------------------------------------------------------------------
# ui.booking
# ---------------------------------------------------------------------------
def test_form_has_26_fields() -> None:
    """El formulario describe las 26 características de entrada."""
    assert len(booking.all_fields()) == 26


def test_form_field_names_match_example() -> None:
    """Los nombres de los campos coinciden con la reserva de ejemplo."""
    field_names = {f.name for f in booking.all_fields()}
    assert field_names == set(booking.EXAMPLE_BOOKING)


def test_field_kinds_are_valid() -> None:
    valid = {"categorical", "int", "float"}
    assert all(f.kind in valid for f in booking.all_fields())


def test_categorical_fields_have_options_key() -> None:
    for fld in booking.all_fields():
        if fld.kind == "categorical":
            assert fld.options_key, f"{fld.name} categórico sin options_key"


def test_build_payload_casts_types() -> None:
    payload = booking.build_payload(booking.EXAMPLE_BOOKING)
    assert isinstance(payload["lead_time"], int)
    assert isinstance(payload["adr"], float)
    assert isinstance(payload["agent"], str)
    assert set(payload) == set(booking.EXAMPLE_BOOKING)


# ---------------------------------------------------------------------------
# ui.config
# ---------------------------------------------------------------------------
def test_base_cancellation_rate_is_a_probability() -> None:
    assert 0.0 < ui_config.BASE_CANCELLATION_RATE < 1.0


def test_class_labels_binary() -> None:
    assert set(ui_config.CLASS_LABELS) == {0, 1}


# ---------------------------------------------------------------------------
# ui.data — cliente API
# ---------------------------------------------------------------------------
def test_is_remote_api_false_for_localhost(monkeypatch) -> None:
    monkeypatch.setattr(ui_data.config, "API_BASE_URL", "http://localhost:8000")
    assert ui_data.is_remote_api() is False


def test_is_remote_api_true_for_remote(monkeypatch) -> None:
    monkeypatch.setattr(ui_data.config, "API_BASE_URL", "https://pontia.onrender.com")
    assert ui_data.is_remote_api() is True


# ---------------------------------------------------------------------------
# ui.data — agregaciones de EDA (con dataset sintético)
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_dataset(monkeypatch):
    df = pd.DataFrame(
        {
            "is_canceled": [0, 1, 1, 0, 1, 0],
            "hotel": ["City", "City", "Resort", "Resort", "City", "Resort"],
            "lead_time": [10, 200, 50, 5, 300, 20],
            "adr": [80.0, 120.0, 95.0, 60.0, 150.0, 70.0],
            "stays_in_week_nights": [2, 5, 3, 1, 4, 2],
            "total_of_special_requests": [1, 0, 2, 1, 0, 3],
        }
    )
    # `load_dataset` está cacheado; lo reemplazamos por una función directa.
    monkeypatch.setattr(ui_data, "load_dataset", lambda *a, **k: df)
    return df


def test_cancellation_rate_by_sorted_desc(fake_dataset) -> None:
    result = ui_data.cancellation_rate_by("hotel")
    assert list(result.columns) == ["categoria", "tasa_cancelacion", "reservas"]
    # Ordenado de mayor a menor tasa.
    assert result["tasa_cancelacion"].is_monotonic_decreasing
    assert result["reservas"].sum() == len(fake_dataset)


def test_class_balance_percentages_sum_100(fake_dataset) -> None:
    result = ui_data.class_balance()
    assert abs(result["porcentaje"].sum() - 100.0) < 0.1
    assert result["reservas"].sum() == len(fake_dataset)


def test_numeric_summary_returns_relevant_columns(fake_dataset) -> None:
    result = ui_data.numeric_summary()
    variables = set(result["variable"])
    assert {"lead_time", "adr"}.issubset(variables)
