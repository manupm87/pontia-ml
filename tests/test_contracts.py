"""Tests de contrato: fuente única de verdad entre `src`, `api` y `ui`.

Estos tests CODIFICAN el estado objetivo de la Fase 1 del plan de mejora
(`agents/code-quality-tasks.md`). Algunos fallan a propósito contra el código
actual (RED) y pasan cuando se elimina la duplicación (GREEN).
"""

from __future__ import annotations

import pandas as pd

from ml_hotel_cancellations import config


# ---------------------------------------------------------------------------
# T1.1 — Etiquetas de clase únicas y coherentes
# ---------------------------------------------------------------------------
def test_canonical_class_labels_exist() -> None:
    """`src.config` define las etiquetas cortas canónicas."""
    assert hasattr(config, "CLASS_LABELS_SHORT")
    assert config.CLASS_LABELS_SHORT == ["No cancelada", "Cancelada"]


def test_api_class_labels_derive_from_config() -> None:
    from ml_hotel_cancellations.api import service

    assert list(service.CLASS_LABELS) == config.CLASS_LABELS_SHORT


def test_ui_class_labels_derive_from_config() -> None:
    from ml_hotel_cancellations.ui import config as ui_config

    assert ui_config.CLASS_LABELS[0] == config.CLASS_LABELS_SHORT[0]
    assert ui_config.CLASS_LABELS[1] == config.CLASS_LABELS_SHORT[1]


# ---------------------------------------------------------------------------
# T1.3 — Umbral de decisión con nombre
# ---------------------------------------------------------------------------
def test_decision_threshold_in_config() -> None:
    assert hasattr(config, "DECISION_THRESHOLD")
    assert 0.0 < config.DECISION_THRESHOLD < 1.0


def test_predict_uses_config_threshold() -> None:
    """`src.predict` no debe llevar el 0.5 hardcodeado."""
    import inspect

    from ml_hotel_cancellations.ml import predict

    source = inspect.getsource(predict.predict_dataframe)
    assert ">= 0.5" not in source
    assert "DECISION_THRESHOLD" in source


# ---------------------------------------------------------------------------
# T1.4 — Reserva de ejemplo con una única definición
# ---------------------------------------------------------------------------
def test_booking_example_single_source() -> None:
    """El ejemplo de reserva vive en `src.config` y lo reutilizan api y ui."""
    assert hasattr(config, "BOOKING_EXAMPLE")
    from ml_hotel_cancellations.api import schemas
    from ml_hotel_cancellations.ui import booking

    assert schemas.BOOKING_EXAMPLE is config.BOOKING_EXAMPLE
    assert booking.EXAMPLE_BOOKING is config.BOOKING_EXAMPLE


def test_booking_example_matches_schema_fields() -> None:
    from ml_hotel_cancellations.api.schemas import Booking

    assert set(config.BOOKING_EXAMPLE) == set(Booking.model_fields)


# ---------------------------------------------------------------------------
# T1.6 — Mapa de familias de modelo único
# ---------------------------------------------------------------------------
def test_model_family_map_in_config() -> None:
    assert hasattr(config, "MODEL_FAMILY")
    expected = {
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "XGBoost",
        "Neural Network (Keras)",
    }
    assert expected.issubset(set(config.MODEL_FAMILY))


def test_modules_reuse_config_model_family() -> None:
    """train/tuning/balancing no redefinen su propio mapa de familias."""
    from ml_hotel_cancellations.ml import balancing, train, tuning

    assert train._MODEL_FAMILY is config.MODEL_FAMILY
    assert tuning._MODEL_FAMILY is config.MODEL_FAMILY
    assert balancing._MODEL_FAMILY is config.MODEL_FAMILY


# ---------------------------------------------------------------------------
# T1.2 — El ROC-AUC reportado procede del artefacto de métricas, no de un literal
# ---------------------------------------------------------------------------
def test_reported_roc_auc_matches_metrics_artifact() -> None:
    metrics = pd.read_csv(config.METRICS_TABLE_PATH, index_col=0)
    best_auc = metrics["roc_auc"].max()

    from ml_hotel_cancellations.api import service

    assert round(service.MODEL_ROC_AUC, 4) == round(float(best_auc), 4)
