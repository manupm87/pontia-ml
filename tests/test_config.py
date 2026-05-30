"""Tests de las invariantes de `src.config`.

Fijan el contrato de la definición del problema: número de características,
columnas, métricas y rangos de las constantes clave.
"""

from __future__ import annotations

from ml_hotel_cancellations import config


def test_feature_counts() -> None:
    """15 numéricas + 12 categóricas = 27 características de entrada."""
    assert len(config.NUMERIC_COLUMNS) == 15
    assert len(config.CATEGORICAL_COLUMNS) == 12
    assert len(config.NUMERIC_COLUMNS) + len(config.CATEGORICAL_COLUMNS) == 27


def test_no_overlap_numeric_categorical() -> None:
    """Una columna no puede ser a la vez numérica y categórica."""
    assert set(config.NUMERIC_COLUMNS).isdisjoint(config.CATEGORICAL_COLUMNS)


def test_arrival_year_excluded() -> None:
    """`arrival_date_year` se excluye a propósito de las features (EDA §6)."""
    assert "arrival_date_year" not in config.NUMERIC_COLUMNS
    assert "arrival_date_year" not in config.CATEGORICAL_COLUMNS
    assert "arrival_date_year" in config.DROP_COLUMNS


def test_parking_is_leakage_and_dropped() -> None:
    """`required_car_parking_spaces` es fuga (EDA §11): fuera de features y se descarta."""
    assert "required_car_parking_spaces" not in config.FEATURE_COLUMNS
    assert "required_car_parking_spaces" in config.DROP_COLUMNS


def test_company_is_a_feature() -> None:
    """`company` se conserva como categórica (no se descarta)."""
    assert "company" in config.CATEGORICAL_COLUMNS
    assert "company" not in config.DROP_COLUMNS


def test_derived_numeric_features() -> None:
    """Las features derivadas (EDA §5) se suman a las numéricas que ve el modelo."""
    assert config.DERIVED_NUMERIC_COLUMNS == ["has_company", "has_agent", "noches"]
    assert config.NUMERIC_FEATURES == [*config.NUMERIC_COLUMNS, *config.DERIVED_NUMERIC_COLUMNS]
    # Las derivadas NO son de entrada (las calcula el preprocesado).
    for col in config.DERIVED_NUMERIC_COLUMNS:
        assert col not in config.FEATURE_COLUMNS


def test_target_not_in_features() -> None:
    """El target no debe figurar como característica de entrada."""
    assert config.TARGET_COLUMN not in config.NUMERIC_COLUMNS
    assert config.TARGET_COLUMN not in config.CATEGORICAL_COLUMNS


def test_leakage_columns_are_dropped() -> None:
    """Las columnas de leakage están incluidas en las que se descartan."""
    for col in config.LEAKAGE_COLUMNS:
        assert col in config.DROP_COLUMNS


def test_class_labels_binary() -> None:
    """Hay exactamente dos etiquetas de clase (problema binario)."""
    assert len(config.CLASS_LABELS) == 2


def test_primary_metric_in_metric_names() -> None:
    """La métrica principal debe estar entre las reportadas."""
    assert config.PRIMARY_METRIC in config.METRIC_NAMES


def test_test_size_is_a_fraction() -> None:
    assert 0.0 < config.TEST_SIZE < 1.0


def test_tuning_grids_not_empty() -> None:
    """Los espacios de búsqueda de hiperparámetros no están vacíos."""
    grids = [
        config.LOGISTIC_REGRESSION_GRID,
        config.DECISION_TREE_GRID,
        config.RANDOM_FOREST_GRID,
        config.XGBOOST_GRID,
    ]
    for grid in grids:
        assert grid, "el grid de tuning no debería estar vacío"
        assert all(k.startswith("model__") for k in grid)
