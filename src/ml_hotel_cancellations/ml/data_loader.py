"""Carga, limpieza y partición train/test del dataset crudo."""

from __future__ import annotations

import logging

import pandas as pd

from ml_hotel_cancellations import config

logger = logging.getLogger(__name__)


def load_raw_data(path=config.RAW_DATASET_PATH) -> pd.DataFrame:
    """Lee el CSV crudo interpretando como ``NaN`` los tokens de valor ausente."""
    logger.info("Cargando datos crudos desde %s", path)
    df = pd.read_csv(path, na_values=config.NA_TOKENS, keep_default_na=True)
    logger.info("Datos cargados: %d filas x %d columnas", df.shape[0], df.shape[1])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza segura fila a fila (no aprende parámetros, EDA §2).

    Descarta columnas con fuga / que no generalizan (leakage, parking, año) y filas
    sin sentido (sin huéspedes, sin noches, ``adr`` extremo, EDA §8). Mantiene
    ``company`` y ``agent`` en crudo: las features derivadas y la reducción de
    cardinalidad las construye el ``Pipeline`` de preprocesado. Devuelve datos
    listos para separar en X / y.
    """
    df = df.copy()

    cols_to_drop = [c for c in config.DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    logger.info("Columnas eliminadas (leakage / no generalizan): %s", cols_to_drop)

    # Saneo de filas (EDA §8): fuera reservas imposibles.
    n_before = len(df)
    keep = pd.Series(True, index=df.index)
    guest_cols = ["adults", "children", "babies"]
    if all(c in df.columns for c in guest_cols):
        keep &= df[guest_cols].fillna(0).sum(axis=1) > 0  # sin huéspedes
    if {"stays_in_week_nights", "stays_in_weekend_nights"} <= set(df.columns):
        nights = df["stays_in_week_nights"].fillna(0) + df["stays_in_weekend_nights"].fillna(0)
        keep &= nights > 0  # sin noches de estancia
    if "adr" in df.columns:
        keep &= ~((df["adr"] < 0) | (df["adr"] >= 5400))  # adr negativo o desorbitado
    df = df.loc[keep].reset_index(drop=True)
    logger.info("Filas eliminadas (sin huéspedes / sin noches / adr extremo): %d", n_before - len(df))

    return df


def get_feature_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa la matriz de características ``X`` de la variable objetivo ``y``."""
    y = df[config.TARGET_COLUMN].astype(int)
    X = df.drop(columns=[config.TARGET_COLUMN])
    return X, y


def split_data(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Partición estratificada train/test (conserva el ~37 % de cancelaciones)."""
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    logger.info(
        "Partición: train=%d filas, test=%d filas (test_size=%.2f, estratificada)",
        len(X_train),
        len(X_test),
        config.TEST_SIZE,
    )
    return X_train, X_test, y_train, y_test


def load_and_prepare() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Ejecuta el flujo completo y devuelve ``(X_train, X_test, y_train, y_test)``."""
    df = load_raw_data()
    df = clean_data(df)
    X, y = get_feature_target(df)
    return split_data(X, y)
