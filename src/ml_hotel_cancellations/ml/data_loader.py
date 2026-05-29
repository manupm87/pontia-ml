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


def normalize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza las categóricas igual en train e inferencia.

    ``agent`` (ID numérico) pasa a texto; los ausentes se marcan como ``"Unknown"``.
    Función única para evitar divergencias entre entrenamiento y predicción.
    """
    df = df.copy()
    if "agent" in df.columns:
        agent = df["agent"].astype("Int64").astype("object")
        df["agent"] = agent.where(agent.notna(), "Unknown").astype(str)
    for col in config.CATEGORICAL_COLUMNS:
        if col == "agent" or col not in df.columns:
            continue
        df[col] = df[col].astype("object").where(df[col].notna(), "Unknown")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia el DataFrame: descarta leakage/baja utilidad, reservas sin huéspedes
    y normaliza las categóricas. Devuelve datos listos para separar en X / y.
    """
    df = df.copy()

    cols_to_drop = [c for c in config.DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    logger.info("Columnas eliminadas (leakage / baja utilidad): %s", cols_to_drop)

    # Reservas sin ningún huésped: registros inconsistentes.
    guest_cols = ["adults", "children", "babies"]
    if all(c in df.columns for c in guest_cols):
        n_before = len(df)
        no_guests = df[guest_cols].fillna(0).sum(axis=1) == 0
        df = df.loc[~no_guests].reset_index(drop=True)
        logger.info("Filas sin huéspedes eliminadas: %d", n_before - len(df))

    df = normalize_categoricals(df)

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
