"""Carga, limpieza y partición de los datos.

Funciones para llevar el dataset crudo (CSV proporcionado en el enunciado) hasta
un par de conjuntos de entrenamiento y test listos para el preprocesado y el
modelado. Toda la lógica de "qué columnas existen y qué significan" vive en
``config`` para que este módulo se centre exclusivamente en *cómo* transformar
los datos.
"""

from __future__ import annotations

import logging

import pandas as pd

from ml_hotel_cancellations import config

logger = logging.getLogger(__name__)


def load_raw_data(path=config.RAW_DATASET_PATH) -> pd.DataFrame:
    """Lee el CSV crudo interpretando los tokens de valor ausente.

    El dataset codifica los valores ausentes con la cadena ``"NULL"`` (por
    ejemplo en ``agent`` o ``company``), por lo que se le indica a pandas que la
    trate como ``NaN``.

    Parameters
    ----------
    path:
        Ruta al fichero CSV. Por defecto, el dataset del enunciado.

    Returns
    -------
    pandas.DataFrame
        DataFrame crudo con los valores ausentes normalizados a ``NaN``.
    """
    logger.info("Cargando datos crudos desde %s", path)
    df = pd.read_csv(path, na_values=config.NA_TOKENS, keep_default_na=True)
    logger.info("Datos cargados: %d filas x %d columnas", df.shape[0], df.shape[1])
    return df


def normalize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza las variables categóricas de forma idéntica en train e inferencia.

    - ``agent`` (identificador numérico) se convierte a texto y sus ausentes se
      marcan como ``"Unknown"``.
    - El resto de categóricas reemplazan sus ausentes por ``"Unknown"``.

    Mantener esta lógica en una única función evita inconsistencias entre el
    preprocesado del entrenamiento y el de la predicción.
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
    """Aplica una limpieza ligera y elimina las columnas problemáticas.

    Pasos:

    1. Elimina columnas de *data leakage* y de baja utilidad (``config.DROP_COLUMNS``):
       ``reservation_status`` / ``reservation_status_date`` revelan el desenlace
       de la reserva, y ``company`` está ausente en ~94 % de los registros.
    2. Descarta reservas imposibles sin ningún huésped
       (``adults + children + babies == 0``).
    3. Normaliza las variables categóricas: ``agent`` (identificador numérico) se
       convierte a texto y los ausentes de las categóricas se marcan como
       ``"Unknown"`` para tratarlos como una categoría más.

    Parameters
    ----------
    df:
        DataFrame crudo devuelto por :func:`load_raw_data`.

    Returns
    -------
    pandas.DataFrame
        DataFrame limpio, listo para separar en X / y.
    """
    df = df.copy()

    # 1) Eliminar columnas que filtran el target o que aportan poca señal.
    cols_to_drop = [c for c in config.DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    logger.info("Columnas eliminadas (leakage / baja utilidad): %s", cols_to_drop)

    # 2) Eliminar reservas sin ningún huésped (registros inconsistentes).
    guest_cols = ["adults", "children", "babies"]
    if all(c in df.columns for c in guest_cols):
        n_before = len(df)
        no_guests = df[guest_cols].fillna(0).sum(axis=1) == 0
        df = df.loc[~no_guests].reset_index(drop=True)
        logger.info("Filas sin huéspedes eliminadas: %d", n_before - len(df))

    # 3) Normalizar las variables categóricas (agent y ausentes -> "Unknown").
    df = normalize_categoricals(df)

    return df


def get_feature_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa la matriz de características ``X`` de la variable objetivo ``y``.

    Returns
    -------
    tuple(pandas.DataFrame, pandas.Series)
        ``X`` (todas las columnas menos el target) e ``y`` (``is_canceled``).
    """
    y = df[config.TARGET_COLUMN].astype(int)
    X = df.drop(columns=[config.TARGET_COLUMN])
    return X, y


def split_data(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Realiza una partición *estratificada* train/test.

    La estratificación mantiene la misma proporción de cancelaciones (~37 %) en
    ambos conjuntos, lo que es importante dado el desbalance de clases.

    Returns
    -------
    tuple
        ``(X_train, X_test, y_train, y_test)``.
    """
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
    """Atajo que ejecuta todo el flujo de datos de principio a fin.

    Returns
    -------
    tuple
        ``(X_train, X_test, y_train, y_test)`` listos para el modelado.
    """
    df = load_raw_data()
    df = clean_data(df)
    X, y = get_feature_target(df)
    return split_data(X, y)
