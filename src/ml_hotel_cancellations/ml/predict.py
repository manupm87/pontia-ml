"""Inferencia con el mejor modelo (``models/best_model.pkl``).

El modelo es un ``Pipeline``, así que recibe el DataFrame en crudo y aplica
internamente el mismo preprocesado que en entrenamiento.

Uso::

    # Predecir sobre un CSV propio
    python -m ml_hotel_cancellations.ml.predict --input mis_reservas.csv --output predicciones.csv

    # Demostración: toma una muestra del dataset original
    python -m ml_hotel_cancellations.ml.predict --sample 10
"""

from __future__ import annotations

import argparse
import logging

import joblib
import pandas as pd

from ml_hotel_cancellations import config
from .data_loader import load_raw_data, normalize_categoricals

logger = logging.getLogger(__name__)


def load_best_model(path=config.BEST_MODEL_PATH):
    """Carga el mejor modelo persistido por el pipeline de entrenamiento."""
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. Ejecuta primero 'python -m ml_hotel_cancellations.ml.train'."
        )
    logger.info("Cargando mejor modelo desde %s", path)
    return joblib.load(path)


def prepare_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara el DataFrame para predecir sin descartar filas.

    A diferencia del entrenamiento, NO elimina registros (una predicción por fila):
    solo normaliza categóricas y descarta el target si viniera incluido.
    """
    df = normalize_categoricals(df)
    if config.TARGET_COLUMN in df.columns:
        df = df.drop(columns=[config.TARGET_COLUMN])
    return df


def predict_dataframe(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """Predice por reserva: devuelve ``prediction`` (0/1) y ``probability_canceled``."""
    if model is None:
        model = load_best_model()
    X = prepare_for_inference(df)
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= config.DECISION_THRESHOLD).astype(int)
    return pd.DataFrame(
        {"prediction": pred, "probability_canceled": proba.round(4)},
        index=df.index,
    )


def main() -> None:
    config.configure_logging()
    parser = argparse.ArgumentParser(description="Inferencia de cancelaciones de reservas.")
    parser.add_argument("--input", type=str, help="CSV de entrada con reservas.")
    parser.add_argument("--output", type=str, help="CSV de salida con predicciones.")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Si se indica, usa N filas de muestra del dataset original.",
    )
    args = parser.parse_args()

    if args.input:
        df = load_raw_data(args.input)
    else:
        n = args.sample or 10
        df = load_raw_data().sample(n=n, random_state=config.RANDOM_STATE)
        logger.info("Usando %d filas de muestra del dataset original.", n)

    model = load_best_model()
    result = predict_dataframe(df, model=model)

    if args.output:
        # Conservamos el índice para casar cada salida con su reserva de origen.
        result.to_csv(args.output, index=True)
        logger.info("Predicciones guardadas en %s", args.output)
    else:
        print(result.to_string())


if __name__ == "__main__":
    main()
