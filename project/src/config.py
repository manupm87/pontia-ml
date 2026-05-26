"""Configuración central del proyecto.

Este módulo concentra todas las rutas, constantes y parámetros del sistema de
modelado para que el resto de módulos no contengan valores "mágicos" dispersos.
De esta forma, cambiar el comportamiento del pipeline (semilla, métrica
principal, hiperparámetros, etc.) se hace desde un único punto.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
# `PROJECT_ROOT` apunta a la carpeta `project/` (un nivel por encima de `src/`).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

# Fichero de datos de entrada (dataset proporcionado en el enunciado).
RAW_DATASET_PATH: Path = RAW_DATA_DIR / "dataset_practica_final.csv"

# Artefactos persistidos por el pipeline de entrenamiento.
BEST_MODEL_PATH: Path = MODELS_DIR / "best_model.pkl"
METRICS_TABLE_PATH: Path = OUTPUTS_DIR / "metricas_modelos.csv"

# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2

# ---------------------------------------------------------------------------
# Definición del problema
# ---------------------------------------------------------------------------
TARGET_COLUMN: str = "is_canceled"

# Cadenas que en el CSV representan ausencia de valor.
NA_TOKENS: list[str] = ["NULL", "NA", "NaN", ""]

# Columnas que provocan *data leakage*: describen el desenlace de la reserva y,
# por tanto, "filtran" la variable objetivo. `reservation_status` toma el valor
# "Canceled" exactamente cuando `is_canceled == 1`, lo que haría que cualquier
# modelo obtuviese ~100 % de acierto de forma artificial. Se eliminan siempre.
LEAKAGE_COLUMNS: list[str] = ["reservation_status", "reservation_status_date"]

# Columnas descartadas por baja utilidad: `company` está ausente en ~94 % de las
# filas, por lo que aporta más ruido que señal.
DROP_COLUMNS: list[str] = ["company", *LEAKAGE_COLUMNS]

# Variables categóricas. Nótese que `agent` es un identificador numérico pero
# semánticamente categórico (ID de agencia), por lo que se trata como categoría.
CATEGORICAL_COLUMNS: list[str] = [
    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "assigned_room_type",
    "deposit_type",
    "customer_type",
    "agent",
]

# Variables numéricas continuas o discretas.
NUMERIC_COLUMNS: list[str] = [
    "lead_time",
    "arrival_date_year",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
]

# Cardinalidad máxima admitida por variable categórica en el OneHotEncoder.
# Variables como `country` (178 valores) o `agent` (334) explotarían el espacio
# de características; agrupamos las categorías poco frecuentes en "infrequent".
MAX_OHE_CATEGORIES: int = 25

# ---------------------------------------------------------------------------
# Métrica principal y secundarias
# ---------------------------------------------------------------------------
# Métrica usada para SELECCIONAR el mejor modelo. Se elige ROC-AUC por ser
# independiente del umbral de decisión y robusta ante el desbalance de clases
# (~37 % de cancelaciones). La justificación completa está en la documentación.
PRIMARY_METRIC: str = "roc_auc"

# Orden en el que se reportan las métricas en las tablas comparativas.
METRIC_NAMES: list[str] = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# Etiquetas legibles para las clases de la variable objetivo.
CLASS_LABELS: list[str] = ["No cancelada (0)", "Cancelada (1)"]

# ---------------------------------------------------------------------------
# Hiperparámetros de los modelos clásicos (scikit-learn / XGBoost)
# ---------------------------------------------------------------------------
LOGISTIC_REGRESSION_PARAMS: dict = {
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
}

DECISION_TREE_PARAMS: dict = {
    "max_depth": 12,
    "min_samples_leaf": 50,
    "random_state": RANDOM_STATE,
}

RANDOM_FOREST_PARAMS: dict = {
    "n_estimators": 200,
    "max_depth": 18,
    "min_samples_leaf": 20,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

XGBOOST_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "eval_metric": "logloss",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

# ---------------------------------------------------------------------------
# Hiperparámetros de la red neuronal (Keras / TensorFlow)
# ---------------------------------------------------------------------------
NN_PARAMS: dict = {
    "hidden_units": [64, 32, 16],
    "dropout": 0.3,
    "epochs": 50,
    "batch_size": 512,
    "learning_rate": 1e-3,
    "early_stopping_patience": 5,
    "validation_split": 0.2,
}


def ensure_directories() -> None:
    """Crea las carpetas de salida si no existen.

    Se invoca al inicio del pipeline para garantizar que los artefactos
    (modelos, gráficos, tablas) tengan dónde escribirse.
    """
    for directory in (PROCESSED_DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
