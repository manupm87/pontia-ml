"""Configuración central: rutas, constantes y parámetros del pipeline.

Punto único de cambio para semilla, métrica, hiperparámetros, etc.
"""

from __future__ import annotations

import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
# Raíz del repo: este módulo está 2 niveles por debajo.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

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

# Columnas de data leakage: filtran el target y se eliminan siempre.
# Ver docs/informe_final.md §EDA.
LEAKAGE_COLUMNS: list[str] = ["reservation_status", "reservation_status_date"]

# Columnas que se descartan en el entrenamiento (hallazgos del EDA):
# - leakage directo (reservation_status*),
# - `required_car_parking_spaces`: fuga sutil, se asigna en el check-in (EDA §11),
# - `arrival_date_year`: años parciales, no generaliza (EDA §6).
# `company` ya NO se descarta: se conserva como categórica de cardinalidad reducida.
DROP_COLUMNS: list[str] = [
    *LEAKAGE_COLUMNS,
    "required_car_parking_spaces",
    "arrival_date_year",
]

# Variables categóricas de entrada. `agent` y `company` son IDs tratados como
# categorías de alta cardinalidad (ver HIGH_CARDINALITY_COLUMNS).
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
    "company",
]

# Variables numéricas de entrada. `arrival_date_year` y `required_car_parking_spaces`
# se excluyen a propósito (el año no generaliza, EDA §6; el parking es fuga, EDA §11).
NUMERIC_COLUMNS: list[str] = [
    "lead_time",
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
    "total_of_special_requests",
]

# Las 27 features de entrada (15 numéricas + 12 categóricas); fuente única que usan
# la API y los esquemas. Las features derivadas (abajo) NO son de entrada: el
# preprocesado las calcula a partir de estas.
FEATURE_COLUMNS: list[str] = [*NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS]

# Features derivadas en el preprocesado (EDA §5, "ausencia informativa"):
# - has_company / has_agent: el nulo de company/agent es señal, no ruido.
# - noches: estancia total = noches de semana + de fin de semana.
DERIVED_NUMERIC_COLUMNS: list[str] = ["has_company", "has_agent", "noches"]

# Columnas numéricas que ve el modelo: entrada + derivadas (las escala el preprocesador).
NUMERIC_FEATURES: list[str] = [*NUMERIC_COLUMNS, *DERIVED_NUMERIC_COLUMNS]

# Categóricas de alta cardinalidad: reducción supervisada fit-on-train (EDA §13).
# Se conservan las categorías con soporte (n >= RARE_MIN_N) y señal extrema de
# cancelación (tasa > RARE_HI_FRAC*max o < RARE_LO_FRAC*max); el resto -> "Otros".
HIGH_CARDINALITY_COLUMNS: list[str] = ["agent", "country", "company"]
RARE_NULL_LABELS: dict[str, str] = {
    "agent": "Desconocido",
    "country": "Desconocido",
    "company": "no_company",  # el nulo de company es un estado real (sin empresa)
}
RARE_MIN_N: int = 100
RARE_HI_FRAC: float = 0.60
RARE_LO_FRAC: float = 0.30

# ---------------------------------------------------------------------------
# Métrica principal y secundarias
# ---------------------------------------------------------------------------
# Métrica de selección: ROC-AUC (independiente del umbral y robusta al
# desbalance ~37 %). Justificación en docs/informe_final.md.
PRIMARY_METRIC: str = "roc_auc"

# Orden en el que se reportan las métricas en las tablas comparativas.
METRIC_NAMES: list[str] = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# Etiquetas de clase (índice = clase predicha: 0 = no cancela, 1 = cancela).
# Fuente única que reutilizan la API y la interfaz.
CLASS_LABELS_SHORT: list[str] = ["No cancelada", "Cancelada"]

# Variante con el código entre paréntesis para los gráficos.
CLASS_LABELS: list[str] = [f"{label} ({i})" for i, label in enumerate(CLASS_LABELS_SHORT)]

# Umbral que convierte la probabilidad en clase 0/1 (centralizado).
DECISION_THRESHOLD: float = 0.5

# Familia de cada modelo (para filtrar runs en MLflow); fuente única
# compartida por train/tuning/balancing.
MODEL_FAMILY: dict[str, str] = {
    "Logistic Regression": "linear",
    "Decision Tree": "tree",
    "Random Forest": "forest",
    "XGBoost": "boosting",
    "Neural Network (Keras)": "neural_net",
}

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

# Valores por defecto hallados en `notebooks/playground/`; el finetuning
# (XGBOOST_GRID) parte de esta zona e intenta mejorarla.
XGBOOST_PARAMS: dict = {
    "n_estimators": 500,
    "max_depth": 14,
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

# ---------------------------------------------------------------------------
# Optimización de hiperparámetros (bonus)
# ---------------------------------------------------------------------------
# CV y métrica de la búsqueda (ROC-AUC, igual que la principal).
TUNING_CV_FOLDS: int = 3
TUNING_SCORING: str = "roc_auc"
TUNING_N_ITER: int = 12  # nº de combinaciones que prueba RandomizedSearchCV

# Espacios de búsqueda; el prefijo "model__" apunta al paso "model" del Pipeline.
# Grids pequeños -> GridSearchCV; grandes -> RandomizedSearchCV (ver §4.5).
LOGISTIC_REGRESSION_GRID: dict = {
    "model__C": [0.01, 0.1, 1.0, 10.0],
    "model__class_weight": [None, "balanced"],
}
DECISION_TREE_GRID: dict = {
    "model__max_depth": [6, 8, 12, None],
    "model__min_samples_leaf": [20, 50, 100],
    "model__criterion": ["gini", "entropy"],
}
RANDOM_FOREST_GRID: dict = {
    "model__n_estimators": [200, 300, 400],
    "model__max_depth": [12, 18, None],
    "model__min_samples_leaf": [5, 10, 20],
    "model__max_features": ["sqrt", "log2"],
}
# Rejilla centrada en la zona de XGBOOST_PARAMS, con margen para mejorarla.
XGBOOST_GRID: dict = {
    "model__n_estimators": [300, 400, 500, 600],
    "model__max_depth": [8, 10, 12, 14, 16],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__subsample": [0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.8, 0.9, 1.0],
}

# Artefacto con los resultados de la búsqueda (informe legible).
TUNING_RESULTS_PATH: Path = OUTPUTS_DIR / "tuning_hiperparametros.md"

# Mejores hiperparámetros (JSON). Si existe, `train` los usa por defecto;
# si no, recurre a los valores base de arriba.
BEST_PARAMS_PATH: Path = OUTPUTS_DIR / "best_hiperparametros.json"


# ---------------------------------------------------------------------------
# Reserva de ejemplo (contrato de entrada)
# ---------------------------------------------------------------------------
# Reserva válida de ejemplo (27 features con los nombres exactos del Pipeline);
# fuente única que reutilizan el esquema Pydantic de la API y el formulario UI.
BOOKING_EXAMPLE: dict = {
    "hotel": "City Hotel",
    "lead_time": 100,
    "arrival_date_month": "August",
    "arrival_date_week_number": 33,
    "arrival_date_day_of_month": 15,
    "stays_in_weekend_nights": 2,
    "stays_in_week_nights": 5,
    "adults": 2,
    "children": 0,
    "babies": 0,
    "meal": "BB",
    "country": "PRT",
    "market_segment": "Online TA",
    "distribution_channel": "TA/TO",
    "is_repeated_guest": 0,
    "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0,
    "reserved_room_type": "A",
    "assigned_room_type": "A",
    "booking_changes": 0,
    "deposit_type": "No Deposit",
    "agent": "9",
    "company": "no_company",
    "days_in_waiting_list": 0,
    "customer_type": "Transient",
    "adr": 100.0,
    "total_of_special_requests": 1,
}


def best_metric_value(metric: str = PRIMARY_METRIC) -> float:
    """Mejor valor de ``metric`` leído de la tabla persistida (lanza si no existe)."""
    import pandas as pd

    tabla = pd.read_csv(METRICS_TABLE_PATH, index_col=0)
    return float(tabla[metric].max())


def ensure_directories() -> None:
    """Crea las carpetas de salida si no existen."""
    for directory in (PROCESSED_DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def configure_logging(level: int = logging.INFO) -> None:
    """Configura un formato de logging legible para todo el pipeline."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def use_agg_backend() -> None:
    """Fija el backend no interactivo ``Agg`` de matplotlib (PNG sin pantalla)."""
    import matplotlib

    matplotlib.use("Agg")
