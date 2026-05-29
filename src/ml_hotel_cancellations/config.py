"""Configuración central del proyecto.

Este módulo concentra todas las rutas, constantes y parámetros del sistema de
modelado para que el resto de módulos no contengan valores "mágicos" dispersos.
De esta forma, cambiar el comportamiento del pipeline (semilla, métrica
principal, hiperparámetros, etc.) se hace desde un único punto.
"""

from __future__ import annotations

import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
# `PROJECT_ROOT` apunta a la raíz del repo. Este módulo vive en
# `src/ml_hotel_cancellations/config.py`, así que la raíz está 2 niveles arriba.
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
#
# `arrival_date_year` se EXCLUYE a propósito (no se usa como característica):
#   - Apenas discrimina: la tasa de cancelación es casi plana entre años
#     (2015: 37.0 %, 2016: 35.9 %, 2017: 38.7 %).
#   - No generaliza: el objetivo es predecir reservas FUTURAS, y un año no visto
#     en el entrenamiento (2018 en adelante) no tiene un valor de "año" con
#     sentido para el modelo (los árboles lo meterían en el último tramo y los
#     modelos lineales extrapolarían una tendencia inexistente).
#   - Está confundida con la estación: el dataset cubre años PARCIALES (2015 solo
#     jul-dic, 2017 solo ene-ago), de ahí su correlación −0.54 con
#     `arrival_date_week_number`. La señal estacional ya la capturan `month` y
#     `week_number`, que SÍ se repiten cada año.
# En una partición aleatoria, incluirla subía el ROC-AUC de XGBoost ~0.003, pero
# esa mejora es *optimismo* que no se trasladaría a producción. Al no figurar en
# estas listas, el `ColumnTransformer` la descarta vía `remainder="drop"`.
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
    "required_car_parking_spaces",
    "total_of_special_requests",
]

# Lista ordenada de las 27 características de entrada (numéricas + categóricas).
# Fuente única para contar/enumerar features (la usan la API y los esquemas).
FEATURE_COLUMNS: list[str] = [*NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS]

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

# Etiquetas legibles (cortas) de las clases. FUENTE ÚNICA DE VERDAD que reutilizan
# la API y la interfaz. El índice de la lista coincide con la clase predicha
# (0 = no cancela, 1 = cancela).
CLASS_LABELS_SHORT: list[str] = ["No cancelada", "Cancelada"]

# Variante con el código de clase entre paréntesis, usada como etiqueta en los
# gráficos (matriz de confusión). Se deriva de la lista corta para no duplicar.
CLASS_LABELS: list[str] = [f"{label} ({i})" for i, label in enumerate(CLASS_LABELS_SHORT)]

# Umbral de decisión que convierte la probabilidad estimada en una clase 0/1.
# Centralizado para no repetir el "0.5" mágico en predict/model_trainer/balancing.
DECISION_THRESHOLD: float = 0.5

# Familia de cada modelo, útil para filtrar runs en MLflow ("solo XGBoost", etc.).
# FUENTE ÚNICA DE VERDAD reutilizada por train/tuning/balancing.
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

# Hiperparámetros por defecto adoptados tras la exploración del notebook
# `notebooks/playground/`: {learning_rate: 0.1, max_depth: 14, n_estimators: 500}
# (AUC≈0.957 en validación cruzada). El finetuning (RandomizedSearchCV sobre
# XGBOOST_GRID) parte de esta zona e intenta mejorarla.
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
# Validación cruzada (CV) y métrica usadas durante la búsqueda. Se optimiza
# ROC-AUC, la misma métrica principal del proyecto, para ser coherentes.
TUNING_CV_FOLDS: int = 3
TUNING_SCORING: str = "roc_auc"
TUNING_N_ITER: int = 12  # nº de combinaciones que prueba RandomizedSearchCV

# Espacios de búsqueda. Las claves llevan el prefijo "model__" porque el
# estimador es el paso llamado "model" dentro del Pipeline. Para espacios
# pequeños usamos GridSearchCV (exhaustivo); para los grandes, RandomizedSearchCV
# (muestreo aleatorio), mucho más eficiente. (RandomizedSearchCV NO se ve en
# `recursos/`, que usa GridSearchCV; mapeo de herramientas en
# docs/informe_final.md §4.5.)
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
# Rejilla CENTRADA en la zona hallada en `notebooks/playground/` (max_depth≈14,
# n_estimators≈500, learning_rate≈0.1) y CON MARGEN para mejorarla: el
# RandomizedSearchCV muestrea esta rejilla buscando una combinación aún mejor que
# los valores por defecto de XGBOOST_PARAMS.
XGBOOST_GRID: dict = {
    "model__n_estimators": [300, 400, 500, 600],
    "model__max_depth": [8, 10, 12, 14, 16],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__subsample": [0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.8, 0.9, 1.0],
}

# Artefacto con los resultados de la búsqueda (informe legible).
TUNING_RESULTS_PATH: Path = OUTPUTS_DIR / "tuning_hiperparametros.md"

# Mejores hiperparámetros encontrados, persistidos en JSON. Si este fichero
# existe, el pipeline por defecto (`python -m ml_hotel_cancellations.ml.train`) los usa automáticamente;
# si no, recurre a los valores base definidos arriba. Así "se buscan una vez y se
# usan por defecto a partir de entonces".
BEST_PARAMS_PATH: Path = OUTPUTS_DIR / "best_hiperparametros.json"

# ---------------------------------------------------------------------------
# Balanceo de clases (bonus)
# ---------------------------------------------------------------------------
# Artefactos de la comparación de estrategias de balanceo (sin balanceo,
# reponderación por class_weight, y sobremuestreo SMOTE).
BALANCING_RESULTS_PATH: Path = OUTPUTS_DIR / "balanceo_clases.md"
BALANCING_PLOT_PATH: Path = OUTPUTS_DIR / "balanceo_clases.png"


# ---------------------------------------------------------------------------
# Reserva de ejemplo (contrato de entrada)
# ---------------------------------------------------------------------------
# Una reserva válida con las 27 características, en los nombres EXACTOS que espera
# el Pipeline. FUENTE ÚNICA DE VERDAD reutilizada por el esquema Pydantic de la
# API (Swagger) y por el formulario de la interfaz Streamlit.
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
    "days_in_waiting_list": 0,
    "customer_type": "Transient",
    "adr": 100.0,
    "required_car_parking_spaces": 0,
    "total_of_special_requests": 1,
}


def best_metric_value(metric: str = PRIMARY_METRIC) -> float:
    """Devuelve el mejor valor de ``metric`` en la tabla de métricas persistida.

    Lee ``METRICS_TABLE_PATH`` (generada por el pipeline de entrenamiento) en vez
    de mantener el número a mano, de forma que el valor reportado nunca se quede
    obsoleto tras un reentrenamiento. Lanza si el artefacto no existe.
    """
    import pandas as pd

    tabla = pd.read_csv(METRICS_TABLE_PATH, index_col=0)
    return float(tabla[metric].max())


def ensure_directories() -> None:
    """Crea las carpetas de salida si no existen.

    Se invoca al inicio del pipeline para garantizar que los artefactos
    (modelos, gráficos, tablas) tengan dónde escribirse.
    """
    for directory in (PROCESSED_DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def configure_logging(level: int = logging.INFO) -> None:
    """Configura un formato de logging legible para todo el pipeline.

    Ubicado aquí (módulo neutral) para que cualquier ``main()`` del paquete lo
    use sin que un CLI tenga que importar a otro.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def use_agg_backend() -> None:
    """Fija el backend no interactivo ``Agg`` de matplotlib en un único sitio.

    Permite guardar PNG sin necesidad de pantalla (CI/headless). Centralizar la
    llamada evita repetir ``matplotlib.use("Agg")`` como efecto de import en
    evaluator/interpretability/visualization_2d.
    """
    import matplotlib

    matplotlib.use("Agg")
