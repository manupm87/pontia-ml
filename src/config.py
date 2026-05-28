"""Configuración central del proyecto.

Este módulo concentra todas las rutas, constantes y parámetros del sistema de
modelado para que el resto de módulos no contengan valores "mágicos" dispersos.
De esta forma, cambiar el comportamiento del pipeline (semilla, métrica
principal, hiperparámetros, etc.) se hace desde un único punto.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
# `PROJECT_ROOT` apunta a la raíz del repo (un nivel por encima de `src/`).
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
# Aceleración por GPU (opt-in)
# ---------------------------------------------------------------------------
# Por defecto, todo se ejecuta en CPU: para este tamaño de datos la GPU no
# acelera (el coste por modelo es pequeño) y la CPU garantiza resultados
# reproducibles. El código es "GPU-aware": poniendo la variable de entorno
# PONTIA_USE_GPU=1 se activa el uso de GPU en XGBoost (device='cuda') cuando
# haya una disponible, con caída automática a CPU si no funciona.
USE_GPU: bool = os.getenv("PONTIA_USE_GPU", "false").lower() in {"1", "true", "yes"}

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
# existe, el pipeline por defecto (`python -m src.train`) los usa automáticamente;
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


def ensure_directories() -> None:
    """Crea las carpetas de salida si no existen.

    Se invoca al inicio del pipeline para garantizar que los artefactos
    (modelos, gráficos, tablas) tengan dónde escribirse.
    """
    for directory in (PROCESSED_DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
