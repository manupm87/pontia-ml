"""Configuración de la interfaz visual.

Centraliza rutas, URLs y constantes para que el resto de módulos no contengan
valores "mágicos". Cambiar dónde está la API o las carpetas de artefactos se
hace desde un único punto.

Términos para estudiantes:
- *API*: programa que expone el modelo por HTTP; la web le envía una reserva y
  recibe la predicción. Aquí es la API FastAPI del proyecto.
- *Artefacto*: fichero que produce el pipeline de ML (gráfico, tabla, modelo).
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
# `PROJECT_ROOT` apunta a la raíz del repo (un nivel por encima de `ui/`).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
RAW_DATASET_PATH: Path = RAW_DATA_DIR / "dataset_practica_final.csv"

# Artefactos concretos que consume la interfaz.
METRICS_CSV_PATH: Path = OUTPUTS_DIR / "metricas_modelos.csv"
BEST_HYPERPARAMS_PATH: Path = OUTPUTS_DIR / "best_hiperparametros.json"
BALANCING_MD_PATH: Path = OUTPUTS_DIR / "balanceo_clases.md"
TUNING_MD_PATH: Path = OUTPUTS_DIR / "tuning_hiperparametros.md"

# ---------------------------------------------------------------------------
# API de predicción (FastAPI)
# ---------------------------------------------------------------------------
# La URL base se puede sobreescribir con la variable de entorno PONTIA_API_URL.
# Así, en local apunta a localhost y en otro entorno (Docker, despliegue) basta
# con cambiar la variable sin tocar el código.
API_BASE_URL: str = os.getenv("PONTIA_API_URL", "http://localhost:8000").rstrip("/")
API_HEALTH_ENDPOINT: str = f"{API_BASE_URL}/health"
API_PREDICT_ENDPOINT: str = f"{API_BASE_URL}/predict"

# Tiempo máximo (segundos) que esperamos a la API antes de darla por caída.
API_TIMEOUT_S: float = 8.0

# ---------------------------------------------------------------------------
# Cifras de cabecera del proyecto (para destacar resultados de un vistazo)
# ---------------------------------------------------------------------------
BEST_MODEL_NAME: str = "XGBoost"
BEST_MODEL_ROC_AUC: float = 0.9614  # ROC-AUC del mejor modelo sobre el test.

# Etiquetas legibles de las clases del problema (clasificación binaria).
CLASS_LABELS: dict[int, str] = {0: "No cancelada", 1: "Cancelada"}

# Tasa de cancelación global del dataset (~37 %). Sirve de referencia ("línea
# base") al interpretar una probabilidad concreta.
BASE_CANCELLATION_RATE: float = 0.37

# Nombre de la columna objetivo en el dataset crudo.
TARGET_COLUMN: str = "is_canceled"
