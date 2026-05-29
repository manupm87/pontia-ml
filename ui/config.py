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

from src import config as _src_config

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

# ROC-AUC del mejor modelo: leído del artefacto de métricas del entrenamiento
# (no un literal) para que no se quede obsoleto. Respaldo si el CSV no estuviera.
try:
    BEST_MODEL_ROC_AUC: float = round(_src_config.best_metric_value("roc_auc"), 4)
except Exception:  # noqa: BLE001 - el artefacto puede faltar en algún entorno
    BEST_MODEL_ROC_AUC = 0.9614

# Etiquetas legibles de las clases (derivadas de la fuente única en src.config).
CLASS_LABELS: dict[int, str] = dict(enumerate(_src_config.CLASS_LABELS_SHORT))

# Tasa de cancelación global del dataset (~37 %). Sirve de referencia ("línea
# base") al interpretar una probabilidad concreta.
BASE_CANCELLATION_RATE: float = 0.37

# Texto legible de la tasa base ("~37 %"), derivado del valor anterior para que
# la cifra y la prosa nunca diverjan.
BASE_CANCELLATION_RATE_TEXT: str = f"~{BASE_CANCELLATION_RATE * 100:.0f} %"

# Cadenas que en el CSV representan ausencia de valor (reexportadas de src.config
# para que la interfaz no mantenga su propia copia de la lista).
NA_TOKENS: list[str] = _src_config.NA_TOKENS

# Nombre de la columna objetivo en el dataset crudo.
TARGET_COLUMN: str = _src_config.TARGET_COLUMN

# ---------------------------------------------------------------------------
# Paleta y proporciones de columnas
# ---------------------------------------------------------------------------
# Colores de las clases (no cancelada / cancelada) para los gráficos. Se
# centralizan aquí —igual que `IMAGE_COLUMN_RATIO` en `layout.py`— para que la
# misma codificación de color se use en todas las secciones.
CLASS_COLORS: dict[str, str] = {
    "No cancelada": "#2c7fb8",
    "Cancelada": "#de2d26",
}

# Color de resaltado de la fila del mejor modelo en la tabla comparativa.
BEST_ROW_HIGHLIGHT: str = "#d4edda"

# Proporciones de columnas usadas con `st.columns(...)` en las secciones.
CHART_TABLE_RATIO: tuple[int, int] = (2, 1)  # gráfico ancho + tabla estrecha

# ---------------------------------------------------------------------------
# Selecciones para el EDA (Análisis Exploratorio de Datos)
# ---------------------------------------------------------------------------
# Variables numéricas que resume la vista de EDA (antelación, tarifa media,
# noches entre semana y peticiones especiales).
EDA_NUMERIC_COLUMNS: list[str] = [
    "lead_time",
    "adr",
    "stays_in_week_nights",
    "total_of_special_requests",
]

# Meses en orden natural (no alfabético), para ordenar el desplegable de mes
# de llegada de forma que tenga sentido.
MONTH_ORDER: list[str] = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# ---------------------------------------------------------------------------
# Catálogo de gráficos exportados a `outputs/`
# ---------------------------------------------------------------------------
# Metadatos (título + descripción didáctica) de cada PNG generado por el
# pipeline. Las secciones seleccionan las claves que les interesan en vez de
# repetir los textos, de modo que un cambio de copy se hace en un único sitio.
PLOTS: dict[str, tuple[str, str]] = {
    "roc_curves.png": (
        "Curvas ROC",
        "Cada curva enfrenta la tasa de **verdaderos positivos** "
        "(cancelaciones detectadas) frente a la de **falsos positivos**. "
        "Cuanto más se acerca al ángulo superior izquierdo, mejor; el área "
        "bajo la curva es la **ROC-AUC**.",
    ),
    "confusion_matrices.png": (
        "Matrices de confusión",
        "Muestran aciertos y errores por clase: verdaderos/falsos positivos y "
        "negativos. Permiten ver si el modelo confunde más cancelaciones con "
        "no-cancelaciones o al revés.",
    ),
    "confusion_matrix_best.png": (
        "Matriz de confusión del mejor modelo (XGBoost)",
        "Detalle del modelo ganador. Arriba-izquierda y abajo-derecha son los "
        "aciertos; las otras dos celdas, los errores (**falsos positivos** y "
        "**falsos negativos**).",
    ),
    "decision_regions_pls.png": (
        "Regiones de decisión en 2D (proyección PLS)",
        "Para *ver* en 2D modelos que se entrenan con ~200 variables, los "
        "proyectamos al plano con **PLS** (un PCA supervisado: elige las 2 "
        "direcciones más correlacionadas con la cancelación). Sobre ese plano "
        "reentrenamos los 5 modelos y pintamos su predicción en cada punto: "
        "**rojo** = predice cancelación, **azul** = no, y la línea negra es la "
        "frontera 0.5.\n\n"
        "**Se ve la *personalidad* de cada modelo**: la regresión logística "
        "traza una frontera recta; el árbol corta en bloques; Random Forest "
        "suaviza esos bloques; XGBoost queda más fragmentado; el MLP es liso. "
        "El panel **Referencia** muestra las clases reales para comparar.",
    ),
    "feature_importance.png": (
        "Importancia de variables",
        "Qué características pesan más en la decisión del modelo ganador "
        "(p. ej. `lead_time`, `deposit_type` o el país suelen ser muy "
        "informativos).",
    ),
    "balanceo_clases.png": (
        "Efecto del balanceo de clases",
        "Compara estrategias para tratar el desbalance "
        f"(~{BASE_CANCELLATION_RATE * 100:.0f} % de cancelaciones): "
        "sin balanceo, reponderación (`class_weight`) y sobremuestreo (SMOTE). "
        "El balanceo sube el *recall* (detecta más cancelaciones) a costa de "
        "algo de precisión, mientras la ROC-AUC apenas cambia.",
    ),
}
