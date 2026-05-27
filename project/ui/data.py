"""Lógica de datos de la interfaz (sin Streamlit en el núcleo).

Aquí vive todo lo que NO es renderizado: cargar la tabla de métricas, leer una
muestra del dataset, calcular agregaciones para el EDA y llamar a la API de
predicción. Mantenerlo separado de las páginas tiene dos ventajas didácticas:

1. Se puede importar y probar sin arrancar Streamlit (no hay efectos de página).
2. Las funciones pesadas se cachean con `st.cache_data`, de modo que el CSV de
   16 MB no se relee en cada interacción del usuario.

Si Streamlit no está disponible (por ejemplo, al ejecutar un test), las
funciones siguen funcionando: el decorador de caché se sustituye por uno que no
hace nada (ver `_cache_data`).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from . import config


# ---------------------------------------------------------------------------
# Caché tolerante a la ausencia de Streamlit
# ---------------------------------------------------------------------------
# `st.cache_data` solo existe dentro de una app Streamlit. Para que estos
# helpers se puedan importar en tests o en la consola, definimos un decorador
# que usa la caché de Streamlit si está, y `functools.lru_cache` como respaldo.
try:  # pragma: no cover - depende del entorno de ejecución
    import streamlit as st

    def _cache_data(func):
        return st.cache_data(show_spinner=False)(func)

except Exception:  # pragma: no cover - entorno sin Streamlit (tests)

    def _cache_data(func):
        return lru_cache(maxsize=None)(func)


# ---------------------------------------------------------------------------
# Carga de artefactos y datos
# ---------------------------------------------------------------------------
@_cache_data
def load_metrics_table() -> pd.DataFrame:
    """Carga la tabla comparativa de los 5 modelos (`metricas_modelos.csv`).

    La primera columna del CSV no tiene cabecera (es el nombre del modelo); la
    renombramos a "Modelo" y la dejamos como columna normal para mostrarla.
    """
    df = pd.read_csv(config.METRICS_CSV_PATH)
    df = df.rename(columns={df.columns[0]: "Modelo"})
    # Ordenamos por la métrica principal (ROC-AUC) de mejor a peor.
    if "roc_auc" in df.columns:
        df = df.sort_values("roc_auc", ascending=False).reset_index(drop=True)
    return df


@_cache_data
def load_best_hyperparams() -> dict[str, Any]:
    """Lee los mejores hiperparámetros encontrados en la búsqueda (tuning)."""
    path = config.BEST_HYPERPARAMS_PATH
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@_cache_data
def load_dataset(nrows: int | None = None) -> pd.DataFrame:
    """Carga el dataset crudo (o las primeras `nrows` filas).

    Se respetan los tokens de nulo del proyecto ("NULL", "NA", ...) para que las
    columnas como `agent`/`company` se interpreten bien.
    """
    return pd.read_csv(
        config.RAW_DATASET_PATH,
        na_values=["NULL", "NA", "NaN", ""],
        keep_default_na=True,
        nrows=nrows,
    )


@_cache_data
def load_dataset_sample(n: int = 200) -> pd.DataFrame:
    """Devuelve una muestra reproducible del dataset (para vistas previas)."""
    df = load_dataset()
    n = min(n, len(df))
    return df.sample(n=n, random_state=42).reset_index(drop=True)


@_cache_data
def get_categorical_options() -> dict[str, list[str]]:
    """Valores únicos de las categóricas, para poblar los desplegables del form.

    Derivar las opciones del propio dataset hace que el formulario sea realista
    (no inventamos categorías que el modelo nunca vio). Para variables de alta
    cardinalidad (`country`, `agent`) devolvemos solo las más frecuentes, que es
    lo que cubre el OneHotEncoder del modelo; el resto cae en "infrequent".
    """
    df = load_dataset()
    options: dict[str, list[str]] = {}

    simple_cats = [
        "hotel",
        "arrival_date_month",
        "meal",
        "market_segment",
        "distribution_channel",
        "reserved_room_type",
        "assigned_room_type",
        "deposit_type",
        "customer_type",
    ]
    for col in simple_cats:
        options[col] = sorted(df[col].dropna().unique().tolist())

    # Meses en orden natural (no alfabético) para que el desplegable tenga sentido.
    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    present = set(options.get("arrival_date_month", []))
    options["arrival_date_month"] = [m for m in month_order if m in present]

    # País: las 30 más frecuentes (el resto agrupado por el modelo).
    options["country"] = df["country"].dropna().value_counts().head(30).index.tolist()

    # Agente: identificador categórico. En el CSV se lee como float (9.0); lo
    # pasamos a string entero ("9") para casar con el contrato de la API.
    agent_counts = df["agent"].dropna()
    agent_strings = agent_counts.map(lambda v: str(int(v))).value_counts()
    options["agent"] = agent_strings.head(30).index.tolist()

    return options


# ---------------------------------------------------------------------------
# Agregaciones para el EDA (Análisis Exploratorio de Datos)
# ---------------------------------------------------------------------------
@_cache_data
def cancellation_rate_by(column: str) -> pd.DataFrame:
    """Tasa de cancelación (media de `is_canceled`) por categoría de `column`.

    Devuelve un DataFrame con la categoría, la tasa de cancelación (0-1) y el
    número de reservas, ordenado de mayor a menor tasa. Es la agregación clave
    del EDA: muestra qué grupos cancelan más.
    """
    df = load_dataset()
    grouped = (
        df.groupby(column)[config.TARGET_COLUMN]
        .agg(tasa_cancelacion="mean", reservas="count")
        .reset_index()
        .rename(columns={column: "categoria"})
        .sort_values("tasa_cancelacion", ascending=False)
        .reset_index(drop=True)
    )
    return grouped


@_cache_data
def class_balance() -> pd.DataFrame:
    """Reparto de la variable objetivo (cuántas reservas se cancelan o no).

    El desbalance (~37 % de cancelaciones) justifica usar ROC-AUC como métrica
    principal y explorar técnicas de balanceo.
    """
    df = load_dataset()
    counts = df[config.TARGET_COLUMN].value_counts().sort_index()
    return pd.DataFrame(
        {
            "clase": [config.CLASS_LABELS.get(int(i), str(i)) for i in counts.index],
            "reservas": counts.values,
            "porcentaje": (counts.values / counts.sum() * 100).round(2),
        }
    )


@_cache_data
def numeric_summary() -> pd.DataFrame:
    """Resumen estadístico de algunas variables numéricas relevantes."""
    df = load_dataset()
    cols = ["lead_time", "adr", "stays_in_week_nights", "total_of_special_requests"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].describe().T.round(2).reset_index().rename(columns={"index": "variable"})


# ---------------------------------------------------------------------------
# Cliente de la API de predicción
# ---------------------------------------------------------------------------
def check_api_health() -> tuple[bool, dict[str, Any] | str]:
    """Comprueba si la API está viva consultando `GET /health`.

    Returns
    -------
    (ok, info)
        `ok` es True si la API responde 200; `info` es el JSON de respuesta o,
        en caso de error, un mensaje explicativo (string).
    """
    try:
        resp = requests.get(config.API_HEALTH_ENDPOINT, timeout=config.API_TIMEOUT_S)
        resp.raise_for_status()
        return True, resp.json()
    except requests.exceptions.ConnectionError:
        return False, "No se pudo conectar con la API (¿está arrancada?)."
    except requests.exceptions.Timeout:
        return False, "La API tardó demasiado en responder (timeout)."
    except Exception as exc:  # noqa: BLE001 - mostramos cualquier fallo al usuario
        return False, f"Error consultando la API: {exc}"


def predict_booking(booking: dict[str, Any]) -> tuple[bool, dict[str, Any] | str]:
    """Envía una reserva a `POST /predict` y devuelve la predicción.

    El cuerpo (`booking`) debe seguir el contrato de la API: las 27 variables de
    la reserva con los nombres EXACTOS. La respuesta esperada es::

        {"prediction": 0|1, "label": "...", "probability": <float>}

    Returns
    -------
    (ok, resultado)
        `ok` True con el JSON de la predicción, o False con un mensaje de error.
    """
    try:
        resp = requests.post(
            config.API_PREDICT_ENDPOINT,
            json=booking,
            timeout=config.API_TIMEOUT_S,
        )
        resp.raise_for_status()
        return True, resp.json()
    except requests.exceptions.ConnectionError:
        return False, "No se pudo conectar con la API (¿está arrancada?)."
    except requests.exceptions.Timeout:
        return False, "La API tardó demasiado en responder (timeout)."
    except requests.exceptions.HTTPError as exc:
        # Intentamos extraer el detalle de validación de FastAPI (422, etc.).
        detail = ""
        try:
            detail = f" Detalle: {exc.response.json()}"
        except Exception:  # noqa: BLE001
            detail = f" Código HTTP {exc.response.status_code}."
        return False, f"La API rechazó la petición.{detail}"
    except Exception as exc:  # noqa: BLE001
        return False, f"Error llamando a la API: {exc}"


# ---------------------------------------------------------------------------
# Utilidades de presentación de artefactos
# ---------------------------------------------------------------------------
def existing_pngs(filenames: list[str]) -> list[Path]:
    """De una lista de nombres de PNG en `outputs/`, devuelve los que existen.

    Útil para iterar sobre gráficos conocidos y mostrar solo los disponibles
    (los SHAP, por ejemplo, pueden no estar generados todavía).
    """
    paths = [config.OUTPUTS_DIR / name for name in filenames]
    return [p for p in paths if p.exists()]


def find_shap_pngs() -> list[Path]:
    """Localiza los gráficos SHAP (`shap_*.png`) si el módulo de interpretabilidad
    ya se ejecutó. Devuelve lista vacía si aún no existen."""
    if not config.OUTPUTS_DIR.exists():
        return []
    return sorted(config.OUTPUTS_DIR.glob("shap_*.png"))
