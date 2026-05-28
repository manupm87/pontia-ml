"""Sección 1 — Resumen y resultados.

Da una foto rápida del proyecto: qué problema resolvemos, qué modelo gana y con
qué métricas, además de las visualizaciones clave (curvas ROC, matrices de
confusión e importancia de variables) con explicaciones para estudiantes.
"""

from __future__ import annotations

import streamlit as st

from .. import config, data
from ..layout import image_card


def render() -> None:
    st.title("Resumen y resultados")

    st.markdown(
        """
        Este proyecto resuelve una **clasificación binaria**: predecir si una
        reserva de hotel se **cancelará** (`1`) o **no** (`0`). Se compararon
        cinco modelos y se eligió el mejor según la **ROC-AUC**, una métrica que
        mide la capacidad de *ordenar* las reservas por riesgo de cancelación,
        independiente del umbral de decisión y robusta ante el desbalance de
        clases (~37 % de cancelaciones).
        """
    )

    # --- Cifras de cabecera ---------------------------------------------------
    col1, col2, col3 = st.columns(3)
    col1.metric(
        label="Mejor modelo",
        value=config.BEST_MODEL_NAME,
        help="Modelo ganador según la métrica principal (ROC-AUC).",
    )
    col2.metric(
        label="ROC-AUC (test)",
        value=f"{config.BEST_MODEL_ROC_AUC:.4f}",
        help="0.5 sería azar; 1.0 sería perfecto. 0.96 es muy bueno.",
    )
    col3.metric(
        label="Modelos comparados",
        value="5",
        help="Regresión logística, árbol, random forest, XGBoost y red neuronal.",
    )

    st.info(
        "El modelo ganador es **XGBoost**, con una **ROC-AUC de "
        f"{config.BEST_MODEL_ROC_AUC:.4f}** sobre el conjunto de test. "
        "Supera al resto en casi todas las métricas y además entrena rápido.",
        icon="🏆",
    )

    # --- Tabla comparativa ----------------------------------------------------
    st.subheader("Comparativa de los 5 modelos")
    st.caption(
        "Ordenados por ROC-AUC (de mejor a peor). `train_time_s` es el tiempo de "
        "entrenamiento en segundos."
    )
    try:
        metrics = data.load_metrics_table()
        # Resaltamos la fila del mejor modelo y formateamos los decimales.
        numeric_cols = [c for c in metrics.columns if c != "Modelo"]
        styled = (
            metrics.style.format({c: "{:.4f}" for c in numeric_cols})
            .highlight_max(subset=["roc_auc"], color="#d4edda")
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)
    except FileNotFoundError:
        st.warning(
            "No se encontró `outputs/metricas_modelos.csv`. Ejecuta el pipeline "
            "de entrenamiento para generarlo."
        )

    # --- Visualizaciones clave ------------------------------------------------
    st.subheader("Visualizaciones clave")

    key_plots: list[tuple[str, str, str]] = [
        (
            "roc_curves.png",
            "Curvas ROC",
            "Cada curva enfrenta la tasa de **verdaderos positivos** "
            "(cancelaciones detectadas) frente a la de **falsos positivos**. "
            "Cuanto más se acerca al ángulo superior izquierdo, mejor; el área "
            "bajo la curva es la **ROC-AUC**.",
        ),
        (
            "confusion_matrices.png",
            "Matrices de confusión",
            "Muestran aciertos y errores por clase: verdaderos/falsos positivos y "
            "negativos. Permiten ver si el modelo confunde más cancelaciones con "
            "no-cancelaciones o al revés.",
        ),
        (
            "feature_importance.png",
            "Importancia de variables",
            "Qué características pesan más en la decisión del modelo ganador "
            "(p. ej. `lead_time`, `deposit_type` o el país suelen ser muy "
            "informativos).",
        ),
    ]

    for filename, title, description in key_plots:
        image_card(
            config.OUTPUTS_DIR / filename,
            title=title,
            description=description,
            not_found_message=(
                f"No se encontró `outputs/{filename}` (aún no generado)."
            ),
        )
