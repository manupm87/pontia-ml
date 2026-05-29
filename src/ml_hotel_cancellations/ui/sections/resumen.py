"""Sección 1 — Resumen y resultados.

Foto rápida del proyecto: problema, modelo ganador y métricas, más las
visualizaciones clave (ROC, matrices de confusión, importancia de variables).
"""

from __future__ import annotations

import streamlit as st

from .. import config, data
from ..layout import image_card


def render() -> None:
    st.title("Resumen y resultados")

    # Nº de modelos, derivado de la tabla de métricas (None si falta el artefacto).
    try:
        metrics = data.load_metrics_table()
        n_models = len(metrics)
    except FileNotFoundError:
        metrics = None
        n_models = None

    st.markdown(
        f"""
        Este proyecto resuelve una **clasificación binaria**: predecir si una
        reserva de hotel se **cancelará** (`1`) o **no** (`0`). Se compararon
        varios modelos y se eligió el mejor según la **ROC-AUC**, una métrica que
        mide la capacidad de *ordenar* las reservas por riesgo de cancelación,
        independiente del umbral de decisión y robusta ante el desbalance de
        clases ({config.BASE_CANCELLATION_RATE_TEXT} de cancelaciones).
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
        value=str(n_models) if n_models is not None else "—",
        help="Regresión logística, árbol, random forest, XGBoost y red neuronal.",
    )

    st.info(
        "El modelo ganador es **XGBoost**, con una **ROC-AUC de "
        f"{config.BEST_MODEL_ROC_AUC:.4f}** sobre el conjunto de test. "
        "Supera al resto en casi todas las métricas y además entrena rápido.",
        icon="🏆",
    )

    # --- Tabla comparativa ----------------------------------------------------
    models_text = f"de los {n_models} modelos" if n_models is not None else "de modelos"
    st.subheader(f"Comparativa {models_text}")
    st.caption(
        "Ordenados por ROC-AUC (de mejor a peor). `train_time_s` es el tiempo de "
        "entrenamiento en segundos."
    )
    if metrics is not None:
        # Resaltamos la fila del mejor modelo y formateamos decimales.
        numeric_cols = [c for c in metrics.columns if c != "Modelo"]
        styled = (
            metrics.style.format({c: "{:.4f}" for c in numeric_cols})
            .highlight_max(subset=["roc_auc"], color=config.BEST_ROW_HIGHLIGHT)
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.warning(
            "No se encontró `outputs/metricas_modelos.csv`. Ejecuta el pipeline "
            "de entrenamiento para generarlo."
        )

    # --- Visualizaciones clave ------------------------------------------------
    st.subheader("Visualizaciones clave")

    # Gráficos de cabecera tomados del catálogo único `config.PLOTS`.
    key_plots = ["roc_curves.png", "confusion_matrices.png", "feature_importance.png"]

    for filename in key_plots:
        title, description = config.PLOTS[filename]
        image_card(
            config.OUTPUTS_DIR / filename,
            title=title,
            description=description,
            not_found_message=(
                f"No se encontró `outputs/{filename}` (aún no generado)."
            ),
        )
