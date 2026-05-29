"""Sección 2 — Visualización de los modelos.

Galería de los gráficos de `outputs/` con explicación didáctica. Itera sobre
los gráficos conocidos y solo pinta los que existen.
"""

from __future__ import annotations

import streamlit as st

from .. import config
from ..layout import image_card

# Claves del catálogo único `config.PLOTS`, en el orden de visualización.
KNOWN_PLOTS: list[str] = [
    "roc_curves.png",
    "confusion_matrices.png",
    "confusion_matrix_best.png",
    "decision_regions_pls.png",
    "feature_importance.png",
    "balanceo_clases.png",
]


def render() -> None:
    st.title("Visualización de los modelos")
    st.markdown(
        "Galería de los gráficos generados por el pipeline de modelado. Cada "
        "imagen va acompañada de una breve explicación para interpretarla."
    )

    shown = 0
    for filename in KNOWN_PLOTS:
        title, explanation = config.PLOTS[filename]
        rendered = image_card(
            config.OUTPUTS_DIR / filename,
            title=title,
            description=explanation,
        )
        if rendered:
            shown += 1
            st.divider()

    if shown == 0:
        st.warning(
            "No se encontraron visualizaciones en `outputs/`. Ejecuta el "
            "pipeline de entrenamiento (`python -m ml_hotel_cancellations.ml.train`) para generarlas."
        )
    else:
        st.success(f"Se muestran {shown} visualizaciones disponibles.", icon="🖼️")
