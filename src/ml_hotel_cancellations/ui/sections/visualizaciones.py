"""Sección 2 — Visualización de los modelos.

Muestra TODAS las visualizaciones disponibles en `outputs/` con una explicación
didáctica de cada una. Es robusta: itera sobre una lista de gráficos conocidos y
solo pinta los que existen (algunos, como los SHAP, pueden aparecer más tarde).
"""

from __future__ import annotations

import streamlit as st

from .. import config
from ..layout import image_card

# Galería completa: todas las claves del catálogo único de plots (`config.PLOTS`),
# en el orden en que se quieren mostrar. Los textos viven en el catálogo para no
# duplicarse con la sección de resumen.
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
            "pipeline de entrenamiento (`python -m src.train`) para generarlas."
        )
    else:
        st.success(f"Se muestran {shown} visualizaciones disponibles.", icon="🖼️")
