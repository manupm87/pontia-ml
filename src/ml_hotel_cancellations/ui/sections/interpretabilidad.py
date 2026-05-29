"""Sección 4 — Interpretabilidad.

"Abre la caja negra" del modelo: muestra los gráficos SHAP (`shap_*.png`) si
existen y, en cualquier caso, la importancia de variables clásica como base.
"""

from __future__ import annotations

import streamlit as st

from .. import config, data
from ..layout import image_card


def render() -> None:
    st.title("Interpretabilidad del modelo")
    st.markdown(
        """
        Un buen modelo no solo debe acertar, también debería ser **explicable**.
        Aquí vemos *por qué* el modelo decide lo que decide.

        - **Importancia de variables**: visión global de qué características pesan
          más en el modelo.
        - **SHAP** (*SHapley Additive exPlanations*): técnica que reparte la
          predicción entre las variables, indicando cuánto empuja cada una hacia
          "cancela" o "no cancela", tanto a nivel global como de una reserva
          concreta.
        """
    )

    shap_plots = data.find_shap_pngs()

    if shap_plots:
        st.subheader("Explicaciones SHAP")
        st.success(
            f"Se encontraron {len(shap_plots)} gráficos SHAP en `outputs/`.",
            icon="🔍",
        )
        for path in shap_plots:
            image_card(
                path,
                description=(
                    f"`{path.name}` — En un gráfico SHAP, cada punto/barra "
                    "indica cuánto contribuye una variable a la predicción; "
                    "valores positivos empujan hacia 'cancela' y negativos hacia "
                    "'no cancela'."
                ),
            )
            st.divider()
    else:
        st.info(
            "Todavía no hay gráficos SHAP (`outputs/shap_*.png`). Aparecerán aquí "
            "automáticamente cuando se ejecute el módulo de interpretabilidad. "
            "Mientras tanto, mostramos la importancia de variables clásica.",
            icon="ℹ️",
        )

    # Importancia de variables como base, siempre presente (catálogo `config.PLOTS`).
    fi_title, fi_description = config.PLOTS["feature_importance.png"]
    image_card(
        config.OUTPUTS_DIR / "feature_importance.png",
        title=fi_title,
        description=fi_description,
        not_found_message="No se encontró `outputs/feature_importance.png`.",
    )
