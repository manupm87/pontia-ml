"""Sección 2 — Visualización de los modelos.

Muestra TODAS las visualizaciones disponibles en `outputs/` con una explicación
didáctica de cada una. Es robusta: itera sobre una lista de gráficos conocidos y
solo pinta los que existen (algunos, como los SHAP, pueden aparecer más tarde).
"""

from __future__ import annotations

import streamlit as st

from .. import config
from ..layout import image_card

# Catálogo de gráficos conocidos: (fichero, título, explicación didáctica).
KNOWN_PLOTS: list[tuple[str, str, str]] = [
    (
        "roc_curves.png",
        "Curvas ROC de los 5 modelos",
        "Comparan la capacidad de cada modelo para separar reservas canceladas "
        "de no canceladas. El área bajo cada curva (**ROC-AUC**) resume su "
        "calidad: cuanto mayor, mejor.",
    ),
    (
        "confusion_matrices.png",
        "Matrices de confusión (todos los modelos)",
        "Para cada modelo, recuento de aciertos y errores por clase. La diagonal "
        "son los aciertos; fuera de ella, las confusiones. Ayuda a ver el "
        "compromiso entre **detectar cancelaciones** y **evitar falsas alarmas**.",
    ),
    (
        "confusion_matrix_best.png",
        "Matriz de confusión del mejor modelo (XGBoost)",
        "Detalle del modelo ganador. Arriba-izquierda y abajo-derecha son los "
        "aciertos; las otras dos celdas, los errores (**falsos positivos** y "
        "**falsos negativos**).",
    ),
    (
        "decision_regions_pls.png",
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
    (
        "feature_importance.png",
        "Importancia de variables",
        "Ranking de las características que más influyen en la predicción del "
        "modelo. Útil para entender el negocio: qué factores disparan el riesgo "
        "de cancelación.",
    ),
    (
        "balanceo_clases.png",
        "Efecto del balanceo de clases",
        "Compara estrategias para tratar el desbalance (~37 % de cancelaciones): "
        "sin balanceo, reponderación (`class_weight`) y sobremuestreo (SMOTE). "
        "El balanceo sube el *recall* (detecta más cancelaciones) a costa de "
        "algo de precisión, mientras la ROC-AUC apenas cambia.",
    ),
]


def render() -> None:
    st.title("Visualización de los modelos")
    st.markdown(
        "Galería de los gráficos generados por el pipeline de modelado. Cada "
        "imagen va acompañada de una breve explicación para interpretarla."
    )

    shown = 0
    for filename, title, explanation in KNOWN_PLOTS:
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
