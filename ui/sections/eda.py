"""Sección 5 — Exploración de datos (EDA).

Análisis Exploratorio de Datos: una mirada a los datos crudos antes de modelar.
Se muestran la tasa de cancelación por distintas categorías (qué grupos cancelan
más), el balance de clases y un resumen de variables numéricas.
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from .. import config, data

# Categorías interesantes para analizar la tasa de cancelación, con su etiqueta.
RATE_DIMENSIONS: dict[str, str] = {
    "deposit_type": "Tipo de depósito",
    "hotel": "Tipo de hotel",
    "customer_type": "Tipo de cliente",
    "market_segment": "Segmento de mercado",
}


def _bar_cancellation_rate(column: str, label: str) -> None:
    """Pinta un gráfico de barras de la tasa de cancelación por categoría."""
    df = data.cancellation_rate_by(column)
    df = df.copy()
    df["pct"] = (df["tasa_cancelacion"] * 100).round(1)
    fig = px.bar(
        df,
        x="categoria",
        y="pct",
        text="pct",
        labels={"categoria": label, "pct": "% cancelaciones"},
        color="pct",
        color_continuous_scale="Reds",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.add_hline(
        y=config.BASE_CANCELLATION_RATE * 100,
        line_dash="dash",
        annotation_text=f"Media global ({config.BASE_CANCELLATION_RATE_TEXT})",
        annotation_position="top right",
    )
    fig.update_layout(coloraxis_showscale=False, yaxis_title="% cancelaciones")
    st.plotly_chart(fig, use_container_width=True)


def render() -> None:
    st.title("Exploración de datos (EDA)")
    st.markdown(
        """
        El **EDA** (Análisis Exploratorio de Datos) consiste en mirar los datos
        antes de modelar para entender patrones y posibles problemas. Aquí
        respondemos a una pregunta clave: **¿qué reservas cancelan más?**
        """
    )

    # --- Balance de clases ----------------------------------------------------
    st.subheader("Balance de clases")
    st.caption(
        "Reparto entre reservas canceladas y no canceladas. El desbalance "
        f"({config.BASE_CANCELLATION_RATE_TEXT} cancelan) justifica usar ROC-AUC "
        "y explorar técnicas de balanceo."
    )
    balance = data.class_balance()
    col_chart, col_table = st.columns(config.CHART_TABLE_RATIO)
    with col_chart:
        fig = px.pie(
            balance,
            names="clase",
            values="reservas",
            color="clase",
            color_discrete_map=config.CLASS_COLORS,
            hole=0.4,
        )
        st.plotly_chart(fig, use_container_width=True)
    with col_table:
        st.dataframe(balance, use_container_width=True, hide_index=True)

    st.divider()

    # --- Tasa de cancelación por categoría ------------------------------------
    st.subheader("Tasa de cancelación por categoría")
    st.caption(
        "La línea discontinua marca la media global. Las barras por encima son "
        "grupos con MÁS riesgo de cancelación que la media."
    )
    dimension = st.selectbox(
        "Analizar por:",
        options=list(RATE_DIMENSIONS.keys()),
        format_func=lambda c: RATE_DIMENSIONS[c],
    )
    _bar_cancellation_rate(dimension, RATE_DIMENSIONS[dimension])

    st.markdown(
        "**Lectura típica.** El tipo de depósito *Non Refund* (no reembolsable) "
        "se asocia paradójicamente a tasas de cancelación altísimas en este "
        "dataset, y los hoteles urbanos (*City*) cancelan más que los "
        "vacacionales (*Resort*)."
    )

    st.divider()

    # --- Resumen numérico -----------------------------------------------------
    st.subheader("Resumen de variables numéricas")
    st.caption(
        "Estadísticos básicos de algunas variables continuas relevantes "
        "(antelación, tarifa media, noches y peticiones especiales)."
    )
    st.dataframe(data.numeric_summary(), use_container_width=True, hide_index=True)

    with st.expander("Ver una muestra del dataset"):
        st.dataframe(data.load_dataset_sample(100), use_container_width=True)
