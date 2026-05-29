"""Interfaz visual del proyecto (Streamlit) — punto de entrada.

Arranque: `streamlit run src/ml_hotel_cancellations/ui/app.py`. Solo gestiona la
estructura general (layout, navegación, estado de la API); cada sección vive en
`ui/sections/` y expone una función `render()`.
"""

from __future__ import annotations

import streamlit as st

from ml_hotel_cancellations.ui import config, data, layout
from ml_hotel_cancellations.ui.sections import (
    eda,
    interpretabilidad,
    prediccion,
    resumen,
    visualizaciones,
)

# Mapa de secciones: etiqueta del menú -> función de renderizado.
SECTIONS: dict[str, callable] = {
    "Resumen y resultados": resumen.render,
    "Visualización de los modelos": visualizaciones.render,
    "Predicción (API)": prediccion.render,
    "Interpretabilidad": interpretabilidad.render,
    "Exploración (EDA)": eda.render,
}


def _configure_page() -> None:
    """Configuración global de la página (debe ir la primera de todo)."""
    st.set_page_config(
        page_title="Predicción de cancelaciones — Pontia ML",
        page_icon="🏨",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def _render_sidebar() -> str:
    """Dibuja la barra lateral (navegación + estado de la API) y devuelve la
    sección elegida."""
    st.sidebar.title("🏨 Pontia ML")
    st.sidebar.markdown(
        "Escaparate del proyecto de **predicción de cancelaciones** de reservas "
        "de hotel (clasificación binaria)."
    )

    choice = st.sidebar.radio(
        "Secciones", list(SECTIONS.keys()), label_visibility="visible"
    )

    st.sidebar.divider()
    st.sidebar.subheader("Estado de la API")
    layout.render_api_status(st.sidebar, verbose=False)
    st.sidebar.caption(f"URL: `{config.API_BASE_URL}`")
    if data.is_remote_api():
        st.sidebar.caption(
            "Servida en Render free: la primera petición tras un rato "
            "de inactividad puede tardar mientras el servicio arranca."
        )
    else:
        st.sidebar.caption(
            "La predicción requiere la API en marcha "
            "(`uvicorn ml_hotel_cancellations.api.main:app`). Cambia la URL con `PONTIA_API_URL`."
        )

    st.sidebar.divider()
    st.sidebar.caption(
        f"Mejor modelo: **{config.BEST_MODEL_NAME}** · "
        f"ROC-AUC {config.BEST_MODEL_ROC_AUC:.4f}"
    )
    return choice


_PREWARM_FLAG = "_api_prewarmed"


def _prewarm_api() -> None:
    """Despierta la API remota una sola vez por sesión.

    Usa un *flag* en `st.session_state` como guard one-shot: dispara la petición
    en la primera carga y la salta en los `rerun`. En localhost es no-op.
    """
    if st.session_state.get(_PREWARM_FLAG):
        return
    data.warm_up_api()
    st.session_state[_PREWARM_FLAG] = True


def main() -> None:
    _configure_page()
    # Pre-warm antes de pintar: arranca la API dormida mientras el usuario lee.
    _prewarm_api()
    choice = _render_sidebar()
    SECTIONS[choice]()


def _running_under_streamlit() -> bool:
    """¿Se está ejecutando dentro del runtime de Streamlit (`streamlit run`)?

    Distingue el arranque real de una importación (p. ej. un test): solo en el
    primero hay *script run context* activo. Captura `ImportError` por si la API
    interna de Streamlit cambia entre versiones.
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ImportError:
        return False
    return get_script_run_ctx() is not None


# `streamlit run` puede ejecutar el script como `__main__` o importarlo como
# módulo: cubrimos ambos casos. Al importarlo desde un test (sin runtime) no se
# renderiza nada.
if __name__ == "__main__" or _running_under_streamlit():
    main()
