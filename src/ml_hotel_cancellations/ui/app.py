"""Interfaz visual del proyecto (Streamlit) — punto de entrada.

Arranque (desde la raíz del repo, con el paquete instalado vía `pip install -e .`):

    streamlit run src/ml_hotel_cancellations/ui/app.py

La navegación entre secciones está en la barra lateral. Cada sección vive en su
propio módulo dentro de `ui/sections/` y expone una función `render()`, de modo
que este fichero solo se ocupa de la estructura general (layout, navegación,
estado de la API), no del contenido.

La sección de predicción consume la **API FastAPI** del proyecto; su URL se lee
de la variable de entorno `PONTIA_API_URL` (por defecto `http://localhost:8000`).
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
    """Dibuja la barra lateral (navegación + estado de la API). Devuelve la
    sección elegida por el usuario."""
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

    Usa un *flag* en `st.session_state` como guard one-shot: la primera vez que
    se carga la aplicación dispara la petición; los `rerun` siguientes la saltan.
    En localhost es no-op (`warm_up_api` devuelve True inmediatamente).
    """
    if st.session_state.get(_PREWARM_FLAG):
        return
    data.warm_up_api()
    st.session_state[_PREWARM_FLAG] = True


def main() -> None:
    _configure_page()
    # Lanza el *pre-warm* antes de pintar nada: si la API estaba dormida
    # en Render, esta llamada se ejecuta mientras el usuario lee la
    # interfaz, de modo que cuando intente predecir la API ya esté lista.
    _prewarm_api()
    choice = _render_sidebar()
    # Renderiza la sección seleccionada.
    SECTIONS[choice]()


def _running_under_streamlit() -> bool:
    """¿Se está ejecutando dentro del runtime de Streamlit (`streamlit run`)?

    Distingue el arranque real de la app de una importación normal (p. ej. un
    test que hace `import ui.app`): solo en el primer caso existe un *script run
    context* activo y tiene sentido renderizar. Capturamos `ImportError` por si
    la API interna de Streamlit cambia de ubicación entre versiones; cualquier
    otro fallo debe propagarse en lugar de enmascararse.
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ImportError:
        return False
    return get_script_run_ctx() is not None


# `streamlit run src/ml_hotel_cancellations/ui/app.py` ejecuta el script con `__name__ == "__main__"`, pero
# algunas versiones lo importan como módulo; cubrimos ambos casos detectando el
# runtime. Al importarlo desde un test (sin runtime) no se renderiza nada.
if __name__ == "__main__" or _running_under_streamlit():
    main()
