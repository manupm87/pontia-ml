"""Interfaz visual del proyecto (Streamlit) — punto de entrada.

Arranque (desde la raíz del repo):

    streamlit run ui/app.py

La navegación entre secciones está en la barra lateral. Cada sección vive en su
propio módulo dentro de `ui/sections/` y expone una función `render()`, de modo
que este fichero solo se ocupa de la estructura general (layout, navegación,
estado de la API), no del contenido.

La sección de predicción consume la **API FastAPI** del proyecto; su URL se lee
de la variable de entorno `PONTIA_API_URL` (por defecto `http://localhost:8000`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Permite ejecutar `streamlit run ui/app.py` desde la raíz del repo: añadimos
# esa raíz al path para que `import ui...` resuelva al paquete local.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui import config, data  # noqa: E402
from ui.sections import (  # noqa: E402
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
    ok, info = data.check_api_health()
    if ok:
        st.sidebar.success("Conectada", icon="✅")
        if isinstance(info, dict) and info.get("model_loaded") is False:
            st.sidebar.warning(
                "La API responde pero el modelo no está cargado.", icon="⚠️"
            )
    elif data.is_remote_api():
        # En hostings free (Render), el servicio se duerme tras 15 min
        # de inactividad. Avisamos al usuario en vez de dar un escueto
        # "no disponible" que parece un error de configuración.
        st.sidebar.warning(
            "Despertando la API… (~30–50 s la primera vez)",
            icon="⏳",
        )
    else:
        st.sidebar.error("No disponible", icon="🚫")
    st.sidebar.caption(f"URL: `{config.API_BASE_URL}`")
    if data.is_remote_api():
        st.sidebar.caption(
            "Servida en Render free: la primera petición tras un rato "
            "de inactividad puede tardar mientras el servicio arranca."
        )
    else:
        st.sidebar.caption(
            "La predicción requiere la API en marcha "
            "(`uvicorn api.main:app`). Cambia la URL con `PONTIA_API_URL`."
        )

    st.sidebar.divider()
    st.sidebar.caption(
        f"Mejor modelo: **{config.BEST_MODEL_NAME}** · "
        f"ROC-AUC {config.BEST_MODEL_ROC_AUC:.4f}"
    )
    return choice


@st.cache_resource(show_spinner=False)
def _prewarm_api() -> bool:
    """Despierta la API remota una sola vez por sesión.

    Cacheado con `st.cache_resource`: la primera vez que se carga la
    aplicación dispara la petición; los `rerun` siguientes la saltan.
    En localhost es no-op (devuelve True inmediatamente).
    """
    return data.warm_up_api()


def main() -> None:
    _configure_page()
    # Lanza el *pre-warm* antes de pintar nada: si la API estaba dormida
    # en Render, esta llamada se ejecuta mientras el usuario lee la
    # interfaz, de modo que cuando intente predecir la API ya esté lista.
    _prewarm_api()
    choice = _render_sidebar()
    # Renderiza la sección seleccionada.
    SECTIONS[choice]()


if __name__ == "__main__":
    main()
else:
    # Streamlit importa el script como módulo (`__name__` != "__main__") al
    # ejecutarlo con `streamlit run`, así que disparamos el renderizado también
    # en ese caso. Al importarlo desde un test, en cambio, NO se ejecuta porque
    # Streamlit no está en modo de script (las llamadas a `st.*` no harían nada
    # útil); para evitar efectos colaterales, comprobamos el contexto.
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx() is not None:
            main()
    except Exception:
        # Sin runtime de Streamlit (importación normal): no renderizamos nada.
        pass
