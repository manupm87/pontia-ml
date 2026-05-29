"""Pequeños helpers de *layout* compartidos por las secciones.

Centralizan decisiones de presentación —como mostrar las imágenes en dos
columnas (gráfico a la izquierda, descripción a la derecha)— para que cambiarlas
no obligue a tocar todas las secciones una a una.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from . import config, data

# Proporción gráfico/texto: con [3, 2] el gráfico ocupa ~60 % del ancho.
IMAGE_COLUMN_RATIO: tuple[int, int] = (3, 2)


def image_card(
    image_path: Path,
    *,
    title: str | None = None,
    description: str | None = None,
    not_found_message: str | None = None,
) -> bool:
    """Muestra una imagen (gráfico de ``outputs/``) junto a su descripción en
    dos columnas. Devuelve ``True`` si la imagen se mostró.

    Si el PNG no existe, muestra ``not_found_message`` o, si es ``None``, omite
    la tarjeta entera (útil para gráficos opcionales como los SHAP).
    """
    if not image_path.exists():
        if not_found_message is not None:
            if title:
                st.subheader(title)
            st.warning(not_found_message)
        return False

    if title:
        st.subheader(title)

    col_img, col_text = st.columns(IMAGE_COLUMN_RATIO)
    with col_img:
        st.image(str(image_path), use_container_width=True)
    with col_text:
        if description:
            st.markdown(description)
    return True


def render_api_status(container, *, verbose: bool) -> bool:
    """Dibuja el estado de la API en `container` y devuelve si está operativa.

    Centraliza la máquina de 3 estados (conectada / dormida / no disponible).
    Con `verbose` (página de predicción) muestra mensajes largos con
    instrucciones de arranque; sin él (barra lateral), avisos breves.
    """
    ok, info = data.check_api_health()
    remote = data.is_remote_api()

    if ok:
        model_ok = isinstance(info, dict) and info.get("model_loaded")
        if verbose:
            container.success(
                f"API conectada en `{config.API_BASE_URL}` "
                f"(modelo cargado: {'sí' if model_ok else 'no'}).",
                icon="✅",
            )
        else:
            container.success("Conectada", icon="✅")
            if isinstance(info, dict) and info.get("model_loaded") is False:
                container.warning(
                    "La API responde pero el modelo no está cargado.", icon="⚠️"
                )
        return True

    if remote:
        # En hostings free el servicio se duerme tras inactividad: avisamos en
        # vez de dar un "no disponible" que parecería un error de configuración.
        if verbose:
            container.warning(
                f"**La API parece estar dormida** en `{config.API_BASE_URL}`.\n\n"
                "Los servicios del tier gratuito de Render se apagan tras 15 min "
                "de inactividad y el primer arranque tarda **~30–50 s**. "
                "Espera medio minuto y recarga la página: si el problema persiste, "
                "es que la API no está desplegada.\n\n"
                f"Detalle técnico: {info}",
                icon="⏳",
            )
        else:
            container.warning(
                "Despertando la API… (~30–50 s la primera vez)",
                icon="⏳",
            )
        return False

    if verbose:
        container.error(
            f"**La API no está disponible** en `{config.API_BASE_URL}`.\n\n"
            f"Motivo: {info}",
            icon="🚫",
        )
        with container.expander("¿Cómo arranco la API?"):
            st.markdown(
                f"""
                La predicción necesita la **API FastAPI** del proyecto en marcha.
                Desde la raíz del repo, en otra terminal:

                ```bash
                uvicorn ml_hotel_cancellations.api.main:app --host 0.0.0.0 --port 8000
                ```

                Si la API corre en otra URL, indícala con la variable de entorno
                antes de lanzar esta interfaz:

                ```bash
                export PONTIA_API_URL="http://mi-host:puerto"
                streamlit run src/ml_hotel_cancellations/ui/app.py
                ```

                URL configurada actualmente: `{config.API_BASE_URL}`
                """
            )
    else:
        container.error("No disponible", icon="🚫")
    return False
