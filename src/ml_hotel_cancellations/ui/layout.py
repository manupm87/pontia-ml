"""Pequeños helpers de *layout* compartidos por las secciones.

Centralizan decisiones de presentación —como mostrar las imágenes en dos
columnas (gráfico a la izquierda, descripción a la derecha)— para que cambiarlas
no obligue a tocar todas las secciones una a una.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from . import config, data

# Proporción de ancho entre la columna del gráfico y la del texto.
# Con [3, 2] el gráfico ocupa ~60 % del ancho disponible: lo suficiente para
# leerse cómodamente sin invadir la pantalla, y el texto al lado tiene su sitio.
IMAGE_COLUMN_RATIO: tuple[int, int] = (3, 2)


def image_card(
    image_path: Path,
    *,
    title: str | None = None,
    description: str | None = None,
    not_found_message: str | None = None,
) -> bool:
    """Muestra una imagen junto a su descripción en dos columnas.

    Pensado para los gráficos exportados a ``outputs/``: en lugar de ocupar el
    ancho completo de la página (lo que los muestra exageradamente grandes), el
    gráfico va en la columna izquierda y la explicación didáctica en la derecha.

    Parameters
    ----------
    image_path:
        Ruta del PNG. Si no existe, se muestra ``not_found_message`` (o se omite
        la tarjeta entera si es ``None``).
    title:
        Título opcional encima del bloque (a ancho completo, para que ancle el
        scroll de la página y se vea claro de qué gráfico se trata).
    description:
        Texto explicativo que acompaña al gráfico.
    not_found_message:
        Mensaje que se muestra si el PNG no existe. Si es ``None``, la tarjeta
        no se renderiza en absoluto cuando falta el fichero (útil para gráficos
        opcionales como los SHAP).

    Returns
    -------
    bool
        ``True`` si la imagen se mostró; ``False`` si no se encontró el PNG.
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

    Centraliza la máquina de 3 estados (conectada / dormida / no disponible) que
    antes vivía duplicada en la barra lateral y en la página de predicción, de
    modo que `check_api_health()` se llama (y cachea) una sola vez por render.

    Parameters
    ----------
    container:
        Destino donde pintar (p. ej. ``st.sidebar`` o ``st``).
    verbose:
        En modo verboso (página de predicción) se muestran mensajes largos con
        instrucciones de arranque; en modo compacto (barra lateral) se usan
        avisos breves.

    Returns
    -------
    bool
        ``True`` si la API responde y está operativa.
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
        # En hostings free (Render), el servicio se duerme tras 15 min de
        # inactividad. Avisamos al usuario en vez de dar un escueto "no
        # disponible" que parece un error de configuración.
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
