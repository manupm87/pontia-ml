"""Pequeños helpers de *layout* compartidos por las secciones.

Centralizan decisiones de presentación —como mostrar las imágenes en dos
columnas (gráfico a la izquierda, descripción a la derecha)— para que cambiarlas
no obligue a tocar todas las secciones una a una.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

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
