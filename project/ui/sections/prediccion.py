"""Sección 3 — Predicción (consume la API FastAPI).

Presenta un formulario con las 27 variables de una reserva y, al enviarlo,
llama a la API (`POST /predict`) para obtener la probabilidad de cancelación.
Si la API no está disponible, se explica en español cómo arrancarla.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import booking, config, data


@st.cache_resource(show_spinner=False)
def _load_best_model():
    """Carga el ``Pipeline`` del mejor modelo una sola vez por sesión."""
    from src.predict import load_best_model

    return load_best_model()


@st.cache_resource(show_spinner=False)
def _load_pls_artifacts():
    """Carga los artefactos de la visualización 2D una sola vez por sesión.

    Pesa ~5 MB en memoria; cargarlo una sola vez evita parar la UI cada vez que
    el usuario predice una reserva.
    """
    from src.visualization_2d import _load_artifacts

    return _load_artifacts()


def _render_shap_explanation(payload: dict) -> None:
    """Pinta el *waterfall* SHAP de la reserva (explicación local).

    Si el modelo o la dependencia SHAP no están disponibles, lo dice y sigue.
    """
    st.subheader("¿Por qué esta predicción? — SHAP local")
    st.caption(
        "El gráfico de cascada (*waterfall*) parte de la predicción media del "
        "modelo y va sumando/restando la contribución de cada variable hasta la "
        "predicción de esta reserva. **Rojo** = empuja hacia *cancela*; **azul** "
        "= empuja hacia *no cancela*."
    )
    try:
        from src.interpretability import explain_booking_to_figure

        with st.spinner("Calculando contribuciones SHAP…"):
            pipeline = _load_best_model()
            booking_df = pd.DataFrame([payload])
            fig = explain_booking_to_figure(booking_df, pipeline)
        st.pyplot(fig, use_container_width=True)
    except FileNotFoundError as exc:
        st.warning(
            "No se encontró el modelo entrenado para explicar la predicción "
            f"localmente ({exc}). Ejecuta `python -m src.train`."
        )
    except Exception as exc:  # noqa: BLE001 - mostramos el error sin romper la UI
        st.error(f"No se pudo calcular la explicación SHAP: {exc}")


def _render_position_on_2d(payload: dict, probability: float) -> None:
    """Sitúa la reserva del usuario en el mapa 2D de regiones de decisión."""
    st.subheader("Tu reserva en el mapa 2D de los modelos")
    st.caption(
        "El mismo mapa de la sección *Visualización de los modelos*, pero con "
        "**tu reserva** marcada con una estrella amarilla. Si cae en zona roja, "
        "todos los modelos coinciden en señalarla como probable cancelación; en "
        "zona azul, en lo contrario; cerca de la línea negra, el modelo duda."
    )
    try:
        from src.visualization_2d import plot_booking_on_2d

        with st.spinner("Proyectando la reserva al plano PLS…"):
            artifacts = _load_pls_artifacts()
            booking_df = pd.DataFrame([payload])
            fig = plot_booking_on_2d(
                booking_df,
                probability_canceled=probability,
                artifacts=artifacts,
            )
        st.pyplot(fig, use_container_width=True)
    except FileNotFoundError as exc:
        st.warning(
            "No se encontraron los artefactos de la visualización 2D "
            f"({exc}). Ejecuta `python -m src.visualization_2d`."
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudo dibujar el mapa 2D: {exc}")


def _render_api_status() -> bool:
    """Muestra el estado de la API y devuelve True si está operativa."""
    ok, info = data.check_api_health()
    if ok:
        modelo_ok = isinstance(info, dict) and info.get("model_loaded")
        st.success(
            f"API conectada en `{config.API_BASE_URL}` "
            f"(modelo cargado: {'sí' if modelo_ok else 'no'}).",
            icon="✅",
        )
        return True

    if data.is_remote_api():
        # En Render free, "no disponible" suele significar "dormida".
        # Mostramos un aviso amable en vez de un error técnico.
        st.warning(
            f"**La API parece estar dormida** en `{config.API_BASE_URL}`.\n\n"
            "Los servicios del tier gratuito de Render se apagan tras 15 min "
            "de inactividad y el primer arranque tarda **~30–50 s**. "
            "Espera medio minuto y recarga la página: si el problema persiste, "
            "es que la API no está desplegada.\n\n"
            f"Detalle técnico: {info}",
            icon="⏳",
        )
        return False

    st.error(
        f"**La API no está disponible** en `{config.API_BASE_URL}`.\n\n"
        f"Motivo: {info}",
        icon="🚫",
    )
    with st.expander("¿Cómo arranco la API?"):
        st.markdown(
            f"""
            La predicción necesita la **API FastAPI** del proyecto en marcha.
            Desde la carpeta `project/`, en otra terminal:

            ```bash
            uvicorn api.main:app --host 0.0.0.0 --port 8000
            ```

            Si la API corre en otra URL, indícala con la variable de entorno
            antes de lanzar esta interfaz:

            ```bash
            export PONTIA_API_URL="http://mi-host:puerto"
            streamlit run ui/app.py
            ```

            URL configurada actualmente: `{config.API_BASE_URL}`
            """
        )
    return False


def _render_form() -> dict | None:
    """Dibuja el formulario de la reserva. Devuelve el payload si se envía."""
    options = data.get_categorical_options()
    defaults = booking.EXAMPLE_BOOKING
    values: dict = {}

    with st.form("form_prediccion"):
        st.caption(
            "El formulario viene precargado con una reserva de ejemplo. Ajusta "
            "los valores y pulsa **Predecir** para consultar al modelo."
        )

        for section_name, fields in booking.FORM_SECTIONS.items():
            st.markdown(f"**{section_name}**")
            cols = st.columns(2)
            for i, fld in enumerate(fields):
                target = cols[i % 2]
                with target:
                    if fld.kind == "categorical":
                        opts = options.get(fld.options_key or fld.name, [])
                        default = defaults.get(fld.name)
                        # Garantizamos que el valor por defecto esté en la lista.
                        if default is not None and default not in opts:
                            opts = [default, *opts]
                        index = opts.index(default) if default in opts else 0
                        values[fld.name] = st.selectbox(
                            fld.label, opts, index=index, help=fld.help,
                            key=f"f_{fld.name}",
                        )
                    elif fld.kind == "int":
                        values[fld.name] = st.number_input(
                            fld.label,
                            min_value=int(fld.min) if fld.min is not None else None,
                            max_value=int(fld.max) if fld.max is not None else None,
                            value=int(defaults.get(fld.name, 0)),
                            step=int(fld.step or 1),
                            help=fld.help,
                            key=f"f_{fld.name}",
                        )
                    else:  # float
                        values[fld.name] = st.number_input(
                            fld.label,
                            min_value=float(fld.min) if fld.min is not None else None,
                            max_value=float(fld.max) if fld.max is not None else None,
                            value=float(defaults.get(fld.name, 0.0)),
                            step=float(fld.step or 1.0),
                            help=fld.help,
                            key=f"f_{fld.name}",
                        )
            st.divider()

        submitted = st.form_submit_button("Predecir", type="primary", use_container_width=True)

    if submitted:
        return booking.build_payload(values)
    return None


def _render_result(result: dict) -> None:
    """Presenta la predicción devuelta por la API de forma visual."""
    prediction = int(result.get("prediction", 0))
    label = result.get("label", config.CLASS_LABELS.get(prediction, str(prediction)))
    proba = float(result.get("probability", 0.0))

    st.subheader("Resultado de la predicción")
    col1, col2 = st.columns(2)
    col1.metric(
        "Decisión del modelo",
        label,
        help="Etiqueta con umbral 0.5 sobre la probabilidad de cancelación.",
    )
    col2.metric(
        "Probabilidad de cancelación",
        f"{proba * 100:.1f} %",
        delta=f"{(proba - config.BASE_CANCELLATION_RATE) * 100:+.1f} pp vs. media",
        help="Comparada con la tasa media de cancelación del dataset (~37 %). "
        "'pp' = puntos porcentuales.",
    )

    # Barra de progreso como "medidor" visual del riesgo.
    st.progress(min(max(proba, 0.0), 1.0))

    if prediction == 1:
        st.warning(
            "El modelo estima un **riesgo alto de cancelación**. Acciones "
            "posibles: pedir depósito, confirmar la reserva o sobre-reservar.",
            icon="⚠️",
        )
    else:
        st.success(
            "El modelo estima que la reserva **probablemente se mantendrá**.",
            icon="👍",
        )

    with st.expander("Ver respuesta completa de la API (JSON)"):
        st.json(result)


def render() -> None:
    st.title("Predicción de cancelación")
    st.markdown(
        "Rellena los datos de una reserva y el modelo (servido por la **API "
        "FastAPI**) estimará la probabilidad de que se cancele."
    )

    api_ok = _render_api_status()

    payload = _render_form()
    if payload is None:
        return

    if not api_ok:
        # Reintentamos el health al enviar, por si la API se levantó mientras tanto.
        api_ok, _ = data.check_api_health()
        if not api_ok:
            st.error(
                "No se puede predecir: la API sigue sin responder. Arráncala y "
                "vuelve a intentarlo.",
                icon="🚫",
            )
            return

    with st.spinner("Consultando al modelo..."):
        ok, result = data.predict_booking(payload)

    if ok and isinstance(result, dict):
        _render_result(result)
        st.divider()
        _render_shap_explanation(payload)
        st.divider()
        _render_position_on_2d(payload, float(result.get("probability", 0.0)))
    else:
        st.error(f"No se pudo obtener la predicción. {result}", icon="🚫")
