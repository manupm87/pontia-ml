"""Interpretabilidad del modelo: ¿por qué predice lo que predice? (bonus).

Un buen ROC-AUC no basta: en problemas reales necesitamos **entender** las
decisiones del modelo para confiar en ellas, detectar sesgos y poder explicarlas
al negocio. Este módulo aporta dos familias de herramientas complementarias:

- **SHAP** (*SHapley Additive exPlanations*): reparte la predicción de cada
  reserva entre sus características usando los *valores de Shapley* (un concepto
  de la teoría de juegos cooperativos). Permite tanto una visión **global**
  (qué variables pesan más en todo el conjunto) como **local** (por qué se
  predijo esta reserva concreta). Para modelos de árboles (XGBoost, Random
  Forest) usamos ``shap.TreeExplainer``, que es exacto y muy rápido.
- **Importancia por permutación** (*permutation importance*): mide cuánto empeora
  una métrica (aquí ROC-AUC) al barajar al azar una variable. Es
  **agnóstica al modelo** (funciona con cualquier estimador, no solo árboles),
  por lo que sirve de complemento y contraste a SHAP.

El modelo que explicamos es un ``Pipeline`` de scikit-learn
``(preprocessor=ColumnTransformer, model)``. Como SHAP necesita trabajar con la
matriz de características YA preprocesada (numéricas estandarizadas + categóricas
codificadas en *one-hot*), separamos ambos pasos: transformamos ``X`` con el
``preprocessor`` y aplicamos el explicador sobre el estimador final.

Uso por línea de comandos (regenera todos los gráficos en ``outputs/``)::

    python -m ml_hotel_cancellations.utils.interpretability
    python -m ml_hotel_cancellations.utils.interpretability --sample 2000 --no-permutation
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ml_hotel_cancellations import config

config.use_agg_backend()  # backend no interactivo: guarda PNG sin necesidad de pantalla.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from .reporting import save_figure

logger = logging.getLogger(__name__)

# Número de filas que muestreamos para SHAP. Calcular los valores de Shapley sobre
# decenas de miles de filas es innecesario: una muestra de ~2000 reservas ya da
# una imagen global estable y reduce mucho el tiempo de cómputo.
DEFAULT_SHAP_SAMPLE: int = 2000


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------
def _transform_features(pipeline: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Aplica el preprocesador del ``Pipeline`` y devuelve la matriz + nombres.

    SHAP necesita los datos en el mismo espacio en el que "vive" el estimador
    final (numéricas escaladas y categóricas en *one-hot*). Aquí ejecutamos solo
    el paso ``preprocessor`` del pipeline y recuperamos los nombres legibles de
    las columnas resultantes con ``get_feature_names_out()``.

    Parameters
    ----------
    pipeline:
        ``Pipeline`` entrenado con los pasos ``preprocessor`` y ``model``.
    X:
        Características en crudo (mismas columnas que en entrenamiento).

    Returns
    -------
    tuple[numpy.ndarray, list[str]]
        Matriz transformada y lista de nombres de característica.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    X_trans = preprocessor.transform(X)
    # Algunos transformadores devuelven matrices dispersas; SHAP trabaja mejor
    # con arrays densos, así que las convertimos si hace falta.
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()
    feature_names = list(preprocessor.get_feature_names_out())
    return np.asarray(X_trans), feature_names


def _subsample(X: pd.DataFrame, n: int, random_state: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Toma una submuestra aleatoria reproducible de ``X`` (si procede)."""
    if n and len(X) > n:
        return X.sample(n=n, random_state=random_state)
    return X


def _patch_shap_xgboost_base_score() -> None:
    """Parche puntual: arregla un fallo de SHAP 0.49 con XGBoost ≥ 2.x.

    Desde XGBoost 2.x, el ``base_score`` que la librería serializa en su volcado
    UBJ es una cadena envuelta en corchetes (p. ej. ``"[3.7076378E-1]"``,
    porque internamente es un array). SHAP 0.49 hace ``float(base_score)``
    directamente, lo que revienta con ``ValueError``. Como las dependencias del
    proyecto fijan ``shap<0.50`` (forzado a su vez por TensorFlow 2.16.2 vía
    ``numpy<2.0``), no podemos saltar a la versión que lo corrige (0.50+) sin
    romper el resto del entorno. Solución: envolver el decodificador UBJ que
    usa SHAP para limpiar la cadena antes de que SHAP la lea.

    Idempotente: solo aplica el parche la primera vez.
    """
    import shap.explainers._tree as _shap_tree

    if getattr(_shap_tree, "_pontia_base_score_patched", False):
        return

    _orig_decode = _shap_tree.decode_ubjson_buffer

    def _decode_and_fix(fd):
        jm = _orig_decode(fd)
        learner = jm.get("learner") if isinstance(jm, dict) else None
        params = learner.get("learner_model_param") if isinstance(learner, dict) else None
        bs = params.get("base_score") if isinstance(params, dict) else None
        if isinstance(bs, str) and bs.startswith("[") and bs.endswith("]"):
            params["base_score"] = bs.strip("[]")
        return jm

    _shap_tree.decode_ubjson_buffer = _decode_and_fix
    _shap_tree._pontia_base_score_patched = True


def _build_tree_explainer(pipeline: Pipeline, X_sample: pd.DataFrame):
    """Construye un ``shap.TreeExplainer`` y calcula los valores SHAP.

    Returns
    -------
    tuple
        ``(shap_values, X_trans, feature_names)`` donde ``shap_values`` es un
        objeto ``shap.Explanation`` con un valor por (fila, característica).
    """
    import shap  # importación perezosa: solo se necesita aquí (dependencia del bonus).

    _patch_shap_xgboost_base_score()

    estimator = pipeline.named_steps["model"]
    X_trans, feature_names = _transform_features(pipeline, X_sample)

    # TreeExplainer es el algoritmo exacto y eficiente para modelos de árboles.
    explainer = shap.TreeExplainer(estimator)
    explanation = explainer(X_trans)
    # Anotamos los nombres de columna para que los gráficos los muestren.
    explanation.feature_names = feature_names
    return explanation, X_trans, feature_names


# ---------------------------------------------------------------------------
# Explicaciones globales (todo el conjunto)
# ---------------------------------------------------------------------------
def explain_global(
    pipeline: Pipeline,
    X: pd.DataFrame,
    output_dir: Path = config.OUTPUTS_DIR,
    sample_size: int = DEFAULT_SHAP_SAMPLE,
) -> dict[str, Path]:
    """Genera las explicaciones SHAP **globales** y las guarda como PNG.

    Produce dos gráficos clásicos de SHAP:

    - **Beeswarm** (*enjambre de abejas*): cada punto es una reserva; el eje X es
      su valor SHAP (cuánto empuja la predicción hacia "cancela" si es positivo,
      o hacia "no cancela" si es negativo) y el color, el valor de la variable
      (rojo = alto, azul = bajo). Permite ver de un vistazo qué variables mandan
      y en qué dirección.
    - **Bar** (*barras*): la media del valor absoluto de SHAP por variable, es
      decir, su importancia global media. Es el resumen más directo de "qué
      variables pesan más".

    Parameters
    ----------
    pipeline:
        ``Pipeline`` entrenado (preprocesador + modelo de árbol).
    X:
        Características en crudo sobre las que explicar (típicamente el test).
    output_dir:
        Carpeta donde guardar los PNG.
    sample_size:
        Nº de filas a muestrear para acelerar el cálculo.

    Returns
    -------
    dict[str, Path]
        Rutas de los gráficos generados (``beeswarm`` y ``bar``).
    """
    import shap

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_sample = _subsample(X, sample_size)
    logger.info("Calculando valores SHAP sobre %d reservas...", len(X_sample))
    explanation, _, _ = _build_tree_explainer(pipeline, X_sample)

    paths: dict[str, Path] = {}

    # 1) Beeswarm
    beeswarm_path = output_dir / "shap_summary_beeswarm.png"
    fig = plt.figure()
    shap.plots.beeswarm(explanation, max_display=20, show=False)
    plt.title("SHAP — Importancia global y dirección (beeswarm)")
    save_figure(fig, beeswarm_path, bbox_inches="tight")
    paths["beeswarm"] = beeswarm_path

    # 2) Bar (importancia media)
    bar_path = output_dir / "shap_summary_bar.png"
    fig = plt.figure()
    shap.plots.bar(explanation, max_display=20, show=False)
    plt.title("SHAP — Importancia media global (|valor SHAP| medio)")
    save_figure(fig, bar_path, bbox_inches="tight")
    paths["bar"] = bar_path

    return paths


# ---------------------------------------------------------------------------
# Explicación local (una reserva concreta)
# ---------------------------------------------------------------------------
def explain_local(
    pipeline: Pipeline,
    X: pd.DataFrame,
    idx: int,
    output_dir: Path = config.OUTPUTS_DIR,
    filename: str = "shap_waterfall.png",
    title: str | None = None,
) -> Path:
    """Explica la predicción de **una sola reserva** con un gráfico *waterfall*.

    El gráfico de cascada (*waterfall*) parte del valor base (la predicción media
    del modelo sobre todo el conjunto) y va sumando/restando la contribución de
    cada variable hasta llegar a la predicción final de ESTA reserva. Responde a
    la pregunta "¿por qué el modelo cree que esta reserva concreta se cancelará
    (o no)?".

    Parameters
    ----------
    pipeline:
        ``Pipeline`` entrenado.
    X:
        Características en crudo. Se usa ``X.iloc[[idx]]`` (la fila ``idx``).
    idx:
        Posición (entera, base 0) de la reserva a explicar dentro de ``X``.
    output_dir:
        Carpeta donde guardar el PNG.
    filename:
        Nombre del fichero de salida.
    title:
        Título opcional para el gráfico.

    Returns
    -------
    pathlib.Path
        Ruta del gráfico generado.
    """
    import shap

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_one = X.iloc[[idx]]
    explanation, _, _ = _build_tree_explainer(pipeline, X_one)

    path = output_dir / filename
    fig = plt.figure()
    # explanation[0] selecciona la única fila calculada.
    shap.plots.waterfall(explanation[0], max_display=15, show=False)
    if title:
        plt.title(title)
    save_figure(fig, path, bbox_inches="tight")
    return path


def explain_booking_to_figure(
    booking_df: pd.DataFrame,
    pipeline: Pipeline,
    max_display: int = 12,
):
    """Devuelve una figura SHAP *waterfall* para UNA reserva, sin guardar a disco.

    Pensada para la página de predicción de la interfaz visual: la UI llama a
    esta función con la reserva que el usuario acaba de enviar y la pinta con
    ``st.pyplot(fig)`` (no se persiste el PNG porque cambia cada vez).

    Parameters
    ----------
    booking_df:
        DataFrame de UNA fila con las mismas columnas que se usaron al
        entrenar (las 27 características en crudo).
    pipeline:
        ``Pipeline`` entrenado (preprocesador + modelo de árbol).
    max_display:
        Nº máximo de variables a mostrar en la cascada. Para una reserva
        suelta, 10-15 ya transmite la idea sin saturar.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import shap

    if len(booking_df) != 1:
        raise ValueError(
            f"Se esperaba una reserva (1 fila); se recibieron {len(booking_df)}."
        )

    explanation, _, _ = _build_tree_explainer(pipeline, booking_df)
    fig = plt.figure(figsize=(9, 6))
    # explanation[0] selecciona la única fila calculada.
    shap.plots.waterfall(explanation[0], max_display=max_display, show=False)
    plt.tight_layout()
    return fig


def find_examples(pipeline: Pipeline, X: pd.DataFrame) -> dict[str, int]:
    """Localiza reservas representativas para las explicaciones locales.

    Devuelve la posición de la reserva con MAYOR probabilidad de cancelación
    estimada (un caso "claramente cancela") y la de MENOR probabilidad (un caso
    "claramente no cancela"), para ilustrar el *waterfall* en ambos extremos.

    Returns
    -------
    dict[str, int]
        ``{"alta_prob": idx, "baja_prob": idx}`` con posiciones base 0.
    """
    proba = pipeline.predict_proba(X)[:, 1]
    return {
        "alta_prob": int(np.argmax(proba)),
        "baja_prob": int(np.argmin(proba)),
    }


# ---------------------------------------------------------------------------
# Importancia por permutación (agnóstica al modelo)
# ---------------------------------------------------------------------------
def permutation_importance_report(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path = config.OUTPUTS_DIR,
    n_repeats: int = 10,
    top_n: int = 20,
    scoring: str = config.PRIMARY_METRIC,
) -> tuple[pd.DataFrame, Path]:
    """Calcula la importancia por permutación y guarda un gráfico de barras.

    A diferencia de SHAP (específico de árboles aquí), esta técnica es
    **agnóstica al modelo**: baraja al azar los valores de una variable y mide
    cuánto cae la métrica (ROC-AUC). Si barajar una variable destroza el
    rendimiento, es que el modelo dependía mucho de ella. Funciona con CUALQUIER
    estimador (regresión logística, red neuronal...), por lo que sirve para
    comparar de forma justa entre modelos distintos.

    Se calcula sobre el ``Pipeline`` completo y las columnas EN CRUDO, de modo que
    la importancia se atribuye a las variables originales (``lead_time``,
    ``deposit_type``...) y no a las columnas one-hot expandidas.

    Parameters
    ----------
    pipeline:
        ``Pipeline`` entrenado.
    X, y:
        Características en crudo y etiquetas reales (típicamente el test).
    output_dir:
        Carpeta donde guardar el PNG.
    n_repeats:
        Nº de barajados por variable (más = estimación más estable, más lento).
    top_n:
        Nº de variables a mostrar en el gráfico.
    scoring:
        Métrica de scikit-learn a usar (por defecto, la principal del proyecto).

    Returns
    -------
    tuple[pandas.DataFrame, pathlib.Path]
        Tabla ordenada (media y desviación de la caída de métrica) y ruta del PNG.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Calculando importancia por permutación (scoring=%s, n_repeats=%d)...",
        scoring,
        n_repeats,
    )
    result = permutation_importance(
        pipeline,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )

    tabla = (
        pd.DataFrame(
            {
                "variable": X.columns,
                "importancia_media": result.importances_mean,
                "importancia_std": result.importances_std,
            }
        )
        .sort_values("importancia_media", ascending=False)
        .reset_index(drop=True)
    )

    top = tabla.head(top_n).iloc[::-1]  # invertir para barh de mayor a menor
    path = output_dir / "permutation_importance.png"
    fig = plt.figure(figsize=(9, 0.42 * len(top) + 1.5))
    plt.barh(top["variable"], top["importancia_media"], xerr=top["importancia_std"], color="#2c7fb8")
    plt.xlabel(f"Caída media en {scoring} al barajar la variable")
    plt.ylabel("Variable")
    plt.title(f"Importancia por permutación (top {len(top)})")
    save_figure(fig, path)

    return tabla, path


# ---------------------------------------------------------------------------
# CLI / orquestador
# ---------------------------------------------------------------------------
def main() -> None:
    """Carga el mejor modelo y los datos, y regenera todos los gráficos.

    Pensado para ejecutarse como ``python -m ml_hotel_cancellations.utils.interpretability`` desde la
    raíz del repo.
    """
    config.configure_logging()
    parser = argparse.ArgumentParser(
        description="Interpretabilidad del mejor modelo (SHAP + permutación)."
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=DEFAULT_SHAP_SAMPLE,
        help="Nº de reservas a muestrear para SHAP (por defecto 2000).",
    )
    parser.add_argument(
        "--no-permutation",
        action="store_true",
        help="Omite el cálculo (más lento) de importancia por permutación.",
    )
    args = parser.parse_args()

    # Importaciones locales para que el módulo se pueda importar sin cargar datos.
    import joblib

    from ml_hotel_cancellations.ml.data_loader import load_and_prepare

    config.ensure_directories()

    if not config.BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {config.BEST_MODEL_PATH}. Entrena con 'python -m ml_hotel_cancellations.ml.train' "
            "o copia el modelo desde el checkout principal."
        )
    logger.info("Cargando mejor modelo desde %s", config.BEST_MODEL_PATH)
    pipeline = joblib.load(config.BEST_MODEL_PATH)

    _, X_test, _, y_test = load_and_prepare()

    # 1) Explicaciones globales (beeswarm + bar).
    explain_global(pipeline, X_test, sample_size=args.sample)

    # 2) Explicaciones locales: una reserva que el modelo da por cancelada y otra
    #    que da por no cancelada. Recorremos ambos extremos en vez de duplicar la
    #    llamada.
    ejemplos = find_examples(pipeline, X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    casos = [
        ("alta_prob", "shap_waterfall_ejemplo1.png", "alta"),
        ("baja_prob", "shap_waterfall_ejemplo2.png", "baja"),
    ]
    for clave, filename, etiqueta in casos:
        idx = ejemplos[clave]
        explain_local(
            pipeline,
            X_test,
            idx,
            filename=filename,
            title=f"Reserva con {etiqueta} probabilidad de cancelación (p={proba[idx]:.3f})",
        )

    # 3) Importancia por permutación (complemento agnóstico al modelo).
    if not args.no_permutation:
        permutation_importance_report(pipeline, X_test, y_test)

    logger.info("Interpretabilidad completada. Gráficos en %s", config.OUTPUTS_DIR)


if __name__ == "__main__":
    main()
