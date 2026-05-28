"""Visualización 2D de los modelos: proyección PLS + regiones de decisión.

Genera la figura del notebook ``07_comparativa_modelos.ipynb`` §6.1 como un PNG
guardado en ``outputs/decision_regions_pls.png``, listo para incrustarlo en la
interfaz visual (Streamlit) sin tener que recomputar cada vez.

Idea
----
Para *ver* en 2D modelos que se entrenan con ~200 variables hace falta proyectar
los datos a un plano. PCA elige las 2 direcciones de **mayor varianza**, pero
**ignora** la clase: si lo que separa las cancelaciones no coincide con lo que
más varía, las clases se mezclan. **PLS** (*Partial Least Squares*) hace lo
mismo que PCA pero **mirando el ``target``**: elige las 2 direcciones más
correlacionadas con ``is_canceled``, y entonces las clases sí se separan.

Procedimiento::

    1. Preprocesamos (ColumnTransformer) y proyectamos a 2 componentes PLS.
    2. Reentrenamos los 5 modelos SOBRE esas 2 componentes (es necesario:
       los modelos originales usan ~200 columnas y no pueden dibujarse en 2D).
       La red neuronal real (Keras) se sustituye por un ``MLPClassifier``
       de scikit-learn, equivalente en espíritu y mucho más ligero.
    3. Evaluamos ``predict_proba`` sobre una rejilla del plano y la pintamos
       (rojo = predice cancelación, azul = no) con la frontera 0.5.

Además del PNG, se persisten los artefactos necesarios para **proyectar nuevas
reservas** sobre el plano sin recomputar nada (preprocesador + PLS + signo de
orientación + rejilla con las probabilidades precalculadas de cada modelo).
Esto permite que la página de predicción de la UI sitúe la reserva del usuario
en el mismo mapa en milisegundos.

Uso::

    python -m src.visualization_2d
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # Backend sin GUI: imprescindible para escribir PNG en CI/headless.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from . import config
from .data_loader import load_and_prepare
from .preprocessing import build_preprocessor

logger = logging.getLogger(__name__)

OUTPUT_PNG_PATH = config.OUTPUTS_DIR / "decision_regions_pls.png"
# Pickle con todo lo necesario para situar una nueva reserva en el plano sin
# volver a entrenar nada. Tamaño esperado: ~5-15 MB (5 rejillas 300x300 float32
# + preprocesador + PLS + submuestra de test).
OUTPUT_ARTIFACTS_PATH = config.OUTPUTS_DIR / "decision_regions_pls.pkl"

# Número de puntos por eje de la rejilla: 300x300 = 90 000 puntos. Suficiente
# para curvas suaves y barato de calcular en 2D.
GRID_RESOLUTION: int = 300
# Submuestreo de puntos de test sobre los que pintamos la clase verdadera.
SCATTER_SAMPLE_SIZE: int = 1500

# Colormap para los puntos de scatter (clase real): azul = no cancela, rojo = cancela.
_POINT_CMAP = ListedColormap(["#0b3d66", "#b32400"])


def _build_2d_models() -> dict:
    """Devuelve los 5 modelos a reentrenar sobre el plano PLS.

    Se usan los hiperparámetros base de ``config`` (excepto la red neuronal,
    que se sustituye por un MLP de scikit-learn equivalente al de Keras pero
    sin dependencia de TensorFlow ni callbacks, ya que aquí solo aprende de 2
    variables y no necesita la maquinaria completa).
    """
    return {
        "Regresión logística": LogisticRegression(
            **config.LOGISTIC_REGRESSION_PARAMS
        ),
        "Árbol de decisión": DecisionTreeClassifier(**config.DECISION_TREE_PARAMS),
        "Random Forest": RandomForestClassifier(**config.RANDOM_FOREST_PARAMS),
        "XGBoost": XGBClassifier(**config.XGBOOST_PARAMS),
        "Red neuronal (MLP)": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=400,
            random_state=config.RANDOM_STATE,
        ),
    }


# ---------------------------------------------------------------------------
# Cálculo de los artefactos (la parte pesada: solo se hace una vez)
# ---------------------------------------------------------------------------
def _compute_artifacts() -> dict:
    """Entrena PLS + los 5 modelos 2D y devuelve todo lo necesario para pintar.

    Es la parte cara (≈30-45 s en CPU). Se ejecuta una sola vez desde el CLI y
    el resultado se persiste en disco (``OUTPUT_ARTIFACTS_PATH``) para que la
    UI no tenga que repetirlo.
    """
    logger.info("Cargando y preparando los datos…")
    X_train, X_test, y_train, y_test = load_and_prepare()

    logger.info("Ajustando el preprocesador (ColumnTransformer)…")
    preprocessor = build_preprocessor().fit(X_train)
    Z_train = preprocessor.transform(X_train)
    Z_test = preprocessor.transform(X_test)

    logger.info("Proyectando a 2 componentes con PLS supervisado…")
    pls = PLSRegression(n_components=2).fit(Z_train, y_train.values)
    C_train = pls.transform(Z_train)
    C_test = pls.transform(Z_test)

    # Orientamos el eje 1 hacia "más riesgo de cancelación" para que la lectura
    # sea siempre la misma (rojo a la derecha) independientemente del signo
    # arbitrario que devuelva PLS. Guardamos el flip como flag para poder
    # aplicarlo también a nuevas reservas.
    mean_pos = C_train[y_train.values == 1, 0].mean()
    mean_neg = C_train[y_train.values == 0, 0].mean()
    flip_first_axis: bool = mean_pos < mean_neg
    if flip_first_axis:
        C_train[:, 0] *= -1
        C_test[:, 0] *= -1

    logger.info("Reentrenando los 5 modelos sobre las 2 componentes PLS…")
    models = _build_2d_models()
    for name, model in models.items():
        logger.debug("  - %s", name)
        model.fit(C_train, y_train.values)

    # Rejilla del plano (recortada a los percentiles 1-99 para evitar que
    # outliers la estiren y oculten la zona interesante).
    x0, x1 = np.percentile(C_train[:, 0], [1, 99])
    y0, y1 = np.percentile(C_train[:, 1], [1, 99])
    xx, yy = np.meshgrid(
        np.linspace(x0, x1, GRID_RESOLUTION),
        np.linspace(y0, y1, GRID_RESOLUTION),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    logger.info("Precomputando ``predict_proba`` sobre la rejilla (300x300)…")
    proba_grids: dict[str, np.ndarray] = {}
    for name, model in models.items():
        proba_grids[name] = (
            model.predict_proba(grid)[:, 1].reshape(xx.shape).astype(np.float32)
        )

    # Submuestra fija de puntos de test para el scatter.
    rng = np.random.RandomState(config.RANDOM_STATE)
    idx = rng.choice(len(C_test), min(SCATTER_SAMPLE_SIZE, len(C_test)), replace=False)

    return {
        "preprocessor": preprocessor,
        "pls": pls,
        "flip_first_axis": flip_first_axis,
        "model_names": list(models.keys()),
        "xx": xx.astype(np.float32),
        "yy": yy.astype(np.float32),
        "proba_grids": proba_grids,
        "scatter_points": C_test[idx].astype(np.float32),
        "scatter_labels": y_test.values[idx].astype(np.int8),
        "x_range": (float(x0), float(x1)),
        "y_range": (float(y0), float(y1)),
    }


# ---------------------------------------------------------------------------
# Renderizado (parte ligera: se puede invocar en caliente desde la UI)
# ---------------------------------------------------------------------------
def _render_figure(
    artifacts: dict,
    sample_2d: tuple[float, float] | None = None,
    sample_label: str | None = None,
):
    """Pinta la figura 2x3 con las regiones de decisión y, opcionalmente, una
    nueva reserva marcada como una estrella.

    Parameters
    ----------
    artifacts:
        Diccionario devuelto por :func:`_compute_artifacts` o cargado de disco.
    sample_2d:
        Coordenadas ``(x, y)`` de la reserva a marcar en el plano PLS. Si es
        ``None``, no se dibuja ningún marcador (modo "figura base").
    sample_label:
        Etiqueta corta que se añadirá al título de la figura (p. ej. la
        probabilidad de cancelación predicha por el modelo principal).

    Returns
    -------
    matplotlib.figure.Figure
        La figura lista para guardar (``fig.savefig``) o pintar en Streamlit
        (``st.pyplot(fig)``).
    """
    xx = artifacts["xx"]
    yy = artifacts["yy"]
    proba_grids = artifacts["proba_grids"]
    scatter_points = artifacts["scatter_points"]
    scatter_labels = artifacts["scatter_labels"]
    x0, x1 = artifacts["x_range"]
    y0, y1 = artifacts["y_range"]
    model_names = artifacts["model_names"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    axes = axes.ravel()
    cf = None
    levels = np.linspace(0, 1, 21)
    for ax, name in zip(axes, model_names):
        Zp = proba_grids[name]
        cf = ax.contourf(xx, yy, Zp, levels=levels, cmap="RdBu_r", vmin=0, vmax=1)
        ax.contour(xx, yy, Zp, levels=[0.5], colors="k", linewidths=1.5)
        ax.scatter(
            scatter_points[:, 0],
            scatter_points[:, 1],
            c=scatter_labels,
            cmap=_POINT_CMAP,
            s=9,
            edgecolor="white",
            linewidth=0.2,
            alpha=0.75,
        )
        if sample_2d is not None:
            ax.scatter(
                [sample_2d[0]],
                [sample_2d[1]],
                marker="*",
                s=380,
                c="#ffd400",
                edgecolor="black",
                linewidth=1.6,
                zorder=5,
            )
        ax.set_title(name)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xticks([])
        ax.set_yticks([])

    # Sexta celda: clase real, como "referencia" visual.
    axes[5].scatter(
        scatter_points[:, 0],
        scatter_points[:, 1],
        c=scatter_labels,
        cmap=_POINT_CMAP,
        s=11,
        alpha=0.85,
    )
    if sample_2d is not None:
        axes[5].scatter(
            [sample_2d[0]],
            [sample_2d[1]],
            marker="*",
            s=380,
            c="#ffd400",
            edgecolor="black",
            linewidth=1.6,
            zorder=5,
        )
    axes[5].set_title("Referencia: clase REAL")
    axes[5].set_xlim(x0, x1)
    axes[5].set_ylim(y0, y1)
    axes[5].set_xticks([])
    axes[5].set_yticks([])

    for k, ax in enumerate(axes):
        if k // 3 == 1:
            ax.set_xlabel("Comp. PLS 1  (→ más riesgo de cancelación)")
        if k % 3 == 0:
            ax.set_ylabel("Comp. PLS 2")

    fig.colorbar(cf, ax=axes.tolist(), shrink=0.6, label="prob. de cancelación")
    if sample_label:
        suptitle = (
            "PLS supervisado: regiones de decisión + reserva del usuario "
            f"({sample_label})"
        )
    else:
        suptitle = (
            "PLS supervisado: regiones de decisión de cada modelo "
            "(las clases SÍ se separan)"
        )
    fig.suptitle(suptitle)
    return fig


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------
def generate_decision_regions_plot(
    png_path: Path = OUTPUT_PNG_PATH,
    artifacts_path: Path = OUTPUT_ARTIFACTS_PATH,
) -> Path:
    """Entrena los artefactos, los persiste y guarda la figura como PNG."""
    config.ensure_directories()
    artifacts = _compute_artifacts()

    logger.info("Guardando artefactos en %s", artifacts_path)
    joblib.dump(artifacts, artifacts_path, compress=3)

    logger.info("Pintando figura base…")
    fig = _render_figure(artifacts)
    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura guardada en %s", png_path)
    return png_path


def _load_artifacts(artifacts_path: Path = OUTPUT_ARTIFACTS_PATH) -> dict:
    """Carga el pickle con los artefactos del plano PLS.

    Se separa para que la UI pueda envolverlo con su propia caché
    (``streamlit.cache_resource``) sin acoplar este módulo a Streamlit.
    """
    if not artifacts_path.exists():
        raise FileNotFoundError(
            f"No se encontró {artifacts_path}. Ejecuta primero "
            "'python -m src.visualization_2d' para generarlo."
        )
    return joblib.load(artifacts_path)


def project_booking(
    booking_df: pd.DataFrame,
    artifacts: dict,
) -> tuple[float, float]:
    """Proyecta UNA reserva al plano PLS (mismo signo que la figura base).

    Parameters
    ----------
    booking_df:
        DataFrame de UNA fila con las MISMAS columnas que en entrenamiento
        (las 27 features en crudo). Se normaliza igual que en
        :func:`src.predict.prepare_for_inference`.
    artifacts:
        Diccionario devuelto por :func:`_load_artifacts`.

    Returns
    -------
    tuple[float, float]
        Coordenadas ``(x, y)`` de la reserva en el plano PLS.
    """
    from .predict import prepare_for_inference

    X_one = prepare_for_inference(booking_df)
    Z = artifacts["preprocessor"].transform(X_one)
    C = artifacts["pls"].transform(Z)
    if artifacts["flip_first_axis"]:
        C = C.copy()
        C[:, 0] *= -1
    return float(C[0, 0]), float(C[0, 1])


def plot_booking_on_2d(
    booking_df: pd.DataFrame,
    probability_canceled: float | None = None,
    artifacts: dict | None = None,
):
    """Renderiza el mapa PLS con UNA reserva marcada como estrella.

    Parameters
    ----------
    booking_df:
        Reserva (una fila) a situar en el plano.
    probability_canceled:
        Probabilidad de cancelación devuelta por el modelo principal, opcional.
        Si se pasa, aparece en el título de la figura.
    artifacts:
        Artefactos ya cargados (para evitar leer el pickle en cada llamada). Si
        es ``None``, se cargan desde disco.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if artifacts is None:
        artifacts = _load_artifacts()
    sample = project_booking(booking_df, artifacts)
    label = None
    if probability_canceled is not None:
        label = f"prob. de cancelación = {probability_canceled * 100:.1f} %"
    return _render_figure(artifacts, sample_2d=sample, sample_label=label)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    generate_decision_regions_plot()


if __name__ == "__main__":
    main()
