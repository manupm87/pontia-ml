"""Visualización 2D de los modelos: proyección PLS supervisada + regiones de decisión.

Proyecta los datos (144 variables) a 2 componentes PLS (correlacionadas con el
target, a diferencia de PCA) para poder dibujar las fronteras de decisión, y
persiste los artefactos para situar nuevas reservas en el plano sin recomputar.

Uso: ``python -m ml_hotel_cancellations.utils.visualization_2d``
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib

from ml_hotel_cancellations import config

config.use_agg_backend()  # Backend sin GUI: imprescindible para escribir PNG en CI/headless.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPClassifier

from ml_hotel_cancellations.ml.data_loader import load_and_prepare
from ml_hotel_cancellations.ml.preprocessing import build_transform_pipeline

logger = logging.getLogger(__name__)

OUTPUT_PNG_PATH = config.OUTPUTS_DIR / "decision_regions_pls.png"
# Pickle con todo lo necesario para situar una nueva reserva sin reentrenar.
OUTPUT_ARTIFACTS_PATH = config.OUTPUTS_DIR / "decision_regions_pls.pkl"

GRID_RESOLUTION: int = 300
SCATTER_SAMPLE_SIZE: int = 1500

# azul = no cancela, rojo = cancela.
_POINT_CMAP = ListedColormap(["#0b3d66", "#b32400"])


def _build_2d_models() -> dict:
    """Devuelve los 5 modelos a reentrenar sobre el plano PLS.

    Para la red neuronal se usa aquí un ``MLPClassifier`` de sklearn **ligero** como
    sustituto rápido (la red de producción es Keras): reentrenar Keras por cada píxel de
    la rejilla sería lento y este gráfico es solo ilustrativo (mismo criterio que el
    notebook ``05``). Aquí solo aprende de 2 componentes.
    """
    from ml_hotel_cancellations.ml.models import build_classic_estimators

    classic_models = build_classic_estimators()
    return {
        "Regresión logística": classic_models["Logistic Regression"],
        "Árbol de decisión": classic_models["Decision Tree"],
        "Random Forest": classic_models["Random Forest"],
        "XGBoost": classic_models["XGBoost"],
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

    Parte cara (~30-45 s); se ejecuta una vez desde el CLI y se persiste a disco.
    """
    logger.info("Cargando y preparando los datos…")
    X_train, X_test, y_train, y_test = load_and_prepare()

    logger.info("Ajustando el preprocesador (features + reducción + ColumnTransformer)…")
    # fit-on-train CON target: la reducción de cardinalidad es supervisada.
    preprocessor = build_transform_pipeline().fit(X_train, y_train)
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
def _draw_sample_star(ax, sample_2d: tuple[float, float]) -> None:
    """Dibuja la reserva del usuario como una estrella amarilla sobre ``ax``."""
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


def _style_axis(ax, title: str, x_range: tuple[float, float], y_range: tuple[float, float]) -> None:
    """Aplica título, límites y ocultado de ticks comunes a cada subgráfica."""
    ax.set_title(title)
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_xticks([])
    ax.set_yticks([])


def _render_figure(
    artifacts: dict,
    sample_2d: tuple[float, float] | None = None,
    sample_label: str | None = None,
) -> Figure:
    """Pinta la figura 2x3 con las regiones de decisión.

    Con ``sample_2d`` marca una reserva como estrella; sin él, genera la figura base.
    """
    xx = artifacts["xx"]
    yy = artifacts["yy"]
    proba_grids = artifacts["proba_grids"]
    scatter_points = artifacts["scatter_points"]
    scatter_labels = artifacts["scatter_labels"]
    x0, x1 = artifacts["x_range"]
    y0, y1 = artifacts["y_range"]
    model_names = artifacts["model_names"]
    x_range, y_range = (x0, x1), (y0, y1)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    axes = axes.ravel()
    cf = None
    levels = np.linspace(0, 1, 21)
    for ax, name in zip(axes, model_names):
        proba_grid = proba_grids[name]
        cf = ax.contourf(xx, yy, proba_grid, levels=levels, cmap="RdBu_r", vmin=0, vmax=1)
        ax.contour(xx, yy, proba_grid, levels=[0.5], colors="k", linewidths=1.5)
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
            _draw_sample_star(ax, sample_2d)
        _style_axis(ax, name, x_range, y_range)

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
        _draw_sample_star(axes[5], sample_2d)
    _style_axis(axes[5], "Referencia: clase REAL", x_range, y_range)

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
    # Guardamos directo (no `save_figure`): su `tight_layout` choca con `constrained_layout`.
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura guardada en %s", png_path)
    return png_path


def load_artifacts(artifacts_path: Path = OUTPUT_ARTIFACTS_PATH) -> dict:
    """Carga el pickle con los artefactos del plano PLS (la consume la UI)."""
    if not artifacts_path.exists():
        raise FileNotFoundError(
            f"No se encontró {artifacts_path}. Ejecuta primero "
            "'python -m ml_hotel_cancellations.utils.visualization_2d' para generarlo."
        )
    return joblib.load(artifacts_path)


# Alias retrocompatible: el código interno seguía llamando a `_load_artifacts`.
_load_artifacts = load_artifacts


def project_booking(
    booking_df: pd.DataFrame,
    artifacts: dict,
) -> tuple[float, float]:
    """Proyecta UNA reserva (1 fila, 26 features crudas) al plano PLS, con el mismo signo que la figura base."""
    from ml_hotel_cancellations.ml.predict import prepare_for_inference

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

    Si ``artifacts`` es ``None`` se cargan de disco; ``probability_canceled``, si se pasa, va en el título.
    """
    if artifacts is None:
        artifacts = _load_artifacts()
    sample = project_booking(booking_df, artifacts)
    label = None
    if probability_canceled is not None:
        label = f"prob. de cancelación = {probability_canceled * 100:.1f} %"
    return _render_figure(artifacts, sample_2d=sample, sample_label=label)


def main() -> None:
    config.configure_logging()
    generate_decision_regions_plot()


if __name__ == "__main__":
    main()
