"""Interpretabilidad del modelo: ¿por qué predice lo que predice? (bonus).

Ofrece dos técnicas complementarias sobre el ``Pipeline`` ganador:
- **SHAP** (TreeExplainer, exacto para árboles): contribución de cada variable,
  global (beeswarm/bar) y local (waterfall). Opera sobre la matriz ya preprocesada.
- **Importancia por permutación**: agnóstica al modelo, mide la caída de ROC-AUC
  al barajar cada variable.

Uso: ``python -m ml_hotel_cancellations.utils.interpretability [--sample N] [--no-permutation]``
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

# ~2000 reservas dan una imagen global estable sin el coste de calcular SHAP sobre todo el conjunto.
DEFAULT_SHAP_SAMPLE: int = 2000


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------
def _transform_features(pipeline: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Aplica el preprocesador del ``Pipeline`` y devuelve la matriz + nombres.

    SHAP necesita los datos en el espacio del estimador final (numéricas escaladas
    y categóricas one-hot), no las columnas en crudo.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    X_trans = preprocessor.transform(X)
    # SHAP trabaja mejor con arrays densos; convertimos si el transformador devuelve sparse.
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
    """Construye un ``shap.TreeExplainer`` y devuelve ``(explanation, X_trans, feature_names)``."""
    import shap  # importación perezosa: dependencia del bonus.

    _patch_shap_xgboost_base_score()

    estimator = pipeline.named_steps["model"]
    X_trans, feature_names = _transform_features(pipeline, X_sample)

    explainer = shap.TreeExplainer(estimator)
    explanation = explainer(X_trans)
    explanation.feature_names = feature_names  # para que los gráficos etiqueten las columnas.
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
    """Genera las explicaciones SHAP globales (beeswarm + bar) y las guarda como PNG.

    Devuelve las rutas con claves ``beeswarm`` y ``bar``.
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
    """Explica la predicción de la reserva ``X.iloc[idx]`` con un gráfico *waterfall*.

    El waterfall parte del valor base (predicción media) y suma/resta la
    contribución de cada variable hasta la predicción de ESTA reserva.
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

    Pensada para la UI: pinta en caliente la reserva enviada por el usuario.
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
    """Devuelve ``{"alta_prob": idx, "baja_prob": idx}``: las reservas de mayor y menor probabilidad estimada."""
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

    Se calcula sobre el ``Pipeline`` completo y las columnas EN CRUDO, así la
    importancia se atribuye a las variables originales, no a las columnas one-hot.
    Devuelve ``(tabla ordenada, ruta del PNG)``.
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

    table = (
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

    top = table.head(top_n).iloc[::-1]  # invertir para barh de mayor a menor
    path = output_dir / "permutation_importance.png"
    fig = plt.figure(figsize=(9, 0.42 * len(top) + 1.5))
    plt.barh(top["variable"], top["importancia_media"], xerr=top["importancia_std"], color="#2c7fb8")
    plt.xlabel(f"Caída media en {scoring} al barajar la variable")
    plt.ylabel("Variable")
    plt.title(f"Importancia por permutación (top {len(top)})")
    save_figure(fig, path)

    return table, path


# ---------------------------------------------------------------------------
# CLI / orquestador
# ---------------------------------------------------------------------------
def main() -> None:
    """Carga el mejor modelo y los datos, y regenera todos los gráficos."""
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

    # 2) Explicaciones locales: una reserva cancelada y otra no cancelada (ambos extremos).
    examples = find_examples(pipeline, X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    cases = [
        ("alta_prob", "shap_waterfall_ejemplo1.png", "alta"),
        ("baja_prob", "shap_waterfall_ejemplo2.png", "baja"),
    ]
    for key, filename, label in cases:
        idx = examples[key]
        explain_local(
            pipeline,
            X_test,
            idx,
            filename=filename,
            title=f"Reserva con {label} probabilidad de cancelación (p={proba[idx]:.3f})",
        )

    # 3) Importancia por permutación (complemento agnóstico al modelo).
    if not args.no_permutation:
        permutation_importance_report(pipeline, X_test, y_test)

    logger.info("Interpretabilidad completada. Gráficos en %s", config.OUTPUTS_DIR)


if __name__ == "__main__":
    main()
