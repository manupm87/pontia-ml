"""Optimización de hiperparámetros por validación cruzada (bonus, opcional).

Para cada modelo clásico busca los mejores hiperparámetros con GridSearchCV
(espacios pequeños) o RandomizedSearchCV (grandes), optimizando ROC-AUC. La red
Keras queda fuera (se ajusta con early stopping). Guarda un informe Markdown y un
JSON con los mejores params, que `train` usa por defecto si existe.

Uso::

    python -m ml_hotel_cancellations.ml.tuning
"""

from __future__ import annotations

import json
import logging
import time

import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from ml_hotel_cancellations import config
from ml_hotel_cancellations.utils import tracking
from ml_hotel_cancellations.utils.reporting import df_to_markdown
from .preprocessing import make_pipeline

logger = logging.getLogger(__name__)

# Fuente única en `config` (compartida con train).
_MODEL_FAMILY: dict[str, str] = config.MODEL_FAMILY


def load_best_params(path=config.BEST_PARAMS_PATH) -> dict[str, dict]:
    """Carga los mejores hiperparámetros persistidos (``{}`` si no existen)."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("No se pudieron leer los hiperparámetros de %s (%s).", path, exc)
        return {}


def _search_space() -> dict[str, tuple]:
    """Por modelo: ``(grid, tipo_de_busqueda)``. Random Forest y XGBoost con
    ``n_jobs=1`` para no competir con el paralelismo de la propia búsqueda.
    """
    from .models import build_classic_estimators

    estimators = build_classic_estimators(n_jobs={"Random Forest": 1, "XGBoost": 1})
    return {
        "Logistic Regression": (estimators["Logistic Regression"], config.LOGISTIC_REGRESSION_GRID, "grid"),
        "Decision Tree": (estimators["Decision Tree"], config.DECISION_TREE_GRID, "grid"),
        "Random Forest": (estimators["Random Forest"], config.RANDOM_FOREST_GRID, "random"),
        "XGBoost": (estimators["XGBoost"], config.XGBOOST_GRID, "random"),
    }


def _build_search(estimator, grid: dict, kind: str):
    """Crea el buscador (Grid o Randomized) sobre el Pipeline preprocesado."""
    pipe = make_pipeline(estimator)
    if kind == "grid":
        search = GridSearchCV(pipe, grid, scoring=config.TUNING_SCORING, cv=config.TUNING_CV_FOLDS, n_jobs=-1)
    else:
        search = RandomizedSearchCV(
            pipe, grid, n_iter=config.TUNING_N_ITER, scoring=config.TUNING_SCORING,
            cv=config.TUNING_CV_FOLDS, n_jobs=-1, random_state=config.RANDOM_STATE,
        )
    return search


def tune(X_train, y_train) -> tuple[dict, pd.DataFrame]:
    """Optimiza cada modelo clásico y devuelve ``(mejores_params, tabla_resultados)``.

    Persiste el informe Markdown y el JSON de mejores params (los usa `train`).
    """
    best_params: dict[str, dict] = {}
    rows = {}
    with tracking.start_run(run_name="tuning_hyperparameters"):
        for name, (estimator, grid, kind) in _search_space().items():
            logger.info("Optimizando %s (%s)...", name, kind)
            start = time.perf_counter()
            search = _build_search(estimator, grid, kind)
            search.fit(X_train, y_train)
            elapsed = time.perf_counter() - start

            # Quitamos el prefijo "model__" para poder reusar los params al reentrenar.
            best = {k.replace("model__", "", 1): v for k, v in search.best_params_.items()}
            best_params[name] = best
            rows[name] = {
                "busqueda": "Grid" if kind == "grid" else "Randomized",
                "combinaciones": len(search.cv_results_["params"]),
                f"cv_{config.TUNING_SCORING}": round(search.best_score_, 4),
                "segundos": round(elapsed, 1),
            }
            logger.info("  -> %s: CV %s=%.4f [%.0fs]", name, config.TUNING_SCORING, search.best_score_, elapsed)
            _log_model_run(name, best, search.best_score_)

    table = pd.DataFrame(rows).T.sort_values(f"cv_{config.TUNING_SCORING}", ascending=False)
    _save_results(table, best_params)
    return best_params, table


def _log_model_run(name: str, best: dict, cv_score: float) -> None:
    """MLflow: un child run por modelo con sus params + score de CV (no-op si off)."""
    with tracking.start_run(run_name=name, nested=True):
        tracking.set_tags({"model_family": _MODEL_FAMILY.get(name, "other"), "phase": "tuning"})
        tracking.log_params(best)
        tracking.log_metrics({f"cv_{config.TUNING_SCORING}": float(cv_score)})


def _save_results(table: pd.DataFrame, best_params: dict) -> None:
    """Escribe el informe Markdown y el JSON con los mejores hiperparámetros."""
    lines = [
        "# Optimización de hiperparámetros\n",
        f"Métrica optimizada: **{config.TUNING_SCORING}** · CV de **{config.TUNING_CV_FOLDS}** particiones.\n",
        df_to_markdown(table, index_label="Modelo", float_fmt="{}"),
        "\n## Mejores hiperparámetros por modelo\n",
        *[f"- **{name}**: `{params}`" for name, params in best_params.items()],
    ]
    config.TUNING_RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    config.BEST_PARAMS_PATH.write_text(json.dumps(best_params, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tracking.log_artifact(config.TUNING_RESULTS_PATH)
    tracking.log_artifact(config.BEST_PARAMS_PATH)
    logger.info("Resultados de la búsqueda guardados en: %s", config.TUNING_RESULTS_PATH)


def main() -> None:
    """Punto de entrada CLI: optimiza de forma independiente al pipeline."""
    from .data_loader import load_and_prepare

    config.configure_logging()
    config.ensure_directories()
    logger.info("=== Optimización de hiperparámetros (CV=%d, métrica=%s) ===",
                config.TUNING_CV_FOLDS, config.TUNING_SCORING)
    X_train, _, y_train, _ = load_and_prepare()
    _, table = tune(X_train, y_train)
    print("\n" + "=" * 70)
    print("RESULTADO DE LA OPTIMIZACIÓN DE HIPERPARÁMETROS")
    print("=" * 70)
    print(table.to_string())
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
