"""Optimización de hiperparámetros por validación cruzada (bonus técnico).

Usa GridSearchCV (espacios pequeños) y RandomizedSearchCV (grandes); optimiza
ROC-AUC y compara CV por defecto vs. tuneada. La red Keras queda fuera (se ajusta
con early stopping). Mapeo de herramientas en docs/informe_final.md §4.5.

Uso::

    python -m ml_hotel_cancellations.ml.tuning
"""

from __future__ import annotations

import json
import logging
import time

import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.pipeline import Pipeline

from ml_hotel_cancellations import config
from ml_hotel_cancellations.utils import tracking
from .preprocessing import make_pipeline
from ml_hotel_cancellations.utils.reporting import df_to_markdown

logger = logging.getLogger(__name__)


# Fuente única en `config` (compartida con train/balancing).
_MODEL_FAMILY: dict[str, str] = config.MODEL_FAMILY


def _pipeline(estimator) -> Pipeline:
    """Envuelve un estimador con el preprocesador del proyecto en un Pipeline."""
    return make_pipeline(estimator)


def _strip_prefix(params: dict) -> dict:
    """Quita el prefijo ``model__`` de las claves (para reusarlas al reentrenar)."""
    return {k.replace("model__", "", 1): v for k, v in params.items()}


def load_best_params(path=config.BEST_PARAMS_PATH) -> dict[str, dict]:
    """Carga los mejores hiperparámetros persistidos (``{}`` si no existen)."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("No se pudieron leer los hiperparámetros de %s (%s).", path, exc)
        return {}


class HyperparameterTuner:
    """Optimiza los hiperparámetros de los modelos clásicos por CV."""

    def __init__(
        self,
        cv: int = config.TUNING_CV_FOLDS,
        scoring: str = config.TUNING_SCORING,
        n_iter: int = config.TUNING_N_ITER,
        random_state: int = config.RANDOM_STATE,
    ):
        self.cv = cv
        self.scoring = scoring
        self.n_iter = n_iter
        self.random_state = random_state
        self.results_: dict[str, dict] = {}
        self.best_params_: dict[str, dict] = {}

    def _catalog(self) -> dict[str, tuple]:
        """Por modelo: (estimador base, grid, tipo de búsqueda, params por defecto).

        Random Forest y XGBoost van con ``n_jobs=1`` para no competir con el
        paralelismo de la propia búsqueda.
        """
        from .model_factory import build_classic_estimators

        estimators = build_classic_estimators(
            overrides={"Logistic Regression": {"random_state": self.random_state},
                       "Decision Tree": {"random_state": self.random_state},
                       "Random Forest": {"random_state": self.random_state},
                       "XGBoost": {"random_state": self.random_state}},
            n_jobs={"Random Forest": 1, "XGBoost": 1},
        )
        return {
            "Logistic Regression": (
                estimators["Logistic Regression"],
                config.LOGISTIC_REGRESSION_GRID,
                "grid",
                config.LOGISTIC_REGRESSION_PARAMS,
            ),
            "Decision Tree": (
                estimators["Decision Tree"],
                config.DECISION_TREE_GRID,
                "grid",
                config.DECISION_TREE_PARAMS,
            ),
            "Random Forest": (
                estimators["Random Forest"],
                config.RANDOM_FOREST_GRID,
                "random",
                config.RANDOM_FOREST_PARAMS,
            ),
            "XGBoost": (
                estimators["XGBoost"],
                config.XGBOOST_GRID,
                "random",
                config.XGBOOST_PARAMS,
            ),
        }

    def tune(self, X_train, y_train) -> dict[str, dict]:
        """Ejecuta la búsqueda para cada modelo y guarda los resultados."""
        # MLflow: parent run propio si se invoca directo; child del run de train
        # si se llama desde `train --tune`. Los runs por modelo cuelgan de aquí.
        with tracking.start_run(run_name="tuning_hyperparameters"):
            tracking.set_tags(
                {
                    "phase": "tuning",
                    "cv_folds": self.cv,
                    "scoring": self.scoring,
                    "n_iter": self.n_iter,
                }
            )
            return self._tune_inner(X_train, y_train)

    def _tune_inner(self, X_train, y_train) -> dict[str, dict]:
        """Bucle de búsqueda (separado del setup de MLflow)."""
        self.results_ = {}
        self.best_params_ = {}
        for name, (estimator, grid, kind, default_params) in self._catalog().items():
            self._tune_one(name, estimator, grid, kind, default_params, X_train, y_train)

        # Persistir antes de subir como artefactos (única escritura por ejecución).
        self.save_results()
        self.save_best_params()
        tracking.log_artifact(config.TUNING_RESULTS_PATH)
        tracking.log_artifact(config.BEST_PARAMS_PATH)
        return self.results_

    def _build_search(self, pipe, grid, kind: str, njobs: int):
        """Crea el buscador (Grid/Randomized) y devuelve ``(search, tipo)``."""
        if kind == "grid":
            search = GridSearchCV(  # búsqueda exhaustiva
                pipe, grid, scoring=self.scoring, cv=self.cv, n_jobs=njobs
            )
            return search, "GridSearchCV"
        # RandomizedSearchCV: muestreo aleatorio para espacios grandes (ver §4.5).
        search = RandomizedSearchCV(
            pipe,
            grid,
            n_iter=self.n_iter,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=njobs,
            random_state=self.random_state,
        )
        return search, "RandomizedSearchCV"

    def _baseline_cv(self, estimator, default_params: dict, njobs: int, X_train, y_train) -> float:
        """CV con los hiperparámetros por defecto del proyecto (baseline)."""
        extra = {"n_jobs": 1} if "n_jobs" in default_params else {}
        base = estimator.__class__(**{**default_params, **extra})
        return cross_val_score(
            _pipeline(base), X_train, y_train, scoring=self.scoring, cv=self.cv, n_jobs=njobs
        ).mean()

    def _log_model_run(self, name: str, search_type: str, best: dict, cv_default: float,
                       search, elapsed: float) -> None:
        """MLflow: un child run por modelo con sus params + métricas de CV."""
        with tracking.start_run(run_name=name, nested=True):
            tracking.set_tags(
                {
                    "model_family": _MODEL_FAMILY.get(name, "other"),
                    "search": search_type,
                    "phase": "tuning",
                }
            )
            tracking.log_params(best)
            tracking.log_metrics(
                {
                    f"cv_{self.scoring}_default": float(cv_default),
                    f"cv_{self.scoring}_tuned": float(search.best_score_),
                    f"cv_{self.scoring}_improvement": float(
                        search.best_score_ - cv_default
                    ),
                    "n_combos_tried": int(len(search.cv_results_["params"])),
                    "elapsed_s": float(elapsed),
                }
            )

    def _tune_one(self, name: str, estimator, grid, kind: str, default_params: dict,
                  X_train, y_train) -> None:
        """Optimiza un modelo: búsqueda + baseline + registro de resultados/MLflow."""
        pipe = _pipeline(estimator)
        njobs = -1  # la búsqueda CV paraleliza sobre todos los núcleos (CPU)
        search, search_type = self._build_search(pipe, grid, kind, njobs)

        logger.info("Optimizando %s con %s...", name, search_type)
        start = time.perf_counter()
        search.fit(X_train, y_train)

        cv_default = self._baseline_cv(estimator, default_params, njobs, X_train, y_train)
        elapsed = time.perf_counter() - start

        best = _strip_prefix(search.best_params_)
        self.best_params_[name] = best
        self.results_[name] = {
            "search": search_type,
            "n_combos": len(search.cv_results_["params"]),
            "cv_default": cv_default,
            "cv_tuned": search.best_score_,
            "mejora": search.best_score_ - cv_default,
            "best_params": best,
            "segundos": elapsed,
        }
        logger.info(
            "  -> %s: CV %s  %.4f (default) -> %.4f (tuned)  [%.0fs]",
            name,
            self.scoring,
            cv_default,
            search.best_score_,
            elapsed,
        )

        self._log_model_run(name, search_type, best, cv_default, search, elapsed)

    def results_table(self) -> pd.DataFrame:
        """Devuelve los resultados como tabla ordenada por CV ``tuned``."""
        rows = {
            name: {
                "busqueda": r["search"],
                "combinaciones": r["n_combos"],
                f"cv_{self.scoring}_default": round(r["cv_default"], 4),
                f"cv_{self.scoring}_tuned": round(r["cv_tuned"], 4),
                "mejora": round(r["mejora"], 4),
                "segundos": round(r["segundos"], 1),
            }
            for name, r in self.results_.items()
        }
        return pd.DataFrame(rows).T.sort_values(
            f"cv_{self.scoring}_tuned", ascending=False
        )

    def save_results(self, path=config.TUNING_RESULTS_PATH) -> None:
        """Escribe un informe Markdown con la tabla y los mejores hiperparámetros."""
        table = self.results_table()
        # Los valores de la tabla ya vienen redondeados (str(v) los respeta).
        lines = [
            "# Optimización de hiperparámetros\n",
            f"Métrica optimizada: **{self.scoring}** · CV de **{self.cv}** particiones.\n",
            df_to_markdown(table, index_label="Modelo", float_fmt="{}"),
            "\n## Mejores hiperparámetros por modelo\n",
        ]
        for name, params in self.best_params_.items():
            lines.append(f"- **{name}**: `{params}`")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Resultados de la búsqueda guardados en: %s", path)

    def save_best_params(self, path=config.BEST_PARAMS_PATH) -> None:
        """Persiste los mejores hiperparámetros en JSON (se usan por defecto)."""
        path.write_text(json.dumps(self.best_params_, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        logger.info("Mejores hiperparámetros guardados en: %s", path)


def main() -> None:
    """Punto de entrada CLI: tuneable de forma independiente al pipeline."""
    from .data_loader import load_and_prepare

    config.configure_logging()
    config.ensure_directories()
    logger.info("=== Optimización de hiperparámetros (CV=%d, métrica=%s) ===",
                config.TUNING_CV_FOLDS, config.TUNING_SCORING)
    X_train, _, y_train, _ = load_and_prepare()
    tuner = HyperparameterTuner()
    # `tune()` ya persiste los artefactos; no los reescribimos aquí.
    tuner.tune(X_train, y_train)
    print("\n" + "=" * 70)
    print("RESULTADO DE LA OPTIMIZACIÓN DE HIPERPARÁMETROS")
    print("=" * 70)
    print(tuner.results_table().to_string())
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
