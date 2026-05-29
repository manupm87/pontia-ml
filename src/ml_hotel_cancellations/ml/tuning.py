"""Optimización de hiperparámetros (bonus técnico).

Busca, mediante **validación cruzada**, la mejor combinación de hiperparámetros
para los modelos clásicos del proyecto. Demuestra las dos técnicas habituales:

- **GridSearchCV** (búsqueda exhaustiva) para espacios pequeños — regresión
  logística y árbol de decisión.
- **RandomizedSearchCV** (muestreo aleatorio de combinaciones) para los espacios
  grandes — Random Forest y XGBoost, donde una búsqueda exhaustiva sería
  inviable. (`RandomizedSearchCV` no aparece en `recursos/`, que usa
  `GridSearchCV`; ver el mapeo de herramientas en `docs/informe_final.md` §4.5.)

Se optimiza ROC-AUC (la métrica principal del proyecto). Para cada modelo se
compara la puntuación de CV de los hiperparámetros **por defecto** del proyecto
con la de los **mejores** encontrados, de modo que se ve si la búsqueda aporta.

La red neuronal (Keras) se deja fuera: su ajuste se hace con *early stopping* y
una búsqueda con CV sería desproporcionadamente costosa.

Uso::

    python -m src.tuning
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


# Fuente única de verdad en `config` (compartida con train/balancing).
_MODEL_FAMILY: dict[str, str] = config.MODEL_FAMILY


def _pipeline(estimator) -> Pipeline:
    """Envuelve un estimador con el preprocesador del proyecto en un Pipeline."""
    return make_pipeline(estimator)


def _strip_prefix(params: dict) -> dict:
    """Quita el prefijo ``model__`` de las claves (para reusarlas al reentrenar)."""
    return {k.replace("model__", "", 1): v for k, v in params.items()}


def load_best_params(path=config.BEST_PARAMS_PATH) -> dict[str, dict]:
    """Carga los mejores hiperparámetros persistidos (``{}`` si no existen).

    Lo usa el pipeline por defecto para entrenar con los hiperparámetros
    optimizados una vez que se han buscado (``--tune`` o ``python -m src.tuning``).
    """
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("No se pudieron leer los hiperparámetros de %s (%s).", path, exc)
        return {}


class HyperparameterTuner:
    """Optimiza los hiperparámetros de los modelos clásicos por CV.

    Parameters
    ----------
    cv:
        Número de particiones de validación cruzada.
    scoring:
        Métrica a optimizar (por defecto, ``roc_auc``).
    n_iter:
        Nº de combinaciones que prueba ``RandomizedSearchCV``.
    random_state:
        Semilla para reproducibilidad.
    """

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
        """Devuelve, por modelo: (estimador base, grid, tipo búsqueda, params por defecto).

        Los estimadores se construyen con la fábrica única
        (:func:`src.model_factory.build_classic_estimators`), leyendo sus
        hiperparámetros de ``config`` (antes ``max_iter`` de la regresión
        logística estaba hardcodeado y divergía). Random Forest y XGBoost llevan
        ``n_jobs=1`` para no competir con el paralelismo de la propia búsqueda
        (que usa ``n_jobs=-1``).
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
        """Ejecuta la búsqueda para cada modelo y guarda los resultados.

        Returns
        -------
        dict[str, dict]
            ``nombre -> {best_params, cv_tuned, cv_default, search, n_combos, segundos}``
        """
        # MLflow: si nos invocan desde `python -m src.tuning` (raíz), abrimos
        # un parent run "tuning_hyperparameters". Si nos llaman desde
        # `python -m src.train --tune`, ``start_run`` detecta el run activo
        # y este se convierte en child del run de entrenamiento. En cualquiera
        # de los dos casos, los runs por modelo cuelgan de aquí.
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
        """Implementación del bucle de búsqueda (lo separamos del setup de MLflow)."""
        self.results_ = {}
        self.best_params_ = {}
        for nombre, (estimador, grid, kind, default_params) in self._catalog().items():
            self._tune_one(nombre, estimador, grid, kind, default_params, X_train, y_train)

        # Persistimos los resultados ANTES de subirlos como artefactos para
        # asegurar que los ficheros existen. Esta es la ÚNICA escritura de
        # artefactos por ejecución (los callers ya no la repiten).
        self.save_results()
        self.save_best_params()
        tracking.log_artifact(config.TUNING_RESULTS_PATH)
        tracking.log_artifact(config.BEST_PARAMS_PATH)
        return self.results_

    def _build_search(self, pipe, grid, kind: str, njobs: int):
        """Crea el buscador (Grid/Randomized) y devuelve ``(search, tipo)``."""
        if kind == "grid":
            # GridSearchCV: búsqueda exhaustiva (la herramienta vista en `recursos/`).
            search = GridSearchCV(
                pipe, grid, scoring=self.scoring, cv=self.cv, n_jobs=njobs
            )
            return search, "GridSearchCV"
        # RandomizedSearchCV: muestreo aleatorio para espacios grandes (no se ve
        # en `recursos/`; equivale a un GridSearchCV pero sin recorrerlo entero).
        # Ver el mapeo de herramientas en `docs/informe_final.md` §4.5.
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

    def _baseline_cv(self, estimador, default_params: dict, njobs: int, X_train, y_train) -> float:
        """CV con los hiperparámetros por defecto del proyecto (baseline)."""
        extra = {"n_jobs": 1} if "n_jobs" in default_params else {}
        base = estimador.__class__(**{**default_params, **extra})
        return cross_val_score(
            _pipeline(base), X_train, y_train, scoring=self.scoring, cv=self.cv, n_jobs=njobs
        ).mean()

    def _log_model_run(self, nombre: str, tipo: str, best: dict, cv_default: float,
                       search, elapsed: float) -> None:
        """MLflow: un child run por modelo con sus params + métricas de CV."""
        with tracking.start_run(run_name=nombre, nested=True):
            tracking.set_tags(
                {
                    "model_family": _MODEL_FAMILY.get(nombre, "other"),
                    "search": tipo,
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

    def _tune_one(self, nombre: str, estimador, grid, kind: str, default_params: dict,
                  X_train, y_train) -> None:
        """Optimiza un modelo: búsqueda + baseline + registro de resultados/MLflow."""
        pipe = _pipeline(estimador)
        njobs = -1  # la búsqueda CV paraleliza sobre todos los núcleos (CPU)
        search, tipo = self._build_search(pipe, grid, kind, njobs)

        logger.info("Optimizando %s con %s...", nombre, tipo)
        inicio = time.perf_counter()
        search.fit(X_train, y_train)

        cv_default = self._baseline_cv(estimador, default_params, njobs, X_train, y_train)
        elapsed = time.perf_counter() - inicio

        best = _strip_prefix(search.best_params_)
        self.best_params_[nombre] = best
        self.results_[nombre] = {
            "search": tipo,
            "n_combos": len(search.cv_results_["params"]),
            "cv_default": cv_default,
            "cv_tuned": search.best_score_,
            "mejora": search.best_score_ - cv_default,
            "best_params": best,
            "segundos": elapsed,
        }
        logger.info(
            "  -> %s: CV %s  %.4f (default) -> %.4f (tuned)  [%.0fs]",
            nombre,
            self.scoring,
            cv_default,
            search.best_score_,
            elapsed,
        )

        self._log_model_run(nombre, tipo, best, cv_default, search, elapsed)

    def results_table(self) -> pd.DataFrame:
        """Devuelve los resultados como tabla ordenada por CV ``tuned``."""
        filas = {
            nombre: {
                "busqueda": r["search"],
                "combinaciones": r["n_combos"],
                f"cv_{self.scoring}_default": round(r["cv_default"], 4),
                f"cv_{self.scoring}_tuned": round(r["cv_tuned"], 4),
                "mejora": round(r["mejora"], 4),
                "segundos": round(r["segundos"], 1),
            }
            for nombre, r in self.results_.items()
        }
        return pd.DataFrame(filas).T.sort_values(
            f"cv_{self.scoring}_tuned", ascending=False
        )

    def save_results(self, path=config.TUNING_RESULTS_PATH) -> None:
        """Escribe un informe Markdown con la tabla y los mejores hiperparámetros."""
        tabla = self.results_table()
        # Los valores de la tabla ya vienen redondeados (str(v) los respeta).
        lineas = [
            "# Optimización de hiperparámetros\n",
            f"Métrica optimizada: **{self.scoring}** · CV de **{self.cv}** particiones.\n",
            df_to_markdown(tabla, index_label="Modelo", float_fmt="{}"),
            "\n## Mejores hiperparámetros por modelo\n",
        ]
        for nombre, params in self.best_params_.items():
            lineas.append(f"- **{nombre}**: `{params}`")
        path.write_text("\n".join(lineas) + "\n", encoding="utf-8")
        logger.info("Resultados de la búsqueda guardados en: %s", path)

    def save_best_params(self, path=config.BEST_PARAMS_PATH) -> None:
        """Persiste los mejores hiperparámetros en JSON para usarlos por defecto."""
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
    # `tune()` ya persiste los artefactos (resultados + mejores params) una sola
    # vez; no los reescribimos aquí.
    tuner.tune(X_train, y_train)
    print("\n" + "=" * 70)
    print("RESULTADO DE LA OPTIMIZACIÓN DE HIPERPARÁMETROS")
    print("=" * 70)
    print(tuner.results_table().to_string())
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
