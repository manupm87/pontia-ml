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

from . import config, gpu
from .preprocessing import build_preprocessor

logger = logging.getLogger(__name__)


def _pipeline(estimator) -> Pipeline:
    """Envuelve un estimador con el preprocesador del proyecto en un Pipeline."""
    return Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])


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

        Los estimadores base llevan ``n_jobs=1`` para no competir con el
        paralelismo de la propia búsqueda (que usa ``n_jobs=-1``).
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from xgboost import XGBClassifier

        from . import gpu

        rs = self.random_state
        return {
            "Logistic Regression": (
                LogisticRegression(max_iter=1000, random_state=rs),
                config.LOGISTIC_REGRESSION_GRID,
                "grid",
                config.LOGISTIC_REGRESSION_PARAMS,
            ),
            "Decision Tree": (
                DecisionTreeClassifier(random_state=rs),
                config.DECISION_TREE_GRID,
                "grid",
                config.DECISION_TREE_PARAMS,
            ),
            "Random Forest": (
                RandomForestClassifier(random_state=rs, n_jobs=1),
                config.RANDOM_FOREST_GRID,
                "random",
                config.RANDOM_FOREST_PARAMS,
            ),
            "XGBoost": (
                XGBClassifier(
                    random_state=rs,
                    n_jobs=1,
                    eval_metric="logloss",
                    **gpu.xgboost_gpu_kwargs(),  # GPU si está disponible
                ),
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
        self.results_ = {}
        self.best_params_ = {}
        for nombre, (estimador, grid, kind, default_params) in self._catalog().items():
            pipe = _pipeline(estimador)
            # En GPU, XGBoost debe ajustarse de forma secuencial (n_jobs=1): varios
            # entrenamientos en paralelo competirían por la misma GPU. El resto de
            # modelos (CPU) sí paralelizan la búsqueda.
            njobs = 1 if (nombre == "XGBoost" and gpu.xgboost_device() == "cuda") else -1
            if kind == "grid":
                # GridSearchCV: búsqueda exhaustiva (la herramienta vista en `recursos/`).
                search = GridSearchCV(
                    pipe, grid, scoring=self.scoring, cv=self.cv, n_jobs=njobs
                )
                tipo = "GridSearchCV"
            else:
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
                tipo = "RandomizedSearchCV"

            logger.info("Optimizando %s con %s...", nombre, tipo)
            inicio = time.perf_counter()
            search.fit(X_train, y_train)

            # Baseline: CV con los hiperparámetros por defecto del proyecto.
            extra = {"n_jobs": 1} if "n_jobs" in default_params else {}
            if estimador.__class__.__name__ == "XGBClassifier":
                extra.update(gpu.xgboost_gpu_kwargs())  # mismo device que la búsqueda
            base = estimador.__class__(**{**default_params, **extra})
            cv_default = cross_val_score(
                _pipeline(base), X_train, y_train, scoring=self.scoring, cv=self.cv, n_jobs=njobs
            ).mean()
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
        return self.results_

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

    @staticmethod
    def _to_markdown(tabla: pd.DataFrame) -> str:
        """Convierte el DataFrame de resultados a tabla Markdown (sin dependencias)."""
        cols = list(tabla.columns)
        encabezado = "| Modelo | " + " | ".join(cols) + " |"
        separador = "|" + "---|" * (len(cols) + 1)
        filas = [
            "| " + str(idx) + " | " + " | ".join(str(v) for v in fila) + " |"
            for idx, fila in zip(tabla.index, tabla.to_numpy())
        ]
        return "\n".join([encabezado, separador, *filas])

    def save_results(self, path=config.TUNING_RESULTS_PATH) -> None:
        """Escribe un informe Markdown con la tabla y los mejores hiperparámetros."""
        tabla = self.results_table()
        lineas = [
            "# Optimización de hiperparámetros\n",
            f"Métrica optimizada: **{self.scoring}** · CV de **{self.cv}** particiones.\n",
            self._to_markdown(tabla),
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
    from .train import configure_logging

    configure_logging()
    config.ensure_directories()
    logger.info("=== Optimización de hiperparámetros (CV=%d, métrica=%s) ===",
                config.TUNING_CV_FOLDS, config.TUNING_SCORING)
    X_train, _, y_train, _ = load_and_prepare()
    tuner = HyperparameterTuner()
    tuner.tune(X_train, y_train)
    tuner.save_results()
    tuner.save_best_params()  # se usarán por defecto en `python -m src.train`
    print("\n" + "=" * 70)
    print("RESULTADO DE LA OPTIMIZACIÓN DE HIPERPARÁMETROS")
    print("=" * 70)
    print(tuner.results_table().to_string())
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
