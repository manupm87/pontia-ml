"""Catálogo de modelos y entrenamiento.

Define los cinco modelos a comparar (4 clásicos de scikit-learn/XGBoost + una red
neuronal **Keras/TensorFlow**), cada uno metido en un ``Pipeline`` con el
preprocesador, y los entrena con un bucle simple. Mismo preprocesador para todos ->
comparación justa y sin *data leakage*.

La red neuronal va envuelta en :class:`KerasMLPClassifier`, un estimador mínimo
compatible con scikit-learn para que encaje en el mismo ``Pipeline`` que el resto.
TensorFlow se importa de forma **perezosa** (solo al entrenar la red): en inferencia
se sirve XGBoost, así que el entorno de ejecución no carga TensorFlow.
"""

from __future__ import annotations

import logging
import re
import time

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from ml_hotel_cancellations import config
from .preprocessing import make_pipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Red neuronal Keras como estimador de scikit-learn
# ---------------------------------------------------------------------------
class KerasMLPClassifier(ClassifierMixin, BaseEstimator):
    """Red neuronal multicapa con Keras/TensorFlow (la que pide el enunciado).

    Misma arquitectura que el notebook ``04_red_neuronal``: densas 64->32->16 con
    *dropout* y salida sigmoide. Se presenta como estimador de scikit-learn
    (``fit``/``predict``/``predict_proba``) para encajar en el ``Pipeline`` común y
    poder compararla y serializarla igual que los demás modelos. TensorFlow se
    importa dentro de los métodos para no cargarlo salvo cuando se entrena la red.
    """

    def __init__(self, epochs=60, batch_size=512, validation_split=0.2,
                 patience=10, random_state=config.RANDOM_STATE):
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.patience = patience
        self.random_state = random_state

    def _build(self, n_features: int):
        import os
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # silencia logs de TF
        import tensorflow as tf
        from tensorflow.keras import layers, models

        tf.keras.utils.set_random_seed(self.random_state)  # reproducibilidad
        model = models.Sequential([
            layers.Input(shape=(n_features,)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def fit(self, X, y):
        from tensorflow.keras.callbacks import EarlyStopping

        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="float32")
        self.classes_ = np.array([0, 1])
        self.model_ = self._build(X.shape[1])
        early = EarlyStopping(monitor="val_loss", patience=self.patience,
                              restore_best_weights=True)
        self.model_.fit(X, y, epochs=self.epochs, batch_size=self.batch_size,
                        validation_split=self.validation_split,
                        callbacks=[early], verbose=0)
        return self

    def predict_proba(self, X):
        proba_pos = self.model_.predict(np.asarray(X, dtype="float32"), verbose=0).ravel()
        return np.column_stack([1.0 - proba_pos, proba_pos])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # --- Serialización: un modelo Keras no se pickea directo, así que guardamos
    #     su formato `.keras` como bytes dentro del propio pickle (y al revés). ---
    def __getstate__(self):
        state = self.__dict__.copy()
        model = state.pop("model_", None)
        if model is not None:
            import os
            import tempfile

            fd, path = tempfile.mkstemp(suffix=".keras")
            os.close(fd)
            model.save(path)
            with open(path, "rb") as fh:
                state["_model_bytes"] = fh.read()
            os.remove(path)
        return state

    def __setstate__(self, state):
        blob = state.pop("_model_bytes", None)
        self.__dict__.update(state)
        if blob is not None:
            import os
            import tempfile

            import keras

            fd, path = tempfile.mkstemp(suffix=".keras")
            os.close(fd)
            with open(path, "wb") as fh:
                fh.write(blob)
            self.model_ = keras.models.load_model(path)
            os.remove(path)


# ---------------------------------------------------------------------------
# Los cuatro modelos clásicos
# ---------------------------------------------------------------------------
def build_classic_estimators(
    *,
    overrides: dict[str, dict] | None = None,
    n_jobs: dict[str, int] | None = None,
) -> dict:
    """Crea los cuatro estimadores clásicos con los hiperparámetros de ``config``.

    ``overrides`` cambia params por modelo (lo usa `tuning`) y ``n_jobs`` ajusta el
    paralelismo por modelo (para no competir con la búsqueda CV).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    overrides = overrides or {}
    n_jobs = n_jobs or {}

    def params(name: str, base: dict) -> dict:
        """Combina los params base con los overrides (y n_jobs) de ese modelo."""
        merged = {**base, **overrides.get(name, {})}
        if name in n_jobs:
            merged["n_jobs"] = n_jobs[name]
        return merged

    return {
        "Logistic Regression": LogisticRegression(
            **params("Logistic Regression", config.LOGISTIC_REGRESSION_PARAMS)
        ),
        "Decision Tree": DecisionTreeClassifier(
            **params("Decision Tree", config.DECISION_TREE_PARAMS)
        ),
        "Random Forest": RandomForestClassifier(
            **params("Random Forest", config.RANDOM_FOREST_PARAMS)
        ),
        "XGBoost": XGBClassifier(**params("XGBoost", config.XGBOOST_PARAMS)),
    }


# ---------------------------------------------------------------------------
# Catálogo completo y entrenamiento
# ---------------------------------------------------------------------------
def build_models(overrides: dict[str, dict] | None = None) -> dict:
    """Devuelve los 5 modelos a comparar como ``{nombre: Pipeline}`` (sin entrenar).

    Cada modelo va en un ``Pipeline`` con su propio preprocesador, de modo que el
    objeto guardado sabe preprocesar reservas en crudo en inferencia.
    """
    estimators = build_classic_estimators(overrides=overrides)
    nn_params = {**config.NN_PARAMS, **(overrides or {}).get("Neural Network (Keras)", {})}
    estimators["Neural Network (Keras)"] = KerasMLPClassifier(**nn_params)
    return {name: make_pipeline(est) for name, est in estimators.items()}


def train_models(models: dict, X_train, y_train) -> tuple[dict, dict]:
    """Entrena cada modelo y devuelve ``(modelos_entrenados, tiempos_en_segundos)``."""
    fitted: dict = {}
    train_times: dict = {}
    for name, pipeline in models.items():
        logger.info("Entrenando %s...", name)
        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_times[name] = time.perf_counter() - start
        fitted[name] = pipeline
        logger.info("  -> %s entrenado en %.1f s", name, train_times[name])
    return fitted, train_times


def save_models(fitted: dict, directory=config.MODELS_DIR) -> None:
    """Guarda cada Pipeline entrenado en disco (un ``.pkl`` por modelo)."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, pipeline in fitted.items():
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        path = directory / f"{slug}.pkl"
        joblib.dump(pipeline, path)
        logger.info("Modelo guardado: %s", path)
