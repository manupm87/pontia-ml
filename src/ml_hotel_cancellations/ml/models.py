"""Catálogo de modelos y entrenamiento.

Define los cinco modelos a comparar (4 clásicos + red neuronal Keras), cada uno
metido en un ``Pipeline`` con el preprocesador, y los entrena con un bucle simple.
Mismo preprocesador para todos -> comparación justa y sin *data leakage*.
"""

from __future__ import annotations

# Silenciar los logs de TensorFlow ANTES de importarlo (la importación es perezosa).
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

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
# Red neuronal Keras con interfaz scikit-learn
# ---------------------------------------------------------------------------
# Keras no se serializa con pickle de forma fiable; lo convertimos a/desde bytes
# con su formato nativo ``.keras`` para que ``joblib.dump`` funcione con la red.
def _serialize_keras_model(model) -> bytes:
    """Vuelca un modelo Keras al formato ``.keras`` y devuelve sus bytes."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "model.keras"
        model.save(tmp_path)
        return tmp_path.read_bytes()


def _deserialize_keras_model(data: bytes):
    """Reconstruye un modelo Keras a partir de los bytes de un fichero ``.keras``."""
    import tempfile
    from pathlib import Path

    import tensorflow as tf

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "model.keras"
        tmp_path.write_bytes(data)
        return tf.keras.models.load_model(tmp_path)


class KerasMLPClassifier(ClassifierMixin, BaseEstimator):
    """Perceptrón multicapa (Keras) con interfaz de scikit-learn.

    Implementa ``fit`` / ``predict`` / ``predict_proba`` para usarse como último
    paso de un ``Pipeline``. Recibe las características ya preprocesadas (matriz densa).
    """

    def __init__(
        self,
        hidden_units=(64, 32, 16),
        dropout: float = 0.3,
        epochs: int = 50,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
        validation_split: float = 0.2,
        early_stopping_patience: int = 5,
        random_state: int = config.RANDOM_STATE,
        verbose: int = 0,
    ):
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.verbose = verbose

    def _build_model(self, n_features: int):
        """Construye y compila la red (Dense+ReLU+Dropout, salida sigmoide)."""
        import tensorflow as tf
        from tensorflow.keras import layers, models

        tf.keras.utils.set_random_seed(self.random_state)  # reproducibilidad

        capas = [layers.Input(shape=(n_features,), name="entrada")]
        for i, units in enumerate(self.hidden_units, start=1):
            capas.append(layers.Dense(units, activation="relu", name=f"oculta_{i}"))
            if self.dropout and self.dropout > 0:
                capas.append(layers.Dropout(self.dropout, name=f"dropout_{i}"))
        capas.append(layers.Dense(1, activation="sigmoid", name="salida"))

        model = models.Sequential(capas)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )
        return model

    def fit(self, X, y):
        """Entrena la red con *early stopping* sobre una partición de validación."""
        import tensorflow as tf

        X = np.asarray(X, dtype="float32")
        y = np.asarray(y).astype("float32")
        self.classes_ = np.unique(y).astype(int)
        self.n_features_in_ = X.shape[1]

        self.model_ = self._build_model(self.n_features_in_)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.early_stopping_patience,
            restore_best_weights=True,
        )
        self.history_ = self.model_.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=[early_stopping],
            verbose=self.verbose,
        )
        return self

    def predict_proba(self, X):
        """Devuelve las probabilidades de cada clase con forma ``(n, 2)``."""
        X = np.asarray(X, dtype="float32")
        p1 = self.model_.predict(X, verbose=0).ravel()
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        """Devuelve la clase predicha aplicando el umbral de decisión del proyecto."""
        return (self.predict_proba(X)[:, 1] >= config.DECISION_THRESHOLD).astype(int)

    # -- Serialización (delegada en los helpers de módulo) ------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        model = state.get("model_", None)
        if model is not None and not isinstance(model, (bytes, bytearray)):
            state["model_"] = _serialize_keras_model(model)
            state["_model_serialized"] = True
        return state

    def __setstate__(self, state):
        serialized = state.pop("_model_serialized", False)
        self.__dict__.update(state)
        if serialized and isinstance(self.model_, (bytes, bytearray)):
            self.model_ = _deserialize_keras_model(self.model_)


# ---------------------------------------------------------------------------
# Catálogo completo y entrenamiento
# ---------------------------------------------------------------------------
def build_models(overrides: dict[str, dict] | None = None) -> dict:
    """Devuelve los 5 modelos a comparar como ``{nombre: Pipeline}`` (sin entrenar).

    Cada modelo va en un ``Pipeline`` con su propio preprocesador, de modo que el
    objeto guardado sabe preprocesar reservas en crudo en inferencia.
    """
    estimators = build_classic_estimators(overrides=overrides)
    estimators["Neural Network (Keras)"] = KerasMLPClassifier(
        **config.NN_PARAMS, random_state=config.RANDOM_STATE
    )
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
