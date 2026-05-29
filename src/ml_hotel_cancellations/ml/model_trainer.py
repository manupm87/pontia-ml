"""Definición y entrenamiento de los modelos de clasificación.

Este módulo expone:

- :class:`KerasMLPClassifier`: un envoltorio compatible con la API de
  scikit-learn (``fit`` / ``predict`` / ``predict_proba``) alrededor de una red
  neuronal multicapa de Keras, de modo que pueda integrarse en un ``Pipeline`` y
  evaluarse exactamente igual que el resto de modelos.
- :class:`ModelTrainer`: clase orquestadora que construye un ``Pipeline``
  (preprocesado + modelo) por cada algoritmo y los entrena de forma homogénea.

Todos los modelos comparten el mismo preprocesador, lo que garantiza una
comparación justa entre algoritmos.
"""

from __future__ import annotations

# Silenciar los logs informativos de TensorFlow ANTES de importarlo (de forma
# diferida en los métodos). Mejora la legibilidad de la salida del pipeline.
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import logging
import time

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from ml_hotel_cancellations import config
from .preprocessing import make_pipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# (De)serialización de la red de Keras
# ---------------------------------------------------------------------------
# Los modelos de Keras no se serializan con pickle de forma fiable, así que se
# convierten a/desde bytes usando el formato nativo ``.keras``. Esto permite que
# ``joblib.dump`` funcione aunque el mejor modelo sea la red. Se usa un
# ``TemporaryDirectory`` para garantizar la limpieza del temporal aunque
# ``save``/``load_model`` lance.
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


# ---------------------------------------------------------------------------
# Envoltorio de la red neuronal de Keras
# ---------------------------------------------------------------------------
class KerasMLPClassifier(ClassifierMixin, BaseEstimator):
    """Perceptrón multicapa (Keras) con interfaz de scikit-learn.

    Implementa ``fit``, ``predict`` y ``predict_proba`` para poder usarse como
    último paso de un ``Pipeline`` de scikit-learn. Recibe ya las características
    preprocesadas (matriz densa de ``float``) producidas por el preprocesador.

    Parameters
    ----------
    hidden_units:
        Número de neuronas de cada capa oculta.
    dropout:
        Fracción de *dropout* tras cada capa oculta (regularización).
    epochs, batch_size, learning_rate, validation_split, early_stopping_patience:
        Hiperparámetros de entrenamiento.
    random_state:
        Semilla para reproducibilidad.
    verbose:
        Verbosidad del entrenamiento de Keras.
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
        """Construye y compila la red neuronal (arquitectura Sequential de Keras).

        Glosario rápido de los términos que aparecen (más detalle en
        ``docs/glosario.md``):

        - *Capa densa (Dense)*: capa donde cada neurona se conecta con todas las
          de la capa anterior.
        - *ReLU*: función de activación de las capas internas (deja pasar los
          positivos, pone a 0 los negativos); aporta "no linealidad".
        - *Dropout*: apaga neuronas al azar durante el entrenamiento para evitar
          el sobreajuste (que la red "se memorice" los datos).
        - *Sigmoide*: activación de la capa final; convierte la salida en una
          probabilidad entre 0 y 1.
        - *Adam*: algoritmo que ajusta los pesos de la red (el "optimizador").
        - *binary_crossentropy*: función de error (pérdida) para clasificación
          binaria; el entrenamiento intenta minimizarla.
        """
        import tensorflow as tf
        from tensorflow.keras import layers, models

        # Fijar la semilla aleatoria de TensorFlow para que el entrenamiento sea
        # reproducible (mismos resultados al repetir la ejecución).
        tf.keras.utils.set_random_seed(self.random_state)

        # `Input`: define cuántas características entran. Luego, capas densas con
        # ReLU y dropout intercalado.
        capas = [layers.Input(shape=(n_features,), name="entrada")]
        for i, unidades in enumerate(self.hidden_units, start=1):
            capas.append(layers.Dense(unidades, activation="relu", name=f"oculta_{i}"))
            if self.dropout and self.dropout > 0:
                capas.append(layers.Dropout(self.dropout, name=f"dropout_{i}"))
        # Capa de salida: 1 neurona sigmoide -> probabilidad de cancelación.
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

    # -- Serialización ------------------------------------------------------
    # La (de)serialización de la red Keras se delega en los helpers de módulo
    # ``_serialize_keras_model`` / ``_deserialize_keras_model``.
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
# Orquestador de entrenamiento
# ---------------------------------------------------------------------------
class ModelTrainer:
    """Construye y entrena el conjunto de modelos a comparar.

    Cada modelo se encapsula en un ``Pipeline`` que incluye el preprocesador, de
    forma que recibe siempre el DataFrame crudo y el preprocesado se ajusta solo
    con los datos de entrenamiento (evitando *data leakage*).
    """

    def __init__(
        self,
        random_state: int = config.RANDOM_STATE,
        param_overrides: dict[str, dict] | None = None,
    ):
        self.random_state = random_state
        # `param_overrides` permite inyectar hiperparámetros (p. ej. los hallados
        # por la optimización con `src.tuning`) sin tocar la configuración base.
        self.param_overrides = param_overrides or {}
        self.models_: dict[str, Pipeline] = {}
        self.train_times_: dict[str, float] = {}

    @staticmethod
    def _make_pipeline(estimator) -> Pipeline:
        """Envuelve un estimador con un preprocesador nuevo en un ``Pipeline``."""
        return make_pipeline(estimator)

    def build_models(self) -> dict[str, Pipeline]:
        """Define el catálogo de modelos a entrenar (los 5 algoritmos exigidos).

        Returns
        -------
        dict[str, Pipeline]
            Diccionario ``nombre -> Pipeline`` sin ajustar.
        """
        from .model_factory import build_classic_estimators

        # Los 4 clásicos salen de la fábrica única.
        modelos = build_classic_estimators(overrides=self.param_overrides)
        modelos["Neural Network (Keras)"] = KerasMLPClassifier(
            hidden_units=config.NN_PARAMS["hidden_units"],
            dropout=config.NN_PARAMS["dropout"],
            epochs=config.NN_PARAMS["epochs"],
            batch_size=config.NN_PARAMS["batch_size"],
            learning_rate=config.NN_PARAMS["learning_rate"],
            validation_split=config.NN_PARAMS["validation_split"],
            early_stopping_patience=config.NN_PARAMS["early_stopping_patience"],
            random_state=self.random_state,
        )
        return {nombre: self._make_pipeline(est) for nombre, est in modelos.items()}

    def train(self, X_train, y_train) -> dict[str, Pipeline]:
        """Entrena todos los modelos y registra sus tiempos de entrenamiento.

        Returns
        -------
        dict[str, Pipeline]
            Diccionario ``nombre -> Pipeline`` ya entrenado.
        """
        self.models_ = {}
        self.train_times_ = {}
        for nombre, pipeline in self.build_models().items():
            logger.info("Entrenando modelo: %s", nombre)
            inicio = time.perf_counter()
            pipeline.fit(X_train, y_train)
            elapsed = time.perf_counter() - inicio
            self.models_[nombre] = pipeline
            self.train_times_[nombre] = elapsed
            logger.info("  -> %s entrenado en %.1f s", nombre, elapsed)
        return self.models_

    def save_models(self, directory=config.MODELS_DIR) -> None:
        """Guarda cada Pipeline entrenado en disco (un ``.pkl`` por modelo)."""
        import re

        import joblib

        directory.mkdir(parents=True, exist_ok=True)
        for nombre, pipeline in self.models_.items():
            slug = re.sub(r"[^a-z0-9]+", "_", nombre.lower()).strip("_")
            ruta = directory / f"{slug}.pkl"
            joblib.dump(pipeline, ruta)
            logger.info("Modelo guardado: %s", ruta)
