"""Envoltorio sklearn de la red Keras, compartido por los notebooks 04 y 05.

Define ``KerasMLPClassifier``, el mismo estimador que vive en
``src/ml_hotel_cancellations/ml/models.py``: envuelve un modelo Keras con la API
de scikit-learn (``fit``/``predict``/``predict_proba``) para tratarlo igual que
los demás modelos y poder serializarlo con ``joblib``.

Lo extraemos a este módulo para **no copiarlo** en los dos notebooks: el `04` lo
usa para exportar la red entrenada a ``models/red_neuronal.pkl`` y el `05` para
recargarla. Que la clase viva aquí (y no en cada notebook) hace además que
``joblib`` reconstruya el ``.pkl`` de forma fiable, porque la referencia de la
clase es ``keras_mlp.KerasMLPClassifier`` y no ``__main__`` (que cambia entre
notebooks).

Nota: TensorFlow se importa de forma **perezosa** dentro de los métodos, así que
basta con ``import keras_mlp`` sin pagar el arranque de TF hasta usarlo.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class KerasMLPClassifier(BaseEstimator, ClassifierMixin):
    """Red Keras (MLP) con API sklearn y serializable con joblib.

    La arquitectura es la del notebook `04` (densas 64→32→16 + Dropout(0.3) +
    sigmoide, Adam 1e-3, EarlyStopping). Un modelo Keras no se *pickea* directo,
    así que en ``__getstate__``/``__setstate__`` guardamos su formato nativo
    ``.keras`` como **bytes dentro del propio pickle** (y lo reconstruimos al
    cargar).
    """

    def __init__(self, epochs=60, batch_size=512, validation_split=0.2,
                 patience=10, random_state=42):
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.patience = patience
        self.random_state = random_state

    def _build(self, n_features):
        import tensorflow as tf
        from tensorflow.keras import layers, models
        tf.keras.utils.set_random_seed(self.random_state)
        red = models.Sequential([
            layers.Input(shape=(n_features,)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ])
        red.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                    loss="binary_crossentropy", metrics=["accuracy"])
        return red

    def fit(self, X, y):
        from tensorflow.keras.callbacks import EarlyStopping
        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="float32")
        self.classes_ = np.array([0, 1])
        self.model_ = self._build(X.shape[1])
        early = EarlyStopping(monitor="val_loss", patience=self.patience,
                              restore_best_weights=True)
        self.history_ = self.model_.fit(
            X, y, epochs=self.epochs, batch_size=self.batch_size,
            validation_split=self.validation_split, callbacks=[early], verbose=0)
        return self

    def predict_proba(self, X):
        p = self.model_.predict(np.asarray(X, dtype="float32"), verbose=0).ravel()
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # --- Serialización: el .keras va como bytes dentro del pickle (y al revés) ---
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("history_", None)            # el History de Keras no se pickea
        model = state.pop("model_", None)
        if model is not None:
            import os, tempfile
            fd, path = tempfile.mkstemp(suffix=".keras"); os.close(fd)
            model.save(path)
            with open(path, "rb") as fh:
                state["_model_bytes"] = fh.read()
            os.remove(path)
        return state

    def __setstate__(self, state):
        blob = state.pop("_model_bytes", None)
        self.__dict__.update(state)
        if blob is not None:
            import os, tempfile, keras
            fd, path = tempfile.mkstemp(suffix=".keras"); os.close(fd)
            with open(path, "wb") as fh:
                fh.write(blob)
            self.model_ = keras.models.load_model(path)
            os.remove(path)
