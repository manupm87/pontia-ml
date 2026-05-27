"""Visualización de la *estructura* o los *parámetros aprendidos* de cada modelo.

Una sola función pública, :func:`visualizar_modelo`, que recibe el ``Pipeline``
entrenado y dibuja lo más representativo de **ese tipo de modelo**:

- Regresión logística → sus **coeficientes** (peso y signo de cada variable).
- Árbol de decisión → el **árbol** (primeros niveles).
- Random Forest → **uno** de sus árboles (es un bosque de muchos).
- XGBoost → **importancia de variables** (son ~cientos de árboles; no se dibuja entero).
- Red neuronal → un **esquema de la arquitectura** (capas y neuronas).

Todo con dependencias ya presentes (matplotlib + sklearn): no requiere graphviz.
Centralizar esto aquí permite que cada notebook de modelo llame exactamente a la
misma función (`visualizar_modelo(modelo, X_train)`), manteniéndolos consistentes.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .preprocessing import get_feature_names


def visualizar_modelo(pipeline, X_train=None, max_features: int = 15):
    """Dibuja la visualización adecuada según el tipo de modelo del pipeline."""
    model = pipeline.named_steps["model"]
    prep = pipeline.named_steps["preprocessor"]
    nombre = type(model).__name__

    if nombre == "LogisticRegression":
        _coeficientes(model, prep, max_features)
    elif nombre == "DecisionTreeClassifier":
        _arbol(model, prep, "Árbol de decisión (primeros niveles)")
    elif nombre == "RandomForestClassifier":
        _arbol(model.estimators_[0], prep,
               f"Un árbol del bosque (1 de {len(model.estimators_)}; primeros niveles)")
    elif nombre == "XGBClassifier":
        _importancia(model, prep, max_features)
    elif nombre == "KerasMLPClassifier":
        _arquitectura(model)
    else:
        print(f"(Sin visualización específica para {nombre}.)")


def _nombres(prep) -> list[str]:
    return list(get_feature_names(prep))


def _coeficientes(model, prep, k: int) -> None:
    nombres = _nombres(prep)
    coef = model.coef_.ravel()
    idx = np.argsort(np.abs(coef))[-k:]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(idx) + 1))
    colores = ["#d95f0e" if c > 0 else "#2c7fb8" for c in coef[idx]]
    ax.barh([nombres[i] for i in idx], coef[idx], color=colores)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("peso (coeficiente)")
    ax.set_title("Coeficientes de la regresión logística\n"
                 "(naranja → empuja a «cancela»; azul → a «no cancela»)")
    plt.tight_layout()
    plt.show()


def _arbol(tree_model, prep, titulo: str) -> None:
    from sklearn.tree import plot_tree

    fig, ax = plt.subplots(figsize=(18, 8))
    plot_tree(tree_model, max_depth=3, feature_names=_nombres(prep),
              class_names=["No cancela", "Cancela"], filled=True, rounded=True,
              proportion=True, impurity=False, fontsize=8, ax=ax)
    ax.set_title(titulo)
    plt.tight_layout()
    plt.show()


def _importancia(model, prep, k: int) -> None:
    nombres = _nombres(prep)
    imp = model.feature_importances_
    idx = np.argsort(imp)[-k:]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(idx) + 1))
    ax.barh([nombres[i] for i in idx], imp[idx], color="#2c7fb8")
    ax.set_xlabel("importancia (ganancia)")
    ax.set_title(f"XGBoost: importancia de variables\n"
                 f"(el modelo son ~{model.n_estimators} árboles encadenados; "
                 f"no se dibuja entero)")
    plt.tight_layout()
    plt.show()


def _arquitectura(model) -> None:
    """Esquema de la red: una caja por capa con su nº de neuronas y activación."""
    n_in = getattr(model, "n_features_in_", "?")
    capas = [("Entrada", f"{n_in} variables", "")]
    for i, u in enumerate(model.hidden_units, 1):
        capas.append((f"Oculta {i}", f"{u} neuronas", "ReLU"))
    capas.append(("Salida", "1 neurona", "sigmoide"))

    fig, ax = plt.subplots(figsize=(2.4 * len(capas), 4))
    for xi, (nombre, detalle, act) in enumerate(capas):
        ax.add_patch(plt.Rectangle((xi - 0.35, 0.25), 0.7, 0.5, fc="#cfe3f3", ec="k"))
        ax.text(xi, 0.63, nombre, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(xi, 0.50, detalle, ha="center", va="center", fontsize=9)
        if act:
            ax.text(xi, 0.38, act, ha="center", va="center", fontsize=8, style="italic", color="#555")
        if xi < len(capas) - 1:
            ax.annotate("", xy=(xi + 1 - 0.35, 0.5), xytext=(xi + 0.35, 0.5),
                        arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set_xlim(-0.6, len(capas) - 0.4)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Arquitectura de la red neuronal (perceptrón multicapa)")
    plt.tight_layout()
    plt.show()
