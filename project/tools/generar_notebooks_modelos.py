"""Generador de los notebooks "un modelo por notebook".

Todos los notebooks de modelo comparten el MISMO esqueleto (plantilla) y solo
cambian los contenidos específicos de cada modelo, definidos en `SPECS`. Así se
garantiza una forma consistente y es trivial regenerarlos:

    python tools/generar_notebooks_modelos.py

Estructura de cada notebook:
  1. Cómo funciona el modelo
  2. Los datos (preprocesado compartido)
  3. Los hiperparámetros: qué controla cada uno
  4. Entrenamiento y evaluación (parámetros base)
  5. Optimización / regularización de hiperparámetros
  6. Resultado final y cuándo usar este modelo
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"

# --------------------------------------------------------------------------- #
# Celdas comunes (idénticas en todos los notebooks)
# --------------------------------------------------------------------------- #
SETUP_CODE = (
    "%matplotlib inline\n"
    "import sys, os\n"
    "sys.path.insert(0, os.path.abspath('..'))\n"
    "import warnings; warnings.filterwarnings('ignore')\n"
    "os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')\n"
    "import numpy as np, pandas as pd\n"
    "import matplotlib.pyplot as plt, seaborn as sns\n"
    "sns.set_theme(style='whitegrid')\n"
    "from sklearn.pipeline import Pipeline\n"
    "from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,\n"
    "                             roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay)\n"
    "from src import config\n"
    "from src.data_loader import load_and_prepare\n"
    "from src.preprocessing import build_preprocessor\n"
    "\n"
    "X_train, X_test, y_train, y_test = load_and_prepare()\n"
    "print('Entrenamiento:', X_train.shape, '| Prueba:', X_test.shape)\n"
    "\n"
    "def evaluar(pipe, etiqueta):\n"
    "    \"\"\"Imprime las métricas de test del pipeline y devuelve su ROC-AUC.\"\"\"\n"
    "    proba = pipe.predict_proba(X_test)[:, 1]\n"
    "    pred = (proba >= 0.5).astype(int)\n"
    "    m = dict(accuracy=accuracy_score(y_test, pred), precision=precision_score(y_test, pred),\n"
    "             recall=recall_score(y_test, pred), f1=f1_score(y_test, pred),\n"
    "             roc_auc=roc_auc_score(y_test, proba))\n"
    "    print(f\"{etiqueta:12s} | \" + ' | '.join(f'{k}={v:.4f}' for k, v in m.items()))\n"
    "    return m['roc_auc']"
)

DATOS_MD = (
    "## 2. Los datos\n\n"
    "Usamos el **mismo preprocesado compartido** por todos los modelos "
    "(`src/preprocessing.py`): imputación de huecos, estandarización de las "
    "numéricas y *one-hot* de las categóricas. El análisis de los datos está en "
    "[`01_eda.ipynb`](01_eda.ipynb). Cargamos y dividimos en train/test "
    "(estratificado, semilla fija) con `load_and_prepare()`."
)

EVAL_BASE_CODE = (
    "# Pipeline = preprocesado + modelo (con los hiperparámetros base del proyecto)\n"
    "__IMPORT__\n"
    "modelo = Pipeline([('preprocessor', build_preprocessor()), ('model', __CTOR_BASE__)])\n"
    "modelo.fit(X_train, y_train)\n"
    "auc_base = evaluar(modelo, 'Base')"
)

PLOT_CODE = (
    "# Matriz de confusión y curva ROC de este modelo\n"
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))\n"
    "ConfusionMatrixDisplay.from_estimator(modelo, X_test, y_test, ax=ax1,\n"
    "    display_labels=['No cancela', 'Cancela'], cmap='Blues', colorbar=False)\n"
    "ax1.set_title('Matriz de confusión')\n"
    "RocCurveDisplay.from_estimator(modelo, X_test, y_test, ax=ax2)\n"
    "ax2.plot([0, 1], [0, 1], '--', color='gray', alpha=0.7)\n"
    "ax2.set_title('Curva ROC')\n"
    "plt.tight_layout(); plt.show()"
)

SEARCH_CODE = (
    "from sklearn.model_selection import __SEARCHCLS__\n"
    "\n"
    "base_pipe = Pipeline([('preprocessor', build_preprocessor()), ('model', __CTOR_TUNE__)])\n"
    "busqueda = __SEARCH_CALL__\n"
    "busqueda.fit(X_train, y_train)\n"
    "print('Mejores hiperparámetros:', busqueda.best_params_)\n"
    "print(f'ROC-AUC (validación cruzada, {config.TUNING_CV_FOLDS} folds): {busqueda.best_score_:.4f}')"
)

EVAL_TUNED_CODE = (
    "auc_tuned = evaluar(busqueda.best_estimator_, 'Optimizado')\n"
    "print(f'\\nROC-AUC en test:  base {auc_base:.4f}  ->  optimizado {auc_tuned:.4f}')"
)

NN_HISTORY_CODE = (
    "# La red no se ajusta con búsqueda en rejilla (sería costosísimo); se regulariza\n"
    "# con dropout y EARLY STOPPING. Vemos su curva de aprendizaje por época.\n"
    "hist = modelo.named_steps['model'].history_.history\n"
    "auc_k = next(k for k in hist if 'auc' in k and not k.startswith('val'))\n"
    "fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))\n"
    "a1.plot(hist['loss'], label='entrenamiento'); a1.plot(hist['val_loss'], label='validación')\n"
    "a1.set_title('Pérdida (loss) por época'); a1.set_xlabel('época'); a1.legend()\n"
    "a2.plot(hist[auc_k], label='entrenamiento'); a2.plot(hist['val_' + auc_k], label='validación')\n"
    "a2.set_title('ROC-AUC por época'); a2.set_xlabel('época'); a2.legend()\n"
    "plt.tight_layout(); plt.show()\n"
    "print(f'La red entrenó {len(hist[\"loss\"])} épocas; early stopping restauró los mejores pesos.')"
)


# --------------------------------------------------------------------------- #
# Fichas por modelo
# --------------------------------------------------------------------------- #
SPECS = [
    {
        "file": "02_modelo_regresion_logistica.ipynb",
        "titulo": "02 · Modelo: Regresión logística",
        "una_frase": "un modelo **lineal** sencillo y muy interpretable; nuestra línea base.",
        "como_funciona": (
            "A pesar de su nombre, la **regresión logística** es un *clasificador*. "
            "Calcula una **suma ponderada** de las variables (como una recta o un plano) "
            "y la pasa por la función **sigmoide**, que la convierte en una "
            "**probabilidad** entre 0 y 1. Durante el entrenamiento aprende los **pesos** "
            "que mejor separan las clases.\n\n"
            "Es un modelo **lineal**: su frontera de decisión es una línea/plano recto. "
            "Por eso es rápido e interpretable (cada coeficiente indica cuánto y en qué "
            "dirección empuja una variable hacia \"cancela\"), pero no captura por sí solo "
            "relaciones complejas o curvas."
        ),
        "hiperparametros": [
            ("`C`", "Inverso de la **regularización**. `C` alto → poca regularización (puede sobreajustar); `C` bajo → modelo más simple y suave."),
            ("`penalty`", "Tipo de regularización (`l2` por defecto): penaliza pesos grandes para evitar el sobreajuste."),
            ("`class_weight`", "Si `'balanced'`, da más peso a la clase minoritaria (ver el notebook de balanceo)."),
            ("`solver`", "Algoritmo de optimización que ajusta los pesos."),
            ("`max_iter`", "Nº máximo de iteraciones del optimizador (que llegue a converger)."),
        ],
        "import": "from sklearn.linear_model import LogisticRegression",
        "ctor_base": "LogisticRegression(**config.LOGISTIC_REGRESSION_PARAMS)",
        "ctor_tune": "LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)",
        "grid": "config.LOGISTIC_REGRESSION_GRID",
        "search": "grid",
        "cuando_usar": (
            "**Cuándo conviene:** como **línea base** rápida e interpretable, y cuando "
            "esperas que las relaciones sean aproximadamente lineales. Es la referencia "
            "contra la que se mide si modelos más complejos merecen la pena."
        ),
    },
    {
        "file": "03_modelo_arbol_decision.ipynb",
        "titulo": "03 · Modelo: Árbol de decisión",
        "una_frase": "preguntas sí/no encadenadas; **no lineal** e interpretable.",
        "como_funciona": (
            "Un **árbol de decisión** hace **preguntas sí/no** encadenadas sobre las "
            "variables (p. ej. *¿`lead_time` > 100?*), dividiendo los datos en ramas "
            "hasta llegar a **hojas** que predicen una clase. En cada división elige la "
            "pregunta que **mejor separa** las clases (según una medida de impureza como "
            "*gini* o *entropy*).\n\n"
            "Es **no lineal** y muy interpretable (se puede dibujar el árbol y leer las "
            "reglas), pero un árbol que crece sin límite tiende a **sobreajustar** "
            "(memorizar el entrenamiento). Por eso se controla su tamaño."
        ),
        "hiperparametros": [
            ("`max_depth`", "Profundidad máxima del árbol. Más profundo → más complejo y mayor riesgo de sobreajuste."),
            ("`min_samples_leaf`", "Mínimo de muestras en una hoja. Más alto → hojas más \"pobladas\" → árbol más simple y robusto."),
            ("`criterion`", "Medida de impureza para elegir las divisiones (`gini` o `entropy`)."),
            ("`min_samples_split`", "Mínimo de muestras para poder dividir un nodo."),
        ],
        "import": "from sklearn.tree import DecisionTreeClassifier",
        "ctor_base": "DecisionTreeClassifier(**config.DECISION_TREE_PARAMS)",
        "ctor_tune": "DecisionTreeClassifier(random_state=config.RANDOM_STATE)",
        "grid": "config.DECISION_TREE_GRID",
        "search": "grid",
        "cuando_usar": (
            "**Cuándo conviene:** cuando quieres **reglas claras e interpretables**, o "
            "como bloque de construcción de modelos de conjunto (Random Forest, XGBoost). "
            "Por sí solo suele rendir menos que esos conjuntos."
        ),
    },
    {
        "file": "04_modelo_random_forest.ipynb",
        "titulo": "04 · Modelo: Random Forest",
        "una_frase": "**muchos árboles que votan**; robusto y potente.",
        "como_funciona": (
            "Un **Random Forest** (bosque aleatorio) es un **conjunto** (*ensemble*) de "
            "muchos árboles de decisión. Cada árbol se entrena con una **muestra "
            "aleatoria** de filas (*bagging*) y, en cada división, solo considera un "
            "**subconjunto aleatorio** de variables. La predicción final es el "
            "**voto/promedio** de todos los árboles.\n\n"
            "Esa doble aleatoriedad hace que los árboles sean **diferentes entre sí**; al "
            "promediarlos se **reduce el sobreajuste** de un árbol individual. Suele dar "
            "muy buen rendimiento sin apenas ajuste y aporta una **importancia de "
            "variables** útil para interpretar."
        ),
        "hiperparametros": [
            ("`n_estimators`", "Nº de árboles. Más árboles → predicción más estable (hasta saturar) y más lento."),
            ("`max_depth`", "Profundidad de cada árbol (complejidad individual)."),
            ("`min_samples_leaf`", "Muestras mínimas por hoja (suaviza/regulariza)."),
            ("`max_features`", "Nº de variables candidatas por división (`sqrt`/`log2`). Controla la **diversidad** entre árboles."),
            ("`class_weight`", "Reponderación de clases (`'balanced'`)."),
        ],
        "import": "from sklearn.ensemble import RandomForestClassifier",
        "ctor_base": "RandomForestClassifier(**config.RANDOM_FOREST_PARAMS)",
        "ctor_tune": "RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1)",
        "grid": "config.RANDOM_FOREST_GRID",
        "search": "random",
        "cuando_usar": (
            "**Cuándo conviene:** como modelo **potente y robusto \"de fábrica\"**, poco "
            "sensible a la escala y con buena importancia de variables. Excelente opción "
            "por defecto cuando no quieres ajustar mucho."
        ),
    },
    {
        "file": "05_modelo_xgboost.ipynb",
        "titulo": "05 · Modelo: XGBoost",
        "una_frase": "**boosting** de árboles; el rey de los datos tabulares (y nuestro mejor modelo).",
        "como_funciona": (
            "**XGBoost** es *gradient boosting*: entrena árboles **en secuencia**, y cada "
            "árbol nuevo se centra en **corregir los errores** del conjunto anterior "
            "(ajusta los gradientes del error). Combinando muchos árboles \"débiles\" "
            "obtiene un modelo **fuerte**.\n\n"
            "A diferencia del Random Forest (árboles independientes que se promedian), "
            "aquí los árboles se construyen **uno tras otro** mejorando lo anterior. "
            "Incluye **regularización** integrada y suele ser el **mejor** en datos "
            "tabulares — como ocurre en este proyecto."
        ),
        "hiperparametros": [
            ("`n_estimators`", "Nº de árboles (rondas de boosting)."),
            ("`learning_rate`", "Cuánto aporta cada árbol. Bajo → aprendizaje lento pero más fino (necesita más árboles)."),
            ("`max_depth`", "Profundidad de cada árbol (complejidad e interacciones que capta)."),
            ("`subsample`", "Fracción de **filas** por árbol (regularización tipo *bagging*)."),
            ("`colsample_bytree`", "Fracción de **columnas** por árbol."),
            ("`min_child_weight` / `gamma` / `reg_lambda`", "Parámetros de **regularización** que frenan el sobreajuste."),
        ],
        "import": "from xgboost import XGBClassifier",
        "ctor_base": "XGBClassifier(**config.XGBOOST_PARAMS)",
        "ctor_tune": "XGBClassifier(random_state=config.RANDOM_STATE, n_jobs=-1, eval_metric='logloss')",
        "grid": "config.XGBOOST_GRID",
        "search": "random",
        "cuando_usar": (
            "**Cuándo conviene:** cuando buscas **el máximo rendimiento** en datos "
            "tabulares. Es el modelo elegido del proyecto (mejor ROC-AUC). A cambio, "
            "tiene **más hiperparámetros** que ajustar."
        ),
    },
    {
        "file": "06_modelo_red_neuronal.ipynb",
        "titulo": "06 · Modelo: Red neuronal (Keras)",
        "una_frase": "capas de neuronas (*deep learning*) que capturan relaciones complejas.",
        "como_funciona": (
            "Una **red neuronal multicapa** (*perceptrón multicapa*) apila **capas de "
            "neuronas**. Cada neurona combina sus entradas con unos **pesos** y aplica una "
            "**función de activación** (ReLU) que aporta no linealidad; la capa final "
            "(sigmoide) devuelve la **probabilidad** de cancelación.\n\n"
            "Aprende ajustando los pesos por **descenso de gradiente** (*backpropagation*), "
            "minimizando el error capa a capa. Puede capturar relaciones **muy complejas**, "
            "pero necesita más datos y cuidado con el sobreajuste, que aquí controlamos con "
            "**dropout** y **early stopping**."
        ),
        "hiperparametros": [
            ("`hidden_units`", "Nº de neuronas por capa oculta (la **capacidad** del modelo)."),
            ("`dropout`", "Fracción de neuronas que se \"apagan\" al azar al entrenar (**regularización**)."),
            ("`learning_rate`", "Tamaño del paso del optimizador (Adam)."),
            ("`epochs` / `batch_size`", "Nº de pasadas por los datos / tamaño del lote."),
            ("`early_stopping_patience`", "Épocas sin mejorar la validación antes de **parar** (evita sobreajuste)."),
        ],
        "import": "from src.model_trainer import KerasMLPClassifier",
        "ctor_base": "KerasMLPClassifier(**config.NN_PARAMS, random_state=config.RANDOM_STATE)",
        "ctor_tune": None,
        "grid": None,
        "search": "nn",
        "cuando_usar": (
            "**Cuándo conviene:** con relaciones **muy no lineales** y abundancia de "
            "datos. Aquí compite de cerca, pero **XGBoost la supera**: en datos tabulares "
            "medianos, el boosting de árboles suele ganar a las redes."
        ),
    },
]


def _hiperparam_md(spec) -> str:
    filas = "\n".join(f"| {p} | {d} |" for p, d in spec["hiperparametros"])
    return (
        "## 3. Los hiperparámetros: ¿qué controla cada uno?\n\n"
        "Los **hiperparámetros** son los ajustes que fijamos *antes* de entrenar "
        "(no se aprenden de los datos). Estos son los principales de este modelo:\n\n"
        "| Hiperparámetro | Qué controla |\n|---|---|\n" + filas
    )


def _search_md(spec) -> str:
    if spec["search"] == "nn":
        return (
            "## 5. Regularización y *early stopping*\n\n"
            "Ajustar una red por **búsqueda en rejilla** sería desproporcionadamente "
            "costoso. En su lugar, la red se **autorregula** durante el entrenamiento "
            "con **dropout** (apaga neuronas al azar) y **early stopping** (para cuando "
            "la validación deja de mejorar y restaura los mejores pesos). La curva de "
            "aprendizaje muestra ese proceso:"
        )
    tecnica = "GridSearchCV (búsqueda exhaustiva)" if spec["search"] == "grid" else "RandomizedSearchCV (muestreo aleatorio de combinaciones)"
    return (
        "## 5. Optimización de hiperparámetros\n\n"
        f"Buscamos la mejor combinación con **{tecnica}**, optimizando **ROC-AUC** por "
        "validación cruzada. (Es el mismo procedimiento, por modelo, que automatiza "
        "`src/tuning.py` para todo el proyecto.)"
    )


def _search_code(spec) -> str:
    cv = "config.TUNING_CV_FOLDS"
    if spec["search"] == "grid":
        call = (f"GridSearchCV(base_pipe, {spec['grid']}, scoring='roc_auc', "
                f"cv={cv}, n_jobs=-1)")
        cls = "GridSearchCV"
    else:
        call = (f"RandomizedSearchCV(base_pipe, {spec['grid']}, n_iter=config.TUNING_N_ITER, "
                f"scoring='roc_auc', cv={cv}, n_jobs=-1, random_state=config.RANDOM_STATE)")
        cls = "RandomizedSearchCV"
    return (SEARCH_CODE
            .replace("__SEARCHCLS__", cls)
            .replace("__SEARCH_CALL__", call)
            .replace("__CTOR_TUNE__", spec["ctor_tune"]))


def build_notebook(spec) -> nbf.NotebookNode:
    md = nbf.v4.new_markdown_cell
    code = nbf.v4.new_code_cell
    cells = [
        md(f"# {spec['titulo']}\n\n"
           "**Proyecto Final — Machine Learning y Deep Learning**\n\n"
           f"Este notebook se centra en **un solo modelo**: {spec['una_frase']} "
           "Sigue la misma estructura que el resto de notebooks de modelo, para poder "
           "compararlos con facilidad.\n\n"
           "> 📖 Términos técnicos explicados en `docs/glosario.md`. El análisis de los "
           "datos está en [`01_eda.ipynb`](01_eda.ipynb); la comparación de todos los "
           "modelos, en [`07_comparativa_modelos.ipynb`](07_comparativa_modelos.ipynb)."),
        md("## 1. Cómo funciona\n\n" + spec["como_funciona"]),
        md(DATOS_MD),
        code(SETUP_CODE),
        md(_hiperparam_md(spec)),
        md("## 4. Entrenamiento y evaluación (parámetros base)\n\n"
           "Entrenamos el modelo con los hiperparámetros base y lo evaluamos sobre el "
           "conjunto de prueba (datos que no vio al entrenar)."),
        code(EVAL_BASE_CODE.replace("__IMPORT__", spec["import"]).replace("__CTOR_BASE__", spec["ctor_base"])),
        code(PLOT_CODE),
        md(_search_md(spec)),
        code(NN_HISTORY_CODE if spec["search"] == "nn" else _search_code(spec)),
    ]
    if spec["search"] == "nn":
        cells.append(md("## 6. Resultado final y cuándo usar este modelo\n\n" + spec["cuando_usar"]))
    else:
        cells.append(md("## 6. Resultado final y cuándo usar este modelo\n\n"
                        "Comparamos el rendimiento en test **antes y después** de optimizar:"))
        cells.append(code(EVAL_TUNED_CODE))
        cells.append(md(spec["cuando_usar"]))

    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    return nb


def main() -> None:
    for spec in SPECS:
        nb = build_notebook(spec)
        out = NB_DIR / spec["file"]
        nbf.write(nb, out)
        print(f"Generado: {out.name} ({len(nb.cells)} celdas)")


if __name__ == "__main__":
    main()
