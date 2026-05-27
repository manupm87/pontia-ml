# Notebooks

Cuadernos del proyecto. Orden de lectura recomendado:

| Notebook | Contenido |
|---|---|
| `01_eda.ipynb` | **EDA compartido**: exploración y decisiones de preprocesado (común a todos los modelos). |
| `02_modelo_regresion_logistica.ipynb` | Modelo: regresión logística. |
| `03_modelo_arbol_decision.ipynb` | Modelo: árbol de decisión. |
| `04_modelo_random_forest.ipynb` | Modelo: Random Forest. |
| `05_modelo_xgboost.ipynb` | Modelo: XGBoost (el mejor). |
| `06_modelo_red_neuronal.ipynb` | Modelo: red neuronal (Keras). |
| `07_comparativa_modelos.ipynb` | Comparación de los 5 modelos + visualización 2D (PLS, t-SNE). |
| `08_balanceo_clases.ipynb` | Desbalance de clases: `class_weight` vs SMOTE. |
| `09_no_supervisado.ipynb` | Clustering (K-Means): segmentos de reserva. |
| `_PLANTILLA_modelo.ipynb` | **Plantilla** para crear un nuevo notebook de modelo. |

## Convención de los notebooks de modelo

Todos los notebooks `0X_modelo_*.ipynb` siguen **la misma estructura**, para poder
leerlos y compararlos con facilidad:

1. **Cómo funciona** — explicación intuitiva del algoritmo.
2. **Los datos** — carga con el preprocesado compartido.
3. **Los hiperparámetros: qué controla cada uno** — tabla explicativa.
4. **Entrenamiento y evaluación (parámetros base)** — métricas + matriz de confusión y ROC.
5. **Visualización del modelo** — su estructura/parámetros (árbol, coeficientes, arquitectura…).
6. **Optimización de hiperparámetros** — Grid/RandomizedSearchCV (la red: *early stopping*).
7. **Resultado final y cuándo usar este modelo.**

### Cómo añadir un nuevo modelo

1. **Copia** `_PLANTILLA_modelo.ipynb` y renómbralo (p. ej. `10_modelo_lightgbm.ipynb`).
2. Rellena los `TODO` manteniendo las 7 secciones.
3. Reutiliza los **helpers compartidos** (no dupliques lógica):
   - `src/notebook_utils.py` → `evaluar(...)` y `plot_confusion_roc(...)`.
   - `src/model_viz.py` → `visualizar_modelo(...)` (añade el caso de tu modelo si es nuevo).

> Estos notebooks se **editan a mano** (no se generan). Mantener la consistencia es
> responsabilidad de la plantilla + esta convención + los helpers de `src/`, no de un
> generador de código.
