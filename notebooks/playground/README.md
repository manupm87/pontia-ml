# 🛝 Playground — aprender practicando (estilo `recursos/`)

Estos cuadernos son nuestro **banco de pruebas**. Aquí **aprendimos haciendo**:
replicamos el *look & feel*, los conceptos y la sintaxis de los notebooks de
clase (`recursos/`) aplicándolos, paso a paso, al problema real de **predicción
de cancelaciones hoteleras**.

> ⚠️ **Son autónomos**: **no** importan nada de `src/`. Todo el código vive en
> cada notebook, igual que en `recursos/`. De *generalizar* y *ordenar* lo que
> probamos aquí nació el paquete `src/ml_hotel_cancellations`. Por eso este es el
> **primer** nivel de la historia del proyecto:
>
> **`playground/` (aprender)  →  `src/` (generalizar)  →  `notebooks/` (mostrar)**

Cada cuaderno imita una **clase** del temario (salvo el `02`, que es el **puente**
entre la exploración y el modelado: aplica los hallazgos del EDA para construir el
dataset listo para entrenar):

| Notebook | Inspirado en | Qué se practica |
|---|---|---|
| [`01_eda_exploracion.ipynb`](01_eda_exploracion.ipynb) | clase 1 | EDA: carga, nulos, fugas de información, correlaciones, tasa de cancelación por categoría |
| [`02_preparacion_datos.ipynb`](02_preparacion_datos.ipynb) | — (puente) | Del crudo al dataset final: aplica los hallazgos del EDA (saneo, *features* `has_*`, reducción de cardinalidad y *one-hot* **ajustados en `train`**) y guarda `data/processed/{train,test}.csv` |
| [`03_modelos_supervisados.ipynb`](03_modelos_supervisados.ipynb) | clase 2 | Regresión logística, árbol, random forest y XGBoost; métricas, matriz de confusión, ROC, importancia, `GridSearchCV`/`RandomizedSearchCV` |
| [`04_red_neuronal.ipynb`](04_red_neuronal.ipynb) | clase 4 | Red neuronal con Keras (`Sequential`, *early stopping*, curvas de aprendizaje) |
| [`05_comparativa_y_visualizacion.ipynb`](05_comparativa_y_visualizacion.ipynb) | clases 2–3 | Comparativa de los 5 modelos (tabla + ROC superpuestas) y **visualización 2D** de sus regiones de decisión con **PLS supervisado** |
| [`06_balanceo_clases.ipynb`](06_balanceo_clases.ipynb) | clase 2 | Desbalance de clases: *baseline* vs `class_weight` vs **SMOTE** |
| [`07_interpretabilidad.ipynb`](07_interpretabilidad.ipynb) | clases 1–2 | Interpretabilidad: **SHAP** (global, *beeswarm*, *waterfall*) e importancia por permutación |

## Estilo

- **Plotly** para los gráficos interactivos (`px.imshow`, `px.bar`, `px.area`,
  `px.scatter`) con `pio.renderers.default = 'iframe'`; *matplotlib* solo para
  `ConfusionMatrixDisplay`, las curvas de Keras y los gráficos propios de SHAP.
- Codificación de categóricas con `pd.get_dummies`, partición estratificada
  (`train_test_split(..., stratify=y, random_state=42)`).
- Prosa, comentarios y títulos en **español**; identificadores en **inglés**.

## Cómo ejecutarlos

Requieren el entorno de desarrollo (`make setup-dev`, que incluye Jupyter). Desde
la raíz del repo:

```bash
.venv/bin/jupyter lab            # y abrir el notebook, o:
cd notebooks/playground && ../../.venv/bin/jupyter nbconvert --to notebook \
    --inplace --execute --ExecutePreprocessor.kernel_name=python3 03_modelos_supervisados.ipynb
```

> ℹ️ Los gráficos de Plotly usan el *renderer* `iframe` (como en `recursos/`):
> generan una carpeta `iframe_figures/` (ignorada por git) y **no** se ven en la
> previsualización de GitHub. Para verlos, ejecuta el notebook en Jupyter.
