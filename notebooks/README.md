# 📓 Notebooks

Los cuadernos del proyecto cuentan una **historia en tres niveles**:

```
playground/  (aprender)   →   src/  (generalizar)   →   API + UI  (mostrar)
```

1. **`playground/` — aprender practicando.** Cuadernos **autónomos** (no importan
   `src/`) que replican el estilo de las clases (`recursos/`) sobre el problema
   real. Es donde experimentamos.
2. **`src/ml_hotel_cancellations` — generalizar.** Lo que funcionó en el
   _playground_ lo ordenamos y lo convertimos en un **paquete instalable**: la
   _fuente única de verdad_ (carga, preprocesado, entrenamiento, inferencia…).
3. **La API + la interfaz — mostrar.** El resultado consolidado ya no se enseña en
   cuadernos finales, sino que se **sirve y se demuestra en vivo**: una API REST
   (FastAPI) que sirve el mejor modelo y una interfaz Streamlit que reúne
   resultados, gráficos, exploración, interpretabilidad (SHAP) y un formulario de
   **predicción que consume la API**.

Así, lo que se ve en la web es **exactamente** lo que se entrenó: mismos datos,
mismo preprocesado, mismo modelo. No hay _drift_.

---

## 🛝 Playground — aprender practicando (estilo `recursos/`)

Estos cuadernos son el **banco de pruebas** del proyecto. Aquí **aprendimos
haciendo**: replicamos el _look & feel_, los conceptos y la sintaxis de los
notebooks de clase (`recursos/`) aplicándolos, paso a paso, al problema real de
**predicción de cancelaciones hoteleras**.

> ⚠️ **Son autónomos**: **no** importan nada de `src/`. Todo el código vive en
> cada notebook, igual que en `recursos/`. De _generalizar_ y _ordenar_ lo que
> probamos aquí nació el paquete `src/ml_hotel_cancellations`.

Cada cuaderno imita una **clase** del temario (salvo el `02`, que es el **puente**
entre la exploración y el modelado: aplica los hallazgos del EDA para construir el
dataset listo para entrenar):

| Notebook                                                                                             | Inspirado en | Qué se practica                                                                                                                                                 |
| ---------------------------------------------------------------------------------------------------- | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`playground/01_eda_exploracion.ipynb`](playground/01_eda_exploracion.ipynb)                         | clase 1      | EDA: carga, nulos, fugas de información, correlaciones, tasa de cancelación por categoría                                                                       |
| [`playground/02_preparacion_datos.ipynb`](playground/02_preparacion_datos.ipynb)                     | — (puente)   | Del crudo al dataset final: saneo, _features_ `has_*`, reducción de cardinalidad y _one-hot_ **ajustados en `train`**; guarda `data/processed/{train,test}.csv` |
| [`playground/03_modelos_supervisados.ipynb`](playground/03_modelos_supervisados.ipynb)               | clase 2      | Regresión logística, árbol, random forest y XGBoost; métricas, matriz de confusión, ROC, importancia, `GridSearchCV`/`RandomizedSearchCV`                       |
| [`playground/04_red_neuronal.ipynb`](playground/04_red_neuronal.ipynb)                               | clase 4      | Red neuronal con Keras (`Sequential`, _early stopping_, curvas de aprendizaje)                                                                                  |
| [`playground/05_comparativa_y_visualizacion.ipynb`](playground/05_comparativa_y_visualizacion.ipynb) | clases 2–3   | Comparativa de los 5 modelos (tabla + ROC superpuestas) y **visualización 2D** con **PLS supervisado**                                                          |
| [`playground/06_balanceo_clases.ipynb`](playground/06_balanceo_clases.ipynb)                         | clase 2      | Desbalance de clases: _baseline_ vs `class_weight` vs **SMOTE**                                                                                                 |
| [`playground/07_interpretabilidad.ipynb`](playground/07_interpretabilidad.ipynb)                     | clases 1–2   | Interpretabilidad: **SHAP** (global, _beeswarm_, _waterfall_) e importancia por permutación                                                                     |

### Estilo del playground

- **Plotly** para gráficos interactivos (`px.imshow`, `px.bar`, `px.area`, `px.scatter`)
  con `pio.renderers.default = 'iframe'`; _matplotlib_ solo para `ConfusionMatrixDisplay`,
  curvas de Keras y gráficos propios de SHAP.
- Codificación de categóricas con `pd.get_dummies`, partición estratificada
  (`train_test_split(..., stratify=y, random_state=42)`).
- Prosa, comentarios y títulos en **español**; identificadores en **inglés**.
  Términos técnicos explicados; glosario en [`../docs/glosario.md`](../docs/glosario.md).

> ℹ️ Los gráficos de Plotly usan el _renderer_ `iframe`: generan una carpeta
> `iframe_figures/` (ignorada por git) y **no** se ven en la previsualización de
> GitHub. Para verlos se ha de ejecutar el notebook en Jupyter.

---

## 🚀 Cómo ejecutar

### Prerrequisitos

```bash
make setup-dev        # crea .venv e instala dependencias [train,dev] (incluye Jupyter)
# o bien:
pip install -e ".[train,dev]"
```

### Notebooks del playground

```bash
.venv/bin/jupyter lab
# o ejecutar directamente desde línea de comandos:
cd notebooks/playground && ../../.venv/bin/jupyter nbconvert \
    --to notebook --inplace --execute \
    --ExecutePreprocessor.kernel_name=python3 03_modelos_supervisados.ipynb
```

### API + interfaz Streamlit

Desde la raíz del repo, con el entorno instalado (`make setup` o `pip install -e .`):

```bash
uvicorn ml_hotel_cancellations.api.main:app --reload      # API en :8000 (/docs)
streamlit run src/ml_hotel_cancellations/ui/app.py        # interfaz Streamlit
```

O todo a la vez con:

```bash
make run
```

Los gráficos y tablas que muestra la interfaz se regeneran desde la CLI:

```bash
make train      # entrena y guarda el modelo
make viz2d      # genera visualización 2D de regiones de decisión
make explain    # genera gráficos SHAP
```
