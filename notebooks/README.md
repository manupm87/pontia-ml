# 📓 Notebooks

Los cuadernos del proyecto cuentan una **historia en dos niveles**:

```
playground/  (aprender)   →   src/  (generalizar)   →   notebooks/  (mostrar)
```

1. **`playground/` — aprender practicando.** Cuadernos **autónomos** (no importan
   `src/`) que replican el estilo de las clases (`recursos/`) sobre el problema
   real. Es donde experimentamos. Ver [`playground/README.md`](playground/README.md).
2. **`src/ml_hotel_cancellations` — generalizar.** Lo que funcionó en el
   *playground* lo ordenamos y lo convertimos en un **paquete instalable**: la
   *fuente única de verdad* (carga, preprocesado, entrenamiento, inferencia…).
3. **Estos notebooks — mostrar.** Usan el paquete `src/` para **enseñar el
   pipeline de producción** de principio a fin: del dato a la predicción servida.

Así, lo que ves en estos cuadernos finales es **exactamente** lo que ejecutan la
API y la interfaz: mismos datos, mismo preprocesado, mismo modelo. No hay *drift*.

## Los cuadernos finales (sobre `src/`)

| Notebook | Usa de `src` | Qué muestra |
|---|---|---|
| [`01_eda.ipynb`](01_eda.ipynb) | `config`, `data_loader.load_raw_data` | EDA que **justifica** las decisiones del pipeline (fugas, desbalance ~37 %, exclusión de `arrival_date_year`) |
| [`02_entrenamiento_y_comparativa.ipynb`](02_entrenamiento_y_comparativa.ipynb) | `train.run_pipeline`, `evaluator`, `visualization_2d` | Entrena y selecciona los 5 modelos; tabla comparativa, curvas ROC y regiones de decisión 2D (PLS). Gana **XGBoost** (ROC-AUC ≈ 0.9614) |
| [`03_balanceo_clases.ipynb`](03_balanceo_clases.ipynb) | `balancing.compare` | Comparativa *baseline* / `class_weight` / SMOTE y por qué producción **no** balancea |
| [`04_interpretabilidad_shap.ipynb`](04_interpretabilidad_shap.ipynb) | `utils.interpretability`, `predict.load_best_model` | SHAP (global, *beeswarm*, *waterfall*) e importancia por permutación del modelo ganador |
| [`05_inferencia.ipynb`](05_inferencia.ipynb) | `predict.load_best_model`, `predict.predict_dataframe` | Inferencia con `models/best_model.pkl`, igual que la API y la UI |

Orden de lectura recomendado: **01 → 02 → 03 → 04 → 05**.

## Estilo (nivel final)

- **matplotlib/seaborn** (no Plotly): gráficos estáticos que se ven bien en GitHub.
- Se importa **solo el contrato** de `src` (`config`, `load_and_prepare`,
  `build_preprocessor`, `run_pipeline`, `predict_dataframe`…); la evaluación y la
  presentación quedan **inline y visibles** en el notebook.
- Prosa y comentarios en **español**; identificadores en **inglés**. Términos
  técnicos explicados; glosario en [`../docs/glosario.md`](../docs/glosario.md).

## Cómo ejecutarlos

Requieren el entorno de desarrollo y el paquete instalado en editable:

```bash
make setup-dev                 # crea .venv e instala [train,dev] (incluye Jupyter)
.venv/bin/jupyter lab          # abrir y ejecutar, o de forma headless:
cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --inplace \
    --execute --ExecutePreprocessor.kernel_name=python3 02_entrenamiento_y_comparativa.ipynb
```

`02_entrenamiento_y_comparativa.ipynb` reentrena los modelos (vía
`run_pipeline`) y regenera `models/` y `outputs/`; el resto reutiliza esos
artefactos. Para regenerarlos desde la CLI: `make train` (+ `make viz2d`,
`make explain`).
