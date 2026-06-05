# Bonus técnicos

El enunciado (§5 del anexo) propone siete *bonus técnicos* opcionales (hasta **+2 puntos**).
Hemos implementado **seis de los siete**. Este documento los resume y explica **cómo
ejecutarlos**; el flujo de **MLflow** se detalla al final por ser el más completo.

> Todos los comandos asumen el entorno de desarrollo instalado (`make setup-dev`, o
> `pip install -e ".[train,dev]"`). Vía `make`, las variables del `.env` se cargan solas.

## Resumen

| Bonus (enunciado) | Estado | Dónde vive | Cómo ejecutarlo |
|---|---|---|---|
| 🛠 Optimización de hiperparámetros | ✅ | `ml/tuning.py` | `make tune` · `make train ARGS="--tune"` |
| ⚖️ Balanceo de clases | ✅ | `notebooks/06_balanceo_clases.ipynb` | abrir el notebook 06 |
| 🔍 Interpretabilidad | ✅ | `utils/interpretability.py` | `make explain` |
| 🌐 API REST con FastAPI | ✅ | `api/` | `make api` → <http://127.0.0.1:8000/docs> |
| 🖥 Interfaz visual | ✅ | `ui/` (Streamlit) | `make run` (API + UI) |
| 🧪 Registro de experimentos (MLflow) | ✅ | `utils/tracking.py`, `utils/register_model.py`, `api/registry.py` | ver [§ MLflow](#-mlflow--dagshub-el-flujo) |
| 🧠 Embeddings personalizados | ❌ | — | no abordado (ver nota final) |

---

## 🛠 Optimización de hiperparámetros

`GridSearchCV` sobre el árbol y `RandomizedSearchCV` sobre XGBoost, optimizando ROC-AUC por
validación cruzada. Los mejores parámetros se persisten en `outputs/best_hiperparametros.json`
y `train` los reutiliza automáticamente.

```bash
make tune                      # solo la búsqueda (bonus)
make train ARGS="--tune"       # entrena ya con los hiperparámetros óptimos
```

## ⚖️ Balanceo de clases

La clase está desbalanceada (~37 % de cancelaciones). En `06_balanceo_clases.ipynb` comparamos
**baseline vs `class_weight` vs SMOTE** (`imbalanced-learn`) sobre Regresión Logística y XGBoost,
**siempre re-muestreando solo en `train`** (el `test` queda intacto). Conclusión: el balanceo
sube el *recall* a costa de precisión, con ROC-AUC casi igual.

## 🔍 Interpretabilidad (SHAP)

Explica **por qué** decide el modelo, a nivel global (qué variables pesan) y local (una reserva
concreta), con `shap.TreeExplainer` sobre XGBoost + *permutation importance*. Genera los gráficos
en `outputs/` (resumen, *beeswarm*, *waterfall*).

```bash
make explain
```

## 🌐 API REST con FastAPI

Sirve el mejor modelo por HTTP. Endpoints: `GET /health`, `GET /model-info`, `POST /predict`,
`POST /predict/batch`.

```bash
make api        # → http://127.0.0.1:8000/docs (Swagger)
```

## 🖥 Interfaz visual (Streamlit)

Web que reúne resultados, gráficos, un formulario de **predicción que consume la API** e
interpretabilidad.

```bash
make run        # arranca API + UI juntas (Ctrl-C detiene ambas)
```

---

## 🧪 MLflow + DagsHub (el flujo)

Trazabilidad de experimentos y **Model Registry**, con servidor MLflow alojado en **DagsHub**
(gratis). Es **100 % opcional**: sin credenciales, todo sigue funcionando (el tracking se queda
en *no-op* silencioso).

### Las tres piezas

| Pieza | Fichero | Qué hace |
|---|---|---|
| **Tracking** | `utils/tracking.py` | Envoltorio fino: si faltan las variables o `mlflow`, **no-op silencioso**. `train` loguea un run `train_all_models` con params, métricas, artefactos y el modelo (con su *signature*). |
| **Registro** | `utils/register_model.py` | Registra el modelo del run en el **Model Registry** y lo pasa a `Production`. Incluye un **gate por métrica**: solo promueve si iguala o supera al modelo actual. |
| **Servir** | `api/registry.py` | La API descarga el modelo del registry por **REST puro** (`requests`), sin importar `mlflow` (~50-150 MB) para no rebasar la RAM de Render. Si falla → *fallback* al `best_model.pkl` local. |

### Configuración (una vez)

```bash
cp .env.example .env     # rellena MLFLOW_TRACKING_URI / _USERNAME / _PASSWORD (token DagsHub)
```

Vía `make`, el `.env` se carga automáticamente. Si lanzas los scripts a mano, expórtalo antes:

```bash
set -a; source .env; set +a
```

### Flujo de extremo a extremo

```
make train ARGS="--tune --register"
   │
   ├─ run "train_all_models": params + métricas + artefacto model/ (con signature)
   └─ gate: registra pontia-cancellations vN → Production SOLO si roc_auc ≥ champion

# (equivalente en dos pasos, registrando el último run manualmente)
make train ARGS="--tune"
make register-model

# La API sirve desde el registry (si no, usa el pickle local):
export MLFLOW_MODEL_URI="models:/pontia-cancellations/Production"
make api          # GET /model-info indica de dónde cargó (registry vs bundled)
```

- **`train --register`** encadena entrenamiento y registro en un solo paso, pero **validado**:
  no promueve a `Production` un modelo peor que el actual (gate por ROC-AUC).
- **`register-model`** (CLI) hace lo mismo a posteriori sobre el último run `train_all_models`.
- La **signature** del modelo (esquema de las 26 *features* crudas) se adjunta al loguearlo, para
  documentar y validar el input al servir.

### Probarlo sin DagsHub (MLflow local)

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000 &
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000 MLFLOW_TRACKING_USERNAME=x MLFLOW_TRACKING_PASSWORD=x
make train ARGS="--register"      # abre http://127.0.0.1:5000 para ver el run y la versión
```

> Nota: los *stages* de MLflow (`Production`/`Staging`) están deprecados desde 2.9 en favor de
> *aliases*; aquí se mantienen por simplicidad y compatibilidad con DagsHub.

---

## Nota: bonus no abordado

**🧠 Embeddings personalizados** (Word2Vec / TF-IDF / *embeddings* categóricos) es el único que
no implementamos: las variables de alta cardinalidad (`agent`, `country`, `company`) se tratan con
**reducción de cardinalidad supervisada** + *one-hot* (ver `ml/preprocessing.py` e informe §2.6),
suficiente para datos tabulares y coherente con el resto del *pipeline*.
