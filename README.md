# 🏨 Predicción de cancelaciones de reservas hoteleras

> Sistema automático que **entrena, evalúa y compara** varios modelos de _Machine
> Learning_ para predecir si una reserva de hotel se cancelará. Entrega final del
> módulo de **Machine Learning y Deep Learning** (Máster en IA, Cloud Computing y
> DevOps).

**¿Qué hace este proyecto, en una frase?** A partir de los datos de una reserva
(antelación, tipo de hotel, país del cliente, precio, etc.) estima la
**probabilidad de que esa reserva se cancele**.

---

## 📋 Resumen del proyecto

**El problema:** Las cancelaciones dejan habitaciones vacías sin tiempo de revenderlas. Prediciendo el riesgo con antelación, el hotel puede actuar (overbooking controlado, depósito, incentivos).

**Los datos:** 119 390 reservas × 32 columnas (City Hotel y Resort Hotel, Portugal 2015–2017). Tras limpieza: **94 850** filas de entrenamiento y **23 713** de test. El pipeline acepta **26 características** (15 numéricas + 11 categóricas), deriva 3 adicionales y llega a **144 columnas** tras one-hot encoding. Clases desbalanceadas (~37 % cancelaciones).

**Lo que está implementado:**

- 5 modelos: Regresión logística, Árbol de decisión, Random Forest, XGBoost, Red neuronal (Keras)
- Búsqueda de hiperparámetros: GridSearchCV (LR, árbol) y RandomizedSearchCV (RF, XGBoost)
- API REST con FastAPI (`/predict`, `/predict/batch`, `/model-info`, `/health`)
- Interfaz web con Streamlit (predicción, métricas, gráficos, SHAP)
- Tracking de experimentos con MLflow + DagsHub
- Interpretabilidad con SHAP (global y local)
- Suite de 65 tests (pipeline, API, UI, contract tests)

**Mejor resultado:** XGBoost — ROC-AUC **0.9529** en test.

> 📖 Términos → [glosario.md](docs/glosario.md) · Sistema → [arquitectura.md](docs/arquitectura.md) · Análisis completo → [informe_final.md](docs/informe_final.md)

---

## 🌐 Demo en vivo

| Componente                            | URL                                                     | Tier                             |
| ------------------------------------- | ------------------------------------------------------- | -------------------------------- |
| 🖥️ **Interfaz web** (Streamlit)       | <https://ml-hotel-cancellations-manupm87.streamlit.app> | Streamlit Community Cloud (free) |
| 🔌 **API REST** (FastAPI + Swagger)   | <https://pontia-api-fi8t.onrender.com/docs>             | Render (free)                    |
| 🧪 **Experimentos MLflow + Registry** | <https://dagshub.com/manupm87/pontia-ml.mlflow>         | DagsHub (free)                   |

> ⏳ La API se duerme tras 15 min sin uso (tier gratis de Render). La primera
> petición tarda ~30-50 s en despertarla; la UI lo indica con un aviso amable.

### Arquitectura, de un vistazo

```mermaid
flowchart LR
    classDef dev fill:#e6f7ff,stroke:#1890ff,color:#003a8c
    classDef ml fill:#f9f0ff,stroke:#722ed1,color:#22075e
    classDef repo fill:#e6fffb,stroke:#13c2c2,color:#002766
    classDef cloud fill:#fff1f0,stroke:#cf1322,color:#5c0011
    classDef user fill:#f6ffed,stroke:#52c41a,color:#135200

    DEV["Entrenamiento<br/>(local · Python)"]:::dev
    MLF[("MLflow / DagsHub<br/>experimentos + registry")]:::ml
    REPO[/"GitHub<br/>manupm87/pontia-ml"/]:::repo
    API["FastAPI<br/>Render"]:::cloud
    UI["Streamlit<br/>Streamlit Cloud"]:::cloud
    USER([Usuario]):::user

    DEV --> MLF
    DEV -- "best_model.pkl" --> REPO
    REPO -- auto-deploy --> API
    REPO -- auto-deploy --> UI
    USER --> UI
    UI -- "POST /predict" --> API
```

> Arquitectura completa: [`docs/arquitectura.md`](docs/arquitectura.md) · Términos: [`docs/glosario.md`](docs/glosario.md)

---

## 👥 Autores

| Autor                                         | Rol principal                                      |
| --------------------------------------------- | -------------------------------------------------- |
| **Manuel Pérez** (manugijon@gmail.com)        | Arquitectura src/, entrenamiento, API, UI, memoria |
| **Joaquin Castro** (jcastrosalas03@gmail.com) | EDA, preprocesado, pruebas de modelos              |

> Reparto detallado de tareas: [`docs/informe_final.md`](docs/informe_final.md)

---

## 🗂️ Estructura

```text
pontia-ml/
├── data/raw/               # Dataset original
├── docs/                   # arquitectura.md · glosario.md · informe_final.md
├── models/best_model.pkl   # Mejor modelo entrenado
├── notebooks/              # 01_eda … 07_interpretabilidad (autónomos)
├── outputs/                # Métricas, gráficos e hiperparámetros
├── src/ml_hotel_cancellations/
│   ├── ml/        # train · predict · tuning · preprocessing · evaluate
│   ├── api/       # FastAPI: main · schemas · service · registry
│   ├── ui/        # Streamlit: app · secciones
│   └── utils/     # SHAP · MLflow · reporting
├── tests/          # 65 tests
└── pyproject.toml  # dependencias y config de pytest
```

---

## ⚙️ Instalación

Requiere **Python 3.11 ó 3.12**. Python 3.13 no soportado aún.

```bash
git clone https://github.com/manupm87/pontia-ml.git && cd pontia-ml
python3 -m venv .venv
source .venv/bin/activate       # Linux/macOS · Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .                 # API + UI + inferencia
pip install -e ".[train,dev]"   # + entrenamiento, MLflow y tests
```

> macOS + XGBoost: si falla `libomp.dylib` → `brew install libomp`

---

## ▶️ Ejecución

Desde la raíz del repo con el venv activado. En Linux/macOS hay atajos `make` (`make help`).

### 1. Entrenar y comparar todos los modelos (proceso completo)

```bash
python -m ml_hotel_cancellations.ml.train            # entrena 5 modelos, guarda best_model.pkl
python -m ml_hotel_cancellations.ml.train --tune     # rehace búsqueda de hiperparámetros antes
python -m ml_hotel_cancellations.ml.tuning           # solo la búsqueda (GridSearchCV/RandomizedSearchCV)
```

### 2. Predecir con el mejor modelo

```bash
# Demostración rápida: usa 10 reservas de ejemplo del propio dataset
python -m ml_hotel_cancellations.ml.predict --sample 10

# Sobre tu propio fichero CSV de reservas
python -m ml_hotel_cancellations.ml.predict --input mis_reservas.csv --output predicciones.csv
```

### 3. Abrir los notebooks (para explorar y aprender)

```bash
# Registrar el entorno como "kernel" de Jupyter (solo la primera vez)
python -m ipykernel install --user --name pontia-ml --display-name "Python (pontia-ml)"
jupyter lab
```

Los notebooks están en `notebooks/` ([`notebooks/README.md`](notebooks/README.md)): EDA, preparación, modelos supervisados, red neuronal, comparativa, balanceo e interpretabilidad. Son autónomos (no importan `src`).

### 4. Interpretabilidad del modelo con SHAP (bonus)

Explica **por qué** el modelo decide, a nivel global (qué variables pesan más) y
local (una reserva concreta). Genera los gráficos en `outputs/`:

```bash
python -m ml_hotel_cancellations.utils.interpretability
```

Exploración inicial en [`notebooks/07_interpretabilidad.ipynb`](notebooks/07_interpretabilidad.ipynb).

### 5. API REST con FastAPI (bonus)

Sirve el mejor modelo por HTTP para consumirlo desde otros sistemas:

```bash
uvicorn ml_hotel_cancellations.api.main:app --reload      # desde la raíz del repo
```

Abre la documentación interactiva en <http://127.0.0.1:8000/docs>. Endpoints:
`GET /health`, `GET /model-info`, `POST /predict`, `POST /predict/batch`. Ejemplo:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"hotel":"City Hotel","lead_time":100,"arrival_date_month":"August","arrival_date_week_number":33,"arrival_date_day_of_month":15,"stays_in_weekend_nights":2,"stays_in_week_nights":5,"adults":2,"children":0,"babies":0,"meal":"BB","country":"PRT","market_segment":"Online TA","distribution_channel":"TA/TO","is_repeated_guest":0,"previous_cancellations":0,"previous_bookings_not_canceled":0,"reserved_room_type":"A","booking_changes":0,"deposit_type":"No Deposit","agent":"9","company":"no_company","days_in_waiting_list":0,"customer_type":"Transient","adr":100.0,"total_of_special_requests":1}'
```

### 6. Interfaz visual con Streamlit (bonus)

Una web que reúne todo: resultados, gráficos de los modelos, un formulario de
**predicción que consume la API**, interpretabilidad (SHAP) y exploración:

```bash
uvicorn ml_hotel_cancellations.api.main:app --reload      # 1) en una terminal: la API (para predecir)
streamlit run src/ml_hotel_cancellations/ui/app.py            # 2) en otra terminal: la interfaz
```

La URL de la API se configura con la variable `PONTIA_API_URL` (por defecto `http://localhost:8000`).

### 7. MLflow + DagsHub (bonus)

```bash
cp .env.example .env && $EDITOR .env   # añade token DagsHub (scope: mlflow)
set -a; source .env; set +a

python -m ml_hotel_cancellations.ml.train             # loguea a DagsHub automáticamente
python -m ml_hotel_cancellations.utils.register_model # registra el ganador en el Model Registry

# API sirve desde el registry:
export MLFLOW_MODEL_URI="models:/pontia-cancellations/Production"
uvicorn ml_hotel_cancellations.api.main:app --reload
```

Sin las variables de entorno los scripts funcionan igual (no-op silencioso). Si falla la descarga, la API cae al pickle local.

### 8. Tests

```bash
.venv/bin/python -m pytest                          # suite completa (65 tests)
.venv/bin/python -m pytest -m "not slow"            # sin tests lentos
.venv/bin/python -m pytest tests/test_contracts.py  # contract tests
```

---

## 📊 Resultados

| Modelo               | Accuracy | Precision | Recall |   F1   | **ROC-AUC** |
| -------------------- | :------: | :-------: | :----: | :----: | :---------: |
| **XGBoost** ⭐       |  0.8811  |  0.8593   | 0.8141 | 0.8361 | **0.9529**  |
| Red neuronal (Keras) |  0.8614  |  0.8450   | 0.7691 | 0.8053 |   0.9353    |
| Random Forest        |  0.8562  |  0.8719   | 0.7198 | 0.7886 |   0.9338    |
| Árbol de decisión    |  0.8455  |  0.8219   | 0.7473 | 0.7828 |   0.9235    |
| Regresión logística  |  0.8031  |  0.7233   | 0.7636 | 0.7429 |   0.8862    |

⭐ **Mejor modelo: XGBoost** (ROC-AUC = 0.9529, hiperparámetros optimizados). Se guarda como `models/best_model.pkl`. Métricas sobre el 20 % de test. ROC-AUC como métrica principal por ser robusta ante el desbalance de clases e independiente del umbral. Ver [glosario](docs/glosario.md) para definiciones.

<p align="center">
  <img src="outputs/roc_curves.png" width="48%" alt="Curva ROC comparativa">
  <img src="outputs/confusion_matrix_best.png" width="48%" alt="Matriz de confusión del mejor modelo">
</p>
<p align="center">
  <img src="outputs/feature_importance.png" width="70%" alt="Importancia de variables">
</p>

---

## 🧰 Tecnologías

Python 3.11–3.12 · scikit-learn · XGBoost · Keras/TensorFlow · FastAPI · Streamlit · MLflow · SHAP · pandas · NumPy · pytest. Versiones exactas en [`requirements.txt`](requirements.txt).
