# Plan: despliegue público + MLflow

> Documento operativo. Lo lee tanto el usuario como cualquier agente que
> retome el trabajo. Está pensado para ejecutarse por fases, marcando las
> tareas como `[x]` a medida que se completan.

## 1. Objetivo

Publicar el proyecto Pontia ML en internet **gratis**, con dos URLs:

- **`api.tudominio.com`** — la API FastAPI sirviendo el modelo (Render).
- **`tu-usuario-pontia-ml.streamlit.app`** — la interfaz Streamlit
  consumiendo esa API (Streamlit Community Cloud).

Y, a la vez, ticar el **bonus de MLflow** (registro de experimentos) usando
**DagsHub** como servidor de tracking público.

Como guinda, la API leerá el modelo desde el **MLflow Model Registry** (no
desde un `best_model.pkl` comprometido en git), demostrando el flujo
*MLOps* completo: **entrenar → registrar → promocionar → servir**.

## 2. Arquitectura de destino

```
    Portátil del usuario                                DagsHub MLflow
   ┌──────────────────────────┐                       ┌───────────────────┐
   │ python -m src.train      │  log params/metrics   │ Experimentos      │
   │ python -m src.tuning     │ ────────────────────▶│ Runs              │
   │ python -m src.balancing  │  log artefactos       │ Model Registry    │
   │ python -m src.tracking   │ ─ (registra v1, etc.) │  pontia-cancela…  │
   └──────────────────────────┘                       └───────────────────┘
                                                              │
                                            descarga al boot  │
                                                              ▼
                                                    ┌──────────────────┐
                                                    │ Render free tier │
                                                    │ FastAPI          │
                                                    │ api.tudominio… │
                                                    └──────────────────┘
                                                              ▲
                                                    HTTPS /predict
                                                              │
                                                    ┌──────────────────┐
                                                    │ Streamlit Cloud  │
                                                    │ tu-app.streamlit │
                                                    │ .app             │
                                                    └──────────────────┘
```

### Variables de entorno (resumen)

| Variable                     | Dónde se define              | Para qué sirve                                  |
|------------------------------|------------------------------|-------------------------------------------------|
| `MLFLOW_TRACKING_URI`        | local (training) + Render    | Servidor MLflow (DagsHub). Si no está, se omite |
| `MLFLOW_TRACKING_USERNAME`   | local (training) + Render    | Usuario de DagsHub                              |
| `MLFLOW_TRACKING_PASSWORD`   | local (training) + Render    | Token de DagsHub                                |
| `MLFLOW_MODEL_URI`           | Render                       | URI del modelo en el registry, p. ej.           |
|                              |                              | `models:/pontia-cancellations/Production`       |
| `PONTIA_API_URL`             | Streamlit Cloud (secrets)    | URL de la API (`https://api.tudominio.com`)     |
| `PONTIA_MODEL_PATH`          | Render (opcional)            | Si se quiere forzar leer un pkl concreto        |

## 3. Decisiones y alternativas

| Componente             | Elegido                        | Alternativas descartadas (motivo)                                  |
|------------------------|--------------------------------|--------------------------------------------------------------------|
| Hosting de la API      | **Render free**                | Fly (sin tier gratis), Cloud Run (Streamlit no encaja serverless)  |
| Hosting de la UI       | **Streamlit Community Cloud**  | HF Spaces (sin custom domain en free), Render (UI OOM en 512 MB)   |
| Tracking + Registry    | **DagsHub MLflow**             | MLflow self-host en Render (DB libre caduca a los 30 d), HF Hub    |
|                        |                                | (no trackea experimentos), local-only (no público)                 |
| Dominio personalizado  | Solo en la API                 | Streamlit Cloud no admite custom domain en el tier gratuito        |
| CORS                   | `allow_origins=["*"]` actual   | No es necesario "evitar" CORS: ya está abierto                     |

### Por qué no Hugging Face para esto

- **HF Spaces** (free): tendría más RAM (16 GB), pero **no admite custom
  domain** en el tier gratuito y, sobre todo, deja menos clara la
  separación API/Frontend que en el enunciado es bonus aparte.
- **HF Hub como registry**: es una opción de respaldo válida (push del
  pickle como repo de modelo público), pero pierde la mitad de "experiment
  tracking" del bonus MLflow. Lo dejamos como *fallback* en la cadena de
  carga del modelo (ver T15).

## 4. Mapa de tareas

> Estado: `[ ]` = pendiente, `[~]` = en curso, `[x]` = hecho.
> Las tareas están ordenadas por **dependencia**: respeta el orden salvo
> que la columna *Depende de* permita el adelanto.

### Fase 0 — Preparación del repo

- [ ] **T01 — Partir `requirements.txt` en runtime vs entrenamiento**
  *Depende de*: ninguna · *Owner*: agente
  - Crear `project/requirements-train.txt` con: `tensorflow-cpu` (o
    `tensorflow` en macOS), `imbalanced-learn`, **`mlflow>=2.16`**,
    `nbformat`, `jupyter`, `seaborn`.
  - Dejar en `project/requirements.txt` solo lo necesario para que la API
    y la UI arranquen (xgboost, sklearn, fastapi, uvicorn, streamlit,
    shap, matplotlib, plotly, joblib, requests, pandas, numpy, scipy).
  - Documentar la división en cabecera de cada fichero (qué incluye, por
    qué).
  - **Aceptación**: `pip install -r requirements.txt` en venv limpio
    instala < 400 MB y permite arrancar API y UI sin tocar `train.py`.

- [ ] **T02 — Crear `src/tracking.py`**
  *Depende de*: T01 · *Owner*: agente
  - Helper en español con dos funciones:
    - `init_tracking(experiment: str) -> bool`: si las variables
      `MLFLOW_TRACKING_URI`/`USERNAME`/`PASSWORD` están definidas,
      configura MLflow y devuelve `True`; si no, no-op y devuelve
      `False`. Idempotente.
    - `tracking_enabled() -> bool`: idem para consultar fuera del setup.
  - Importación perezosa de `mlflow` (que `import src.tracking` no
    explote si mlflow no está instalado en runtime — no lo estará en la
    API).
  - **Aceptación**: `python -c "from src.tracking import init_tracking;
    print(init_tracking('test'))"` devuelve `False` sin lanzar errores
    cuando MLflow no está instalado o las vars no están.

### Fase 1 — MLflow tracking en DagsHub

- [ ] **T03 — Crear cuenta DagsHub + mirror del repo + token**
  *Depende de*: ninguna · *Owner*: **usuario**
  - Pasos:
    1. Crear cuenta en dagshub.com (gratis).
    2. "Connect a repository" → conectar el repo de GitHub.
    3. Generar un **personal access token** en *Settings → Tokens*.
    4. Anotar la URL del tracking server: suele ser
       `https://dagshub.com/<usuario>/<repo>.mlflow`.
  - **Aceptación**: el usuario tiene a mano los 3 valores que irán a
    `MLFLOW_TRACKING_URI`, `..._USERNAME`, `..._PASSWORD`.

- [ ] **T04 — Instrumentar `src/train.py`**
  *Depende de*: T02 · *Owner*: agente
  - Llamar a `tracking.init_tracking("pontia-cancellations-train")` al
    inicio de `main()`.
  - Envolver el bucle de entrenamiento de los 5 modelos en
    `mlflow.start_run(run_name="train_all_models")` (parent run).
  - Por cada modelo: `mlflow.start_run(nested=True, run_name=name)` y
    loguear:
    - **params**: los hiperparámetros del modelo (de `config.*_PARAMS`).
    - **metrics**: las 5 del informe (accuracy, precision, recall, f1,
      roc_auc) sobre test.
    - **tags**: `model_family={logreg|tree|rf|xgb|nn}`.
    - **artefactos**: si el entrenamiento ya generó PNGs en
      `outputs/`, subirlos con `mlflow.log_artifact(...)`.
  - Para el modelo ganador: `mlflow.sklearn.log_model(pipeline,
    artifact_path="model")` (esto es lo que más adelante registraremos).
  - **Aceptación**: con las variables de entorno definidas,
    `python -m src.train` deja un parent run con 5 child runs visibles
    en DagsHub.

- [ ] **T05 — Instrumentar `src/tuning.py`**
  *Depende de*: T02 · *Owner*: agente
  - Igual que T04 pero parent run = `tuning_xgboost`, child run por cada
    combinación que prueba `RandomizedSearchCV`.
  - Loguear `best_estimator_` como modelo. Marcar el child run ganador
    con `tags={"best": True}`.
  - **Aceptación**: `python -m src.tuning` deja un nuevo experimento
    con un único parent run y `TUNING_N_ITER` child runs.

- [ ] **T06 — Instrumentar `src/balancing.py`**
  *Depende de*: T02 · *Owner*: agente
  - Parent run = `balanceo_clases`, child run por estrategia
    (`baseline`, `class_weight`, `smote`). Tags coherentes.
  - **Aceptación**: `python -m src.balancing` deja un parent run con 3
    child runs en DagsHub.

- [ ] **T07 — Ejecutar los 3 scripts y registrar el modelo v1**
  *Depende de*: T03, T04 · *Owner*: **usuario** (corre los scripts en
  local con sus credenciales DagsHub)
  - Con las 3 variables MLflow en el entorno, ejecutar en orden:
    `python -m src.train`, `python -m src.tuning`,
    `python -m src.balancing`.
  - En la UI de DagsHub: seleccionar el run XGBoost ganador → "Register
    Model" → nombre `pontia-cancellations` → versión `1`.
  - Transicionar la versión 1 a stage `Production`.
  - **Aceptación**: en `https://dagshub.com/<user>/<repo>.mlflow/
    #/models/pontia-cancellations` figura la versión 1 en `Production`.

### Fase 2 — Despliegue público de la API y la UI

- [ ] **T08 — Endurecer CORS en `api/main.py`**
  *Depende de*: ninguna · *Owner*: agente
  - Cambiar `allow_origins=["*"]` por la lista concreta:
    ```python
    allow_origins=[
        "https://tu-usuario-pontia-ml.streamlit.app",
        "https://api.tudominio.com",
        "http://localhost:8501",  # desarrollo
    ]
    ```
  - Permitir override por env var (`PONTIA_CORS_ORIGINS`, separados por
    coma) para no tener que tocar código si cambia la URL pública.
  - **Aceptación**: tests de la API (`api/tests/`) siguen pasando;
    `curl -H "Origin: https://otra.com" ... -i /predict` ya no muestra
    `Access-Control-Allow-Origin: *`.

- [ ] **T09 — Añadir `project/render.yaml`**
  *Depende de*: T01 · *Owner*: agente
  - Spec declarativo de Render (un único servicio web):
    ```yaml
    services:
      - type: web
        name: pontia-api
        runtime: python
        rootDir: project
        plan: free
        buildCommand: pip install -r requirements.txt
        startCommand: uvicorn api.main:app --host 0.0.0.0 --port $PORT
        healthCheckPath: /health
        envVars:
          - key: PYTHON_VERSION
            value: 3.12
          - key: PONTIA_CORS_ORIGINS
            sync: false        # lo pone el usuario en Render
    ```
  - **Aceptación**: render.com → "New Blueprint" usando el repo crea el
    servicio sin errores.

- [ ] **T10 — Conectar el dominio del usuario a la API**
  *Depende de*: T09 · *Owner*: **usuario**
  - En Render → service → *Custom Domains* → añadir
    `api.tudominio.com`.
  - En el proveedor DNS: CNAME `api → <servicio>.onrender.com`.
  - Esperar a que el SSL se aprovisione (~5 min). Verificar con
    `curl https://api.tudominio.com/health`.
  - **Aceptación**: `/health` responde 200 sobre el dominio del usuario
    con cert válido.

- [ ] **T11 — Tarjeta "warm up" en la UI Streamlit**
  *Depende de*: ninguna · *Owner*: agente
  - En `ui/app.py`, antes de renderizar la barra lateral, lanzar un
    `data.check_api_health()` en *fire-and-forget* para despertar al
    servicio dormido de Render.
  - Si la API tarda > 5 s en responder, mostrar un `st.info("Despertando
    la API…  (~30 s la primera vez)")` mientras tanto.
  - **Aceptación**: visitar la URL cuando la API está dormida muestra el
    aviso y, tras la espera, la app funciona.

- [ ] **T12 — Crear la app en Streamlit Community Cloud**
  *Depende de*: T01, T10 · *Owner*: **usuario**
  - share.streamlit.io → *New app*.
  - Repo + branch `main`, *Main file*: `project/ui/app.py`.
  - *Advanced settings → Secrets*:
    ```toml
    PONTIA_API_URL = "https://api.tudominio.com"
    ```
  - Python 3.12, requirements file `project/requirements.txt`.
  - **Aceptación**: la URL `tu-usuario-pontia-ml.streamlit.app` carga la
    UI, conecta con la API y permite predecir reservas.

### Fase 3 — La API lee el modelo desde el registry

- [ ] **T13 — Refactorizar `api/service.py` para cargar desde MLflow**
  *Depende de*: T07 · *Owner*: agente
  - Modificar `get_model()` para que use esta cadena de carga, en orden:
    1. Si `MLFLOW_MODEL_URI` está definida, descargar desde el registry
       a `/tmp/pontia_models/<hash(uri)>` y cargar el pickle.
    2. Si falla (cualquier excepción), loguear *warning* y caer a
       `joblib.load(get_model_path())` (el pkl comprometido en repo).
  - Cachear en `/tmp` para no re-descargar en cada warm start.
  - Hacer **lazy import** de `mlflow` dentro de la función (que la API
    pueda arrancar sin tener mlflow instalado).
  - **Aceptación**: con `MLFLOW_MODEL_URI` puesto en local funciona y
    se cachea; sin la var, sigue sirviendo el pkl bundled.

- [ ] **T14 — Enriquecer `/model-info`**
  *Depende de*: T13 · *Owner*: agente
  - Añadir al schema y al endpoint:
    - `source`: `"registry"` | `"bundled"`.
    - `registry_uri`: la URI usada cuando `source=registry`.
    - `version`: número de versión del modelo en el registry (None si
      bundled).
  - **Aceptación**: `GET /model-info` muestra el origen del modelo y la
    versión cuando aplica.

- [ ] **T15 — *Fallback chain* opcional con HF Hub**
  *Depende de*: T13 · *Owner*: agente (opcional, posponer si no da
  tiempo)
  - Antes de caer al pkl bundled, intentar `huggingface_hub.hf_hub_
    download(repo_id="<user>/pontia-cancellations", filename="best_
    model.pkl")` si `PONTIA_HF_MODEL_REPO` está definida.
  - Justificación: si DagsHub se cae, HF Hub es un segundo registry
    público que también podemos usar (capítulo "MLOps avanzado" en la
    defensa).
  - **Aceptación**: con `MLFLOW_MODEL_URI` mal formada pero
    `PONTIA_HF_MODEL_REPO` válida, la API sigue funcionando.

- [ ] **T16 — Promocionar el registry en producción de Render**
  *Depende de*: T13, T14, T10 · *Owner*: **usuario**
  - En Render → service → *Environment*, añadir:
    ```
    MLFLOW_TRACKING_URI=https://dagshub.com/<user>/pontia-ml.mlflow
    MLFLOW_TRACKING_USERNAME=<user>
    MLFLOW_TRACKING_PASSWORD=<token>
    MLFLOW_MODEL_URI=models:/pontia-cancellations/Production
    ```
  - Re-desplegar.
  - **Aceptación**: `GET https://api.tudominio.com/model-info` reporta
    `source=registry, version=1`.

### Fase 4 — Documentación

- [ ] **T17 — Actualizar `docs/informe_final.md` §6**
  *Depende de*: T07, T12, T16 · *Owner*: agente
  - Añadir subsección **6.6 Registro de experimentos con MLflow**.
  - Explicar (didáctico):
    - Qué es MLflow (tracking + registry) y por qué es un bonus.
    - Qué se rastrea (params/metrics/artefactos por modelo).
    - Cómo se sirvió el modelo desde el registry vs pickle bundled.
    - Cadena de fallback (registry → bundled → opcionalmente HF Hub).
  - Añadir URLs:
    - DagsHub MLflow: `https://dagshub.com/<user>/<repo>.mlflow`.
    - API pública: `https://api.tudominio.com/docs`.
    - UI pública: `https://tu-app.streamlit.app`.
  - **Aceptación**: subsección presente con todos los enlaces vivos.

- [ ] **T18 — Actualizar `project/README.md` y `CLAUDE.md`**
  *Depende de*: T17 · *Owner*: agente
  - En `README.md`: bloque "Despliegue público" con las 3 URLs.
  - En `CLAUDE.md`: nuevo *gotcha* sobre la cadena de carga del modelo
    (registry → bundled) y la división de requirements.
  - **Aceptación**: ambos documentos coherentes con el estado real.

- [ ] **T19 — Marcar bonus MLflow como cumplido en §7 del informe**
  *Depende de*: T17 · *Owner*: agente
  - En la subsección de "líneas de mejora", eliminar el bullet "Registro
    de experimentos con MLflow (otro bonus pendiente)" y, si procede,
    moverlo a "bonus técnicos implementados".
  - **Aceptación**: el informe ya no cita MLflow como pendiente.

## 5. Convenciones para los agentes

Cualquier agente que recoja una tarea debe cumplir las reglas de
`CLAUDE.md`:

- Docs, comentarios y prosa de notebooks en **español didáctico**.
- Usar el venv `project/.venv` (Python 3.12). Nada de crear otros venvs.
- Commits en español con la coletilla:
  `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.
- **No** hacer push a `main` salvo que el usuario lo pida.
- `gh` no está disponible: los PRs se abren por la web.

### Cómo verificar el final feliz

```bash
# Local (con MLflow): un nuevo run aparece en DagsHub
MLFLOW_TRACKING_URI=... MLFLOW_TRACKING_USERNAME=... \
  MLFLOW_TRACKING_PASSWORD=... \
  python -m src.train

# API en producción carga del registry
curl https://api.tudominio.com/model-info | jq
# → {"source":"registry","registry_uri":"models:/...","version":1, ...}

# UI llama a la API
open https://tu-usuario-pontia-ml.streamlit.app
# rellenar el formulario → ver predicción + SHAP + mapa 2D
```

### Cómo retomar si una tarea quedó a medias

1. Lee el estado de la tarea aquí (busca `[~]`).
2. Lee los ficheros que tocaba (la sección los enumera).
3. Continúa donde se dejó. Si no está claro, consulta el commit más
   reciente que mencione la tarea.
4. Cuando termines, marca `[x]` y haz un commit pequeño que cite el ID
   (`feat(T05): instrumentar tuning con MLflow…`).
