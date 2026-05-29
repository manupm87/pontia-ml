# Reestructuración a src-layout (`ml_hotel_cancellations`) — Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrar `src/` + `api/` + `ui/` (planos en la raíz) a un único paquete instalable `ml_hotel_cancellations` con subpaquetes `ml/`, `api/`, `ui/`, `utils/` bajo `src/`, consolidando dependencias en `pyproject.toml`, sin perder funcionalidad y con los 62 tests en verde.

**Architecture:** src-layout PyPA. `config.py` en la raíz del paquete (fuente única). `ml/` = pipeline (entrenamiento+inferencia+experimentos). `api/` = FastAPI (con cliente MLflow extraído a `registry.py`). `ui/` = Streamlit. `utils/` = transversales (reporting, viz, SHAP, MLflow infra). Imports absolutos entre subpaquetes, relativos dentro del mismo subpaquete.

**Tech Stack:** Python 3.12, setuptools/pyproject, pytest, FastAPI, Streamlit, scikit-learn/XGBoost.

**Naturaleza del cambio:** es un *rename* atómico. Las Tareas 2-6 forman el movimiento estructural y la suite **solo vuelve a verde al final de la Tarea 6**. A partir de la Tarea 7 cada tarea queda verde. Se ejecuta en la rama actual `refactor/quality-tests-and-gpu-removal`.

**Tabla maestra de reescritura de imports** (se usa en todas las tareas):

| Import antiguo | Import nuevo |
|---|---|
| `from src import config` | `from ml_hotel_cancellations import config` |
| `from src.predict import X` | `from ml_hotel_cancellations.ml.predict import X` |
| `from src.visualization_2d import X` | `from ml_hotel_cancellations.utils.visualization_2d import X` |
| `from src.interpretability import X` | `from ml_hotel_cancellations.utils.interpretability import X` |
| `from src import <ml_mod>` (data_loader, train, tuning, balancing…) | `from ml_hotel_cancellations.ml import <ml_mod>` |
| `from api.main import app` | `from ml_hotel_cancellations.api.main import app` |
| `from api.schemas import X` | `from ml_hotel_cancellations.api.schemas import X` |
| `from ui import X` / `from ui.sections import X` | `from ml_hotel_cancellations.ui[...] import X` |
| `from .X import` **dentro del mismo subpaquete** | **se mantiene relativo** |
| `from . import config` (en módulos que pasan a `ml/` o `utils/`) | `from ml_hotel_cancellations import config` |
| `from . import tracking` (en `ml/`) | `from ml_hotel_cancellations.utils import tracking` |
| `from .reporting import X` (en `ml/`) | `from ml_hotel_cancellations.utils.reporting import X` |
| `from .data_loader/.model_factory/.preprocessing import X` (en `utils/`) | `from ml_hotel_cancellations.ml.<mod> import X` |

Verificación global (debe quedar VACÍA al final): `grep -rnE "from (src|api|ui)[ .]|^import (src|api|ui)\b|python -m src\." --include=*.py src/ tests/ conftest.py`

---

### Task 1: `pyproject.toml` + esqueleto del paquete

**Files:**
- Create: `pyproject.toml`
- Create: `src/ml_hotel_cancellations/__init__.py`, `src/ml_hotel_cancellations/{ml,api,ui,utils}/__init__.py`, `src/ml_hotel_cancellations/ui/sections/__init__.py`
- Delete (al final de la tarea): `pytest.ini`

- [ ] **Step 1: Crear `pyproject.toml`** con el siguiente contenido exacto:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ml-hotel-cancellations"
version = "1.0.0"
description = "Predicción de cancelaciones de reservas hoteleras (proyecto final ML)."
requires-python = ">=3.11,<3.13"
dependencies = [
    "numpy>=1.26,<2.0",
    "pandas>=2.2,<3.0",
    "scikit-learn>=1.5,<1.7",
    "scipy>=1.13,<1.15",
    "joblib>=1.4",
    "xgboost>=2.1,<3.3",
    "shap>=0.46,<0.50",
    "numba>=0.59,<0.63",
    "llvmlite>=0.42,<0.46",
    "matplotlib>=3.8,<3.11",
    "plotly>=5.20",
    "fastapi>=0.110,<0.120",
    "uvicorn[standard]>=0.30",
    "requests>=2.32",
    "streamlit>=1.36,<1.50",
]

[project.optional-dependencies]
train = [
    "tensorflow-cpu==2.16.2; sys_platform != 'darwin'",
    "tensorflow==2.16.2; sys_platform == 'darwin'",
    "keras>=3.3,<3.6",
    "imbalanced-learn>=0.12,<0.14",
    "seaborn>=0.13",
    "mlflow>=2.16,<3.0",
    "jupyter>=1.0",
    "ipykernel>=6.29",
    "nbformat>=5.10",
    "nbconvert>=7.16",
    "notebook>=7.2",
]
dev = ["pytest>=8.0", "httpx>=0.27"]

[project.scripts]
train = "ml_hotel_cancellations.ml.train:main"
predict = "ml_hotel_cancellations.ml.predict:main"
tune = "ml_hotel_cancellations.ml.tuning:main"
balance = "ml_hotel_cancellations.ml.balancing:main"
register-model = "ml_hotel_cancellations.utils.register_model:main"
explain = "ml_hotel_cancellations.utils.interpretability:main"
viz2d = "ml_hotel_cancellations.utils.visualization_2d:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
markers = [
    "slow: tests que cargan el modelo bundled o son lentos por otros motivos.",
    "integration: tests que ejercitan varios componentes a la vez.",
]
filterwarnings = [
    "ignore::DeprecationWarning",
    "ignore::FutureWarning",
]
```

- [ ] **Step 2: Crear los directorios y `__init__.py`**

Run:
```bash
mkdir -p src/ml_hotel_cancellations/ml src/ml_hotel_cancellations/api \
         src/ml_hotel_cancellations/ui/sections src/ml_hotel_cancellations/utils
touch src/ml_hotel_cancellations/ml/__init__.py \
      src/ml_hotel_cancellations/api/__init__.py \
      src/ml_hotel_cancellations/ui/__init__.py \
      src/ml_hotel_cancellations/ui/sections/__init__.py \
      src/ml_hotel_cancellations/utils/__init__.py
printf '"""Paquete del proyecto: predicción de cancelaciones de reservas hoteleras."""\n\n__version__ = "1.0.0"\n' > src/ml_hotel_cancellations/__init__.py
```

- [ ] **Step 3: Eliminar `pytest.ini`** (su contenido ya está en `pyproject.toml`)

Run: `git rm pytest.ini`

- [ ] **Step 4: Instalar el paquete editable**

Run: `.venv/bin/pip install -e ".[train,dev]"`
Expected: `Successfully installed ml-hotel-cancellations-1.0.0` (sin errores de resolución).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/ml_hotel_cancellations/
git rm pytest.ini
git commit -m "build: pyproject.toml + esqueleto del paquete ml_hotel_cancellations"
```

---

### Task 2: Mover `config.py` y los módulos `ml/`

**Files (git mv):** `src/config.py` → `src/ml_hotel_cancellations/config.py`; `src/{data_loader,preprocessing,model_factory,model_trainer,evaluator,train,predict,tuning,balancing}.py` → `src/ml_hotel_cancellations/ml/`

- [ ] **Step 1: Mover los ficheros**

```bash
git mv src/config.py src/ml_hotel_cancellations/config.py
for m in data_loader preprocessing model_factory model_trainer evaluator train predict tuning balancing; do
  git mv src/$m.py src/ml_hotel_cancellations/ml/$m.py
done
```

- [ ] **Step 2: Arreglar las rutas en `config.py`**

En `src/ml_hotel_cancellations/config.py`, cambiar el cálculo de la raíz (el módulo bajó un nivel):

```python
# ANTES:  PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
# DESPUÉS:
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
```

- [ ] **Step 3: Reescribir imports en los 9 módulos `ml/`** según la tabla maestra. Reglas concretas:
  - `from . import config` → `from ml_hotel_cancellations import config`
  - `from . import config, tracking` (train, tuning) → dos líneas: `from ml_hotel_cancellations import config` + `from ml_hotel_cancellations.utils import tracking`
  - `from .reporting import df_to_markdown` / `save_figure` → `from ml_hotel_cancellations.utils.reporting import ...`
  - imports entre módulos de `ml/` (`from .data_loader import`, `from .preprocessing import`, `from .model_factory import`, `from .evaluator import`, `from .model_trainer import`, `from .tuning import`) → **se mantienen relativos**.
  - `configure_logging` ya vive en `config`: cualquier `from .train import configure_logging` (en tuning/balancing) → usar `config.configure_logging()` con `from ml_hotel_cancellations import config`.

- [ ] **Step 4: Verificar imports residuales en `ml/`**

Run: `grep -rnE "from (src|api|ui)[ .]|from \.train import|from \. import config" src/ml_hotel_cancellations/`
Expected: VACÍO (solo deben quedar relativos válidos entre módulos `ml/`).
Nota: la suite aún NO pasa (api/ui siguen apuntando a `src`). Es esperado.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor: mover config + pipeline ml/ al paquete"
```

---

### Task 3: Mover los módulos `utils/`

**Files (git mv):** `src/{reporting,visualization_2d,interpretability,tracking,register_model}.py` → `src/ml_hotel_cancellations/utils/`

- [ ] **Step 1: Mover los ficheros**

```bash
for m in reporting visualization_2d interpretability tracking register_model; do
  git mv src/$m.py src/ml_hotel_cancellations/utils/$m.py
done
```

- [ ] **Step 2: Reescribir imports en `utils/`**
  - `from . import config` → `from ml_hotel_cancellations import config`
  - `from .data_loader import` / `.preprocessing` / `.model_factory` / `.predict` → `from ml_hotel_cancellations.ml.<mod> import ...`
  - imports entre módulos de `utils/` (p. ej. `from .reporting import save_figure` en visualization_2d/interpretability) → **relativos**.

- [ ] **Step 3: Arreglar rutas `__file__` en `utils/` si las hay**

Run: `grep -rn "Path(__file__)" src/ml_hotel_cancellations/utils/`
Para cada coincidencia, recalcular el nivel: estos módulos están en `…/utils/`, dos niveles bajo el paquete. Si construían rutas relativas a la raíz del repo vía `parents[N]`, sumar 1 al índice respecto al original (estaban en `src/`). (En la práctica usan `config.OUTPUTS_DIR`, así que probablemente no haya ninguna.)

- [ ] **Step 4: Verificar**

Run: `grep -rnE "from (src|api|ui)[ .]|from \. import config" src/ml_hotel_cancellations/utils/`
Expected: VACÍO.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor: mover módulos transversales a utils/"
```

---

### Task 4: Mover `api/`

**Files (git mv):** `api/{__init__,main,schemas,service}.py` → `src/ml_hotel_cancellations/api/`

- [ ] **Step 1: Mover (conservando el `__init__.py` ya creado en Task 1)**

```bash
git mv api/main.py src/ml_hotel_cancellations/api/main.py
git mv api/schemas.py src/ml_hotel_cancellations/api/schemas.py
git mv api/service.py src/ml_hotel_cancellations/api/service.py
git rm api/__init__.py   # ya existe uno nuevo en el paquete
```

- [ ] **Step 2: Reescribir imports en `api/`**
  - `from src import config` → `from ml_hotel_cancellations import config`
  - `from src.predict import load_best_model, predict_dataframe` (service.py) → `from ml_hotel_cancellations.ml.predict import load_best_model, predict_dataframe`
  - `from . import service`, `from .schemas import ...` (main.py) → **se mantienen relativos**.

- [ ] **Step 3: Verificar**

Run: `grep -rnE "from (src|api|ui)[ .]" src/ml_hotel_cancellations/api/`
Expected: VACÍO.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor: mover api/ al paquete"
```

---

### Task 5: Mover `ui/`

**Files (git mv):** `ui/{app,config,data,booking,layout}.py` + `ui/sections/*.py` → `src/ml_hotel_cancellations/ui/`

- [ ] **Step 1: Mover**

```bash
for m in app config data booking layout; do git mv ui/$m.py src/ml_hotel_cancellations/ui/$m.py; done
for s in eda resumen visualizaciones prediccion interpretabilidad; do
  git mv ui/sections/$s.py src/ml_hotel_cancellations/ui/sections/$s.py
done
git rm ui/__init__.py ui/sections/__init__.py   # ya existen nuevos
git mv ui/README.md src/ml_hotel_cancellations/ui/README.md
```

- [ ] **Step 2: Reescribir imports en `ui/`**
  - `ui/config.py`: `from src import config as _src_config` → `from ml_hotel_cancellations import config as _src_config`. Recalcular ruta: `PROJECT_ROOT = Path(__file__).resolve().parents[1]` → `parents[3]` (ahora en `…/ui/config.py`).
  - `ui/booking.py`: `from src import config` → `from ml_hotel_cancellations import config`.
  - `ui/sections/prediccion.py`: `from src.predict import load_best_model` → `from ml_hotel_cancellations.ml.predict import load_best_model`; `from src.visualization_2d import load_artifacts` → `from ml_hotel_cancellations.utils.visualization_2d import load_artifacts`; `from src.interpretability import explain_booking_to_figure` → `from ml_hotel_cancellations.utils.interpretability import explain_booking_to_figure`.
  - imports relativos internos de ui (`from . import config, data, booking, layout`, `from .. import ...` en sections) → **se mantienen**.

- [ ] **Step 3: Verificar `__file__` y residuales**

Run: `grep -rnE "from (src|api|ui)[ .]|Path\(__file__\)" src/ml_hotel_cancellations/ui/`
Expected: sin `from src/api/ui`; las rutas `__file__` recalculadas a `parents[3]`.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor: mover ui/ al paquete"
```

---

### Task 6: Consolidar tests y conftest → PRIMER CHECKPOINT VERDE

**Files:**
- Modify: `conftest.py`
- Move: `api/tests/test_api.py` → `tests/test_api.py`
- Modify: todos los `tests/*.py`

- [ ] **Step 1: Mover la suite de la API a `tests/` y borrar el paquete `api/tests`**

```bash
git mv api/tests/test_api.py tests/test_api.py
git rm api/tests/__init__.py
rmdir api/tests 2>/dev/null || true
```

- [ ] **Step 2: Reescribir imports en `conftest.py`**
  - `from api.schemas import BOOKING_EXAMPLE` → `from ml_hotel_cancellations.api.schemas import BOOKING_EXAMPLE`
  - `from src.predict import load_best_model` → `from ml_hotel_cancellations.ml.predict import load_best_model`
  - `from api.main import app` → `from ml_hotel_cancellations.api.main import app`

- [ ] **Step 3: Reescribir imports en todos los `tests/*.py`** según la tabla maestra. Concretamente:
  - `from src import config` → `from ml_hotel_cancellations import config`
  - `from src import config, data_loader` → `from ml_hotel_cancellations import config` + `from ml_hotel_cancellations.ml import data_loader`
  - `from src import predict` → `from ml_hotel_cancellations.ml import predict`
  - `from src import gpu` → (ya no existe; test_gpu se borró)
  - `from src.evaluator import Evaluator, compute_metrics` → `from ml_hotel_cancellations.ml.evaluator import ...`
  - `from src import preprocessing` → `from ml_hotel_cancellations.ml import preprocessing`
  - `from api import service`, `from api import schemas`, `from api.schemas import Booking` → `from ml_hotel_cancellations.api ...`
  - `from ui import booking`, `from ui import config as ui_config`, `from ui import data as ui_data` → `from ml_hotel_cancellations.ui ...`
  - `from src import balancing, train, tuning` (test_contracts) → `from ml_hotel_cancellations.ml import balancing, train, tuning`
  - En `test_contracts.py`, `inspect.getsource(predict.predict_dataframe)` y los `is` de identidad siguen válidos con los nuevos imports.

- [ ] **Step 4: Reinstalar (nuevos entry points) y correr la suite COMPLETA**

Run: `.venv/bin/pip install -e ".[train,dev]" -q && .venv/bin/python -m pytest -q`
Expected: `62 passed`.

- [ ] **Step 5: Verificación global de imports**

Run: `grep -rnE "from (src|api|ui)[ .]|^import (src|api|ui)\b" --include=*.py src/ tests/ conftest.py`
Expected: VACÍO.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "refactor: consolidar tests en tests/ y completar la migración a src-layout"
```

---

### Task 7: Extraer `api/registry.py` (SOLID)

**Files:**
- Create: `src/ml_hotel_cancellations/api/registry.py`
- Modify: `src/ml_hotel_cancellations/api/service.py`

- [ ] **Step 1: Crear `registry.py`** moviendo desde `service.py` el cliente MLflow: la dataclass `LoadInfo`, `_REGISTRY_CACHE_ROOT`, `_registry_cache_dir`, `_require_env`, `_mlflow_auth`, `_resolve_registry_uri`, `_download_artifact_file`, `_load_from_registry`. Mantener sus imports (`hashlib`, `json`, `os`, `requests`, `joblib`, `Path`, `logging`). `_load_from_registry` devuelve `(pipeline, LoadInfo)`.

- [ ] **Step 2: Actualizar `service.py`** para importar del nuevo módulo:

```python
from .registry import LoadInfo, load_from_registry  # nombre público
```
Renombrar `_load_from_registry` → `load_from_registry` (público, ya que cruza módulo). `service.get_model()` llama a `registry.load_from_registry(uri)` en la rama del registry. `service.py` conserva: `get_model_path`, `get_registry_uri`, `_set_load_info`, `get_load_info`, `_load_bundled`, `get_model`, `is_model_loaded`, `get_model_type`, `get_model_info_payload`, `predict_one/many`, `_format_result`.

- [ ] **Step 3: Correr la suite**

Run: `.venv/bin/python -m pytest -q`
Expected: `62 passed`.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor(api): extraer el cliente MLflow Registry a api/registry.py"
```

---

### Task 8: Despliegue y dependencias (render, requirements, scripts)

**Files:**
- Modify: `render.yaml`
- Replace: `requirements.txt` (→ una línea), `git rm requirements-train.txt`

- [ ] **Step 1: `requirements.txt` → instalación del paquete**

Sustituir TODO el contenido de `requirements.txt` por:
```
# Instala el paquete y sus dependencias de runtime desde pyproject.toml.
# (Streamlit Community Cloud y otros entornos que solo leen requirements.txt.)
-e .
```

- [ ] **Step 2: Eliminar `requirements-train.txt`** (su contenido vive en el extra `[train]`)

Run: `git rm requirements-train.txt`

- [ ] **Step 3: Actualizar `render.yaml`**
  - `buildCommand`: `pip install --upgrade pip && pip install -e .`
  - `startCommand`: `uvicorn ml_hotel_cancellations.api.main:app --host 0.0.0.0 --port $PORT`

- [ ] **Step 4: Smoke de entrypoints**

```bash
.venv/bin/python -c "from ml_hotel_cancellations.api.main import app; print('api ok')"
.venv/bin/python -c "import ml_hotel_cancellations.ui.app; print('ui import ok')"
.venv/bin/train --help 2>/dev/null || .venv/bin/python -m ml_hotel_cancellations.ml.train --help
```
Expected: `api ok`, `ui import ok`, y la ayuda del CLI `train`.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "build: deps en pyproject (extra train); render + requirements al nuevo paquete"
```

---

### Task 9: Documentación (README, devcontainer, árbol)

**Files:**
- Modify: `README.md`, `.devcontainer/devcontainer.json`

- [ ] **Step 1: Actualizar el árbol de estructura del README** a la nueva jerarquía `src/ml_hotel_cancellations/{config.py,ml/,api/,ui/,utils/}`.

- [ ] **Step 2: Actualizar los comandos del README**
  - Entorno: `pip install -e ".[train,dev]"` (reentrenar/tests) y `pip install -e .` (solo runtime).
  - CLIs: `python -m ml_hotel_cancellations.ml.train` o los console scripts (`train`, `predict`, `tune`, `balance`).
  - API: `uvicorn ml_hotel_cancellations.api.main:app --reload`.
  - UI: `streamlit run src/ml_hotel_cancellations/ui/app.py`.
  - Sección de tests: `pytest` (ya no menciona `api/tests`).

- [ ] **Step 3: Actualizar `.devcontainer/devcontainer.json`** (rutas a `ui/app.py` y cualquier comando `python -m src.*` → nuevas rutas).

- [ ] **Step 4: Revisar notebooks (manual, rápido)**

Run: `grep -rn "from src\|import src\|python -m src" notebooks/ 2>/dev/null | head`
Si hay referencias, actualizarlas a `ml_hotel_cancellations`. (No bloquean tests.)

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "docs: actualizar README y devcontainer a la nueva estructura"
```

---

### Task 10: Verificación de integración end-to-end

- [ ] **Step 1: Suite completa**

Run: `.venv/bin/python -m pytest -q`
Expected: `62 passed`.

- [ ] **Step 2: Entrenamiento end-to-end (reproduce métricas, restaura artefactos)**

```bash
.venv/bin/python -m ml_hotel_cancellations.ml.train 2>&1 | grep -iE "mejor modelo|roc_auc=0.9614|error|traceback" | tail
git checkout -- outputs/ models/   # restaurar artefactos regenerados
```
Expected: "Mejor modelo … XGBoost", `roc_auc=0.9614`, sin tracebacks.

- [ ] **Step 3: Smoke API + render local**

```bash
.venv/bin/python -c "from fastapi.testclient import TestClient; from ml_hotel_cancellations.api.main import app; from ml_hotel_cancellations import config; c=TestClient(app); print(c.get('/model-info').json()['model_type']); print(c.post('/predict', json=config.BOOKING_EXAMPLE).status_code)"
```
Expected: `XGBoost` y `200`.

- [ ] **Step 4: Verificación final sin referencias antiguas**

Run: `grep -rnE "from (src|api|ui)[ .]|python -m src\.|uvicorn api\.main|ui/app.py" --include=*.py --include=*.md --include=*.yaml --include=*.json . | grep -v docs/superpowers`
Expected: VACÍO (salvo specs/plans históricos en docs/superpowers).

- [ ] **Step 5: Commit final (si quedaron cambios) y resumen**

```bash
git add -A && git commit -m "chore: verificación final de la reestructuración" --allow-empty
```

---

## Self-review notes
- **Cobertura de la spec:** estructura objetivo (T1-T5), imports absolutos (tabla maestra, T2-T6), recálculo de rutas `__file__` (T2 config, T5 ui), extracción `registry.py` (T7), pyproject con deps/extras/scripts/pytest (T1, T8), entrypoints render/streamlit/CLI (T8-T9), consolidación de tests (T6), criterios de aceptación (T10). ✓
- **Sin placeholders:** comandos y contenido de `pyproject.toml` literales; reglas de import concretas con tabla. ✓
- **Checkpoint verde:** explícito en T6 (primer verde tras el movimiento atómico) y en T7/T10.
