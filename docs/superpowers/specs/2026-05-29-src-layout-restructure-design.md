# Diseño: reestructuración a `src/`-layout con paquete `ml_hotel_cancellations`

Fecha: 2026-05-29
Estado: aprobado el diseño de alto nivel; pendiente revisión de esta spec.

## Objetivo

Pasar de tres paquetes planos en la raíz (`src/`, `api/`, `ui/`) a una estructura
profesional de **src-layout** con un único paquete instalable
`ml_hotel_cancellations`, dividido en subpaquetes por responsabilidad. Meta:
estructura limpia, fácil de recorrer en una clase, sin perder funcionalidad y con
los 62 tests en verde en cada paso.

No-objetivos (fases posteriores, fuera de esta spec):
- Recortar la prosa de docstrings/comentarios (acordado, pero es otra fase).
- Eliminar funcionalidad.

## Estructura objetivo

```
ml_hotel_cancellations/                      # raíz del repo (sin cambios de nombre)
├── pyproject.toml                           # metadatos, deps, pytest, console scripts
├── README.md
├── data/  models/  outputs/  docs/  notebooks/  recursos/   # SIN cambios
├── conftest.py                              # fixtures (raíz, estándar de src-layout)
├── tests/                                   # TODA la suite (incl. la de la API)
└── src/
    └── ml_hotel_cancellations/
        ├── __init__.py
        ├── config.py                        # fuente única: rutas, columnas, constantes, BOOKING_EXAMPLE
        ├── ml/
        │   ├── __init__.py
        │   ├── data_loader.py
        │   ├── preprocessing.py
        │   ├── model_factory.py
        │   ├── model_trainer.py
        │   ├── evaluator.py
        │   ├── train.py
        │   ├── predict.py
        │   ├── tuning.py
        │   └── balancing.py
        ├── api/
        │   ├── __init__.py
        │   ├── main.py
        │   ├── schemas.py
        │   ├── service.py
        │   └── registry.py                  # NUEVO: cliente REST MLflow extraído de service.py
        ├── ui/
        │   ├── __init__.py
        │   ├── app.py
        │   ├── config.py
        │   ├── data.py
        │   ├── booking.py
        │   ├── layout.py
        │   └── sections/
        │       ├── __init__.py
        │       ├── eda.py · resumen.py · visualizaciones.py · prediccion.py · interpretabilidad.py
        └── utils/
            ├── __init__.py
            ├── reporting.py
            ├── visualization_2d.py
            ├── interpretability.py
            ├── tracking.py
            └── register_model.py
```

## Mapa de migración (fichero → destino)

| Actual | Destino |
|---|---|
| `src/config.py` | `src/ml_hotel_cancellations/config.py` |
| `src/data_loader.py` | `…/ml/data_loader.py` |
| `src/preprocessing.py` | `…/ml/preprocessing.py` |
| `src/model_factory.py` | `…/ml/model_factory.py` |
| `src/model_trainer.py` | `…/ml/model_trainer.py` |
| `src/evaluator.py` | `…/ml/evaluator.py` |
| `src/train.py` | `…/ml/train.py` |
| `src/predict.py` | `…/ml/predict.py` |
| `src/tuning.py` | `…/ml/tuning.py` |
| `src/balancing.py` | `…/ml/balancing.py` |
| `src/reporting.py` | `…/utils/reporting.py` |
| `src/visualization_2d.py` | `…/utils/visualization_2d.py` |
| `src/interpretability.py` | `…/utils/interpretability.py` |
| `src/tracking.py` | `…/utils/tracking.py` |
| `src/register_model.py` | `…/utils/register_model.py` |
| `src/__init__.py` | contenido → `…/__init__.py` (raíz del paquete) |
| `api/{__init__,main,schemas,service}.py` | `…/api/…` |
| (nuevo) | `…/api/registry.py` (extraído de `service.py`) |
| `api/tests/test_api.py` | `tests/test_api.py` (consolidado en la suite raíz) |
| `ui/**` | `…/ui/**` (misma estructura interna) |

`git mv` para preservar el historial. `data/`, `models/`, `outputs/`,
`notebooks/`, `recursos/`, `docs/` **no se mueven**.

## Convención de imports

**Imports absolutos enraizados en el paquete** (explícito, ideal para explicar en
clase: se ve de dónde viene cada cosa):

- `from ml_hotel_cancellations import config`
- `from ml_hotel_cancellations.ml import preprocessing, model_factory`
- `from ml_hotel_cancellations.utils import reporting, tracking`
- `from ml_hotel_cancellations.utils.reporting import df_to_markdown, save_figure`

Dentro de un mismo subpaquete se permite el relativo corto cuando aporta concisión
(p. ej. en `api/`: `from .schemas import Booking`, `from . import service`; en
`ui/sections`: `from .. import layout, data`). Regla práctica: **relativo solo
entre hermanos del mismo subpaquete; absoluto para cruzar subpaquetes.**

Sustituciones clave:
- `from . import config` (en módulos que pasan a `ml/`) → `from ml_hotel_cancellations import config`.
- `from .train import configure_logging` → `from ml_hotel_cancellations import config` (ya vive en config) — revisar; `configure_logging` está en `config`.
- `from src import config` (en `api/` y `ui/`) → `from ml_hotel_cancellations import config`.
- En `ui/`: `from src.predict import …` → `from ml_hotel_cancellations.ml.predict import …`; `from src.visualization_2d import …` → `from ml_hotel_cancellations.utils.visualization_2d import …`; `from src.interpretability import …` → `from ml_hotel_cancellations.utils.interpretability import …`.

## Recalcular rutas en `config.py` (CRÍTICO)

`data/`, `models/`, `outputs/` siguen en la raíz del repo, pero `config.py` baja dos
niveles. Hay que recalcular `PROJECT_ROOT`:

- `src/config.py` actual: `PROJECT_ROOT = Path(__file__).resolve().parents[1]` (raíz).
- `src/ml_hotel_cancellations/config.py`: pasa a `parents[2]`
  (`…/config.py` → `parents[0]`=paquete, `parents[1]`=`src`, `parents[2]`=raíz).
- `ui/config.py` actual usa `parents[1]`; nuevo `src/ml_hotel_cancellations/ui/config.py`
  → `parents[3]`.
- Revisar cualquier otro `Path(__file__).parents[...]` (p. ej. en `visualization_2d`,
  rutas de artefactos). Los tests de `config`/`predict` detectan un cálculo erróneo
  (no encontrarían el dataset/modelo).

## Extracción `api/registry.py` (SOLID, sin perder funcionalidad)

Mover desde `api/service.py` a `api/registry.py` el cliente REST de MLflow:
`LoadInfo`, `_registry_cache_dir`, `_require_env`, `_mlflow_auth`,
`_resolve_registry_uri`, `_download_artifact_file`, `_load_from_registry`. `service.py`
queda con: carga del pickle bundled, caché del modelo (`get_model`), `predict_one/many`,
`get_model_info_payload`, `is_model_loaded`. `service.py` importa de `registry.py`.
Resultado: `service.py` ~200 líneas (de 444), centrado y explicable.

## `pyproject.toml`

Se convierte en la **fuente de verdad de dependencias y build**, sustituyendo a los
dos `requirements*.txt`:

- `[build-system]`: `setuptools` + `wheel`.
- `[project]`: `name = "ml-hotel-cancellations"`, `version`, `requires-python`,
  `dependencies = [...]` (las de runtime de `requirements.txt`, con sus topes:
  `numpy<2`, `scikit-learn<1.7`, `shap<0.50`, `numba<0.63`, `llvmlite<0.46`,
  `fastapi`, `uvicorn`, `streamlit`, `xgboost`, `pandas`, `joblib`, `requests`, …).
- `[project.optional-dependencies]`:
  - `train = [tensorflow…, keras, imbalanced-learn, mlflow, seaborn, …]` (lo de
    `requirements-train.txt`).
  - `dev = ["pytest", "httpx"]`.
- `[project.scripts]` (console scripts, para una CLI limpia en clase):
  `train = ml_hotel_cancellations.ml.train:main`, `predict = …ml.predict:main`,
  `tune = …ml.tuning:main`, `balance = …ml.balancing:main`,
  `register-model = …utils.register_model:main`,
  `explain = …utils.interpretability:main`, `viz2d = …utils.visualization_2d:main`.
- `[tool.setuptools.packages.find]`: `where = ["src"]`.
- `[tool.pytest.ini_options]`: `testpaths = ["tests"]`, `pythonpath = ["src"]`,
  `markers`, `filterwarnings` (se elimina `pytest.ini`).

`requirements.txt` / `requirements-train.txt`: se sustituyen por instalación del
paquete. **Decisión a confirmar** (ver Riesgos): mantener un `requirements.txt`
mínimo con una sola línea `-e .` para plataformas que solo leen ese fichero
(Streamlit Cloud), y usar `pip install -e .` en Render.

## Cambios de entrypoint / despliegue

| Dónde | Antes | Después |
|---|---|---|
| Render `buildCommand` | `pip install -r requirements.txt` | `pip install -e .` |
| Render `startCommand` | `uvicorn api.main:app …` | `uvicorn ml_hotel_cancellations.api.main:app …` |
| Streamlit Cloud (dashboard) | `ui/app.py` | `src/ml_hotel_cancellations/ui/app.py` |
| Streamlit Cloud deps | `requirements.txt` | `requirements.txt` = `-e .` (instala el paquete) |
| CLIs (README) | `python -m src.train` | `python -m ml_hotel_cancellations.ml.train` o `train` (console script) |
| Tests | `pytest` (pythonpath=.) | `pip install -e ".[dev]"` y `pytest` (pythonpath=src) |
| `.devcontainer/devcontainer.json` | rutas a `ui/app.py` | actualizar a la nueva ruta |

README: actualizar el árbol de estructura, los comandos y la sección de entorno
(añadir `pip install -e ".[train]"` para reentrenar).

## Estrategia de ejecución y verificación

1. Crear `pyproject.toml` y la estructura de carpetas; `pip install -e ".[train,dev]"`.
2. `git mv` de los módulos a su destino (preservando historial).
3. Reescribir imports (absolutos cruzando subpaquetes) y recalcular rutas en `config`.
4. Extraer `api/registry.py`.
5. Consolidar `api/tests/` en `tests/`; actualizar imports de los tests al paquete.
6. Actualizar `render.yaml`, `requirements.txt`(→`-e .`), README, devcontainer.
7. `pytest` (62 en verde) + smoke de los 3 entrypoints (uvicorn import, streamlit
   import, `python -m …train --help`) + un `python -m …ml.train` end-to-end
   (reproduce métricas) restaurando artefactos al terminar.

Los 62 tests son la red de seguridad de los pasos 2-5. Es un cambio mecánico pero
amplio: se hará en una rama y se validará en bloque.

## Riesgos y mitigaciones

- **Despliegue (alto).** src-layout obliga a instalar el paquete en cada entorno.
  Mitigación: `pip install -e .` en Render; `requirements.txt = -e .` en Streamlit
  Cloud. Confirmar que Streamlit Community Cloud instala desde `requirements.txt`
  con `-e .` (es lo habitual). Probar el deploy tras el merge.
- **Rutas `__file__` (medio).** Recalcular `parents[...]`; cubierto por tests.
- **Consolidar requirements en pyproject (medio).** Cambia cómo se resuelven deps.
  Alternativa de menor riesgo: conservar `requirements.txt`/`-train.txt` tal cual y
  hacer `pip install -r requirements.txt && pip install -e . --no-deps`. **A decidir
  en la revisión.**
- **Notebooks (bajo).** Si algún notebook hace `from src import …`, actualizar; no
  forma parte de la suite, revisión manual rápida.

## Criterios de aceptación

- `pip install -e ".[train,dev]"` instala sin errores.
- `pytest` → 62 passed.
- `uvicorn ml_hotel_cancellations.api.main:app` arranca y `/predict` responde 200.
- `streamlit run src/ml_hotel_cancellations/ui/app.py` importa sin error.
- `python -m ml_hotel_cancellations.ml.train` reproduce las métricas comprometidas.
- No quedan referencias a `from src`/`from api`/`from ui` ni a `python -m src.*`.
