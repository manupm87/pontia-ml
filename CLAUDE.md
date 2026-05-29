# CLAUDE.md — guía para trabajar en este repositorio

Predicción de cancelaciones de reservas de hotel (clasificación binaria). Proyecto
final de ML: entrena/compara modelos, sirve el mejor por una API FastAPI y lo
muestra en una interfaz Streamlit.

## ⚠️ Convención de idioma (IMPORTANTE)

- **El código va en INGLÉS**: nombres de variables, funciones, métodos, clases,
  parámetros y constantes.
- **En ESPAÑOL**: los comentarios, los *docstrings*, los textos visibles de la UI
  (Streamlit), los mensajes de log/error, las cabeceras de los informes generados,
  y **todos los ficheros `.md`** (incluido este).
- Las **claves de columnas de salida** (p. ej. `"tasa_cancelacion"`, `"reservas"`,
  `"clase"`, `"modelo"`, `"estrategia"`) y los **nombres de campo del contrato de la
  API** (Pydantic en `api/schemas.py`) son datos/salida de cara al usuario: **se
  dejan en español** aunque sean *strings* dentro del código.

Regla práctica: traduce identificadores de Python a inglés; **nunca** toques
*strings*, comentarios ni docstrings (son la capa en español).

## Estructura (src-layout, paquete instalable)

```
src/ml_hotel_cancellations/
  config.py     # fuente única: rutas, columnas, constantes, BOOKING_EXAMPLE, umbral…
  ml/           # pipeline: data_loader, preprocessing, model_factory, model_trainer,
                # evaluator, train, predict + (bonus) tuning, balancing
  api/          # FastAPI: main, schemas, service, registry (cliente MLflow)
  ui/           # Streamlit: app, config, data, booking, layout, sections/
  utils/        # transversales: reporting, visualization_2d, interpretability,
                # tracking, register_model
tests/          # suite pytest (incluye contract tests de "fuente única de verdad")
conftest.py     # fixtures compartidas (datos sintéticos, modelo bundled, TestClient)
```

`data/`, `models/`, `outputs/`, `docs/`, `notebooks/` viven en la raíz del repo
(NO dentro del paquete). `config.PROJECT_ROOT = Path(__file__).parents[2]` los
localiza. **Por eso el paquete debe instalarse en modo editable** (`pip install -e .`):
una instalación normal copiaría el paquete a `site-packages` y rompería esas rutas.

## Entorno y comandos

```bash
pip install -e ".[train,dev]"   # dev/entrenamiento (TensorFlow, MLflow, pytest…)
pip install -e .                # solo runtime (API + UI + inferencia)
```

Las dependencias están en `pyproject.toml` (extras `[train]` y `[dev]`).
`requirements.txt` es solo `-e .` (para plataformas que solo leen ese fichero).
**Python 3.12** (`requires-python = ">=3.11,<3.13"`; TF 2.16 y numba/llvmlite topan en 3.12).

CLIs (console scripts en `pyproject.toml`, o forma `python -m`):

| Script | Equivalente módulo | Qué hace |
|---|---|---|
| `train` | `python -m ml_hotel_cancellations.ml.train` | entrena los 5 modelos y guarda el mejor (`--tune` opcional) |
| `predict` | `…ml.predict` | inferencia con `models/best_model.pkl` |
| `tune` | `…ml.tuning` | búsqueda de hiperparámetros (bonus) |
| `balance` | `…ml.balancing` | comparación de balanceo (bonus) |
| `register-model` | `…utils.register_model` | registra el modelo en MLflow (bonus) |

```bash
uvicorn ml_hotel_cancellations.api.main:app --reload        # API en :8000 (/docs)
streamlit run src/ml_hotel_cancellations/ui/app.py          # interfaz web
```

## Tests

```bash
pytest                  # suite completa (62 tests)
pytest -m "not slow"    # omite los que cargan el modelo bundled
```

- Config de pytest en `pyproject.toml` (`[tool.pytest.ini_options]`, `pythonpath=["src"]`).
- `tests/test_contracts.py` es clave: garantiza que las constantes compartidas
  (etiquetas de clase, umbral, `MODEL_FAMILY`, `BOOKING_EXAMPLE`, ROC-AUC) tengan
  **una única fuente de verdad en `config.py`**. Si tocas esas constantes, no las
  dupliques: derívalas de `config`.
- Tras cambios en el pipeline, conviene además un `python -m …ml.train` de extremo
  a extremo (reproduce XGBoost ROC-AUC 0.9614); restaura artefactos con
  `git checkout -- outputs/ models/`.

## Despliegue

- **API → Render** (`render.yaml`): `buildCommand: pip install -e .`,
  `startCommand: uvicorn ml_hotel_cancellations.api.main:app`. El servicio está
  gestionado por *blueprint*: los cambios en `render.yaml` requieren **sync** del
  blueprint (no se aplican solos).
- **UI → Streamlit Community Cloud**: *main file path* =
  `src/ml_hotel_cancellations/ui/app.py`, Python **3.12**, e instala `requirements.txt`
  (`-e .`). Necesita el *secret* **`PONTIA_API_URL`** = URL base de la API en Render
  (p. ej. `https://pontia-api-fi8t.onrender.com`); sin él, la UI apunta a localhost.

## Documentación

- `README.md` — punto de entrada (demo, problema, estructura, cómo ejecutar, resultados).
- `docs/arquitectura.md` — arquitectura y diagramas. `docs/informe_final.md` — informe
  académico. `docs/glosario.md` — términos. `docs/interpretabilidad.md` — SHAP.
- `agents/` y `docs/superpowers/` — artefactos de trabajo (análisis, specs, planes);
  no son documentación de usuario.

## Notas

- El modelo ganador es **XGBoost** (ROC-AUC ≈ 0.9614). El umbral de decisión
  (`config.DECISION_THRESHOLD = 0.5`) y la métrica principal (`roc_auc`) están en `config`.
- No hay soporte de GPU (se retiró: todo en CPU, reproducible con `RANDOM_STATE=42`).
- Al añadir un modelo: regístralo en `config.MODEL_FAMILY` y en `model_factory`/`model_trainer`.
