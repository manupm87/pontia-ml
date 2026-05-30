# CLAUDE.md — guide for working in this repository

Hotel booking cancellation prediction (binary classification). Final ML project:
trains/compares models, serves the best one via a FastAPI service, and showcases it
in a Streamlit UI.

> This file (the agent guide) is intentionally written in **English**. It is the one
> exception to the language convention below.

## ⚠️ Language convention (IMPORTANT)

- **Code is in ENGLISH**: variable, function, method, class, parameter and constant names.
- **In SPANISH**: comments, *docstrings*, all visible UI text (Streamlit), log/error
  messages, generated-report headers, and **all `.md` files** (except this CLAUDE.md).
- **Output column keys** (e.g. `"tasa_cancelacion"`, `"reservas"`, `"clase"`,
  `"modelo"`, `"estrategia"`) are user-facing data: **keep them in Spanish** even
  though they are string literals in the code.
- **API JSON field names** (Pydantic in `api/schemas.py`, e.g. `prediction`, `label`,
  `probability`) follow the standard REST convention and stay in **English** — but
  their *values* are the Spanish user-facing layer (e.g. `label: "No cancelada"`).

Rule of thumb: translate Python identifiers to English; **never** touch string
literals, comments or docstrings (that is the Spanish layer).

## Layout (src-layout, installable package)

```
src/ml_hotel_cancellations/
  config.py     # single source of truth: paths, columns, constants, BOOKING_EXAMPLE, threshold…
  ml/           # pipeline: data_loader, preprocessing (FeatureBuilder +
                # RareCategoryGrouper + ColumnTransformer), models, evaluate,
                # train, predict + (bonus) tuning
  api/          # FastAPI: main, schemas, service, registry (MLflow client)
  ui/           # Streamlit: app, config, data, booking, layout, sections/
  utils/        # cross-cutting: reporting, visualization_2d, interpretability,
                # tracking, register_model
tests/          # pytest suite (incl. "single source of truth" contract tests)
conftest.py     # shared fixtures (synthetic data, bundled model, TestClient)
```

`data/`, `models/`, `outputs/`, `docs/`, `notebooks/` live at the repo root (NOT
inside the package). `config.PROJECT_ROOT = Path(__file__).parents[2]` locates them.
**This is why the package must be installed editable** (`pip install -e .`): a regular
install would copy the package into `site-packages` and break those paths.

## Environment and commands

```bash
pip install -e ".[train,dev]"   # dev/training (TensorFlow, MLflow, pytest…)
pip install -e .                # runtime only (API + UI + inference)
```

Dependencies live in `pyproject.toml` (extras `[train]` and `[dev]`).
`requirements.txt` is just `-e .` (for platforms that only read that file).
**Python 3.12** (`requires-python = ">=3.11,<3.13"`; TF 2.16 and numba/llvmlite cap at 3.12).

### Makefile (recommended dev entrypoint — Unix/macOS/WSL)

A root `Makefile` wraps everything below; `make help` lists targets. It uses a venv at
`.venv` and **validates the Python version** (3.11/3.12) before creating it — *guard
only, it does not install Python*; point it at a specific interpreter with
`make setup PYTHON=python3.12`.

| Target | What it does |
|---|---|
| `make setup` / `setup-dev` | venv + `pip install -e .` (runtime) / `.[train,dev]` (dev+train) |
| `make run` | API + UI together (Ctrl-C stops both, no orphans) |
| `make api` / `ui` | either one alone |
| `make train` / `tune` / `register-model` | pipeline/bonus CLIs (need `setup-dev`); pass flags with `ARGS="--tune"` |
| `make predict` / `explain` / `viz2d` | inference/interpretability (runtime install is enough); `ARGS="--sample 5"` |
| `make test` | pytest (needs `setup-dev`) |
| `make clean` | remove `.venv` + build artifacts |

**Windows has no `make`**: use the `python -m …` / `uvicorn` / `streamlit` commands
directly (identical to what each target runs), or work inside WSL. `register-model`
needs MLflow env vars (`MLFLOW_TRACKING_URI/USERNAME/PASSWORD`) or it exits with a clear
"not configured" error by design.

CLIs (console scripts in `pyproject.toml`, or the `python -m` form):

| Script | Module form | What it does |
|---|---|---|
| `train` | `python -m ml_hotel_cancellations.ml.train` | trains the 5 models and saves the best one (`--tune` optional) |
| `predict` | `…ml.predict` | inference with `models/best_model.pkl` |
| `tune` | `…ml.tuning` | hyperparameter search (bonus) |
| `register-model` | `…utils.register_model` | registers the model in MLflow (bonus) |

```bash
uvicorn ml_hotel_cancellations.api.main:app --reload        # API on :8000 (/docs)
streamlit run src/ml_hotel_cancellations/ui/app.py          # web UI
```

## Tests

```bash
pytest                  # full suite (62 tests)
pytest -m "not slow"    # skip the ones that load the bundled model
```

- pytest config is in `pyproject.toml` (`[tool.pytest.ini_options]`, `pythonpath=["src"]`).
- `tests/test_contracts.py` is key: it enforces that shared constants (class labels,
  threshold, `MODEL_FAMILY`, `BOOKING_EXAMPLE`, ROC-AUC) have **a single source of
  truth in `config.py`**. If you touch those constants, don't duplicate them — derive
  them from `config`.
- After pipeline changes, also run an end-to-end `python -m …ml.train` (reproduces
  XGBoost ROC-AUC ≈ 0.9564); restore artifacts with `git checkout -- outputs/ models/`.

## Deployment

- **API → Render** (`render.yaml`): `buildCommand: pip install -e .`,
  `startCommand: uvicorn ml_hotel_cancellations.api.main:app`. The service is
  blueprint-managed: changes to `render.yaml` require a blueprint **sync** (they are
  not applied automatically).
- **UI → Streamlit Community Cloud**: *main file path* =
  `src/ml_hotel_cancellations/ui/app.py`, Python **3.12**, installs `requirements.txt`
  (`-e .`). It needs the secret **`PONTIA_API_URL`** = the API base URL on Render
  (e.g. `https://pontia-api-fi8t.onrender.com`); without it the UI points at localhost.

## Documentation

- `README.md` — entry point (demo, problem, layout, how to run, results).
- `docs/arquitectura.md` — architecture and diagrams. `docs/informe_final.md` —
  academic report. `docs/glosario.md` — glossary. `docs/interpretabilidad.md` — SHAP.
  `docs/visualizacion_2d.md` — PLS 2D decision-region visualization.
- `agents/deep_analysis_report.md` — forward-looking improvement ideas (not yet done).
- `docs/superpowers/` — work artifacts (specs, plans); not user documentation.

## Notes

- Winning model is **XGBoost** (ROC-AUC ≈ 0.9564, leakage-free: the EDA dropped
  `required_car_parking_spaces` as a check-in leak — see notebooks/playground/01–02).
  The decision threshold (`config.DECISION_THRESHOLD = 0.5`) and primary metric
  (`roc_auc`) live in `config`.
- No GPU support (removed: CPU-only, reproducible with `RANDOM_STATE=42`).
- When adding a model: register it in `config.MODEL_FAMILY` and in `ml/models.py` (`build_classic_estimators`/`build_models`).
- Preprocessing lives in the sklearn `Pipeline` (`ml/preprocessing.py`): `FeatureBuilder`
  (derives `has_company`/`has_agent`/`noches`) + `RareCategoryGrouper` (supervised
  fit-on-train cardinality reduction for `agent`/`country`/`company`) → `ColumnTransformer`.
  Input contract = 27 features (15 numeric + 12 categorical); derived features are NOT input.
- Editable installs recreate a gitignored `build/` dir; safe to delete.
