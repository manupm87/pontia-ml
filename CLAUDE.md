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
  `"modelo"`, `"estrategia"`) and the **API contract field names** (Pydantic in
  `api/schemas.py`) are user-facing data/output: **keep them in Spanish** even though
  they are string literals in the code.

Rule of thumb: translate Python identifiers to English; **never** touch string
literals, comments or docstrings (that is the Spanish layer).

## Layout (src-layout, installable package)

```
src/ml_hotel_cancellations/
  config.py     # single source of truth: paths, columns, constants, BOOKING_EXAMPLE, threshold…
  ml/           # pipeline: data_loader, preprocessing, model_factory, model_trainer,
                # evaluator, train, predict + (bonus) tuning, balancing
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

CLIs (console scripts in `pyproject.toml`, or the `python -m` form):

| Script | Module form | What it does |
|---|---|---|
| `train` | `python -m ml_hotel_cancellations.ml.train` | trains the 5 models and saves the best one (`--tune` optional) |
| `predict` | `…ml.predict` | inference with `models/best_model.pkl` |
| `tune` | `…ml.tuning` | hyperparameter search (bonus) |
| `balance` | `…ml.balancing` | class-balancing comparison (bonus) |
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
  XGBoost ROC-AUC 0.9614); restore artifacts with `git checkout -- outputs/ models/`.

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
- `agents/deep_analysis_report.md` — forward-looking improvement ideas (not yet done).
- `docs/superpowers/` — work artifacts (specs, plans); not user documentation.

## Notes

- Winning model is **XGBoost** (ROC-AUC ≈ 0.9614). The decision threshold
  (`config.DECISION_THRESHOLD = 0.5`) and primary metric (`roc_auc`) live in `config`.
- No GPU support (removed: CPU-only, reproducible with `RANDOM_STATE=42`).
- When adding a model: register it in `config.MODEL_FAMILY` and in `model_factory`/`model_trainer`.
- Editable installs recreate a gitignored `build/` dir; safe to delete.
