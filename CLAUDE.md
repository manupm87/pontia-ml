# CLAUDE.md

Guidance for Claude Code when working in this repository. Keep it concise and current.

## What this is

PontIA Master's **final ML project**: binary classification of **hotel booking
cancellations** (`is_canceled`, ~119k rows). The deliverable is a modular Python
package + notebooks + docs + a FastAPI service + a Streamlit UI.

**The project lives in `project/`. Run almost everything from there.** The repo root
also has `recursos/` (read-only class reference notebooks) and `2.Proyecto Final de
Módulo/` (the assignment statement / original dataset).

## Hard rules (MANDATORY)

- **Language:** write ALL docs, comments, markdown and notebook prose in **Spanish**,
  in a **didactic** style that explains every technical term (the audience is
  students). Code identifiers may stay in their natural form.
- **Virtualenv:** use `project/.venv` (Python 3.12). Never create another venv. If a
  command needs Python, run it from `project/` with the venv active, or prefix
  `.venv/bin/python`.
- **Commits:** structured messages in Spanish. End every commit message with:
  `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`
- **Push to `main` only when the user explicitly asks.** `gh` is NOT installed (open
  PRs via the GitHub web UI).

## Common commands (run from `project/`)

```bash
# --- Inference + UI (only needs requirements.txt) ---
uvicorn api.main:app --reload     # bonus: REST API at http://127.0.0.1:8000 (Swagger at /docs)
python -m pytest api/tests -q     # API tests (6 cases)
streamlit run ui/app.py           # bonus: Streamlit UI (needs the API running for predictions)
python -m src.predict --sample 10 # predict on 10 sample bookings (or --input file.csv)
python -m src.visualization_2d    # bonus: regenerate the PLS decision-regions PNG + pickle

# --- Training + experiments (also needs requirements-train.txt) ---
python -m src.train               # train + evaluate the 5 models, pick best, save artefacts
python -m src.train --tune        # also re-run hyperparameter search (slower)
python -m src.tuning              # only the search (writes best_hiperparametros.json + report)
python -m src.balancing           # compare class-balancing strategies (bonus)
python -m src.interpretability    # regenerate SHAP + permutation-importance plots (bonus)
python -m src.register_model      # register the latest XGBoost run in the MLflow registry (DagsHub)

# Execute a notebook in place (note the matplotlib gotcha below):
cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --inplace --execute \
  --ExecutePreprocessor.kernel_name=python3 <notebook>.ipynb
```

Environment variables the project reads:

```
PONTIA_USE_GPU=1                    # opt in to GPU for XGBoost (default: CPU)
PONTIA_MODEL_PATH=/path/to.pkl      # override the bundled best_model.pkl
PONTIA_API_URL=http://host:8000     # UI -> API URL (default: localhost:8000)
PONTIA_REGISTRY_CACHE=/tmp/...      # where to cache models pulled from the MLflow registry
MLFLOW_TRACKING_URI=...             # DagsHub MLflow URL — enables experiment tracking
MLFLOW_TRACKING_USERNAME=...        # DagsHub user
MLFLOW_TRACKING_PASSWORD=...        # DagsHub token (scope: mlflow)
MLFLOW_MODEL_URI=models:/...        # If set, API loads from the registry instead of the bundled pickle
```

Local convention: put the four MLflow vars in `project/.env` (gitignored). Load
with `set -a; source .env; set +a`. There's an `.env.example` template.

## Architecture

- **`src/`** is the production package and the single source of truth:
  `config.py` (paths, constants, column lists, model params, search grids),
  `data_loader.py` (load/clean/stratified split), `preprocessing.py`
  (`ColumnTransformer`), `model_trainer.py`, `evaluator.py`, `train.py`, `predict.py`,
  `tuning.py`, `balancing.py`, `interpretability.py`, `visualization_2d.py`,
  `tracking.py` (MLflow), `register_model.py` (MLflow registry CLI), `gpu.py`.
- Each model is a scikit-learn `Pipeline(preprocessor, model)`. The best model
  (XGBoost) is persisted to `models/best_model.pkl`. 5 models compared; metric =
  **ROC-AUC**. Determinism via `RANDOM_STATE = 42`, `TEST_SIZE = 0.2`.
- **`api/`** (FastAPI) loads the model once via a fallback chain
  (`MLflow registry → bundled pickle`) and reuses `src.predict` so inference is
  identical to training. `GET /model-info` reports which source (and which
  registry version) the API ended up serving.
- **`ui/`** (Streamlit, modular) calls the API for predictions, also calls
  `src.interpretability` + `src.visualization_2d` in-process to render the SHAP
  waterfall + 2D PLS map for the booking on the prediction page. See
  `api/README.md` and `ui/README.md`.
- **MLflow + DagsHub** (bonus): `src/tracking.py` instruments `train.py`,
  `tuning.py`, `balancing.py` to log runs to a free hosted MLflow server on
  DagsHub. `src/register_model.py` registers the winning XGBoost run as
  `pontia-cancellations:vN` and promotes it to stage `Production`. Helpers are
  no-op without `MLFLOW_TRACKING_URI`. Full design in
  `project/docs/plan_despliegue_mlflow.md`.

### Notebook conventions (important)

- Notebooks are **exploratory**. Rule: **import only the _contract_ from `src`**
  (`config`, `load_and_prepare`, `build_preprocessor`, the model definition) so they
  use the exact same data/preprocessing as the pipeline; keep evaluation and
  presentation **inline and visible**. Do NOT move notebook-presentation helpers into
  `src`. One notebook per model (`02`–`06`), shared `01_eda`, comparative `07`.
- **`notebooks/playground/` must NOT import from `src`** — it deliberately replicates
  the `recursos/` class style (`pd.get_dummies`, `GridSearchCV`, plotly).
- Where `src` uses tools not seen in `recursos/` (`ColumnTransformer`,
  `OneHotEncoder`, `SimpleImputer`, `Pipeline`, `RandomizedSearchCV`), they are mapped
  to their class equivalents in **`docs/informe_final.md` §4.5** — keep that table
  updated if you add such a tool.

## Gotchas / dead-ends (don't re-discover these)

- **Two requirements files.** `requirements.txt` is inference-only (~350 MB);
  `requirements-train.txt` adds tensorflow/keras, imbalanced-learn, mlflow, jupyter.
  Hosted runtimes (Render, Streamlit Cloud) only install the first. To train or
  log to MLflow locally, install both.
- **API loader chain.** `api/service.py::get_model()` tries
  `MLFLOW_MODEL_URI` first (cached in `/tmp/pontia_models/<hash>`), falls back
  to `models/best_model.pkl` on any error. `GET /model-info` exposes which
  path won (`source: "registry" | "bundled"`) and `fallback_reason` if the
  registry path failed. Default behaviour (no env var) = bundled pickle.
- **DagsHub MLflow UI hides Register/Stage buttons.** This is a known
  limitation of their forked frontend; the backend works. Use
  `python -m src.register_model` instead of clicking.
- **SHAP 0.49 + XGBoost ≥ 2.x crash.** XGBoost's UBJ dump wraps `base_score`
  in `"[…]"`; SHAP 0.49 does `float()` on it and fails. We can't bump SHAP
  past 0.49 because TensorFlow 2.16 pins `numpy<2` which pins shap. Fix is a
  small monkey-patch in `src/interpretability.py::_patch_shap_xgboost_base_score`.
- **`outputs/best_hiperparametros.json` overrides `config.*_PARAMS` at train time.**
  `ModelTrainer` merges `{**config_params, **json}`, so editing only `config.py` has
  no effect if the JSON has that model. To change the *effective* defaults, update
  both (or re-run `--tune`, which regenerates the JSON).
- **Notebook plots not rendering:** some `src` modules call `matplotlib.use("Agg")` at
  import, which kills the inline backend. Put `%matplotlib inline` in the plotting
  cell **after** the `src` imports.
- **Dropped / dangerous columns:** `reservation_status` & `reservation_status_date`
  are **data leakage** (they encode the outcome) — always dropped. `company` (~94%
  missing) and `arrival_date_year` (doesn't generalize: partial years, confounded with
  season) are also excluded. `is_repeated_guest` is a `0/1` **boolean kept numeric on
  purpose** (one-hot would be redundant). Reasoning is in `01_eda` + the informe.
- **GPU is opt-in** via `PONTIA_USE_GPU=1` (XGBoost `device='cuda'`); default is CPU
  (reproducible). TensorFlow is the CPU build.
- **pandas 3.0:** `astype('Int64').astype(str)` does NOT yield `'<NA>'` for missing
  values — use `fillna(...)` first.

## Git / files

- Gitignored (regenerable): `.venv/`, `models/*.pkl`, `data/processed/`,
  `iframe_figures/` (Plotly exports, ~5 MB each), `outputs/*.pkl` (the
  visualization_2d artefact cache, ~5 MB), `.env` (local secrets).
  **Versioned:** `outputs/*.png` and `outputs/metricas_*`. **Tracked secret
  templates:** `.env.example`.
- Current best result: **XGBoost ROC-AUC 0.9614** (`{n_estimators:600, max_depth:16,
  learning_rate:0.03}`), registered in DagsHub as
  `pontia-cancellations:1@Production`.

## Read more

- `project/README.md` — overview + how to run each piece.
- `project/docs/informe_final.md` — report (roles, EDA §3, design §4, tool mapping
  §4.5, results §5, bonuses §6, limitations §7).
- `project/docs/glosario.md` — every technical term explained.
- `project/docs/interpretabilidad.md`, `project/api/README.md`,
  `project/ui/README.md`, `project/notebooks/README.md`.
- `project/docs/plan_despliegue_mlflow.md` — operational spec for the MLflow
  bonus + the planned public deploy (Render + Streamlit Cloud + DagsHub). The
  file uses `[x]/[~]/[ ]` task statuses so progress is at-a-glance.
