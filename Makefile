# Makefile — atajos de desarrollo (entorno aislado en .venv).
# Uso: `make setup` y luego `make run`. `make help` lista los objetivos.

# Intérprete base usado para crear el venv. Override: `make setup PYTHON=python3.12`.
PYTHON  ?= python3
VENV    := .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
API_APP := ml_hotel_cancellations.api.main:app
UI_APP  := src/ml_hotel_cancellations/ui/app.py
API_URL := http://127.0.0.1:8000

# Argumentos extra para los CLIs: `make train ARGS="--tune"`, `make predict ARGS="--sample 5"`.
ARGS    ?=

.DEFAULT_GOAL := help
.PHONY: help venv setup setup-dev api ui run test clean \
        train tune predict register-model explain viz2d memo

# Tectonic: motor LaTeX autocontenido (un único binario, sin instalar TeX Live).
# Se descarga en .venv/bin la primera vez (binario estático Linux x86_64).
TECTONIC_VER := 0.16.9
TECTONIC     := $(VENV)/bin/tectonic

help: ## Muestra esta ayuda
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

$(PY): ## (interno) valida la versión de Python y crea el venv si no existe
	@$(PYTHON) -c 'import sys; v=sys.version_info; sys.exit(0 if (3,11) <= v[:2] < (3,13) else 1)' 2>/dev/null \
		|| { printf '❌ Se requiere Python 3.11 o 3.12 (requires-python = ">=3.11,<3.13").\n   Intérprete actual: %s -> %s\n   Instálalo (p. ej. `uv python install 3.12` o el gestor de paquetes de tu sistema)\n   y reintenta apuntando a esa versión: `make setup PYTHON=python3.12`.\n' \
			"$(PYTHON)" "$$($(PYTHON) --version 2>&1)" >&2; exit 1; }
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

venv: $(PY) ## Crea el entorno virtual en .venv

setup: $(PY) ## Instala el paquete en modo runtime (API + UI + inferencia)
	$(PIP) install -e .

setup-dev: $(PY) ## Instala con extras de entrenamiento y desarrollo (TF, MLflow, pytest…)
	$(PIP) install -e ".[train,dev]"

api: ## Arranca solo la API FastAPI en :8000 (/docs)
	$(VENV)/bin/uvicorn $(API_APP) --reload

ui: ## Arranca solo la UI Streamlit en :8501 (apunta a la API local)
	PONTIA_API_URL=$(API_URL) $(VENV)/bin/streamlit run $(UI_APP)

run: ## Arranca API + UI juntas; Ctrl-C para ambas
	@echo "API → $(API_URL)  ·  UI → http://localhost:8501"
	@$(VENV)/bin/uvicorn $(API_APP) --host 127.0.0.1 --port 8000 & \
		API_PID=$$!; \
		trap "kill $$API_PID 2>/dev/null" EXIT INT TERM; \
		PONTIA_API_URL=$(API_URL) $(VENV)/bin/streamlit run $(UI_APP)

test: ## Ejecuta la batería de tests (pytest). Requiere setup-dev
	$(VENV)/bin/pytest

# --- Pipeline / CLIs (ARGS="..." para pasar argumentos) -----------------------

train: ## Entrena los 5 modelos y guarda el mejor (ARGS="--tune"). Requiere setup-dev
	$(VENV)/bin/train $(ARGS)

tune: ## Búsqueda de hiperparámetros (bonus). Requiere setup-dev
	$(VENV)/bin/tune $(ARGS)

register-model: ## Registra el modelo en MLflow (bonus). Requiere setup-dev
	$(VENV)/bin/register-model $(ARGS)

predict: ## Inferencia con models/best_model.pkl (ARGS="--sample 5")
	$(VENV)/bin/predict $(ARGS)

explain: ## Genera los gráficos SHAP de interpretabilidad
	$(VENV)/bin/explain $(ARGS)

viz2d: ## Genera las regiones de decisión 2D (proyección PLS)
	$(VENV)/bin/viz2d $(ARGS)

# --- Memoria académica (PDF) --------------------------------------------------

$(TECTONIC): | $(PY) ## (interno) descarga el binario estático de Tectonic en .venv
	@echo "⬇️  Descargando Tectonic $(TECTONIC_VER) (binario estático, sin TeX Live)…"
	@curl -fsSL "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%40$(TECTONIC_VER)/tectonic-$(TECTONIC_VER)-x86_64-unknown-linux-musl.tar.gz" -o /tmp/tectonic-dl.tar.gz
	@tar -xzf /tmp/tectonic-dl.tar.gz -C $(VENV)/bin && chmod +x $(TECTONIC)
	@echo "✓ Tectonic instalado en $(TECTONIC)"

memo: $(TECTONIC) ## Genera memoria/memoria.pdf (figuras EDA + LaTeX vía Tectonic). Requiere setup
	$(PY) memoria/generar_figuras_eda.py
	cd memoria && ../$(TECTONIC) memoria.tex
	@echo "✓ memoria/memoria.pdf"

clean: ## Borra el venv y artefactos de build
	rm -rf $(VENV) build *.egg-info src/*.egg-info
