# Plan de mejora de calidad de código — pontia-ml

> Análisis de legibilidad, simplicidad, DRY y SOLID sobre `src/`, `api/` y `ui/`
> (~5.600 líneas de Python). Las notebooks y `recursos/` quedan fuera del alcance.
>
> Generado a partir de una revisión en paralelo por módulo. El idioma español en
> nombres y comentarios es una convención del proyecto y **no** se considera un defecto.
>
> Cada tarea indica: archivo(s) afectados, problema, fix sugerido y severidad.
> Marca `[x]` al completar.

## Resumen ejecutivo

El código está bien documentado y modularizado. El problema dominante es la
**duplicación de constantes de negocio y de "fábricas" de modelos** entre los tres
paquetes. El mismo dato vive en 3-4 sitios y ya está empezando a divergir:

| Dato duplicado | Lugares | Riesgo |
|---|---|---|
| `CLASS_LABELS` | `src/config.py:135`, `api/service.py:65`, `ui/config.py:55` | 3 formatos distintos |
| ROC-AUC `0.9614` | `api/service.py:69`, `ui/config.py:52`, copiado de `outputs/metricas_modelos.csv` | número mágico que se queda obsoleto al reentrenar |
| Umbral de decisión `0.5` | `src/predict.py:66`, `src/model_trainer.py:161`, `src/balancing.py:109` | número mágico disperso |
| `EXAMPLE_BOOKING` (27 campos) | `ui/booking.py:16`, `api/schemas.py:25` | copia carácter a carácter |
| Tokens NA del CSV | `ui/data.py:83`, `src/config.py:55` (`NA_TOKENS`) | la UI diverge en silencio |
| Mapa `MODEL_FAMILY` | `src/train.py:40`, `src/tuning.py:46`, `src/balancing.py:180` | 5 vs 4 entradas |
| Catálogo de modelos clásicos | `src/model_trainer.py:252`, `src/tuning.py:122`, `src/balancing.py:61`, `src/visualization_2d.py:87` | `max_iter` hardcodeado en `tuning`, params divergen |

**La inversión de mayor retorno es la Fase 1 (fuente única de verdad).** Resuelve la
mayoría de hallazgos High/Medium de los tres módulos con cambios pequeños y localizados.

**Estrategia de ejecución:** primero se construye la **red de seguridad de tests
(Fase 0)** que fija el comportamiento actual; los *contract tests* de fuente única de
verdad fallan a propósito contra el código actual (RED) y pasan tras la Fase 1 (GREEN).
Después se ejecutan las fases de refactor manteniendo toda la suite en verde.

---

## Fase 0 — Tests (red de seguridad) 🧪

Solo existían 6 tests de integración de la API (`api/tests/test_api.py`). Se construye
una suite pytest que cubre la lógica pura (sin entrenar modelos pesados ni generar
figuras). Entorno: `.venv/bin/python -m pytest`.

- [x] **T0.1 — Infraestructura pytest.** `pyproject.toml`/`pytest.ini` con `testpaths`,
  `pythonpath = .`, marcadores (`slow`, `integration`); `tests/conftest.py` con fixtures
  compartidas (DataFrame sintético de reservas, modelo bundled cargado una vez por sesión,
  `TestClient`). — **base**

- [x] **T0.2 — Tests de `src/config.py` (contract).** Invariantes: 16 numéricas + 11
  categóricas = 27; `CLASS_LABELS` con 2 elementos; grids de tuning no vacíos;
  `BASE_CANCELLATION_RATE` en (0,1). Pin de la lista de features. — **base**

- [x] **T0.3 — Tests de `src/data_loader.py`.** `clean_data`, `normalize_categoricals`
  (NA→"Unknown", caso `agent`), `get_feature_target`, `split_data` (estratificado,
  shapes, índices) con DataFrames sintéticos pequeños. — **base**

- [x] **T0.4 — Tests de `src/preprocessing.py`.** `build_preprocessor` produce el número
  de columnas esperado tras fit/transform; one-hot de categóricas; passthrough/escala de
  numéricas. — **base**

- [x] **T0.5 — Tests de `src/predict.py`.** Con el modelo bundled: `predict_dataframe`
  devuelve probabilidades en [0,1], `prediction ∈ {0,1}`, **preserva el índice de entrada**
  (caracteriza el bug T4.2 antes de arreglarlo). — **base**

- [x] **T0.6 — Tests de `src/gpu.py` y `src/evaluator.py`.** `xgboost_device`/
  `xgboost_gpu_kwargs` con CUDA simulada (monkeypatch); cálculo de métricas del evaluador
  y selección del mejor modelo con datos sintéticos. — **base**

- [x] **T0.7 — Tests contract de fuente única de verdad (RED→GREEN).** Estos **fallan
  contra el código actual** y guían la Fase 1:
  - todas las fuentes de `CLASS_LABELS` (src/api/ui) coinciden (T1.1)
  - `set(EXAMPLE_BOOKING) == set(Booking.model_fields)` (T1.4)
  - el ROC-AUC reportado por la API procede del artefacto, no de un literal (T1.2)
  - `DECISION_THRESHOLD` existe en config y se usa en los 3 sitios (T1.3)
  - `MODEL_FAMILY` único cubre todos los modelos (T1.6)
  - la UI usa `NA_TOKENS` de config (T1.5). — **base**

- [x] **T0.8 — Ampliar tests de `api/`.** Mantener los 6 actuales; añadir batch grande,
  errores de validación por rango (límites Pydantic), y `/model-info` derivando
  `model_type` del modelo cargado (tras T3.2). — **base**

- [x] **T0.9 — Tests de `ui/`.** Transformaciones EDA de `ui/data.py`
  (`cancellation_rate_by`, `class_balance`, `numeric_summary`) con datos sintéticos;
  `is_remote_api`; definición de campos de `booking.py`. Sin levantar Streamlit. — **base**

- [x] **T0.10 — Suite verde + documentación.** `pytest` completo en verde; documentar
  cómo ejecutar la suite en el README; marcar tests lentos con `@pytest.mark.slow`. — **base**

---

## Fase 1 — Fuente única de verdad (mayor retorno) 🔴

Centralizar en `src/config.py` y consumir desde todas partes.

- [x] **T1.1 — Unificar `CLASS_LABELS`.** Definir un mapa canónico corto en
  `src/config.py` (p.ej. `CLASS_LABELS_SHORT = ["No cancelada", "Cancelada"]`) y derivar
  la variante con sufijos `(0)/(1)` de ahí. `api/service.py:65` y `ui/config.py:55`
  importan/derivan de la versión canónica. Documentar la invariante índice==clase.
  *(src M4, api M4, ui #2)* — **High**

- [x] **T1.2 — Centralizar el ROC-AUC.** Eliminar el literal `0.9614` de
  `api/service.py:69` y `ui/config.py:52`. Leerlo en tiempo de ejecución de
  `outputs/metricas_modelos.csv` (fila del modelo cargado) o de los metadatos del
  registry. Como mínimo, moverlo a `src/config.py` con un comentario "actualizar al
  reentrenar". *(api H1, ui nit)* — **High**

- [x] **T1.3 — Nombrar el umbral de decisión.** Añadir `DECISION_THRESHOLD = 0.5` a
  `src/config.py` y referenciarlo en `src/predict.py:66`, `src/model_trainer.py:161`,
  `src/balancing.py:109`. *(src X8, api M3)* — **Medium**

- [x] **T1.4 — Una sola `EXAMPLE_BOOKING`.** Definir el ejemplo de reserva una vez
  (mover a `src/config.py` o importar `api.schemas.BOOKING_EXAMPLE` desde
  `ui/booking.py`). Añadir test `set(EXAMPLE_BOOKING) == set(Booking.model_fields)`.
  *(ui #1)* — **High**

- [x] **T1.5 — Reutilizar `NA_TOKENS`.** `ui/data.py:83` debe usar `src.config.NA_TOKENS`
  en vez de la lista hardcodeada `["NULL","NA","NaN",""]`. *(ui #3)* — **Medium**

- [x] **T1.6 — `MODEL_FAMILY` único.** Definir el mapa nombre→familia una vez en
  `src/config.py` e importarlo en `train.py:40`, `tuning.py:46`, `balancing.py:180`
  (que ya tiene 5 vs 4 entradas). *(src X1)* — **Medium**

- [x] **T1.7 — `FEATURE_COLUMNS` derivado.** Exponer
  `FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS` en `src/config.py` y usar
  `len(config.FEATURE_COLUMNS)` en `api/main.py:91` en vez del cálculo inline.
  *(api L6)* — **Low**

---

## Fase 2 — Fábricas y helpers compartidos (DRY estructural) 🟠

- [x] **T2.1 — Fábrica única de estimadores clásicos.** El catálogo
  (`LogisticRegression`/`DecisionTree`/`RandomForest`/`XGBClassifier`) se reconstruye en
  4 sitios (`model_trainer.py:252`, `tuning.py:122`, `balancing.py:61`,
  `visualization_2d.py:87`) y ya diverge: `tuning.py:124` hardcodea `max_iter=1000` en
  vez de leer `config.LOGISTIC_REGRESSION_PARAMS`. Crear
  `model_factory.build_classic_estimators(overrides, n_jobs, class_weight, scale_pos_weight)`
  que también encapsule el merge de kwargs GPU de XGBoost. *(src X4, X7, TU4)* — **Medium**

- [x] **T2.2 — `make_pipeline(estimator)` compartido.** El
  `Pipeline([("preprocessor", build_preprocessor()), ("model", est)])` se reimplementa en
  `model_trainer.py:220`, `tuning.py:54`, `balancing.py:93`. Extraer a
  `preprocessing.make_pipeline()` (+ variante SMOTE). *(src X2)* — **Medium**

- [x] **T2.3 — `df_to_markdown()` único.** Tres conversores DataFrame→Markdown caseros:
  `balancing.py:132`, `tuning.py:292`, `train.py:58`. Extraer uno solo o usar
  `df.to_markdown()` (con `tabulate`). Elimina ~30 líneas. *(src X3)* — **Medium**

- [x] **T2.4 — `configure_logging()` neutral.** Cinco formatos `basicConfig` distintos;
  `balancing.py:230` y `tuning.py:328` importan el helper desde `train.py` (un CLI
  importando otro CLI). Mover `configure_logging` a `config.py` o `logging_setup.py` y que
  todos los `main()` lo llamen. *(src X5)* — **Medium**

- [x] **T2.5 — `save_figure(fig, path, dpi=120)` compartido.** El bloque
  `tight_layout(); savefig(...); close(); logger.info(...)` se repite ~4× en
  `evaluator.py` (173/195/210/247) y muchas más en `interpretability.py`, `balancing.py`,
  `visualization_2d.py`, con dpi inconsistente (110/120/130). Extraer y estandarizar dpi.
  *(src E3)* — **Medium**

- [x] **T2.6 — Backend matplotlib en un solo sitio.** `matplotlib.use("Agg")` como efecto
  de import en `evaluator.py:15`, `interpretability.py:38`, `visualization_2d.py:45`.
  Fijarlo una vez en el entry point o vía `MPLBACKEND`/`matplotlibrc`. *(src X6)* — **Low**

- [x] **T2.7 — `LoadInfo` tipado en la API.** El dict de info de carga (claves `source`,
  `registry_uri`, `version`, `stage`, `run_id`, `path`, `fallback_reason`) se deletrea 3×
  en `service.py` (`_set_load_info:101`, registry:236, bundled:253) + 4ª copia en el
  schema. `_set_load_info(**kwargs)` acepta claves mal escritas en silencio. Crear un
  `@dataclass LoadInfo` con defaults; los loaders devuelven instancias. *(api M1, L5)* — **Medium**

---

## Fase 3 — SOLID / separación de responsabilidades 🟠

- [x] **T3.1 — La página de predicción rompe el contrato "consume la API".** README dice
  que consume la API, pero `ui/sections/prediccion.py:16-33,49-92` importa
  `src.predict.load_best_model`, `src.visualization_2d._load_artifacts` (¡función
  privada!), `src.interpretability...` y ejecuta el modelo **en proceso**. Esto obliga al
  despliegue Streamlit a empaquetar el pickle + SHAP + PLS + deps pesadas de `src/`.
  Decidir: o mover SHAP/2D a endpoints de la API, o (si es intencional) actualizar docs y
  exponer `load_artifacts()` público. *(ui #8)* — **High**

- [x] **T3.2 — El handler `model_info()` conoce el dict interno del servicio.**
  `api/main.py:86-102` desempaqueta 7 claves del dict interno de `service`. Que el
  servicio exponga `get_model_info_payload()` y el handler sea
  `return ModelInfo(**service.get_model_info_payload())`. Resuelve también T1.2.
  *(api H2)* — **High**

- [x] **T3.3 — Dividir el estado de health-status duplicado en la UI.**
  `ui/app.py:72-100` y `ui/sections/prediccion.py:95-147` reimplementan la misma máquina
  de 3 estados y llaman `check_api_health()` por separado (HTTP duplicado). Extraer
  `render_api_status(container, *, verbose)`. *(ui #6, ligado a #13)* — **Medium**

- [x] **T3.4 — Trocear `train.run_pipeline` (god-function).** ~90 líneas con muchas
  responsabilidades (`train.py:69-158`). Extraer `_resolve_param_overrides(tune)`,
  `_write_metric_tables(tabla)`, `_generate_plots(...)`. *(src T1)* — **Medium**

- [x] **T3.5 — Trocear `tuning._tune_inner` y `interpretability.main`.** Métodos de ~95 y
  ~65 líneas mezclando búsqueda/baseline/MLflow/persistencia (`tuning.py:178`) y
  setup/carga/3 familias de plots (`interpretability.py:441`). Extraer subfunciones; en
  `interpretability` recorrer los 2 ejemplos en vez de copiar la llamada. *(src TU1, I1)* — **Medium**

- [ ] **T3.6 — Dividir `ui/data.py` (4 responsabilidades).** Loader + EDA + cliente HTTP +
  helper de PNGs en un módulo. Separar en `data_loaders.py`, `eda.py`, `api_client.py`,
  `artifacts.py`. *(ui #9)* — **Medium** (baja urgencia)

- [x] **T3.7 — `register_model` no debe usar `SystemExit` como flujo.**
  `register_model.py:62-111` lanza `SystemExit` desde funciones reutilizables. Lanzar
  excepción de dominio (`RuntimeError`) y convertir a `SystemExit` solo en `main()`.
  *(src R1)* — **Low**

- [x] **T3.8 — Aislar (de)serialización Keras del estimador.** `KerasMLPClassifier`
  (`model_trainer.py:41-194`) mezcla API sklearn + construcción Keras + pickle. Extraer
  `_serialize/_deserialize_keras_model`. *(src M1)* — **Low**

---

## Fase 4 — Correctness-adjacent (revisar pronto) 🟡

- [x] **T4.1 — Doble escritura de artefactos de tuning.** `train.py --tune` y
  `tuning.main()` llaman `save_results`/`save_best_params`, y `_tune_inner` ya los llama
  internamente (`tuning.py:269`): los resultados se escriben **dos veces** por ejecución.
  Decidir un único responsable. *(src T5)* — **Medium**

- [x] **T4.2 — `predict.py --output` descarta el índice.** `predict_dataframe` preserva
  `index=df.index` (`predict.py:69`) pero `to_csv(..., index=False)` (`:97`) lo tira: no
  se pueden casar las filas de salida con las de entrada. *(src P2)* — **Medium**

- [x] **T4.3 — Fuga de fichero temporal en serialización Keras.**
  `model_trainer.py:172-178,190-193` usan `NamedTemporaryFile(delete=False)` + `os.remove`
  manual: si `save`/`load_model` lanza, el temporal queda huérfano. Usar `try/finally` o
  `TemporaryDirectory`. *(src M2)* — **Medium**

- [x] **T4.4 — `get_load_info()` puede devolver `{}` y mentir.** Si la carga falla,
  `/model-info` reporta "bundled XGBoost" sano con el ROC-AUC hardcodeado. Propagar estado
  de fallo o consultar `is_model_loaded()`. *(api L1)* — **Low**

- [x] **T4.5 — Errores crípticos por credenciales MLflow.** `os.environ[...]` directo en
  `service.py:134-135,157,200` produce `KeyError` opaco. Usar `.get()` y lanzar
  `RuntimeError` explicativo. *(api M2)* — **Medium**

---

## Fase 5 — Simplicidad / anti-patrones Streamlit 🟡

- [x] **T5.1 — Health check sin cache, disparado 2-3×/render.**
  `check_api_health()` no está cacheado y cada rerun hace 2-3 llamadas HTTP síncronas
  (hasta 8s c/u, `config.py:46`). Cachear con `@st.cache_data(ttl=10, show_spinner=False)`
  y deduplicar vía T3.3. *(ui #13)* — **Medium (rendimiento)**

- [x] **T5.2 — `st.cache_resource` cacheando un `True` constante.**
  `_prewarm_api` (`app.py:110`) cachea el bool constante que devuelve `warm_up_api()`. Usar
  `st.session_state` para el one-shot. *(ui #11)* — **Medium**

- [x] **T5.3 — Efectos de import / doble entry en `app.py`.** `app.py:121-147` detecta el
  runtime Streamlit y re-invoca `main()` en import, tragando excepciones
  (`except Exception: pass`). Usar `if __name__ == "__main__"` estándar o documentar y
  estrechar el `except`. *(ui #10)* — **Medium**

- [x] **T5.4 — Eliminar código muerto.** `ui/data.py:292-299` `existing_pngs` nunca se
  usa. `tuning.py:119` reimporta `gpu` ya importado en `:39`. `train.py:96`
  `param_overrides = None` se reasigna siempre. `booking.py:13` importa `field` sin usar.
  *(ui #21, src TU3/T2, ui #22)* — **Low**

- [x] **T5.5 — Cachear globs de filesystem.** `find_shap_pngs`/`existing_pngs` hacen glob
  en cada render. Cachear con TTL corto. *(ui #14)* — **Low**

- [x] **T5.6 — `__import__("pandas").option_context` ofuscado.** `train.py:238-241`. Usar
  `import pandas as pd` arriba. *(src T3)* — **Low**

- [x] **T5.7 — `predict.main` reusa `load_raw_data`.** `predict.py:87` duplica el
  `read_csv(... na_values=...)` que ya encapsula `data_loader.load_raw_data`. *(src P1)* — **Low**

---

## Fase 6 — Legibilidad / números mágicos / nits 🟢

- [x] **T6.1 — Derivar el "~37 %" de `BASE_CANCELLATION_RATE`.** Literal "~37 %" en
  `resumen.py:26`, `prediccion.py:226`, `eda.py:63`, `data.py:170`; `eda.py:42` etiqueta
  "37 %" independiente de la línea que dibuja. Usar
  `f"~{config.BASE_CANCELLATION_RATE*100:.0f} %"` o calcular de `class_balance()`. *(ui #15)* — **Medium**

- [x] **T6.2 — Catálogo único de metadatos de plots.** Las tarjetas
  `roc_curves.png`/`confusion_matrices.png`/`feature_importance.png` se duplican en
  `resumen.py:79`, `visualizaciones.py:16` (e `interpretabilidad.py:61`). Definir
  `PLOTS: dict[str, (title, desc)]` en `ui/config.py`. *(ui #5)* — **Medium**

- [x] **T6.3 — Extraer `_render_field` en el formulario.** `prediccion.py:150-205`: ~55
  líneas, 4 niveles de anidación, ramas `int`/`float` casi idénticas. Colapsar con
  `cast = int if fld.kind=="int" else float`. *(ui #16)* — **Medium**

- [x] **T6.4 — Deduplicar `plot_confusion_matrix` vs `plot_confusion_matrices`.**
  `evaluator.py:178-213`: el single es el grid con n=1. Delegar en `_plot_cm(ax, name, res)`.
  *(src E1)* — **Medium**

- [x] **T6.5 — Extraer helpers de `visualization_2d._render_figure`.** El bloque "estrella
  amarilla" `ax.scatter(marker="*", s=380, ...)` y el setup de ejes se copian 2×
  (`:233-243` vs `:259-269`). Extraer `_draw_sample_star` / `_style_axis`. *(src V1)* — **Medium**

- [x] **T6.6 — Paleta y ratios de columnas como constantes.** Hex de clases
  (`eda.py:73` `#2c7fb8`/`#de2d26`), `#d4edda` (`resumen.py:67`), ratios `st.columns([2,1])`.
  Bloque `CLASS_COLORS`/ratios en `ui/config.py` (ya existe `IMAGE_COLUMN_RATIO`). *(ui #17)* — **Low**

- [x] **T6.7 — Constantes con nombre para selecciones EDA.** `numeric_summary` whitelist
  inline (`data.py:188`), `month_order` inline (`data.py:124`), "5 modelos" en prosa
  (`resumen.py:44`). Hoist a `EDA_NUMERIC_COLUMNS`, `MONTH_ORDER`; derivar el "5" de
  `len(metrics)`. *(ui #18, #19, #20)* — **Low**

- [x] **T6.8 — Anotaciones de tipo de retorno faltantes.** Tuplas sin tipar en
  `data_loader.py:110,123,152`; `-> Figure` en `visualization_2d._render_figure`; renombrar
  `Zp`→`proba_grid`. *(src D3, V4)* — **Low**

- [x] **T6.9 — Estrechar `except Exception` amplios.** `gpu.py:35,61`, `service.py:278`
  (la del health-check, `service.py:300`, está bien). Capturar familias esperadas para no
  enmascarar bugs como "fallback". *(src G1, api L4)* — **Low**

- [x] **T6.10 — Defaults mutables como argumentos.** `evaluator.py:86-88` usa listas de
  config como default de parámetro. Default `None` y asignar en `__init__`. *(src E2)* — **Low**

- [ ] **T6.11 — Memoización con `lru_cache`.** `gpu.py` usa `global _xgb_device_cache`;
  sustituir por `@functools.lru_cache`. *(src G2)* — **Low**

- [ ] **T6.12 — Mover ensayos largos de `config.py` a docs.** Comentarios multi-párrafo
  inline en `config.py:84-98` etc. Dejar puntero de una línea a `docs/`. *(src C1)* — **Low**

- [ ] **T6.13 — Límites de campo UI derivados del schema.** `ui/booking.py:72-144` y
  `api/schemas.py:71-102` describen a mano los mismos tipos/rangos y ya difieren
  (`lead_time` cap 800 en UI, sin tope en schema). Documentar que el schema es la fuente o
  derivar los bounds del metadata Pydantic. *(ui #4)* — **Low**

- [x] **T6.14 — Nits varios.** `data.py:90` default `n=200` nunca usado;
  `data.py:136` `agent_counts` mal nombrado; `is_remote_api()` recomputado de env estático
  (→ constante); docstring de `src/__init__.py` omite módulos nuevos. *(ui nits, #12, src __init__)* — **Low**

---

## Orden recomendado de ejecución

1. **Fase 1** (fuente única de verdad) — máximo retorno, cambios pequeños y aislados.
2. **Fase 4** (correctness-adjacent) — doble escritura, índice perdido, fuga de temporal.
3. **Fase 2** (fábricas/helpers) — apoya y simplifica lo demás.
4. **Fase 3** (SOLID) — empezar por T3.1/T3.2 (contrato API).
5. **Fase 5** (Streamlit/simplicidad) y **Fase 6** (legibilidad) — limpieza incremental.

## Notas

- Hay un `.pytest_cache/` presente: existe una suite de tests (`api/tests/test_api.py` al
  menos). **Ejecutar los tests tras cada fase** y añadir tests de no-regresión para las
  fuentes únicas de verdad (T1.1, T1.4).
- El monkey-patch de SHAP (`interpretability.py:96-127`) es frágil pero está justificado;
  añadirle un guard de versión para que se autodesactive al actualizar el entorno.

---

## Estado final de ejecución

**Completadas:** Fase 0 (tests), Fase 1 (fuente única de verdad) y la práctica
totalidad de las Fases 2–6. Suite: **67 tests en verde**. Verificación de
integración: `python -m src.train`, `python -m src.balancing` y
`python -m src.tuning` se ejecutan de extremo a extremo y reproducen las métricas
comprometidas (XGBoost, ROC-AUC 0.9614); render SHAP + mapa 2D de la página de
predicción comprobados; API (`/model-info`, `/predict`, `/predict/batch`) OK.

**Pendientes a propósito (4):**

- **T3.6** — Dividir `ui/data.py` en `data_loaders/eda/api_client/artifacts`.
  *Diferida:* baja urgencia, el fichero ya está bien organizado por secciones y
  el cambio es puramente organizativo (riesgo > beneficio ahora mismo).
- **T6.11** — `@functools.lru_cache` en `gpu.xgboost_device`. *Omitida a
  propósito:* la suite de tests depende de poder resetear el cache de módulo
  (`_xgb_device_cache`) y de monkeypatchear `cuda_available`; `lru_cache`
  rompería el aislamiento entre tests por un beneficio nulo.
- **T6.12** — Mover los ensayos largos en comentarios de `config.py` a `docs/`.
  *Diferida:* cosmética; el contenido es valioso y mover documentación tiene
  riesgo de desincronización.
- **T6.13** — Derivar los límites numéricos del formulario UI del metadata
  Pydantic. *Diferida:* baja prioridad; documentado que el esquema de la API es
  la fuente de verdad y los límites de la UI son presentacionales.
