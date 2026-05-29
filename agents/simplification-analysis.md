# Análisis de simplificación y legibilidad — pontia-ml

> Objetivo del encargo: **reducir líneas de código** y dejar el repo **fácil de
> recorrer en una clase / examen**, con un equilibrio entre SOLID y legibilidad,
> DRY pero explicable. **Sin eliminar funcionalidad.**
>
> Análisis hecho sobre el estado tras el refactor de calidad + tests + retirada de
> GPU (rama `refactor/quality-tests-and-gpu-removal`).

## Hallazgo principal: el código ya está podado; el peso son los comentarios

Medición real de `src/ + api/ + ui/` (sin tests):

| Paquete | Total | **Código** | Comentarios | Docstrings | % prosa |
|---|---:|---:|---:|---:|---:|
| `src/` | 3.586 | **1.447** | 280 | 1.243 | **42 %** |
| `api/` | 749 | **314** | 35 | 264 | **40 %** |
| `ui/` | 1.529 | **693** | 114 | 469 | **38 %** |
| **Total** | **5.864** | **2.454** | 429 | 1.976 | **41 %** |

Dos conclusiones:

1. **El código real son ~2.450 líneas.** Es perfectamente abarcable en una clase.
   La sensación de "mucho código" viene de que **~3.400 líneas (≈58 %) son prosa**
   (comentarios + docstrings).
2. **No hay grasa de código:** el análisis estático (AST) no encontró imports sin
   usar y solo **1 función muerta** en todo el repo. El refactor anterior ya dejó
   el código limpio. Por tanto, *reducir LOC = ajustar la prosa y la estructura*,
   no borrar lógica.

> La prosa no es mala —es didáctica— pero **un fichero al 66 % de comentarios es
> difícil de *recorrer* en pantalla**. La estrategia es **prosa a su justa medida
> en el código + el "porqué" profundo en `docs/`** (que ya existe).

---

## Eje 1 — Ajustar la prosa (la mayor reducción de LOC, sin tocar lógica)

Ficheros con más densidad de prosa (candidatos nº1):

| Fichero | Total | Código | % prosa | Comentario |
|---|---:|---:|---:|---|
| `src/config.py` | 346 | **69** | **66 %** | ensayos de 10-15 líneas inline |
| `ui/config.py` | 169 | 42 | 62 % | idem |
| `src/data_loader.py` | 165 | 45 | 51 % | docstrings de párrafo por función |
| `src/preprocessing.py` | 107 | 36 | 50 % | |
| `src/interpretability.py` | 499 | 173 | 47 % | 205 líneas de docstring |
| `api/service.py` | 444 | 176 | 43 % | |

**Política propuesta (consistente en todo el repo):**

- **Docstring = 1-3 líneas**: *qué* hace y *por qué* existe. Nada de secciones
  `Parameters/Returns` de estilo NumPy en funciones triviales (los type hints ya
  documentan la firma).
- **El "porqué" largo va a `docs/`**. Ejemplos concretos a mover:
  - El ensayo de ~15 líneas sobre `arrival_date_year` en `config.py:85-98` →
    `docs/informe_final.md` (deja un puntero de 1 línea).
  - Los mapeos "§4.5 con `recursos/`" repetidos en `preprocessing.py`,
    `tuning.py`, etc. → ya están en `docs/informe_final.md §4.5`; basta el puntero.
  - Las decisiones de despliegue/MLflow en la cabecera de `api/service.py:14-44`
    (30 líneas) → `docs/arquitectura.md`.
- **Comentarios inline solo donde el código no se explica solo** (un *por qué*, no
  un *qué*).

**Ahorro estimado: ~1.000-1.300 líneas** de los ficheros de código, sin perder la
explicación (se reubica). Cada fichero del núcleo bajaría a la mitad o menos.

> Resultado para la clase: abres `config.py` y ves **69 líneas de constantes
> claras** en vez de 346; el alumno que quiera el detalle lo tiene en `docs/`.

---

## Eje 2 — Estructura: separar el **núcleo** de los **bonus**

El grafo de imports revela dos niveles nítidos:

```
NÚCLEO (las 4 fases del enunciado):
  config → data_loader → preprocessing → model_factory → model_trainer
         → evaluator → reporting → train → predict          (9 módulos, ~1.150 código)

BONUS (extensiones; dependen del núcleo, el núcleo no depende de ellas*):
  tuning, balancing, interpretability, visualization_2d,
  tracking, register_model                                   (6 módulos)
  *única arista núcleo→bonus: train importa tuning (solo con --tune)
```

`api/` y `ui/` **no importan ningún bonus de entrenamiento** (la UI sí usa
`interpretability`/`visualization_2d` para el render en proceso).

**Propuesta (elige una):**

- **Opción A — subpaquete `src/bonus/`** (`tuning`, `balancing`, `interpretability`,
  `visualization_2d`, `tracking`, `register_model`). Deja el núcleo en `src/` con
  9 ficheros. *Pro:* el recorrido de la lección es "estos 9 son el proyecto; la
  carpeta `bonus/` son extras". *Contra:* mover módulos toca imports (de ahí los
  tests, que blindan el cambio) y referencias en notebooks/README.
- **Opción B — dejar plano + documentar los dos niveles** en el README y un
  comentario de cabecera. Cero churn. Menos explícito que A.

Recomendación: **Opción A** si quieres que la estructura "cuente la historia" en el
examen; **B** si prefieres no tocar rutas. (Los 62 tests cubren el núcleo, así que A
es seguro.)

---

## Eje 3 — SOLID ↔ legibilidad (ajustar la altitud, no más capas)

El refactor previo aplicó SRP con ganas. Dos sitios a revisar para que **no esté
*demasiado* troceado** para una explicación lineal:

- **`api/service.py` (444 líneas, SRP real pendiente):** mezcla 2 responsabilidades
  —servir el pickle local + un **cliente REST de MLflow Model Registry** (~150
  líneas: `_resolve_registry_uri`, `_download_artifact_file`, `_load_from_registry`,
  `LoadInfo`). **Extraer el cliente del registry a `api/registry.py`** deja
  `service.py` en ~200 líneas centradas en "cargar modelo + predecir" (mucho más
  explicable) **sin perder funcionalidad**. Es el cambio SOLID de mayor valor.
- **`src/tuning.py`:** `_tune_inner` se dividió en `_tune_one` / `_build_search` /
  `_baseline_cv` / `_log_model_run`. Está bien para producción, pero para *recorrer
  en clase* quizá sean demasiados saltos. Sugerencia: revisar si `_build_search` +
  `_baseline_cv` se reintegran en `_tune_one` (que quedaría ~25 líneas legibles de
  arriba a abajo). Es una decisión de gusto; señalado para que la tomes tú.
- **`ui/data.py` (325 líneas):** sigue mezclando 4 cosas (loaders + EDA + cliente
  HTTP + artefactos). Para una clase, **un fichero de 325 líneas con 4 temas es más
  difícil de seguir que 2-3 ficheros temáticos**. Dividir en `ui/data_loaders.py`,
  `ui/eda.py`, `ui/api_client.py` ayudaría (era la tarea T3.6, aplazada). Opcional.

> Principio para el equilibrio: **una función debe caber en una pantalla y leerse
> de arriba abajo**; extrae solo cuando el bloque tiene nombre propio y se reutiliza
> o esconde complejidad. No extraer "porque sí".

---

## Eje 4 — Código muerto y micro-limpieza

- **`ui/data.py:load_best_hyperparams`** está definida pero **no se llama en ningún
  sitio**; su única consumidora de `config.BEST_HYPERPARAMS_PATH`. Ninguna sección
  de la UI muestra los hiperparámetros. → **Eliminar la función y la constante.**
- **`src/__init__.py`** (y `api`/`ui` `__init__`) son casi 100 % docstring; el de
  `src` lista módulos de forma incompleta. Ajustar a un docstring de 2 líneas.
- Revisar los `__init__.py` de `ui/sections` (7 líneas, todo docstring).

---

## Recorrido propuesto para la lección (la "espina dorsal")

Con la prosa ajustada y (opcional) el núcleo separado, una clase recorre **8
ficheros, ~1.150 líneas de código**, en orden de pipeline:

1. `config.py` — qué columnas, qué métrica, qué constantes (la "fuente de verdad").
2. `data_loader.py` — cargar → limpiar → partir.
3. `preprocessing.py` — el `ColumnTransformer` (num + cat) y `make_pipeline`.
4. `model_factory.py` — los 4 modelos clásicos en un sitio.
5. `model_trainer.py` — entrenar (incl. la red Keras).
6. `evaluator.py` — métricas, tabla, selección del mejor.
7. `train.py` — el orquestador que une las 4 fases.
8. `predict.py` — inferencia con el mejor modelo.

Los bonus (tuning/balancing/SHAP/2D/MLflow) se explican "a la carta" después.

---

## Plan priorizado (sin perder funcionalidad)

| # | Acción | Eje | Ahorro LOC | Riesgo |
|---|---|---|---:|---|
| 1 | Eliminar `load_best_hyperparams` + constante muerta | 4 | ~15 | nulo |
| 2 | **Ajustar prosa**: docstrings 1-3 líneas, ensayos → `docs/` | 1 | **~1.000-1.300** | bajo (lo cubren los tests) |
| 3 | Extraer cliente MLflow Registry a `api/registry.py` | 3 | 0 (reubica) | bajo |
| 4 | (Opción A) Subpaquete `src/bonus/` | 2 | 0 (reubica) | medio (imports) |
| 5 | Revisar altitud de helpers en `tuning.py` | 3 | ~20 | bajo |
| 6 | (Opcional) Dividir `ui/data.py` por temas | 3 | 0 (reubica) | bajo |

**Recomendación de orden:** 1 → 2 (el gran ahorro) → 3 → 5, y decidir 4/6 según
cuánto quieras tocar la estructura. Tras cada paso, `pytest` (62 tests) valida que
nada se rompe.
