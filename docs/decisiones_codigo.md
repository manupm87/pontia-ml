# Decisiones de ingeniería en el código

Cuatro piezas del `src` cuyo **diseño no es obvio** a primera vista. Aquí se explica el
**porqué** de cómo están escritas, no solo qué hacen. El hilo conductor de las cuatro es el
mismo: **separar lo determinista de lo aprendido** (para no tener fuga) y **no cargar
dependencias pesadas en runtime**.

---

## 1. `KerasMLPClassifier`: la red Keras vestida de estimador sklearn

📄 `ml/models.py`

El enunciado **exige** una red multicapa con **Keras/TensorFlow**, pero todo el resto del
sistema —comparación de modelos, `Pipeline` de preprocesado, serialización, MLflow— está
construido sobre la **API de scikit-learn**. En vez de tratar la red como un caso especial
por todas partes, la **envolvemos** en una clase que se comporta como un estimador sklearn
(`fit` / `predict` / `predict_proba`, heredando de `ClassifierMixin, BaseEstimator`).

**Qué nos da esa decisión:**

- **Encaja en el `Pipeline` común** → recibe el mismo preprocesado *fit-on-train* que los
  otros cuatro modelos, sin código aparte.
- **Se compara igual** → el mismo `evaluate` y la misma tabla de métricas sirven para los 5.
- **Se serializa igual** → `joblib.dump` como los demás. Un modelo Keras no se *pickea*
  directo, así que `__getstate__`/`__setstate__` guardan su formato nativo `.keras` **como
  bytes dentro del propio pickle** (y lo reconstruyen al cargar). Detalle invisible para
  quien usa la clase.
- **Entra en MLflow igual** → se loguea y registra como cualquier `Pipeline` sklearn.

**La decisión clave de deployment:** TensorFlow se importa **dentro de los métodos**, no en
la cabecera del módulo. Importar el paquete (lo que hace la **API en runtime**) **no carga
TF** (cientos de MB). Como el modelo servido es XGBoost, TF solo se necesita al **entrenar**
la red → la API en Render (512 MB) nunca lo toca. Misma idea que replican los notebooks 04/05
(ver [`bonus.md`](bonus.md) y el módulo `notebooks/keras_mlp.py`).

---

## 2. `FeatureBuilder`: features derivadas + normalización de IDs (Paso 1)

📄 `ml/preprocessing.py`

Primer paso del `Pipeline` de preprocesado. Es un *transformer* sklearn, pero **row-wise y que
no aprende nada** (`fit` devuelve `self`): hace lo mismo en entrenamiento e inferencia, sin
riesgo de fuga. ¿Por qué vive en el `Pipeline` entonces? Para que se aplique **idéntico al
servir** y se **persista con el modelo**: la API solo manda las **26 features crudas** y el
`Pipeline` deriva el resto.

**Qué hace en `transform`:**
1. `children` nulos → 0 (apenas 4 en el crudo).
2. `noches` = noches entre semana + de fin de semana (feature derivada, EDA §11.5).
3. **Ausencia informativa** (EDA §5): `has_company` / `has_agent`, *flags* binarios que marcan
   que la reserva **no** tenga empresa/agente. Aquí el nulo **es señal**, no ruido, y se captura
   **antes** de que el valor crudo se agrupe o se descarte.
4. **Normaliza los IDs de alta cardinalidad**: `9.0` (float del CSV) → `"9"` (clave limpia para
   el *one-hot*); los ausentes (NaN o centinela como `"no_company"`) → `np.nan`, para que el
   `RareCategoryGrouper` (Paso 2) les ponga su `null_label` propio.

**Las decisiones:** (a) separar *"hay / no hay"* (los `has_*`) del *"cuál es"* (la categoría
reducida); (b) los helpers `_is_absent` / `_key_str` toleran tanto el `NaN` del CSV como los
centinelas de texto que manda la **UI** en inferencia → el mismo transformer sirve para entrenar
y para servir; (c) va **antes** del `RareCategoryGrouper` porque produce justo los IDs
normalizados (y los nulos marcados) que el Paso 2 necesita para agrupar.

---

## 3. `RareCategoryGrouper`: reducción supervisada de cardinalidad (Paso 2)

📄 `ml/preprocessing.py`

`agent`, `country` y `company` tienen **cientos de categorías**; un *one-hot* directo crea
miles de columnas dispersas que perjudican a varios modelos. La idea del EDA (§13) es
**quedarse con las categorías que de verdad tienen señal** —tasa de cancelación extrema **y**
soporte suficiente— y agrupar el resto en `"Otros"`.

**Por qué un *transformer* sklearn (y no limpiarlo en `clean_data`):** porque esta reducción
**mira el target** (`y`), y elegir categorías por su tasa de cancelación usando *todo* el
dataset sería **fuga train/test**. Al implementarlo como un paso del `Pipeline`:

- se ajusta **solo con `train`** (la lista de categorías a conservar sale únicamente del
  entrenamiento) → **sin fuga**;
- se **persiste con el modelo** → en inferencia aplica exactamente la misma reducción.

**Qué aprende en `fit` (por columna):**
1. agrupa por categoría y calcula su **tasa** de cancelación (`mean`) y su **soporte** (`n`);
2. descarta las que no llegan a `min_n` (sin soporte, ruido);
3. de las que quedan, conserva las de **tasa extrema**: `> hi_frac · max` (muy canceladoras) o
   `< lo_frac · max` (muy fiables). Los umbrales son **adaptativos**: se normalizan por el
   máximo de *esa* columna, porque cada variable tiene su propio rango de tasas.

En `transform`, lo conservado mantiene su clave, el resto pasa a `"Otros"` y los nulos a una
etiqueta propia (p. ej. `no_company`, que en este dominio **es señal**, no ausencia neutra).

**La rama sin `y`** (no supervisada) conserva todas las categorías vistas: permite **reutilizar
el preprocesador sin target** (p. ej. para la visualización 2D), donde no toca re-aprender nada.

---

## 4. `clean_data`: limpieza segura, paso a paso

📄 `ml/data_loader.py`

La decisión de diseño es la **frontera** entre dos tipos de trabajo:

- **`clean_data`** hace lo **determinista y fila-a-fila** (no aprende parámetros) → va **antes
  del split**, no puede causar fuga.
- Lo que **aprende de los datos** (imputación, escalado, reducción de cardinalidad) vive en el
  **`Pipeline`** y se ajusta **solo con `train`**.

Por eso `clean_data` es deliberadamente "tonta": no imputa, no escala, no toca `company`/`agent`
(de eso se encargan `FeatureBuilder` y `RareCategoryGrouper` dentro del `Pipeline`).

**Paso a paso:**

1. **`df.copy()`** — no mutar el DataFrame de entrada (evita efectos colaterales).
2. **Eliminar columnas de `config.DROP_COLUMNS`** que estén presentes. Son columnas con
   **fuga** o que **no generalizan** (EDA §2):
   - `reservation_status` / `reservation_status_date` → contienen el desenlace de la reserva;
   - `required_car_parking_spaces`, `assigned_room_type` → se asignan en el **check-in** (fuga
     de información futura);
   - `arrival_date_year` → no generaliza a años nuevos.
3. **Saneo de filas imposibles** (EDA §8), construyendo una máscara booleana `keep`:
   - **sin huéspedes**: `adults + children + babies == 0` (rellenando nulos a 0);
   - **sin noches**: `stays_in_week_nights + stays_in_weekend_nights == 0`;
   - **`adr` absurdo**: negativo o `>= 5400` (tarifa diaria desorbitada).
4. **`df.loc[keep].reset_index(drop=True)`** — se queda con las filas válidas y reindexa.
5. **Devuelve** el DataFrame listo para separar en `X` / `y` (`get_feature_target`) y partir
   de forma estratificada (`split_data`).

### Las líneas clave de la máscara

`keep` empieza como `pd.Series(True, index=df.index)` (una fila por registro, todas a `True`);
cada regla le hace un **AND** (`&=`), así que solo puede **quitar** filas, nunca re-añadir.

```python
keep &= df[guest_cols].fillna(0).sum(axis=1) > 0      # sin huéspedes
```
`df[guest_cols]` (adults/children/babies) → `.fillna(0)` (un nulo cuenta como 0) → `.sum(axis=1)`
suma **a lo ancho** (huéspedes por fila) → `> 0` da `True` si hay alguien. Descarta reservas con
**0 personas**.

```python
nights = df["stays_in_week_nights"].fillna(0) + df["stays_in_weekend_nights"].fillna(0)
keep &= nights > 0                                    # sin noches
```
El `fillna(0)` es clave: en pandas `NaN + 3 = NaN` y `NaN > 0` es `False`; sin él, una reserva
con una columna de noches nula se borraría por error. Solo caen las que suman **0 noches**.

```python
keep &= ~((df["adr"] < 0) | (df["adr"] >= 5400))      # adr inválido
```
`(adr < 0) | (adr >= 5400)` marca el `adr` **malo** (negativo o disparatado); `~(...)` lo niega →
`True` donde es **válido** (por las leyes de De Morgan, `adr >= 0` **y** `adr < 5400`).

```python
df = df.loc[keep].reset_index(drop=True)
```
`df.loc[keep]` → indexado booleano (solo las filas `True`); `.reset_index(drop=True)` renumera el
índice (que quedó con huecos al borrar filas) y **descarta** el viejo en vez de guardarlo como
columna.

Todo es **vectorizado** (sin bucles de Python), por eso es rápido sobre las ~119 000 filas.

Lo importante no es solo *qué* filtra, sino **dónde**: al ir antes del *split* y no aprender
nada, garantiza que la partición y el *fit-on-train* posteriores sean **honestos**.

---

> **El patrón común.** Las cuatro piezas responden a dos principios que recorren todo el `src`:
> (1) **fit-on-train / sin fuga** — lo que aprende de los datos vive en el `Pipeline` y se ajusta
> solo con `train`; lo determinista va aparte; (2) **runtime esbelto** — TensorFlow (y MLflow,
> SHAP…) se importan de forma perezosa para no cargarlos al servir. Ver también
> [`arquitectura.md`](arquitectura.md) y [`informe_final.md`](informe_final.md).
